# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project.
# TorchNEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# TorchNEP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with TorchNEP.  If not, see <http://www.gnu.org/licenses/>.

r"""
Extrapolation grade of atomic environments (MaxVol active set).

An atom of element ``e`` is represented by the gradient of its NEP energy
with respect to that element's network parameters,

    b = dE_i / d(w0, b0, w1)          (K = H * (D + 2) numbers per atom),

i.e. the features of the NEP energy linearised in those parameters. For
every element the training set gives a matrix of such rows; its active set
is the set of rows spanning the largest volume (MaxVol). A new environment
is written in the basis of the active rows, ``b = c A``, and its
extrapolation grade is ``gamma = max_j |c_j|``: gamma <= 1 inside the region
spanned by the training set, gamma > 1 outside (replacing active row j by b
would multiply the volume by |c_j|).

The rows are numerically far from full rank — their singular values decay
smoothly over many orders of magnitude — so the grade is defined in the
leading subspace of each element: the eigenvectors of the training Gram
matrix ``B^T B`` whose singular value is above ``rcond`` times the largest.
Rows are projected onto that subspace and whitened (unit variance per
direction), which keeps the MaxVol matrices well conditioned. What the
projection drops is reported separately as ``gamma_res``: the norm of the
part of b outside the subspace divided by the largest such norm met in the
training set (> 1: more weight outside the training subspace than any
training atom).

Workflow: :func:`build_active_set` (training set -> active set file, once
per model), then :func:`select_structures` (greedy D-optimal choice of new
structures: it grades the candidate frames itself, visits them from the
highest grade down and takes a structure only if it still extends the active
set after the structures taken before it were added, so near-duplicates are
skipped). :func:`compute_gamma` only grades, to inspect structures (per
frame, per atom, gamma_res) without choosing. The ``*_sharded`` variants run
the same on several GPUs / nodes (``torchrun`` or ``srun``) and give the
same guarantees; with one process they are the plain functions.

Theory: Podryabinkin & Shapeev, Comput. Mater. Sci. 140, 171 (2017);
Gubaev et al., Comput. Mater. Sci. 156, 148 (2019); Podryabinkin et al.,
J. Chem. Phys. 159, 084112 (2023); Lysogorskiy et al., Phys. Rev. Materials
7, 043801 (2023); MaxVol: Goreinov et al., in Matrix Methods: Theory,
Algorithms and Applications (2010). The subspace / gamma_res treatment and
the re-graded greedy choice are this implementation's own. Not compatible
with GPUMD's ``compute_extrapolation`` (see the guide).
"""

import copy
import hashlib
import os
import pickle
import time

import numpy as np
import torch

from . import ops
from .nep import NEPCalculator
from .predict import _Progress, _chunk_bounds, _pick_device

_FORMAT = "torchnep-active-set"
_VERSION = 1
_F64 = torch.float64


# ---------------------------------------------------------------------------
# Per-atom parameter gradients
# ---------------------------------------------------------------------------

def b_vectors(q, w0, b0, w1):
    """Gradient of the atomic energy with respect to one element's network.

    ``q`` (n, D): scaled descriptors (``q * q_scaler``) of atoms of that
    element; ``w0`` (H, D), ``b0`` (H,), ``w1`` (H,): its network in the
    nep.txt layout (``E = w1 . tanh(w0 q - b0) - b1``). Returns (n, H*(D+2)),
    per neuron ``[dE/dw0 (D numbers), dE/db0, dE/dw1]``.
    """
    n, D = q.shape
    H = w1.shape[0]
    h = torch.tanh(torch.addmm(-b0, q, w0.T))
    g = w1 * (1.0 - h * h)
    out = torch.empty(n, H, D + 2, dtype=q.dtype, device=q.device)
    torch.mul(g.unsqueeze(2), q.unsqueeze(1), out=out[:, :, :D])
    out[:, :, D] = -g
    out[:, :, D + 1] = h
    return out.view(n, H * (D + 2))


def _load_calculators(model_file, device, dtype):
    """(float64 calculator, calculator for descriptors in ``dtype``): the
    network weights and the model fingerprint always come from the float64
    parse, the descriptors may run in float32."""
    calc = NEPCalculator(model_file, dtype=_F64, device=device)
    dt = _F64 if dtype == "float64" else torch.float32
    if dt == _F64:
        return calc, calc
    desc = copy.copy(calc)
    desc.dtype = dt
    for k, v in vars(calc).items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            setattr(desc, k, v.to(dt))
        elif isinstance(v, dict):
            setattr(desc, k, {kk: vv.to(dt) if isinstance(vv, torch.Tensor) else vv
                              for kk, vv in v.items()})
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            setattr(desc, k, [vv.to(dt) for vv in v])
    return calc, desc


def _model_fingerprint(calc):
    """SHA-1 of the parameters the grades depend on (networks, descriptor
    coefficients, q_scaler), so an active set is never used with another model."""
    h = hashlib.sha1()
    tensors = list(calc.w0) + list(calc.b0) + list(calc.w1) + [calc.c2, calc.q_scaler]
    if getattr(calc, "c3", None) is not None:
        tensors.append(calc.c3)
    for t in tensors:
        h.update(t.detach().to("cpu", _F64).numpy().tobytes())
    return h.hexdigest()


def _free_bytes(device):
    device = torch.device(device)
    if device.type == "cuda":
        return int(torch.cuda.mem_get_info(device)[0])
    return 8 * 1024 ** 3


# ---------------------------------------------------------------------------
# Several processes (torchrun / srun)
# ---------------------------------------------------------------------------

class _Comm:
    """Collectives of a torchrun / srun launch. With one process every call
    returns its input, so the single-process and sharded paths share code."""

    def __init__(self, dist=None, rank=0, world=1, device=None):
        self.dist, self.rank, self.world, self.dev = dist, rank, world, device

    @property
    def active(self):
        return self.world > 1

    def owner(self, t):
        """Rank that computes element ``t`` (elements are dealt round-robin)."""
        return t % self.world

    def bcast(self, t, src):
        if not self.active:
            return t
        c = t.to(self.dev).contiguous()
        self.dist.broadcast(c, src)
        return c.to(t.device)

    def allreduce(self, t, op="sum"):
        if not self.active:
            return t
        c = t.to(self.dev).contiguous()
        self.dist.all_reduce(c, op=self.dist.ReduceOp.MAX if op == "max" else self.dist.ReduceOp.SUM)
        return c.to(t.device)

    def reduce(self, t, dst):
        """Sum of ``t`` over the ranks, on rank ``dst`` (the others keep theirs)."""
        if not self.active:
            return t
        c = t.to(self.dev).contiguous()
        self.dist.reduce(c, dst)
        return c.to(t.device)

    def allgather_rows(self, t):
        """The rows of ``t`` (n_rank, ...) of every rank, concatenated in rank order."""
        if not self.active:
            return t
        c = t.to(self.dev).contiguous()
        n = torch.tensor([c.shape[0]], dtype=torch.long, device=self.dev)
        ns = [torch.zeros_like(n) for _ in range(self.world)]
        self.dist.all_gather(ns, n)
        ns = [int(k) for k in ns]
        pad = torch.zeros((max(ns),) + tuple(c.shape[1:]), dtype=c.dtype, device=self.dev)
        pad[:c.shape[0]] = c
        parts = [torch.empty_like(pad) for _ in range(self.world)]
        self.dist.all_gather(parts, pad)
        return torch.cat([p[:k] for p, k in zip(parts, ns)]).to(t.device)

    def gather_objects(self, obj):
        """Every rank's ``obj`` on rank 0, one rank at a time (a list in rank
        order; None elsewhere) — rank 0 never holds more than one part in transit."""
        if not self.active:
            return [obj]
        if self.rank == 0:
            out = [obj]
            for r in range(1, self.world):
                n = torch.zeros(1, dtype=torch.long, device=self.dev)
                self.dist.recv(n, src=r)
                buf = torch.empty(int(n), dtype=torch.uint8, device=self.dev)
                self.dist.recv(buf, src=r)
                out.append(pickle.loads(buf.cpu().numpy().tobytes()))
                del buf
            return out
        b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        t = torch.frombuffer(bytearray(b), dtype=torch.uint8).to(self.dev)
        self.dist.send(torch.tensor([t.numel()], dtype=torch.long, device=self.dev), dst=0)
        self.dist.send(t, dst=0)
        return None

    def bcast_object(self, obj, src=0):
        if not self.active:
            return obj
        box = [obj]
        self.dist.broadcast_object_list(box, src=src)
        return box[0]

    def allgather_objects(self, obj):
        """Every rank's (small) ``obj``: a list in rank order on every rank."""
        if not self.active:
            return [obj]
        out = [None] * self.world
        self.dist.all_gather_object(out, obj)
        return out

    def barrier(self):
        if self.active:
            self.dist.barrier()


def _comm_setup(device):
    """(comm, device): the process group of a torchrun / srun launch
    (``WORLD_SIZE`` > 1, initialised once, one GPU per process), or a
    single-process comm and the requested device."""
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world <= 1:
        return _Comm(), torch.device(_pick_device(device))
    import torch.distributed as dist
    from .train_sharded import _register_pg_atexit
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cuda = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if cuda else 0
    if cuda:
        gpu = local_rank % max(1, n_gpus)
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    if not dist.is_initialized():
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE",
                                         os.environ.get("SLURM_NTASKS_PER_NODE", world)))
        backend = "nccl" if cuda and local_world <= n_gpus else "gloo"
        # the owners' MaxVol merges keep the other ranks waiting in a
        # collective; same timeout as the sharded trainer
        from datetime import timedelta
        timeout = timedelta(minutes=float(os.environ.get("TORCHNEP_DIST_TIMEOUT_MIN", 180)))
        dist.init_process_group(backend=backend, timeout=timeout,
                                device_id=device if cuda else None)
        _register_pg_atexit()
    comm_dev = device if dist.get_backend() == "nccl" else torch.device("cpu")
    return _Comm(dist, dist.get_rank(), dist.get_world_size(), comm_dev), device


def _index(comm, xyz_file):
    """(offsets, natoms) of ``xyz_file``: indexed once on rank 0, broadcast."""
    from .data import index_xyz
    if not comm.active:
        return index_xyz(xyz_file)
    n = torch.zeros(1, dtype=torch.long)
    if comm.rank == 0:
        off, nat = index_xyz(xyz_file)
        n[0] = len(off)
    n = int(comm.bcast(n, 0))
    idx = torch.zeros(2, n, dtype=torch.long)
    if comm.rank == 0:
        idx[0], idx[1] = torch.from_numpy(off), torch.from_numpy(nat)
    idx = comm.bcast(idx, 0)
    return idx[0].numpy(), idx[1].numpy()


def _parse_workers(comm):
    """Parser processes of this rank: its CPUs shared among the ranks of the
    node whose CPUs overlap them (``torchrun``: every rank sees all the
    node's CPUs; ``srun`` binds each rank to its own). The environment
    variable ``TORCHNEP_PREPROC_WORKERS`` overrides."""
    env = os.environ.get("TORCHNEP_PREPROC_WORKERS")
    if env is not None:
        return max(1, int(env))
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:                    # not Linux
        return max(1, (os.cpu_count() or 1) // comm.world)
    import socket
    host, mine = socket.gethostname(), set(cpus)
    sharing = sum(1 for h, c in comm.allgather_objects((host, cpus))
                  if h == host and mine.intersection(c))
    return max(1, len(cpus) // max(1, sharing))


def _split_by_atoms(frame_ids, natoms, world):
    """``frame_ids`` (ascending) cut into ``world`` contiguous parts of about
    equal atom counts (each rank reads one byte range of the file)."""
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    cum = np.concatenate([[0], np.cumsum(natoms[frame_ids])])
    edges = np.searchsorted(cum, cum[-1] * np.arange(world + 1) / world)
    edges[0], edges[-1] = 0, len(frame_ids)
    return [frame_ids[edges[k]:edges[k + 1]] for k in range(world)]


# ---------------------------------------------------------------------------
# Streamed descriptors
# ---------------------------------------------------------------------------

def _read_block(args):
    """Pool worker: parse the frames at byte ``offsets`` into flat arrays —
    atom types (N,), positions wrapped into the cell (N, 3), cells (B, 3, 3)
    and atom counts (B,)."""
    from .data import read_xyz_at, wrap_positions
    xyz_file, offsets, energy_key, type_names, np_dtype = args
    frames = read_xyz_at(xyz_file, offsets, energy_key=energy_key)
    index = {s: t for t, s in enumerate(type_names)}
    types, pos = [], []
    cells = np.empty((len(frames), 3, 3), dtype=np_dtype)
    nat = np.empty(len(frames), dtype=np.int64)
    for k, f in enumerate(frames):
        try:
            types.append(np.fromiter((index[s] for s in f["species"]), np.int64,
                                     len(f["species"])))
        except KeyError as err:
            raise ValueError(f"element {err.args[0]} (frame at byte {offsets[k]} of "
                             f"{xyz_file}) is not an element of the model") from None
        cell = f["cell"].astype(np_dtype)
        p, _ = wrap_positions(f["positions"].astype(np_dtype), cell)
        pos.append(p.astype(np_dtype, copy=False))
        cells[k], nat[k] = cell, f["natoms"]
    if not frames:
        return (np.zeros(0, np.int64), np.zeros((0, 3), np_dtype), cells, nat)
    return np.concatenate(types), np.concatenate(pos), cells, nat


def _batch_descriptors(calc, at, pos, cells, nat, backend):
    """Scaled descriptors (N, D) of a batch of frames on the device (atom
    types, wrapped positions, cells, host atom counts): device neighbor
    search, then the NEP descriptors. Also returns the number of pairs."""
    from .train import search_neighbors_batched
    dt = calc.dtype
    N = pos.shape[0]
    off = np.concatenate([[0], np.cumsum(nat)[:-1]])
    pi, pj, rij, d = search_neighbors_batched(pos, cells, nat, off,
                                              max(calc.rc_radial, calc.rc_angular), dt)
    rc_r, rc_a = calc.cutoff_args()
    m_r = d < ops.pair_cutoff(rc_r, at, pi, pj)
    m_a = d < ops.pair_cutoff(rc_a, at, pi, pj)
    pi_r, pj_r, d_r = pi[m_r], pj[m_r], d[m_r]
    pi_a, pj_a, rij_a, da = pi[m_a], pj[m_a], rij[m_a], d[m_a]
    del pi, pj, rij, d
    fk_r = ops.chebyshev_basis(d_r, ops.pair_cutoff(rc_r, at, pi_r, pj_r),
                               calc.basis_size_radial)
    if rij_a.shape[0] > 0:
        fk_a = ops.chebyshev_basis(da, ops.pair_cutoff(rc_a, at, pi_a, pj_a),
                                   calc.basis_size_angular)
        inv = 1.0 / da.clamp(min=1e-10)
        blm = ops.angular_basis(rij_a[:, 0] * inv, rij_a[:, 1] * inv, rij_a[:, 2] * inv,
                                calc.l_max_3b)
    else:
        fk_a = torch.zeros(0, calc.basis_size_angular + 1, dtype=dt, device=pos.device)
        blm = torch.zeros(0, calc.num_lm, dtype=dt, device=pos.device)
    q = ops.compute_descriptors_cached(
        fk_r, fk_a, blm, pi_r, pj_r, pi_a, pj_a, at, N, calc.c2, getattr(calc, "c3", None),
        calc.n_max_radial, calc.n_max_angular, calc.l_max_3b,
        calc.has_q_222, calc.has_q_1111, calc.has_q_112,
        calc.num_lm, calc._c3b, calc._c4b, calc._c5b, calc._c4b2,
        dt, pos.device, backend=backend,
        has_q_123=calc.has_q_123, has_q_233=calc.has_q_233, has_q_134=calc.has_q_134)
    return q * calc.q_scaler, pi_r.shape[0] + pi_a.shape[0]


class _DescriptorStream:
    """Scaled per-atom descriptors of the frames of an xyz file, chunk by chunk.

    Worker processes parse the next chunk while the device computes the
    current one (neighbor search and descriptors both run on the device).
    Host memory is bounded by the chunk (``chunk_atoms``), device memory by
    the pair budget of one descriptor batch (sized from free memory, halved
    on OOM). ``index`` = (offsets, natoms) skips indexing the file;
    ``workers``: parser processes (default: all CPUs of the process)."""

    def __init__(self, calc, xyz_file, chunk_atoms=None, energy_key="energy", index=None,
                 workers=None):
        from .data import index_xyz
        from .train import make_preproc_pool
        self.calc, self.xyz_file, self.device = calc, xyz_file, calc.device
        self.np_dtype = np.float64 if calc.dtype == _F64 else np.float32
        self.energy_key = energy_key
        if chunk_atoms is None:
            chunk_atoms = int(os.environ.get("TORCHNEP_PREDICT_CHUNK_ATOMS", 200_000))
        self.chunk_atoms = max(1, int(chunk_atoms))
        self.offsets, self.natoms = index if index is not None else index_xyz(xyz_file)
        self.backend = ops.resolve_backend("auto", num_types=calc.num_types,
                                           device_type=self.device.type)
        self.workers = _parse_workers(_Comm()) if workers is None else max(1, int(workers))
        self.pool = (make_preproc_pool(self.workers)
                     if len(self.offsets) >= 64 and self.workers > 1 else None)
        self.max_pairs = None
        self.pairs_per_atom = None            # largest seen, sizes the batches
        self._cache = None                    # (frame ids, [(ids, nat, types, q) on the host])
        budget = os.environ.get("TORCHNEP_GAMMA_CACHE_GB")
        if budget is None:
            from .nep import _available_memory_bytes
            self.cache_bytes = int(0.25 * _available_memory_bytes(torch.device("cpu")))
        else:
            self.cache_bytes = int(float(budget) * 1e9)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
        self._cache = None

    def release_uncached(self):
        """Stop the parser workers when every later pass is served from the cache."""
        if self._cache is not None and self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def _pair_budget(self):
        c = self.calc
        esize = 8 if c.dtype == _F64 else 4
        per_pair = esize * (3 * (c.n_max_angular + 1) * c.num_lm + 2 * c.num_lm
                            + 2 * (c.basis_size_angular + 1) + 8) + 24
        return max(10_000, int(0.3 * _free_bytes(self.device) / (2.0 * per_pair)))

    def _parse(self, ids):
        """Start parsing frames ``ids``: an async pool job, or the arguments
        to parse them here (no pool)."""
        type_names = list(self.calc.type_names)
        if self.pool is None:
            return None, (self.xyz_file, self.offsets[ids], self.energy_key, type_names,
                          self.np_dtype)
        parts = np.array_split(ids, max(1, min(4 * self.workers, len(ids) // 16)))
        return self.pool.map_async(_read_block, [
            (self.xyz_file, self.offsets[p], self.energy_key, type_names, self.np_dtype)
            for p in parts if len(p)]), None

    @staticmethod
    def _parsed(job):
        pending, args = job
        parts = pending.get() if pending is not None else [_read_block(args)]
        return [np.concatenate(x) for x in zip(*parts)]

    def _descriptors(self, types, pos, cells, nat):
        """Descriptors of one chunk, in device batches of about ``max_pairs`` pairs."""
        q = torch.empty(pos.shape[0], self.calc.dim, dtype=self.calc.dtype, device=self.device)
        start = np.concatenate([[0], np.cumsum(nat)])
        f0 = 0
        while f0 < len(nat):
            # the pair count is known only after the search: batch by atoms
            # with the largest pairs per atom seen so far (20k atoms at first)
            atoms = (20_000 if self.pairs_per_atom is None
                     else max(1, int(self.max_pairs / (1.2 * self.pairs_per_atom))))
            f1 = max(f0 + 1, int(np.searchsorted(start, start[f0] + atoms, side="right")) - 1)
            f1 = min(f1, len(nat))
            a0, a1 = int(start[f0]), int(start[f1])
            try:
                with torch.no_grad():
                    qb, pairs = _batch_descriptors(self.calc, types[a0:a1], pos[a0:a1],
                                                   cells[f0:f1], nat[f0:f1], self.backend)
            except torch.OutOfMemoryError:
                if f1 - f0 == 1 and self.max_pairs < 10_000:
                    raise
                torch.cuda.empty_cache()
                self.max_pairs = max(1, self.max_pairs // 2)
                if self.pairs_per_atom is None:
                    self.pairs_per_atom = self.max_pairs / 10_000
                continue
            q[a0:a1] = qb
            ppa = pairs / max(1, a1 - a0)
            self.pairs_per_atom = ppa if self.pairs_per_atom is None else max(
                self.pairs_per_atom, ppa)
            f0 = f1
        return q

    def chunks(self, frame_ids=None, progress=None, cache=False):
        """Yield ``(frame_ids, natoms, types, q)`` per chunk: frame indices
        into the file (ascending), their atom counts, and on the device the
        per-atom types (N,) and scaled descriptors (N, D), atoms in file order.
        ``cache=True``: keep the chunks in host memory (when they fit in the
        budget) and serve the same ``frame_ids`` from there next time."""
        if frame_ids is None:
            frame_ids = np.arange(len(self.offsets))
        frame_ids = np.sort(np.asarray(frame_ids, dtype=np.int64))
        if len(frame_ids) == 0:
            return
        if cache and self._cache is not None and np.array_equal(self._cache[0], frame_ids):
            for ids, nat, types, q in self._cache[1]:
                yield (ids, nat, types.to(self.device, non_blocking=True),
                       q.to(self.device, non_blocking=True))
                if progress is not None:
                    progress.update(len(ids))
            return
        if cache:
            self._cache = None                # another frame list: free the old cache first
        store, used = ([], 0) if cache and self.cache_bytes > 0 else (None, 0)
        pin = self.device.type == "cuda"
        if self.max_pairs is None:
            self.max_pairs = self._pair_budget()
        bounds = _chunk_bounds(self.natoms[frame_ids], self.chunk_atoms, max_frames=50_000)
        job = self._parse(frame_ids[bounds[0][0]:bounds[0][1]])
        for k, (lo, hi) in enumerate(bounds):
            ids = frame_ids[lo:hi]
            at, pos, cells, nat = self._parsed(job)
            if k + 1 < len(bounds):            # parsed while the device works on this chunk
                job = self._parse(frame_ids[bounds[k + 1][0]:bounds[k + 1][1]])
            types = torch.from_numpy(at).to(self.device)
            q = self._descriptors(types, torch.from_numpy(pos).to(self.device),
                                  torch.from_numpy(cells).to(self.device), nat)
            nat = self.natoms[ids]
            if store is not None:
                used += q.numel() * q.element_size() + types.numel() * 8
                if used > self.cache_bytes:
                    store = None              # does not fit: no cache at all
                else:
                    qc, tc = q.cpu(), types.cpu()
                    store.append((ids, nat, tc.pin_memory() if pin else tc,
                                  qc.pin_memory() if pin else qc))
            yield ids, nat, types, q
            if progress is not None:
                progress.update(len(ids))
        if store is not None:
            self._cache = (frame_ids, store)


def _atom_sources(ids, nat, device):
    """(N, 2) int64 on ``device``: (frame index in file, atom index in frame)."""
    nat_t = torch.from_numpy(np.asarray(nat, dtype=np.int64)).to(device)
    ids_t = torch.from_numpy(np.asarray(ids, dtype=np.int64)).to(device)
    frame = torch.repeat_interleave(ids_t, nat_t)
    first = torch.repeat_interleave(torch.cumsum(nat_t, 0) - nat_t, nat_t)
    return torch.stack([frame, torch.arange(frame.shape[0], device=device) - first], 1)


def _frame_max(values, nat, device):
    """Per-frame maximum of per-atom ``values`` (atoms of consecutive frames)."""
    nat_t = torch.from_numpy(np.asarray(nat, dtype=np.int64)).to(device)
    frame = torch.repeat_interleave(torch.arange(len(nat), device=device), nat_t)
    out = torch.full((len(nat),), -1.0, dtype=values.dtype, device=device)
    return out.scatter_reduce_(0, frame, values, "amax"), frame


# ---------------------------------------------------------------------------
# Active set of one element
# ---------------------------------------------------------------------------

class _Element:
    """Subspace, whitening and MaxVol active rows of one element (float64).

    ``V`` (K, r) orthonormal basis of the leading subspace, ``scale`` (r,)
    per-direction whitening (1 / rms of the training rows), ``X`` (r, r) the
    whitened active rows, ``Ainv`` = X^-1, ``q`` (r, D) and ``src`` (r, 2) the
    descriptors and (frame, atom) origin of the active rows, ``res_max`` the
    largest out-of-subspace norm of a training row."""

    def __init__(self, name, w0, b0, w1):
        self.name = name
        self.w0, self.b0, self.w1 = w0, b0, w1
        self.V = self.scale = self.X = self.Ainv = self.q = self.src = None
        self.res_max = 0.0
        self.n_rows = 0
        self.swaps = 0
        self._since_inv = 0
        self._f32 = None

    @property
    def r(self):
        return 0 if self.V is None else self.V.shape[1]

    def project(self, q, batch=None):
        """Whitened coordinates (n, r) and out-of-subspace norm (n,) of rows
        given by their descriptors ``q`` (float64), in batches of ``batch`` rows."""
        n = q.shape[0]
        batch = batch or max(n, 1)
        x = torch.empty(n, self.r, dtype=_F64, device=q.device)
        res = torch.empty(n, dtype=_F64, device=q.device)
        for a in range(0, n, batch):
            B = b_vectors(q[a:a + batch], self.w0, self.b0, self.w1)
            y = torch.mm(B, self.V, out=x[a:a + batch])
            r2 = torch.linalg.vector_norm(B, dim=1) ** 2 - torch.linalg.vector_norm(y, dim=1) ** 2
            res[a:a + batch] = torch.sqrt(torch.clamp(r2, min=0.0))
            del B
        x.mul_(self.scale)
        return x, res

    def coefficients(self, x):
        return x @ self.Ainv

    def grade32(self, q):
        """(gamma, out-of-subspace norm) of rows in float32: one product with
        R = V diag(scale) A^-1 for the grade (within 1e-4 relative of the
        float64 grade), and the residual b - b V V^T built explicitly (within
        a few 1e-3; the float64 shortcut |b|^2 - |b V|^2 would cancel every digit)."""
        if self._f32 is None:
            R = (self.V * self.scale) @ self.Ainv
            self._f32 = (R.float(), self.V.float(), self.w0.float(), self.b0.float(),
                         self.w1.float())
        R, V, w0, b0, w1 = self._f32
        B = b_vectors(q.float(), w0, b0, w1)
        g = (B @ R).abs().amax(1)
        res = torch.linalg.vector_norm(B - (B @ V) @ V.T, dim=1)
        return g.to(_F64), res.to(_F64)

    def reinvert(self):
        self.Ainv = torch.linalg.inv(self.X)
        self._since_inv = 0
        self._f32 = None                      # float32 grading matrices are stale

    def init_from_rows(self, x, q, src):
        """First active rows: LU with partial pivoting picks r well-spread rows
        of ``x`` (n >= r). Directions these rows cannot span are dropped from
        the subspace. Returns the kept column mask."""
        keep = torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
        while True:
            try:
                LU, piv = torch.linalg.lu_factor(x[:, keep])
            except RuntimeError:              # backends without rectangular LU
                LU, piv = torch.linalg.lu_factor(x[:, keep].cpu())
                LU = LU.to(x.device)
            diag = torch.diagonal(LU).abs()
            bad = diag <= 1e-9 * diag.max()
            if not bool(bad.any()):
                break
            keep[torch.nonzero(keep).squeeze(1)[bad]] = False
        self.V, self.scale = self.V[:, keep].contiguous(), self.scale[keep]
        perm = np.arange(x.shape[0])
        for k, p in enumerate(piv.cpu().numpy() - 1):
            perm[k], perm[p] = perm[p], perm[k]
        sel = torch.from_numpy(perm[:self.r]).to(x.device)
        self.X = x[sel][:, keep].contiguous()
        self.q, self.src = q[sel].clone(), src[sel].clone()
        self.reinvert()
        return keep

    def maxvol(self, x, q, src, tol, C=None, pivot=None, reinvert_every=256):
        """Greedy MaxVol swaps of rows ``x`` (n, r) into the active set until no
        coefficient exceeds ``tol``; the row leaving the active set takes the
        slot of the row that entered, so it can come back. ``pivot`` = (a, b)
        lets only rows a..b-1 enter (all rows' coefficients stay current).
        ``x``, ``q``, ``src`` and ``C`` (precomputed coefficients) are
        updated in place. Returns (number of swaps, C)."""
        if C is None:
            C = self.coefficients(x)
        a, b = (0, C.shape[0]) if pivot is None else pivot
        r = C.shape[1]
        swaps = 0
        while True:
            flat = int(torch.argmax(C[a:b].abs()))
            i, j = divmod(flat, r)
            i += a
            piv = float(C[i, j])
            if abs(piv) <= tol:
                break
            v = C[i].clone()
            v[j] -= 1.0
            C.addr_(C[:, j].clone(), v, alpha=-1.0 / piv)
            self.Ainv.addr_(self.Ainv[:, j].clone(), v, alpha=-1.0 / piv)
            C[i] = v * (-1.0 / piv)
            C[i, j] += 1.0
            xi, qi, si = x[i].clone(), q[i].clone(), src[i].clone()
            x[i], q[i], src[i] = self.X[j], self.q[j], self.src[j]
            self.X[j], self.q[j], self.src[j] = xi, qi, si
            swaps += 1
            self._since_inv += 1
            if self._since_inv >= reinvert_every:
                self.reinvert()
                C.copy_(self.coefficients(x))
        if swaps:
            self.reinvert()
            C.copy_(self.coefficients(x))
        self.swaps += swaps
        return swaps, C

    def maxvol_screened(self, x, q, src, tol, max_rounds=50):
        """MaxVol over rows ``x`` that hands only the rows whose grade exceeds
        ``tol`` to :meth:`maxvol` — a swap then costs (rows above tol) x r
        instead of (all rows) x r. Rows swapped out go back into their slots,
        so they can return, and the grades are recomputed until no row
        exceeds ``tol``. Returns the number of swaps."""
        total = 0
        for _ in range(max_rounds):
            C = self.coefficients(x)
            hot = torch.nonzero(C.abs().amax(1) > tol).squeeze(1)
            if hot.numel() == 0:
                break
            xh, qh, sh = x[hot], q[hot], src[hot]
            n, _ = self.maxvol(xh, qh, sh, tol, C=C[hot])
            x[hot], q[hot], src[hot] = xh, qh, sh
            total += n
        return total

    def state(self):
        return {"V": self.V.cpu(), "scale": self.scale.cpu(), "X": self.X.cpu(),
                "q": self.q.cpu(), "src": self.src.cpu(), "res_max": self.res_max,
                "n_rows": self.n_rows, "swaps": self.swaps}


# ---------------------------------------------------------------------------
# Active set of a model
# ---------------------------------------------------------------------------

class ActiveSet:
    """MaxVol active sets of every element of a NEP model.

    Built by :func:`build_active_set`, reloaded with :meth:`load`; grades of
    descriptors with :meth:`gamma`. ``calc`` holds the float64 weights,
    ``desc_calc`` computes the descriptors (float64 or float32)."""

    def __init__(self, calc, desc_calc, rcond, tol):
        self.calc, self.desc_calc = calc, desc_calc
        self.device = calc.device
        self.rcond, self.tol = float(rcond), float(tol)
        self.fingerprint = _model_fingerprint(calc)
        self.elements = [_Element(n, calc.w0[t], calc.b0[t], calc.w1[t])
                         for t, n in enumerate(calc.type_names)]
        self.meta = {}

    @property
    def K(self):
        return self.calc.num_neurons * (self.calc.dim + 2)

    def save(self, path):
        torch.save({
            "format": _FORMAT, "version": _VERSION,
            "type_names": list(self.calc.type_names), "fingerprint": self.fingerprint,
            "rcond": self.rcond, "tol": self.tol, "meta": self.meta,
            "elements": {e.name: e.state() for e in self.elements if e.r > 0},
        }, path)

    @classmethod
    def load(cls, path, model_file, device=None, dtype="float64"):
        """Load an active set for ``model_file`` (the model it was built with)."""
        device = torch.device(_pick_device(device))
        d = torch.load(path, map_location="cpu", weights_only=False)
        if d.get("format") != _FORMAT:
            raise ValueError(f"{path} is not a torchnep active set")
        self = cls(*_load_calculators(model_file, device, dtype), d["rcond"], d["tol"])
        if d["fingerprint"] != self.fingerprint:
            raise ValueError(f"{path} was built with a different model than {model_file}")
        self.meta = d.get("meta", {})
        f64 = dict(device=device, dtype=_F64)
        for e in self.elements:
            s = d["elements"].get(e.name)
            if s is None:
                continue
            e.V, e.scale, e.X = s["V"].to(**f64), s["scale"].to(**f64), s["X"].to(**f64)
            e.q, e.src = s["q"].to(**f64), s["src"].to(device)
            e.res_max, e.n_rows, e.swaps = float(s["res_max"]), int(s["n_rows"]), int(s["swaps"])
            e.reinvert()
        return self

    def row_batch(self, r=None):
        """Rows per projection batch: B (n, K) plus a few (n, r) float64
        arrays within ~20% of the free device memory."""
        r = self.K if r is None else max(r, 1)
        per_row = 8 * (2 * self.K + 3 * r)
        return max(256, min(1 << 20, int(0.2 * _free_bytes(self.device) / per_row)))

    @torch.no_grad()
    def gamma(self, types, q, precision="float64"):
        """Per-atom grades of scaled descriptors ``q`` (N, D) with ``types`` (N,).

        Returns ``(gamma, gamma_res)`` (N,) float64 tensors; atoms of elements
        without an active set get ``inf``. ``precision="float32"`` grades in
        float32 (gamma within 1e-4, gamma_res within a few 1e-3 relative; faster
        on GPUs with slow float64)."""
        N = types.shape[0]
        g = torch.full((N,), float("inf"), dtype=_F64, device=self.device)
        gr = torch.full((N,), float("inf"), dtype=_F64, device=self.device)
        q = q.to(_F64)
        for t in torch.unique(types).tolist():
            e = self.elements[t]
            if e.r == 0:
                continue
            idx = torch.nonzero(types == t).squeeze(1)
            nb = self.row_batch(e.r)
            for a in range(0, idx.shape[0], nb):
                sub = idx[a:a + nb]
                if precision == "float32":
                    g[sub], res = e.grade32(q[sub])
                else:
                    x, res = e.project(q[sub])
                    g[sub] = e.coefficients(x).abs().amax(1)
                gr[sub] = res / e.res_max if e.res_max > 0 else res
        return g, gr


def _bcast_element(comm, e, src, K, D):
    """Copy element ``e``'s subspace and active rows from rank ``src`` to all."""
    if not comm.active:
        return
    dev = e.w0.device
    r = int(comm.bcast(torch.tensor([e.r if comm.rank == src else 0]), src))
    if comm.rank != src:
        e.V = torch.empty(K, r, dtype=_F64, device=dev)
        e.scale = torch.empty(r, dtype=_F64, device=dev)
        e.X = torch.empty(r, r, dtype=_F64, device=dev)
        e.q = torch.empty(r, D, dtype=_F64, device=dev)
        e.src = torch.empty(r, 2, dtype=torch.long, device=dev)
    e.V, e.scale, e.X = comm.bcast(e.V, src), comm.bcast(e.scale, src), comm.bcast(e.X, src)
    e.q, e.src = comm.bcast(e.q, src), comm.bcast(e.src, src)
    e.reinvert()                              # every rank inverts the same X


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _log_fn(verbose):
    def _log(msg):
        if verbose:
            print(msg, flush=True)
    return _log


def _gram_pass(aset, stream, frames, cap, gen, progress):
    """Pass 1 over ``frames``: per element the Gram matrix of its rows
    (float64, lazily allocated), their count and a uniform random sample of
    ``cap`` rows (random key, descriptors, source)."""
    dev, K, T = aset.device, aset.K, len(aset.elements)
    G, count, sample = [None] * T, torch.zeros(T, dtype=torch.long), [None] * T
    nb = aset.row_batch()
    with torch.no_grad():
        for ids, nat, types, q in stream.chunks(frames, progress, cache=True):
            src = _atom_sources(ids, nat, dev)
            q = q.to(_F64)
            for t in torch.unique(types).tolist():
                e = aset.elements[t]
                idx = torch.nonzero(types == t).squeeze(1)
                if G[t] is None:
                    G[t] = torch.zeros(K, K, dtype=_F64, device=dev)
                for a in range(0, idx.shape[0], nb):
                    B = b_vectors(q[idx[a:a + nb]], e.w0, e.b0, e.w1)
                    G[t].addmm_(B.T, B)
                    del B
                count[t] += idx.shape[0]
                new = (torch.rand(idx.shape[0], generator=gen, device=dev), q[idx], src[idx])
                if sample[t] is not None:
                    new = tuple(torch.cat([u, w]) for u, w in zip(sample[t], new))
                if new[0].shape[0] > cap:
                    top = torch.topk(new[0], cap).indices
                    new = tuple(u[top] for u in new)
                sample[t] = new
    return G, count, sample


def _merge_sample(comm, sample_t, cap, D, dev):
    """The ``cap`` rows with the largest random keys over all ranks' samples
    of one element (every rank gets them, in key order): (q, src)."""
    if sample_t is None:
        keys = torch.zeros(0, dtype=_F64, device=dev)
        rows = torch.zeros(0, D + 3, dtype=_F64, device=dev)
    else:
        keys = sample_t[0].to(_F64)
        rows = torch.cat([keys[:, None], sample_t[1], sample_t[2].to(_F64)], 1)
    if comm.active:
        allk = comm.allgather_rows(keys)
        if allk.numel() > cap:
            thr = torch.topk(allk, cap).values.min()
            rows = rows[keys >= thr]
        rows = comm.allgather_rows(rows)
    rows = rows[torch.argsort(rows[:, 0], descending=True)]
    return rows[:, 1:1 + D].contiguous(), rows[:, 1 + D:].round().long()


def _seed_element(e, G, n_rows, q_s, src_s, rcond, init_rows, tol, nb):
    """Subspace of element ``e`` from its Gram matrix and first active rows
    from the row sample: LU pivoting, then MaxVol over the whole sample."""
    lam, vec = torch.linalg.eigh(G)
    lam, vec = lam.flip(0).clamp(min=0.0), vec.flip(1)
    r = int((torch.sqrt(lam / lam[0]) > rcond).sum())
    r = min(r, q_s.shape[0])
    e.n_rows = int(n_rows)
    e.V = vec[:, :r].contiguous()
    e.scale = 1.0 / torch.sqrt(lam[:r] / e.n_rows)
    del vec
    x, res = e.project(q_s, nb)
    e.res_max = max(e.res_max, float(res.max()))
    n_init = min(x.shape[0], max(init_rows * r, r))
    keep = e.init_from_rows(x[:n_init], q_s[:n_init], src_s[:n_init])
    x = x[:, keep].contiguous()
    e.maxvol_screened(x, q_s.clone(), src_s.clone(), tol)


_SCREEN_MARGIN = 0.01    # float32 screening: rows graded above near*tol - this are regraded in float64


def _stream_pass(aset, stream, frames, tol, progress, collect_cap=None, near=0.65,
                 precision="float64"):
    """One pass over ``frames``. Without ``collect_cap``: MaxVol swaps into
    the active sets of the rows whose grade exceeds ``tol`` (rows above
    ``near`` x ``tol`` take part, so the swaps rarely push a row that was
    left out over the threshold: with 0.95 the check passes kept finding a
    few such rows and did not converge in 10 passes on two training sets,
    with 0.65 they converged in 6). With it: no swap, collect per element
    the rows above ``near`` x ``tol`` — at most ``collect_cap``, those with
    the largest grades — as (q, src). Returns (rows, number of rows above
    ``tol``, largest grade); tracks ``res_max``.

    ``precision="float32"`` grades every row in float32 first and regrades in
    float64 only the rows within ``_SCREEN_MARGIN`` of ``near`` x ``tol`` or
    above — every decision still uses the float64 grade."""
    dev = aset.device
    hot = [None] * len(aset.elements)
    n_above, g_max = 0, 0.0
    with torch.no_grad():
        for ids, nat, types, q in stream.chunks(frames, progress, cache=True):
            src = _atom_sources(ids, nat, dev)
            q = q.to(_F64)
            for t in torch.unique(types).tolist():
                e = aset.elements[t]
                if e.r == 0:
                    continue
                idx = torch.nonzero(types == t).squeeze(1)
                nb = aset.row_batch(e.r)
                for a in range(0, idx.shape[0], nb):
                    sub = idx[a:a + nb]
                    if precision == "float32":
                        g32, res = e.grade32(q[sub])
                        e.res_max = max(e.res_max, float(res.max()))
                        g_max = max(g_max, float(g32.max()))
                        sub = sub[g32 > near * tol - _SCREEN_MARGIN]
                        if sub.numel() == 0:
                            continue
                    x, res = e.project(q[sub])
                    e.res_max = max(e.res_max, float(res.max()))
                    g = e.coefficients(x).abs().amax(1)
                    n_above += int((g > tol).sum())
                    g_max = max(g_max, float(g.max()))
                    cand = torch.nonzero(g > near * tol).squeeze(1)
                    if cand.numel() == 0:
                        continue
                    if collect_cap is None:
                        e.maxvol_screened(x[cand], q[sub][cand], src[sub][cand], tol)
                        continue
                    new = (g[cand], q[sub][cand], src[sub][cand])
                    if hot[t] is not None:
                        new = tuple(torch.cat([u, w]) for u, w in zip(hot[t], new))
                    if new[0].shape[0] > collect_cap:
                        top = torch.topk(new[0], collect_cap).indices
                        new = tuple(u[top] for u in new)
                    hot[t] = new
    return [None if h is None else (h[1], h[2]) for h in hot], n_above, g_max


def _merge_rows(aset, comm, rows, D):
    """Every rank's candidate rows of each element go to the element's owner,
    which swaps them into its active set by MaxVol; the result is broadcast.
    ``rows[t]`` = (q, src) or None. Returns the number of swaps."""
    dev, tol, K = aset.device, aset.tol, aset.K
    swaps = 0
    for t, e in enumerate(aset.elements):
        if e.r == 0:
            continue
        own = comm.owner(t)
        if rows[t] is None:
            payload = torch.zeros(0, D + 2, dtype=_F64, device=dev)
        else:
            payload = torch.cat([rows[t][0], rows[t][1].to(_F64)], 1)
        allrows = comm.allgather_rows(payload)
        n = 0
        if comm.rank == own and allrows.shape[0]:
            q_u, src_u = allrows[:, :D].contiguous(), allrows[:, D:].round().long()
            x_u, res = e.project(q_u, aset.row_batch(e.r))
            e.res_max = max(e.res_max, float(res.max()))
            n = e.maxvol_screened(x_u, q_u, src_u, tol)
        swaps += int(comm.allreduce(torch.tensor([n]), "sum"))
        _bcast_element(comm, e, own, K, D)
    return swaps


def _build(model_file, xyz_file, output_file, rcond, tol, sample_frames, init_rows,
           max_passes, device, dtype, chunk_atoms, energy_key, seed, verbose, comm,
           precision="float64"):
    from .train import _default_alloc_conf
    _default_alloc_conf()
    main = comm.rank == 0
    _log = _log_fn(verbose and main)
    t_total = time.time()
    aset = ActiveSet(*_load_calculators(model_file, device, dtype), rcond, tol)
    calc = aset.calc
    T, K, D = calc.num_types, aset.K, calc.dim
    offsets, natoms = _index(comm, xyz_file)
    stream = _DescriptorStream(aset.desc_calc, xyz_file, chunk_atoms, energy_key,
                               index=(offsets, natoms), workers=_parse_workers(comm))
    n_frames = len(offsets)
    _log(f"  active set: {n_frames} frames, {int(natoms.sum())} atoms, {T} elements, "
         f"K = {calc.num_neurons} x ({D} + 2) = {K}, rcond {rcond:g}, tol {tol:g}"
         + (f", {comm.world} processes" if comm.active else ""))

    # ---- pass 1: Gram matrices and a row sample (on a frame sample) --------
    t0 = time.time()
    rng = np.random.default_rng(seed)
    sample = np.arange(n_frames) if n_frames <= sample_frames else \
        np.sort(rng.choice(n_frames, sample_frames, replace=False))
    mine = _split_by_atoms(sample, natoms, comm.world)[comm.rank]
    cap = max(4096, 2 * K)                    # rows per element to seed the active set
    gen = torch.Generator(device=aset.device).manual_seed(seed + 7919 * comm.rank)
    progress = _Progress(len(mine), verbose and main, "Gram")
    G, count, rsample = _gram_pass(aset, stream, mine, cap, gen, progress)
    progress.close()
    count = comm.allreduce(count)
    _log(f"  pass 1 (Gram matrices, {len(sample)} frames): {time.time() - t0:.1f}s")

    # ---- subspaces and seed active sets, every element on its owner ---------
    # all the collectives first, then the owners seed their elements side by
    # side (a collective between two seeds would make them run one by one)
    t0 = time.time()
    nb = aset.row_batch()
    with torch.no_grad():
        own_work = []
        for t, e in enumerate(aset.elements):
            if int(count[t]) == 0:
                continue
            own = comm.owner(t)
            g = G[t] if G[t] is not None else torch.zeros(K, K, dtype=_F64, device=aset.device)
            G[t] = None
            g = comm.reduce(g, own)
            q_s, src_s = _merge_sample(comm, rsample[t], cap, D, aset.device)
            rsample[t] = None
            if comm.rank == own:
                own_work.append((e, g, count[t], q_s, src_s))
            del g, q_s, src_s
        while own_work:
            e, g, n, q_s, src_s = own_work.pop(0)
            _seed_element(e, g, n, q_s, src_s, rcond, init_rows, tol, nb)
            del g, q_s, src_s
        for t, e in enumerate(aset.elements):
            if int(count[t]):
                e.n_rows = int(count[t])
                _bcast_element(comm, e, comm.owner(t), K, D)
                _log(f"    {e.name:3s} rows {e.n_rows:9d}  rank {e.r:5d} / {K}")
    _log(f"  subspaces and seed active sets: {time.time() - t0:.1f}s")

    # ---- pass 2: MaxVol over every frame -----------------------------------
    t0 = time.time()
    mine = _split_by_atoms(np.arange(n_frames), natoms, comm.world)[comm.rank]
    progress = _Progress(len(mine), verbose and main, "MaxVol")
    _stream_pass(aset, stream, mine, tol, progress, precision=precision)
    progress.close()
    stream.release_uncached()
    if comm.active:                           # the ranks' active sets -> one
        merged = _merge_rows(aset, comm, [(e.q, e.src) if e.r else None
                                          for e in aset.elements], D)
        _log(f"  pass 2 (MaxVol on {comm.world} shards + merge, {merged} swaps): "
             f"{time.time() - t0:.1f}s")
    else:
        _log(f"  pass 2 (MaxVol, {n_frames} frames): {time.time() - t0:.1f}s")

    # ---- check passes: every training row inside the active set ------------
    converged, g_train = False, float("nan")
    cap_hot = max(4096, 4 * max(e.r for e in aset.elements))
    for p in range(max_passes):
        t0 = time.time()
        progress = _Progress(len(mine), verbose and main, f"check {p + 1}")
        rows, n_above, g_train = _stream_pass(aset, stream, mine, tol, progress,
                                              collect_cap=cap_hot, precision=precision)
        progress.close()
        n_above = int(comm.allreduce(torch.tensor([n_above]), "sum"))
        g_train = float(comm.allreduce(torch.tensor([g_train], dtype=_F64), "max"))
        if n_above == 0:
            _log(f"  check {p + 1}: every training atom has gamma <= {tol:g} "
                 f"(largest {g_train:.4f}; {time.time() - t0:.1f}s)")
            converged = True
            break
        swaps = _merge_rows(aset, comm, rows, D)
        _log(f"  check {p + 1}: {n_above} atoms above {tol:g} (largest {g_train:.4f}) -> "
             f"{swaps} swaps ({time.time() - t0:.1f}s)")
    if not converged:                         # where the last swaps left the training set
        progress = _Progress(len(mine), verbose and main, "final check")
        _, n_above, g_train = _stream_pass(aset, stream, mine, tol, progress, collect_cap=1,
                                           precision=precision)
        progress.close()
        n_above = int(comm.allreduce(torch.tensor([n_above]), "sum"))
        g_train = float(comm.allreduce(torch.tensor([g_train], dtype=_F64), "max"))
        converged = n_above == 0
        _log(f"  final check: {n_above} training atoms above {tol:g}, largest gamma "
             f"{g_train:.4f}" + ("" if converged else " (raise max_passes to tighten)"))
    stream.close()
    for e in aset.elements:                   # largest out-of-subspace norm of any rank
        e.res_max = float(comm.allreduce(torch.tensor([e.res_max], dtype=_F64), "max"))
        e.swaps = int(comm.allreduce(torch.tensor([e.swaps]), "sum"))
    aset.meta = {"xyz_file": os.path.abspath(xyz_file), "n_frames": int(n_frames),
                 "sample_frames": int(len(sample)), "model_file": os.path.abspath(model_file),
                 "converged": converged, "train_gamma_max": g_train}
    if main:
        aset.save(output_file)
    comm.barrier()
    _log(f"  TOTAL: {time.time() - t_total:.1f}s -> {output_file}")
    return aset


def build_active_set(
    model_file: str,
    xyz_file: str,
    output_file: str = "active_set.pt",
    rcond: float = 1e-4,
    tol: float = 1.01,
    sample_frames: int = 50_000,
    init_rows: int = 4,
    max_passes: int = 10,
    device: str = None,
    dtype: str = "float64",
    precision: str = "float64",
    chunk_atoms: int = None,
    energy_key: str = "energy",
    seed: int = 0,
    verbose: bool = True,
):
    """Build the MaxVol active set of a NEP model from its training set.

    Streams ``xyz_file``:

    1. over ``sample_frames`` random frames (all frames if fewer): the Gram
       matrix ``B^T B`` of every element, whose eigenvectors with singular
       value above ``rcond`` x the largest span the subspace of the grade,
       and a random sample of rows that seeds the active set (LU pivoting on
       ``init_rows`` x r rows, then MaxVol over the whole sample);
    2. over all frames: MaxVol swaps of every row whose grade exceeds ``tol``;
    3. check passes (at most ``max_passes``): rows that exceed ``tol`` against
       the final active set (a row passed early can, after later swaps) are
       swapped in, until a pass finds none — then every training atom has a
       grade <= ``tol``. The log (and ``meta``) reports whether it converged
       and the largest training grade.

    The descriptors of pass 2 are kept in host memory for the check passes
    when they fit in 25% of the free memory (``TORCHNEP_GAMMA_CACHE_GB`` sets
    another budget, 0 disables it).

    ``dtype`` is the descriptor precision; subspaces and MaxVol always run
    in float64. ``precision="float32"`` screens the rows of passes 2 and 3 in
    float32 and regrades in float64 only those near or above the threshold —
    the same decisions, many times faster on GPUs with slow float64 (most
    consumer cards). Saves the :class:`ActiveSet` to ``output_file`` and
    returns it.
    """
    comm, dev = _Comm(), torch.device(_pick_device(device))
    return _build(model_file, xyz_file, output_file, rcond, tol, sample_frames, init_rows,
                  max_passes, dev, dtype, chunk_atoms, energy_key, seed, verbose, comm,
                  precision)


def build_active_set_sharded(
    model_file: str,
    xyz_file: str,
    output_file: str = "active_set.pt",
    rcond: float = 1e-4,
    tol: float = 1.01,
    sample_frames: int = 50_000,
    init_rows: int = 4,
    max_passes: int = 10,
    dtype: str = "float64",
    precision: str = "float64",
    chunk_atoms: int = None,
    energy_key: str = "energy",
    seed: int = 0,
    verbose: bool = True,
):
    """Multi-GPU / multi-node :func:`build_active_set`: one process per GPU,
    launched with ``torchrun`` or ``srun`` (which set RANK / LOCAL_RANK /
    WORLD_SIZE / MASTER_ADDR / MASTER_PORT).

    Every rank streams its own contiguous share of the frames. The Gram
    matrices are summed on the rank that owns each element (elements are
    dealt round-robin), which computes its subspace and seed active set; the
    ranks then run MaxVol on their shares, the owners merge the ranks' active
    rows and the check passes run as in :func:`build_active_set`, so the same
    guarantee holds: every training atom has a grade <= ``tol``. Rank 0 writes
    ``output_file``; every rank returns the same :class:`ActiveSet`. With one
    process this is :func:`build_active_set`.
    """
    comm, dev = _comm_setup(None)
    return _build(model_file, xyz_file, output_file, rcond, tol, sample_frames, init_rows,
                  max_passes, dev, dtype, chunk_atoms, energy_key, seed, verbose, comm,
                  precision)


# ---------------------------------------------------------------------------
# Grades of structures
# ---------------------------------------------------------------------------

def _as_active_set(active_set, model_file, device, dtype):
    if isinstance(active_set, ActiveSet):
        return active_set
    return ActiveSet.load(active_set, model_file, device=device, dtype=dtype)


def _grade_frames(aset, stream, frames, per_atom, progress, window=None, precision="float64"):
    """Grades of ``frames``: dict of numpy arrays ``frames``, ``gamma``,
    ``gamma_res``, ``atom`` (per frame) and, with ``per_atom``, ``gamma_atoms``
    / ``gamma_res_atoms`` (the frames' atoms in order). ``window`` =
    (gamma_min, gamma_max): also the rows of the frames whose grade lies in
    (gamma_min, gamma_max] — ``rows_q``, ``rows_t``, ``rows_src`` (CPU tensors)."""
    dev = aset.device
    out = {k: [] for k in ("frames", "gamma", "gamma_res", "atom", "gamma_atoms",
                           "gamma_res_atoms", "rows_q", "rows_t", "rows_src")}
    for ids, nat, types, q in stream.chunks(frames, progress):
        g, gr = aset.gamma(types, q, precision)
        gmax, frame = _frame_max(g, nat, dev)
        grmax, _ = _frame_max(gr, nat, dev)
        nat_t = torch.from_numpy(np.asarray(nat)).to(dev)
        first = torch.cumsum(nat_t, 0) - nat_t
        hit = torch.nonzero(g == gmax[frame]).squeeze(1)
        arg = torch.zeros(len(ids), dtype=torch.long, device=dev)
        arg.scatter_reduce_(0, frame[hit], hit, "amin", include_self=False)
        out["frames"].append(np.asarray(ids))
        out["gamma"].append(gmax.cpu().numpy())
        out["gamma_res"].append(grmax.cpu().numpy())
        out["atom"].append((arg - first).cpu().numpy())
        if per_atom:
            out["gamma_atoms"].append(g.cpu().numpy())
            out["gamma_res_atoms"].append(gr.cpu().numpy())
        if window is not None:
            ok = gmax > window[0]
            if window[1] is not None:
                ok &= gmax <= window[1]
            rows = ok[frame]
            if bool(rows.any()):
                out["rows_q"].append(q[rows].to(_F64).cpu())
                out["rows_t"].append(types[rows].cpu())
                out["rows_src"].append(_atom_sources(ids, nat, dev)[rows].cpu())
    res = {}
    for k in ("frames", "gamma", "gamma_res", "atom", "gamma_atoms", "gamma_res_atoms"):
        if out[k]:
            res[k] = np.concatenate(out[k])
    if window is not None:
        res["rows"] = None if not out["rows_q"] else (
            torch.cat(out["rows_q"]), torch.cat(out["rows_t"]), torch.cat(out["rows_src"]))
    return res


def _assemble_gamma(parts, natoms, per_atom):
    """Per-frame (and per-atom) arrays in file order from graded parts."""
    n = len(natoms)
    out = {"gamma": np.empty(n), "gamma_res": np.empty(n),
           "atom": np.empty(n, dtype=np.int64), "natoms": np.asarray(natoms).copy()}
    if per_atom:
        n_atoms = int(np.sum(natoms))
        out["gamma_atoms"], out["gamma_res_atoms"] = np.empty(n_atoms), np.empty(n_atoms)
        atom_start = np.concatenate([[0], np.cumsum(natoms)])
    for p in parts:
        if "frames" not in p:
            continue
        ids = p["frames"]
        out["gamma"][ids], out["gamma_res"][ids], out["atom"][ids] = \
            p["gamma"], p["gamma_res"], p["atom"]
        if per_atom:
            nat = natoms[ids]
            dst = np.repeat(atom_start[ids], nat) + (np.arange(int(nat.sum()))
                                                     - np.repeat(np.cumsum(nat) - nat, nat))
            out["gamma_atoms"][dst] = p["gamma_atoms"]
            out["gamma_res_atoms"][dst] = p["gamma_res_atoms"]
    return out


def _gamma(model_file, active_set, xyz_file, output_file, per_atom, device, dtype,
           chunk_atoms, energy_key, verbose, comm, precision="float64"):
    main = comm.rank == 0
    _log = _log_fn(verbose and main)
    t_total = time.time()
    aset = _as_active_set(active_set, model_file, device, dtype)
    offsets, natoms = _index(comm, xyz_file)
    stream = _DescriptorStream(aset.desc_calc, xyz_file, chunk_atoms, energy_key,
                               index=(offsets, natoms), workers=_parse_workers(comm))
    mine = _split_by_atoms(np.arange(len(offsets)), natoms, comm.world)[comm.rank]
    progress = _Progress(len(mine), verbose and main, "gamma")
    part = _grade_frames(aset, stream, mine, per_atom, progress, precision=precision)
    progress.close()
    stream.close()
    parts = comm.gather_objects(part)
    if not main:
        comm.barrier()
        return None
    out = _assemble_gamma(parts, natoms, per_atom)
    if output_file is not None:
        np.savez(output_file, **out)
    comm.barrier()
    _log(f"  gamma: {len(offsets)} frames in {time.time() - t_total:.1f}s; frames with gamma > "
         f"{aset.tol:g}: {int((out['gamma'] > aset.tol).sum())}, median "
         f"{np.median(out['gamma']):.3g}, max {out['gamma'].max():.3g}")
    return out


def compute_gamma(
    model_file: str,
    active_set,
    xyz_file: str,
    output_file: str = None,
    per_atom: bool = False,
    device: str = None,
    dtype: str = "float64",
    precision: str = "float64",
    chunk_atoms: int = None,
    energy_key: str = "energy",
    verbose: bool = True,
):
    """Extrapolation grades of every frame of ``xyz_file``.

    Grades only, to inspect structures; :func:`select_structures` grades the
    candidates itself when choosing. ``active_set``: an :class:`ActiveSet` or
    the path of a saved one. Returns
    a dict of numpy arrays: ``gamma`` / ``gamma_res`` (per frame, maximum over
    its atoms), ``atom`` (atom with the largest gamma), ``natoms``, and with
    ``per_atom=True`` also ``gamma_atoms`` / ``gamma_res_atoms`` (every atom,
    file order). Saved as ``.npz`` when ``output_file`` is given.

    ``dtype`` is the descriptor precision; ``precision="float32"`` also grades
    in float32 — gamma within 1e-4 and gamma_res within a few 1e-3 relative of the
    float64 values, much faster on GPUs with slow float64 (consumer cards).
    """
    return _gamma(model_file, active_set, xyz_file, output_file, per_atom, device, dtype,
                  chunk_atoms, energy_key, verbose, _Comm(), precision)


def compute_gamma_sharded(
    model_file: str,
    active_set: str,
    xyz_file: str,
    output_file: str = None,
    per_atom: bool = False,
    dtype: str = "float64",
    precision: str = "float64",
    chunk_atoms: int = None,
    energy_key: str = "energy",
    verbose: bool = True,
):
    """Multi-GPU / multi-node :func:`compute_gamma` (``torchrun`` / ``srun``,
    one process per GPU). Every rank grades a contiguous share of the frames;
    rank 0 assembles them in file order, writes ``output_file`` and returns
    the dict — the other ranks return None. The result is identical to
    :func:`compute_gamma`."""
    comm, dev = _comm_setup(None)
    return _gamma(model_file, active_set, xyz_file, output_file, per_atom, dev, dtype,
                  chunk_atoms, energy_key, verbose, comm, precision)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _greedy_select(aset, rows, gamma, n_frames, max_frames, gamma_min, block_rows, log):
    """The greedy D-optimal choice over the candidate rows (q, types, src) of
    the frames whose grade is ``gamma``. Returns (chosen frames, grade at choice)."""
    dev, tol = aset.device, aset.tol
    cq, ct, csrc = rows
    frames = np.unique(csrc[:, 0].numpy())
    frames = frames[np.argsort(-gamma[frames], kind="stable")]     # visiting order
    rank = np.full(n_frames, -1, dtype=np.int64)
    rank[frames] = np.arange(len(frames))
    row_rank = rank[csrc[:, 0].numpy()]
    order = np.argsort(row_rank, kind="stable")
    cq, ct, csrc, row_rank = cq[order], ct[order], csrc[order], row_rank[order]
    frame_row0 = np.searchsorted(row_rank, np.arange(len(frames) + 1))
    log(f"  select: {len(frames)} candidate frames ({cq.shape[0]} atoms)")
    if block_rows is None:                     # rows whose x and C fit in ~30% of free memory
        r_max = max(e.r for e in aset.elements)
        block_rows = max(1000, int(0.3 * _free_bytes(dev) / (8 * (2 * r_max + 64))))
    chosen, gamma_at = [], []
    f0 = 0
    with torch.no_grad():
        while f0 < len(frames) and (max_frames is None or len(chosen) < max_frames):
            f1 = int(np.searchsorted(frame_row0, frame_row0[f0] + block_rows, side="right")) - 1
            f1 = min(max(f1, f0 + 1), len(frames))
            r0, r1 = int(frame_row0[f0]), int(frame_row0[f1])
            bt = ct[r0:r1]
            blocks = []                       # per element: rows sorted by frame rank
            for t in torch.unique(bt).tolist():
                e = aset.elements[t]
                if e.r == 0:
                    continue
                idx = torch.nonzero(bt == t).squeeze(1) + r0
                ranks = row_rank[idx.numpy()] - f0
                q_e = cq[idx].to(dev)
                x, _ = e.project(q_e, aset.row_batch(e.r))
                blocks.append({"e": e, "x": x, "q": q_e, "src": csrc[idx].to(dev),
                               "C": e.coefficients(x),
                               "ranks": torch.from_numpy(ranks).to(dev),
                               "start": np.searchsorted(ranks, np.arange(f1 - f0 + 1))})
            f = 0                              # next frame of the block to visit
            while f < f1 - f0:
                cur = torch.full((f1 - f0,), -1.0, dtype=_F64, device=dev)
                for b in blocks:
                    a = int(b["start"][f])
                    if a < b["C"].shape[0]:
                        cur.scatter_reduce_(0, b["ranks"][a:], b["C"][a:].abs().amax(1), "amax")
                cur_np = cur[f:].cpu().numpy()
                hit = np.nonzero(cur_np > gamma_min)[0]
                if len(hit) == 0:
                    break
                f += int(hit[0])
                for b in blocks:
                    a, z = int(b["start"][f]), int(b["start"][f + 1])
                    if z > a:
                        b["e"].maxvol(b["x"], b["q"], b["src"], tol, C=b["C"], pivot=(a, z))
                chosen.append(int(frames[f0 + f]))
                gamma_at.append(float(cur_np[hit[0]]))
                f += 1
                if max_frames is not None and len(chosen) >= max_frames:
                    break
            f0 = f1
    chosen = np.asarray(chosen, dtype=np.int64)
    # every atom of the chosen frames now belongs to the training data: MaxVol
    # over all of them (the greedy pass only swapped rows of the frame being
    # taken, and later swaps can lift earlier rows above tol again)
    if len(chosen):
        sel = torch.from_numpy(np.isin(csrc[:, 0].numpy(), chosen))
        _maxvol_rows(aset, cq[sel], ct[sel], csrc[sel])
    return chosen, np.asarray(gamma_at)


def _select(model_file, active_set, xyz_file, output_xyz, max_frames, gamma_min, gamma_max,
            block_rows, output_active_set, device, dtype, chunk_atoms, energy_key, verbose,
            comm):
    main = comm.rank == 0
    _log = _log_fn(verbose and main)
    t_total = time.time()
    aset = _as_active_set(active_set, model_file, device, dtype)
    if gamma_min is None:
        gamma_min = aset.tol
    offsets, natoms = _index(comm, xyz_file)
    n = len(offsets)
    stream = _DescriptorStream(aset.desc_calc, xyz_file, chunk_atoms, energy_key,
                               index=(offsets, natoms), workers=_parse_workers(comm))
    mine = _split_by_atoms(np.arange(n), natoms, comm.world)[comm.rank]
    progress = _Progress(len(mine), verbose and main, "gamma")
    part = _grade_frames(aset, stream, mine, False, progress, window=(gamma_min, gamma_max))
    progress.close()
    stream.close()
    parts = comm.gather_objects(part)
    result = None
    if main:
        gamma = _assemble_gamma(parts, natoms, False)["gamma"]
        cand = [p["rows"] for p in parts if p.get("rows") is not None]
        if not cand:
            _log(f"  select: no frame with gamma > {gamma_min}")
            chosen, gamma_at = np.zeros(0, dtype=np.int64), np.zeros(0)
        else:
            rows = tuple(torch.cat([c[k] for c in cand]) for k in range(3))
            chosen, gamma_at = _greedy_select(aset, rows, gamma, n, max_frames, gamma_min,
                                              block_rows, _log)
        _log(f"  select: {len(chosen)} frames chosen in {time.time() - t_total:.1f}s")
        if output_xyz is not None:
            _write_frames(xyz_file, offsets, chosen, output_xyz)
        if output_active_set is not None:
            aset.save(output_active_set)
        result = {"index": chosen, "gamma_at_choice": gamma_at, "gamma": gamma}
    return comm.bcast_object(result, 0)


def select_structures(
    model_file: str,
    active_set,
    xyz_file: str,
    output_xyz: str = None,
    max_frames: int = None,
    gamma_min: float = None,
    gamma_max: float = None,
    block_rows: int = None,
    output_active_set: str = None,
    device: str = None,
    dtype: str = "float64",
    chunk_atoms: int = None,
    energy_key: str = "energy",
    verbose: bool = True,
):
    """Greedy D-optimal choice of structures that extend the active set.

    Frames of ``xyz_file`` whose grade lies in ``(gamma_min, gamma_max]`` are
    candidates (``gamma_min`` defaults to the ``tol`` of the active set). They
    are visited from the highest grade down; a frame is taken when its grade,
    recomputed against the active set extended by the frames taken before it,
    still exceeds ``gamma_min``, and its atoms then enter the active set by
    MaxVol swaps. Near-duplicates of a taken frame fall below the threshold
    and are skipped. Stops after ``max_frames``. The frames are graded here
    (no :func:`compute_gamma` call is needed first), by gamma only: frames
    with gamma <= ``gamma_min`` are not candidates whatever their gamma_res.

    Writes the chosen frames verbatim to ``output_xyz`` and, if given, the
    extended active set to ``output_active_set``. Returns a dict with
    ``index`` (chosen frames, in order of choice), ``gamma_at_choice`` and
    ``gamma`` (grades of all frames against the original active set).
    """
    return _select(model_file, active_set, xyz_file, output_xyz, max_frames, gamma_min,
                   gamma_max, block_rows, output_active_set, device, dtype, chunk_atoms,
                   energy_key, verbose, _Comm())


def select_structures_sharded(
    model_file: str,
    active_set: str,
    xyz_file: str,
    output_xyz: str = None,
    max_frames: int = None,
    gamma_min: float = None,
    gamma_max: float = None,
    block_rows: int = None,
    output_active_set: str = None,
    dtype: str = "float64",
    chunk_atoms: int = None,
    energy_key: str = "energy",
    verbose: bool = True,
):
    """Multi-GPU / multi-node :func:`select_structures` (``torchrun`` /
    ``srun``, one process per GPU): the ranks grade contiguous shares of the
    candidate frames, rank 0 makes the (sequential) greedy choice and writes
    the outputs; every rank returns the same dict. The choice is identical to
    :func:`select_structures`."""
    comm, dev = _comm_setup(None)
    return _select(model_file, active_set, xyz_file, output_xyz, max_frames, gamma_min,
                   gamma_max, block_rows, output_active_set, dev, dtype, chunk_atoms,
                   energy_key, verbose, comm)


def _maxvol_rows(aset, q, types, src):
    """MaxVol swaps of host-side rows (descriptors ``q``, ``types``, ``src``)
    into the active set, element by element in device-sized batches."""
    dev = aset.device
    for t in torch.unique(types).tolist():
        e = aset.elements[t]
        if e.r == 0:
            continue
        idx = torch.nonzero(types == t).squeeze(1)
        nb = aset.row_batch(e.r)
        for a in range(0, idx.shape[0], nb):
            sub = idx[a:a + nb]
            qd = q[sub].to(dev)
            x, _ = e.project(qd)
            e.maxvol_screened(x, qd, src[sub].to(dev), aset.tol)


def _write_frames(xyz_file, offsets, index, output_xyz):
    """Copy frames ``index`` of ``xyz_file`` byte for byte, in that order."""
    ends = np.append(offsets[1:], os.path.getsize(xyz_file))
    with open(xyz_file, "rb") as src, open(output_xyz, "wb") as dst:
        for i in index:
            src.seek(int(offsets[i]))
            dst.write(src.read(int(ends[i] - offsets[i])))
