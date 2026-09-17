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

Workflow: :func:`build_active_set` (training set -> active set file),
:func:`compute_gamma` (grades of any structures) and
:func:`select_structures` (greedy D-optimal choice of new structures:
candidates are visited from the highest grade down, and a structure is taken
only if it still extends the active set after the structures taken before
it were added, so near-duplicates are skipped).
"""

import copy
import hashlib
import os
import time

import numpy as np
import torch

from . import ops
from .nep import NEPCalculator
from .predict import _Progress, _chunk_bounds, _pick_device

_FORMAT = "torchnep-active-set"
_VERSION = 1


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
    calc = NEPCalculator(model_file, dtype=torch.float64, device=device)
    dt = torch.float64 if dtype == "float64" else torch.float32
    if dt == torch.float64:
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
        h.update(t.detach().to("cpu", torch.float64).numpy().tobytes())
    return h.hexdigest()


def _free_bytes(device):
    device = torch.device(device)
    if device.type == "cuda":
        return int(torch.cuda.mem_get_info(device)[0])
    return 8 * 1024 ** 3


# ---------------------------------------------------------------------------
# Streamed descriptors
# ---------------------------------------------------------------------------

def _batch_descriptors(calc, structures, device, backend):
    """Scaled descriptors (N, D) of a list of preprocessed structures."""
    dt = calc.dtype
    nat = np.asarray([s["natoms"] for s in structures], dtype=np.int64)
    start = np.concatenate([[0], np.cumsum(nat)[:-1]])
    at = torch.from_numpy(np.concatenate([s["atom_types"] for s in structures])).to(device)
    N = int(nat.sum())

    def cat(key, shift):
        parts = [s[key] + start[i] if shift else s[key] for i, s in enumerate(structures)]
        return torch.from_numpy(np.concatenate(parts)).to(device)

    pi_r, pj_r = cat("pair_i_rad", True), cat("pair_j_rad", True)
    pi_a, pj_a = cat("pair_i_ang", True), cat("pair_j_ang", True)
    rij_r, rij_a = cat("rij_rad", False).to(dt), cat("rij_ang", False).to(dt)
    rc_r, rc_a = calc.cutoff_args()
    fk_r = ops.chebyshev_basis(torch.norm(rij_r, dim=-1),
                               ops.pair_cutoff(rc_r, at, pi_r, pj_r), calc.basis_size_radial)
    if rij_a.shape[0] > 0:
        da = torch.norm(rij_a, dim=-1)
        fk_a = ops.chebyshev_basis(da, ops.pair_cutoff(rc_a, at, pi_a, pj_a),
                                   calc.basis_size_angular)
        inv = 1.0 / da.clamp(min=1e-10)
        blm = ops.angular_basis(rij_a[:, 0] * inv, rij_a[:, 1] * inv, rij_a[:, 2] * inv,
                                calc.l_max_3b)
    else:
        fk_a = torch.zeros(0, calc.basis_size_angular + 1, dtype=dt, device=device)
        blm = torch.zeros(0, calc.num_lm, dtype=dt, device=device)
    q = ops.compute_descriptors_cached(
        fk_r, fk_a, blm, pi_r, pj_r, pi_a, pj_a, at, N, calc.c2, getattr(calc, "c3", None),
        calc.n_max_radial, calc.n_max_angular, calc.l_max_3b,
        calc.has_q_222, calc.has_q_1111, calc.has_q_112,
        calc.num_lm, calc._c3b, calc._c4b, calc._c5b, calc._c4b2,
        dt, device, backend=backend,
        has_q_123=calc.has_q_123, has_q_233=calc.has_q_233, has_q_134=calc.has_q_134)
    return q * calc.q_scaler


class _DescriptorStream:
    """Scaled per-atom descriptors of the frames of an xyz file, chunk by chunk.

    Host memory is bounded by the chunk (``chunk_atoms``), device memory by
    the pair budget of one descriptor batch (sized from free memory, halved
    on OOM)."""

    def __init__(self, calc, xyz_file, chunk_atoms=None, energy_key="energy"):
        from .data import index_xyz
        from .train import make_preproc_pool
        self.calc, self.xyz_file, self.device = calc, xyz_file, calc.device
        self.np_dtype = np.float64 if calc.dtype == torch.float64 else np.float32
        self.energy_key = energy_key
        if chunk_atoms is None:
            chunk_atoms = int(os.environ.get("TORCHNEP_PREDICT_CHUNK_ATOMS", 200_000))
        self.chunk_atoms = max(1, int(chunk_atoms))
        self.offsets, self.natoms = index_xyz(xyz_file)
        self.backend = ops.resolve_backend("auto", num_types=calc.num_types,
                                           device_type=self.device.type)
        self.pool = make_preproc_pool() if len(self.offsets) >= 64 else None
        self.max_pairs = None
        self.pp_config = {"cutoff_radial": calc.rc_radial, "cutoff_angular": calc.rc_angular,
                          "cutoff_radial_per_type": calc.rc_radial_per_type,
                          "cutoff_angular_per_type": calc.rc_angular_per_type,
                          "type_names": calc.type_names}

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def _pair_budget(self):
        c = self.calc
        esize = 8 if c.dtype == torch.float64 else 4
        per_pair = esize * (3 * (c.n_max_angular + 1) * c.num_lm + 2 * c.num_lm
                            + 2 * (c.basis_size_angular + 1) + 8) + 24
        return max(10_000, int(0.3 * _free_bytes(self.device) / (2.0 * per_pair)))

    def chunks(self, frame_ids=None, progress=None):
        """Yield ``(frame_ids, natoms, types, q)`` per chunk: frame indices
        into the file (ascending), their atom counts, and on the device the
        per-atom types (N,) and scaled descriptors (N, D), atoms in file order."""
        from .data import read_xyz_at
        from .train import preprocess_structures
        if frame_ids is None:
            frame_ids = np.arange(len(self.offsets))
        frame_ids = np.sort(np.asarray(frame_ids, dtype=np.int64))
        if self.max_pairs is None:
            self.max_pairs = self._pair_budget()
        for lo, hi in _chunk_bounds(self.natoms[frame_ids], self.chunk_atoms, max_frames=50_000):
            ids = frame_ids[lo:hi]
            frames = read_xyz_at(self.xyz_file, self.offsets[ids], energy_key=self.energy_key)
            structures = preprocess_structures(frames, self.pp_config, self.np_dtype,
                                               pool=self.pool)
            del frames
            nat = self.natoms[ids]
            n_pairs = np.asarray([len(s["pair_i_ang"]) + len(s["pair_i_rad"])
                                  for s in structures])
            q = torch.empty(int(nat.sum()), self.calc.dim, dtype=self.calc.dtype,
                            device=self.device)
            s0, a0 = 0, 0
            while s0 < len(structures):
                s1, pairs = s0, 0
                while s1 < len(structures) and (s1 == s0 or pairs + n_pairs[s1] <= self.max_pairs):
                    pairs += n_pairs[s1]
                    s1 += 1
                try:
                    with torch.no_grad():
                        qb = _batch_descriptors(self.calc, structures[s0:s1], self.device,
                                                self.backend)
                except torch.OutOfMemoryError:
                    if s1 - s0 == 1 and self.max_pairs < 10_000:
                        raise
                    torch.cuda.empty_cache()
                    self.max_pairs = max(1, self.max_pairs // 2)
                    continue
                q[a0:a0 + qb.shape[0]] = qb
                a0 += qb.shape[0]
                s0 = s1
            types = torch.from_numpy(
                np.concatenate([s["atom_types"] for s in structures])).to(self.device)
            del structures
            yield ids, nat, types, q
            if progress is not None:
                progress.update(len(ids))


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

    @property
    def r(self):
        return 0 if self.V is None else self.V.shape[1]

    def project(self, q, batch=None):
        """Whitened coordinates (n, r) and out-of-subspace norm (n,) of rows
        given by their descriptors ``q`` (float64), in batches of ``batch`` rows."""
        n = q.shape[0]
        batch = batch or n
        x = torch.empty(n, self.r, dtype=torch.float64, device=q.device)
        res = torch.empty(n, dtype=torch.float64, device=q.device)
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

    def reinvert(self):
        self.Ainv = torch.linalg.inv(self.X)
        self._since_inv = 0

    def init_from_rows(self, x, q, src):
        """First active rows: LU with partial pivoting picks r well-spread rows
        of ``x`` (n >= r). Directions these rows cannot span are dropped from
        the subspace. Returns the kept column mask."""
        keep = torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
        while True:
            LU, piv = torch.linalg.lu_factor(x[:, keep])
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
        f64 = dict(device=device, dtype=torch.float64)
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
    def gamma(self, types, q):
        """Per-atom grades of scaled descriptors ``q`` (N, D) with ``types`` (N,).

        Returns ``(gamma, gamma_res)`` (N,) float64 tensors; atoms of elements
        without an active set get ``inf``."""
        N = types.shape[0]
        g = torch.full((N,), float("inf"), dtype=torch.float64, device=self.device)
        gr = torch.full((N,), float("inf"), dtype=torch.float64, device=self.device)
        q = q.to(torch.float64)
        for t in torch.unique(types).tolist():
            e = self.elements[t]
            if e.r == 0:
                continue
            idx = torch.nonzero(types == t).squeeze(1)
            nb = self.row_batch(e.r)
            for a in range(0, idx.shape[0], nb):
                sub = idx[a:a + nb]
                x, res = e.project(q[sub])
                g[sub] = e.coefficients(x).abs().amax(1)
                gr[sub] = res / e.res_max if e.res_max > 0 else res
        return g, gr


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _log_fn(verbose):
    def _log(msg):
        if verbose:
            print(msg, flush=True)
    return _log


def build_active_set(
    model_file: str,
    xyz_file: str,
    output_file: str = "active_set.pt",
    rcond: float = 1e-4,
    tol: float = 1.01,
    sample_frames: int = 50_000,
    init_rows: int = 4,
    max_passes: int = 3,
    device: str = None,
    dtype: str = "float64",
    chunk_atoms: int = None,
    energy_key: str = "energy",
    seed: int = 0,
    verbose: bool = True,
):
    """Build the MaxVol active set of a NEP model from its training set.

    Two streamed passes over ``xyz_file``:

    1. on ``sample_frames`` random frames (all frames if fewer): the Gram
       matrix ``B^T B`` of every element, whose eigenvectors with singular
       value above ``rcond`` x the largest span the subspace of the grade,
       and a random sample of rows that seeds the active set (LU pivoting on
       ``init_rows`` x r rows, then MaxVol over the whole sample);
    2. on all frames: MaxVol swaps of every row whose grade exceeds ``tol``,
       repeated (at most ``max_passes`` times) until a pass makes no swap,
       since a row passed early can exceed ``tol`` after later swaps.

    ``dtype`` is the descriptor precision; subspaces and MaxVol always run
    in float64. Saves the :class:`ActiveSet` to ``output_file`` and returns it.
    """
    from .train import _default_alloc_conf
    _default_alloc_conf()
    _log = _log_fn(verbose)
    device = torch.device(_pick_device(device))
    t_total = time.time()
    aset = ActiveSet(*_load_calculators(model_file, device, dtype), rcond, tol)
    calc = aset.calc
    T, K = calc.num_types, aset.K
    stream = _DescriptorStream(aset.desc_calc, xyz_file, chunk_atoms, energy_key)
    n_frames = len(stream.offsets)
    _log(f"  active set: {n_frames} frames, {int(stream.natoms.sum())} atoms, {T} elements, "
         f"K = {calc.num_neurons} x ({calc.dim} + 2) = {K}, rcond {rcond:g}, tol {tol:g}")

    # ---- pass 1: Gram matrices and a row sample ----------------------------
    rng = np.random.default_rng(seed)
    sample = np.arange(n_frames) if n_frames <= sample_frames else \
        np.sort(rng.choice(n_frames, sample_frames, replace=False))
    G = [None] * T
    cap = max(4096, (init_rows + 4) * K)      # sampled rows kept per element
    reservoir = [None] * T                    # (random key, q, src)
    gen = torch.Generator(device=device).manual_seed(seed)
    nb = aset.row_batch()
    t0 = time.time()
    progress = _Progress(len(sample), verbose, "Gram")
    with torch.no_grad():
        for ids, nat, types, q in stream.chunks(sample, progress):
            src = _atom_sources(ids, nat, device)
            q = q.to(torch.float64)
            for t in torch.unique(types).tolist():
                e = aset.elements[t]
                idx = torch.nonzero(types == t).squeeze(1)
                if G[t] is None:
                    G[t] = torch.zeros(K, K, dtype=torch.float64, device=device)
                for a in range(0, idx.shape[0], nb):
                    B = b_vectors(q[idx[a:a + nb]], e.w0, e.b0, e.w1)
                    G[t].addmm_(B.T, B)
                    del B
                e.n_rows += idx.shape[0]
                new = (torch.rand(idx.shape[0], generator=gen, device=device), q[idx], src[idx])
                if reservoir[t] is not None:
                    new = tuple(torch.cat([u, w]) for u, w in zip(reservoir[t], new))
                if new[0].shape[0] > cap:
                    top = torch.topk(new[0], cap).indices
                    new = tuple(u[top] for u in new)
                reservoir[t] = new
    progress.close()
    _log(f"  pass 1 (Gram matrices, {len(sample)} frames): {time.time() - t0:.1f}s")

    t0 = time.time()
    with torch.no_grad():
        for t, e in enumerate(aset.elements):
            if G[t] is None:
                continue
            lam, vec = torch.linalg.eigh(G[t])
            G[t] = None
            lam, vec = lam.flip(0).clamp(min=0.0), vec.flip(1)
            _, q_s, src_s = reservoir[t]
            reservoir[t] = None
            r = int((torch.sqrt(lam / lam[0]) > rcond).sum())
            r = min(r, q_s.shape[0])
            e.V = vec[:, :r].contiguous()
            e.scale = 1.0 / torch.sqrt(lam[:r] / e.n_rows)
            del vec
            x, res = e.project(q_s, nb)
            e.res_max = float(res.max())
            n_init = min(x.shape[0], max(init_rows * r, r))
            keep = e.init_from_rows(x[:n_init], q_s[:n_init], src_s[:n_init])
            x = x[:, keep].contiguous()
            e.maxvol(x, q_s.clone(), src_s.clone(), tol)
            _log(f"    {e.name:3s} rows {e.n_rows:9d}  rank {e.r:5d} / {K}")
    _log(f"  subspaces and seed active sets: {time.time() - t0:.1f}s")

    # ---- pass 2: MaxVol over every frame, until no swap --------------------
    for p in range(max_passes):
        t0 = time.time()
        swaps0 = sum(e.swaps for e in aset.elements)
        progress = _Progress(n_frames, verbose, f"MaxVol {p + 1}")
        with torch.no_grad():
            for ids, nat, types, q in stream.chunks(None, progress):
                src = _atom_sources(ids, nat, device)
                q = q.to(torch.float64)
                for t in torch.unique(types).tolist():
                    e = aset.elements[t]
                    if e.r == 0:
                        continue
                    idx = torch.nonzero(types == t).squeeze(1)
                    nbr = aset.row_batch(e.r)
                    for a in range(0, idx.shape[0], nbr):
                        sub = idx[a:a + nbr]
                        x, res = e.project(q[sub])
                        e.res_max = max(e.res_max, float(res.max()))
                        C = e.coefficients(x)
                        if float(C.abs().max()) > tol:
                            e.maxvol(x, q[sub], src[sub], tol, C=C)
        progress.close()
        swaps = sum(e.swaps for e in aset.elements) - swaps0
        _log(f"  pass 2.{p + 1} (MaxVol, {n_frames} frames): {time.time() - t0:.1f}s, {swaps} swaps")
        if swaps == 0:
            break
    stream.close()
    for e in aset.elements:
        if e.r:
            _log(f"    {e.name:3s} active rows {e.r:5d}  swaps {e.swaps:7d}")
    aset.meta = {"xyz_file": os.path.abspath(xyz_file), "n_frames": int(n_frames),
                 "sample_frames": int(len(sample)), "model_file": os.path.abspath(model_file)}
    aset.save(output_file)
    _log(f"  TOTAL: {time.time() - t_total:.1f}s -> {output_file}")
    return aset


# ---------------------------------------------------------------------------
# Grades of structures
# ---------------------------------------------------------------------------

def _as_active_set(active_set, model_file, device, dtype):
    if isinstance(active_set, ActiveSet):
        return active_set
    return ActiveSet.load(active_set, model_file, device=device, dtype=dtype)


def compute_gamma(
    model_file: str,
    active_set,
    xyz_file: str,
    output_file: str = None,
    per_atom: bool = False,
    device: str = None,
    dtype: str = "float64",
    chunk_atoms: int = None,
    energy_key: str = "energy",
    verbose: bool = True,
):
    """Extrapolation grades of every frame of ``xyz_file``.

    ``active_set``: an :class:`ActiveSet` or the path of a saved one. Returns
    a dict of numpy arrays: ``gamma`` / ``gamma_res`` (per frame, maximum over
    its atoms), ``atom`` (atom with the largest gamma), ``natoms``, and with
    ``per_atom=True`` also ``gamma_atoms`` / ``gamma_res_atoms`` (every atom,
    file order). Saved as ``.npz`` when ``output_file`` is given.
    """
    _log = _log_fn(verbose)
    t_total = time.time()
    aset = _as_active_set(active_set, model_file, device, dtype)
    dev = aset.device
    stream = _DescriptorStream(aset.desc_calc, xyz_file, chunk_atoms, energy_key)
    n = len(stream.offsets)
    out = {"gamma": np.empty(n), "gamma_res": np.empty(n),
           "atom": np.empty(n, dtype=np.int64), "natoms": stream.natoms.copy()}
    if per_atom:
        n_atoms = int(stream.natoms.sum())
        out["gamma_atoms"], out["gamma_res_atoms"] = np.empty(n_atoms), np.empty(n_atoms)
        atom_start = np.concatenate([[0], np.cumsum(stream.natoms)])
    progress = _Progress(n, verbose, "gamma")
    for ids, nat, types, q in stream.chunks(None, progress):
        g, gr = aset.gamma(types, q)
        gmax, frame = _frame_max(g, nat, dev)
        grmax, _ = _frame_max(gr, nat, dev)
        nat_t = torch.from_numpy(nat).to(dev)
        first = torch.cumsum(nat_t, 0) - nat_t
        hit = torch.nonzero(g == gmax[frame]).squeeze(1)
        arg = torch.zeros(len(ids), dtype=torch.long, device=dev)
        arg.scatter_reduce_(0, frame[hit], hit, "amin", include_self=False)
        out["gamma"][ids] = gmax.cpu().numpy()
        out["gamma_res"][ids] = grmax.cpu().numpy()
        out["atom"][ids] = (arg - first).cpu().numpy()
        if per_atom:
            dst = np.repeat(atom_start[ids], nat) + (np.arange(int(nat.sum()))
                                                     - np.repeat(np.cumsum(nat) - nat, nat))
            out["gamma_atoms"][dst] = g.cpu().numpy()
            out["gamma_res_atoms"][dst] = gr.cpu().numpy()
    progress.close()
    stream.close()
    if output_file is not None:
        np.savez(output_file, **out)
    _log(f"  gamma: {n} frames in {time.time() - t_total:.1f}s; frames with gamma > 1: "
         f"{int((out['gamma'] > 1).sum())}, median {np.median(out['gamma']):.3g}, "
         f"max {out['gamma'].max():.3g}")
    return out


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

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
    candidates (``gamma_min`` defaults to the ``tol`` of the active set). They are visited from the highest grade down; a frame is
    taken when its grade, recomputed against the active set extended by the
    frames taken before it, still exceeds ``gamma_min``, and its atoms then
    enter the active set by MaxVol swaps. Near-duplicates of a taken frame
    fall below the threshold and are skipped. Stops after ``max_frames``.

    Writes the chosen frames verbatim to ``output_xyz`` and, if given, the
    extended active set to ``output_active_set``. Returns a dict with
    ``index`` (chosen frames, in order of choice), ``gamma_at_choice`` and
    ``gamma`` (grades of all frames against the original active set).
    """
    _log = _log_fn(verbose)
    t_total = time.time()
    aset = _as_active_set(active_set, model_file, device, dtype)
    dev, tol = aset.device, aset.tol
    if gamma_min is None:
        gamma_min = tol

    # ---- grades against the active set; keep the rows of candidate frames --
    stream = _DescriptorStream(aset.desc_calc, xyz_file, chunk_atoms, energy_key)
    offsets = stream.offsets
    n = len(offsets)
    gamma = np.empty(n)
    keep_q, keep_t, keep_src = [], [], []
    progress = _Progress(n, verbose, "gamma")
    for ids, nat, types, q in stream.chunks(None, progress):
        g, _ = aset.gamma(types, q)
        gmax, frame = _frame_max(g, nat, dev)
        gamma[ids] = gmax.cpu().numpy()
        ok = gmax > gamma_min
        if gamma_max is not None:
            ok &= gmax <= gamma_max
        rows = ok[frame]
        if bool(rows.any()):
            keep_q.append(q[rows].to(torch.float64).cpu())
            keep_t.append(types[rows].cpu())
            keep_src.append(_atom_sources(ids, nat, dev)[rows].cpu())
    progress.close()
    stream.close()
    empty = {"index": np.zeros(0, dtype=np.int64), "gamma_at_choice": np.zeros(0),
             "gamma": gamma}
    if not keep_q:
        _log(f"  select: no frame with gamma > {gamma_min}")
        return empty
    cq, ct, csrc = torch.cat(keep_q), torch.cat(keep_t), torch.cat(keep_src)
    frames = np.unique(csrc[:, 0].numpy())
    frames = frames[np.argsort(-gamma[frames], kind="stable")]     # visiting order
    rank = np.full(n, -1, dtype=np.int64)
    rank[frames] = np.arange(len(frames))
    row_rank = rank[csrc[:, 0].numpy()]
    order = np.argsort(row_rank, kind="stable")
    cq, ct, csrc, row_rank = cq[order], ct[order], csrc[order], row_rank[order]
    frame_row0 = np.searchsorted(row_rank, np.arange(len(frames) + 1))
    _log(f"  select: {len(frames)} candidate frames ({cq.shape[0]} atoms)")

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
                cur = torch.full((f1 - f0,), -1.0, dtype=torch.float64, device=dev)
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
    _log(f"  select: {len(chosen)} frames chosen in {time.time() - t_total:.1f}s")
    if output_xyz is not None:
        _write_frames(xyz_file, offsets, chosen, output_xyz)
    if output_active_set is not None:
        aset.save(output_active_set)
    return {"index": chosen, "gamma_at_choice": np.asarray(gamma_at), "gamma": gamma}


def _maxvol_rows(aset, q, types, src, max_passes=5):
    """MaxVol swaps of host-side rows (descriptors ``q``, ``types``, ``src``)
    into the active set, in device-sized batches, until a pass makes no swap."""
    dev = aset.device
    for _ in range(max_passes):
        swaps = 0
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
                swaps += e.maxvol(x, qd, src[sub].to(dev), aset.tol)[0]
        if swaps == 0:
            return


def _write_frames(xyz_file, offsets, index, output_xyz):
    """Copy frames ``index`` of ``xyz_file`` byte for byte, in that order."""
    ends = np.append(offsets[1:], os.path.getsize(xyz_file))
    with open(xyz_file, "rb") as src, open(output_xyz, "wb") as dst:
        for i in index:
            src.seek(int(offsets[i]))
            dst.write(src.read(int(ends[i] - offsets[i])))
