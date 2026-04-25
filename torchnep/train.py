# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Yongchao Wu.
# Part of the torchnep project — https://github.com/mushroomfire/torchnep.

"""
NEP training with PyTorch — single-GPU / CPU entry point.

Use ``train_nep`` launched with plain ``python`` for one-GPU, CPU, or Mac
workloads. For multi-GPU training, use ``train_nep_sharded`` launched with
``torchrun`` — it shards the dataset across ranks so each GPU only holds
1/world_size of the structures, enabling datasets much larger than any one
card's memory.

    # single GPU / CPU / Mac
    python run_train.py                        # calls train_nep(...)

    # multi-GPU, single node
    torchrun --standalone --nproc_per_node=N run_train.py   # train_nep_sharded

    # multi-node (via SLURM)
    see example/run_multi_node.sbatch
"""

import os
import platform
import time
import torch
import numpy as np
from datetime import datetime
from typing import List, Dict
from torch.optim.swa_utils import AveragedModel

from .model import NEPModel, slim_model
from .data import read_xyz, parse_nep_in, build_neighbor_list_np
from . import ops
from . import __version__
from .predict import predict_dataset, predict_from_store


# ---------------------------------------------------------------------------
# Banner & environment info
# ---------------------------------------------------------------------------

_BANNER = r"""
████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗███╗   ██╗███████╗██████╗
╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║████╗  ██║██╔════╝██╔══██╗
   ██║   ██║   ██║██████╔╝██║     ███████║██╔██╗ ██║█████╗  ██████╔╝
   ██║   ██║   ██║██╔══██╗██║     ██╔══██║██║╚██╗██║██╔══╝  ██╔═══╝
   ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║██║ ╚████║███████╗██║
   ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝
"""

_AUTHOR = "Yongchao Wu, yongchao.wu@aalto.fi"


def _backend_info(dev: torch.device, world_size: int = 1) -> List[str]:
    """Describe compute backend for the startup banner.

    ``world_size`` is used by ``train_nep_sharded`` to surface DDP info;
    single-GPU ``train_nep`` keeps the default of 1.

    PyTorch-ROCm exposes AMD GPUs through the ``torch.cuda`` namespace, so
    the cuda branch also handles ROCm. Intel GPUs use ``torch.xpu``.
    """
    lines = []
    if dev.type == "cuda":
        n_visible = torch.cuda.device_count()
        is_rocm = getattr(torch.version, "hip", None) is not None
        tag = "ROCm" if is_rocm else "CUDA"
        if world_size > 1:
            lines.append(f"Backend  : {tag} (DDP, {world_size} processes)")
        else:
            lines.append(f"Backend  : {tag}")
        names = {torch.cuda.get_device_name(i) for i in range(n_visible)}
        name_str = ", ".join(sorted(names))
        total_gb = sum(torch.cuda.get_device_properties(i).total_memory
                       for i in range(n_visible)) / 1e9
        lines.append(f"Devices  : {n_visible} x {name_str}  ({total_gb:.1f} GB total)")
    elif dev.type == "xpu":
        n_visible = torch.xpu.device_count()
        lines.append(f"Backend  : XPU (Intel){' (DDP, ' + str(world_size) + ' processes)' if world_size > 1 else ''}")
        names = {torch.xpu.get_device_name(i) for i in range(n_visible)}
        lines.append(f"Devices  : {n_visible} x {', '.join(sorted(names))}")
    elif dev.type == "mps":
        lines.append("Backend  : MPS (Apple Silicon)")
    else:
        lines.append(f"Backend  : CPU ({platform.processor() or platform.machine()})")
    lines.append(f"PyTorch  : {torch.__version__}")
    return lines


def _default_device() -> str:
    """Select best available device: CUDA/ROCm → XPU → MPS → CPU.

    PyTorch-ROCm routes AMD GPUs through the cuda namespace, so the cuda
    probe catches both. Other torch backends (e.g. XPU for Intel) are
    detected if their namespace is present and reports an available device.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Bucket batching
#
# Frames are sorted by natoms, then cut into buckets of ``world_size × batch_size``
# contiguous frames. Each rank's share from each bucket is a ``batch_size``-long
# slice. Per-epoch, the BUCKET ORDER is shuffled (same seed on every rank), so
# within any DDP iteration all ranks process frames from the same natoms
# quantile of the dataset → per-batch atom counts are matched across ranks,
# eliminating straggler waste.
#
# For single-GPU (world_size == 1) we still sort + bucket; the last bucket
# may be partial. This gives a slight stability gain (within-batch natoms
# variance is small) at zero compute cost.
# ---------------------------------------------------------------------------

def _sort_and_shard(frames: List[Dict], rank: int, world_size: int,
                     batch_size: int):
    """Sort frames by natoms and return this rank's local slice + bucket layout.

    Parameters
    ----------
    frames       : list of frame dicts (as returned by read_xyz).
    rank         : int in [0, world_size).
    world_size   : int  ≥ 1.
    batch_size   : int  per-rank batch size.

    Returns
    -------
    local_frames : list of frames this rank should preprocess / store.
    boundaries   : list of (start, end) tuples into ``local_frames``. One entry
                   per bucket (== one per DDP iteration per epoch). All ranks
                   agree on ``len(boundaries)``; for the last partial bucket
                   different ranks may get slightly different batch sizes, but
                   every rank still does the same number of iterations, which
                   is all DDP requires. The loss normalisation already uses
                   global sample counts, so unequal rank batch sizes are
                   accounted for.
    global_idx   : list[int] — for each entry of ``local_frames``, the index
                   it held in the input ``frames`` list (i.e. its xyz-file
                   row). Needed by sharded prediction to emit output files in
                   input order after gathering ranks' results.
    dropped      : number of frames dropped. Only non-zero when the trailing
                   partial bucket has fewer than ``world_size`` frames total
                   (at most W-1 frames — unavoidable because every rank needs
                   at least one frame per iteration).
    """
    n = len(frames)
    B = batch_size
    W = world_size

    # Sort by natoms while carrying along each frame's original index so a
    # rank can later recover which xyz row each of its local frames came from.
    paired = sorted(enumerate(frames), key=lambda it: it[1]["natoms"])
    sorted_idx   = [i for i, _ in paired]
    sorted_frames = [f for _, f in paired]

    # ----- Single-GPU: one bucket per B-sized slice; trailing one may be partial.
    if W == 1:
        boundaries = [(b * B, min((b + 1) * B, n))
                      for b in range((n + B - 1) // B)]
        return sorted_frames, boundaries, sorted_idx, 0

    # ----- DDP: pack full buckets of W*B frames, then split the leftover
    # evenly across ranks (possibly unequal by 1). Ranks all do the same
    # number of iterations; only the last iteration's batch size may differ.
    num_full = n // (W * B)
    if num_full == 0:
        # No full bucket even once → dataset < W*B. Very rare.
        remaining = n
    else:
        remaining = n - num_full * W * B

    local = []
    local_idx = []
    boundaries = []

    for b in range(num_full):
        bucket_start = b * W * B
        rank_start = bucket_start + rank * B
        off = len(local)
        local.extend(sorted_frames[rank_start:rank_start + B])
        local_idx.extend(sorted_idx[rank_start:rank_start + B])
        boundaries.append((off, off + B))

    dropped = 0
    if remaining >= W:
        # Split ``remaining`` frames: first (remaining % W) ranks get one
        # extra frame so no frame is wasted.
        base = remaining // W
        extras = remaining % W
        n_rank = base + (1 if rank < extras else 0)
        if rank < extras:
            skip = rank * (base + 1)
        else:
            skip = extras * (base + 1) + (rank - extras) * base
        partial_start = num_full * W * B + skip
        off = len(local)
        local.extend(sorted_frames[partial_start:partial_start + n_rank])
        local_idx.extend(sorted_idx[partial_start:partial_start + n_rank])
        boundaries.append((off, off + n_rank))
    elif remaining > 0:
        # Fewer than W frames left — can't give every rank at least one,
        # so these are unavoidably lost. At most W-1 frames.
        dropped = remaining

    if not boundaries:
        raise RuntimeError(
            f"dataset too small for bucket batching with DDP: got {n} frames "
            f"and world_size={W}, need ≥ {W} frames so every rank has work.")

    return local, boundaries, local_idx, dropped



# ---------------------------------------------------------------------------
# GPU data store — all data pre-loaded to device
# ---------------------------------------------------------------------------

class GPUDataStore:
    """Pre-loads all structure data to GPU for zero-copy batch collation.

    When ``config`` is given, also caches Chebyshev basis functions and
    angular basis on GPU so training never recomputes them.
    """

    def __init__(self, structures: List[Dict], device: torch.device,
                 dtype: torch.dtype, config: dict = None):
        self.device = device
        self.dtype = dtype
        self.n = len(structures)
        self.has_cached_basis = config is not None

        n_rad = np.array([len(s["pair_i_rad"]) for s in structures], dtype=np.int64)
        n_ang = np.array([len(s["pair_i_ang"]) for s in structures], dtype=np.int64)
        self.natoms = [int(s["natoms"]) for s in structures]

        # preprocess_structures already returns arrays with the right dtype
        # (int64 for indices, float32 for rij). Skip the defensive astype —
        # it was creating an extra copy of every per-frame array.
        at_cat   = np.concatenate([s["atom_types"]  for s in structures])
        pi_r_cat = np.concatenate([s["pair_i_rad"]  for s in structures])
        pj_r_cat = np.concatenate([s["pair_j_rad"]  for s in structures])
        rij_r_cat = np.concatenate([s["rij_rad"]    for s in structures])
        pi_a_cat = np.concatenate([s["pair_i_ang"]  for s in structures])
        pj_a_cat = np.concatenate([s["pair_j_ang"]  for s in structures])
        rij_a_cat = np.concatenate([s["rij_ang"]    for s in structures])

        at_all   = torch.from_numpy(at_cat).to(device=device, non_blocking=True)

        pi_r_all = torch.from_numpy(pi_r_cat).to(device=device, non_blocking=True)
        pj_r_all = torch.from_numpy(pj_r_cat).to(device=device, non_blocking=True)
        rij_r_all = torch.from_numpy(rij_r_cat).to(device=device, dtype=dtype,
                                                    non_blocking=True)
        pi_a_all = torch.from_numpy(pi_a_cat).to(device=device, non_blocking=True)
        pj_a_all = torch.from_numpy(pj_a_cat).to(device=device, non_blocking=True)
        rij_a_all = torch.from_numpy(rij_a_cat).to(device=device, dtype=dtype,
                                                    non_blocking=True)

        self.energy = [float(s["energy"]) if "energy" in s else 0.0
                       for s in structures]
        self.has_energy_flag = ["energy" in s for s in structures]
        self.has_forces_flag = ["forces" in s for s in structures]
        self.has_virial_flag = ["virial" in s for s in structures]

        f_parts = []
        for s in structures:
            if "forces" in s:
                f_parts.append(np.asarray(s["forces"]).reshape(-1, 3))
            else:
                f_parts.append(np.zeros((s["natoms"], 3), dtype=np.float32))
        f_cat = np.concatenate(f_parts).astype(np.float32 if dtype == torch.float32
                                               else np.float64, copy=False)
        f_all = torch.from_numpy(f_cat).to(device=device, dtype=dtype,
                                            non_blocking=True)

        v_parts = []
        for s in structures:
            if "virial" in s:
                v = np.asarray(s["virial"]).reshape(-1)
                if v.shape[0] == 6:
                    v9 = np.array([v[0], v[3], v[5],
                                   v[3], v[1], v[4],
                                   v[5], v[4], v[2]])
                    v_parts.append(v9)
                else:
                    v_parts.append(v[:9])
            else:
                v_parts.append(np.zeros(9))
        v_cat = np.stack(v_parts).astype(np.float32 if dtype == torch.float32
                                         else np.float64, copy=False)
        v_all = torch.from_numpy(v_cat).to(device=device, dtype=dtype,
                                            non_blocking=True)

        # Per-frame cell volume (Å³) — needed for stress RMSE. Same order as
        # frames, so a batch slice follows the same indexing as .energy etc.
        vol_cat = np.asarray([s.get("volume", 0.0) for s in structures],
                             dtype=np.float32 if dtype == torch.float32
                             else np.float64)
        self.volumes = torch.from_numpy(vol_cat).to(device=device, dtype=dtype,
                                                     non_blocking=True)

        if config is not None:
            dr_all = torch.norm(rij_r_all, dim=-1)
            fk_r_all, fkp_r_all = ops.chebyshev_basis_and_deriv(
                dr_all, config["cutoff_radial"], config["basis_size_radial"])
            d12inv_r_all = 1.0 / dr_all

            if rij_a_all.shape[0] > 0:
                da_all = torch.norm(rij_a_all, dim=-1)
                fk_a_all, fkp_a_all = ops.chebyshev_basis_and_deriv(
                    da_all, config["cutoff_angular"], config["basis_size_angular"])
                d12inv_a_all = 1.0 / da_all
                blm_all = ops.angular_basis(
                    rij_a_all[:, 0] * d12inv_a_all,
                    rij_a_all[:, 1] * d12inv_a_all,
                    rij_a_all[:, 2] * d12inv_a_all,
                    config["l_max"][0])
            else:
                fk_a_all = torch.zeros(0, config["basis_size_angular"] + 1,
                                       dtype=dtype, device=device)
                fkp_a_all = torch.zeros(0, config["basis_size_angular"] + 1,
                                        dtype=dtype, device=device)
                d12inv_a_all = torch.zeros(0, dtype=dtype, device=device)
                num_lm = sum(2 * ll + 1 for ll in range(1, config["l_max"][0] + 1))
                blm_all = torch.zeros(0, num_lm, dtype=dtype, device=device)

        nr_list = n_rad.tolist()
        na_list = n_ang.tolist()
        nat_list = [int(x) for x in self.natoms]

        self.atom_types = list(torch.split(at_all, nat_list))
        self.pi_rad = list(torch.split(pi_r_all, nr_list))
        self.pj_rad = list(torch.split(pj_r_all, nr_list))
        self.rij_rad = list(torch.split(rij_r_all, nr_list))
        self.pi_ang = list(torch.split(pi_a_all, na_list))
        self.pj_ang = list(torch.split(pj_a_all, na_list))
        self.rij_ang = list(torch.split(rij_a_all, na_list))
        self.forces = list(torch.split(f_all, nat_list))
        self.virial = list(torch.unbind(v_all, dim=0))

        if config is not None:
            self.fk_rad = list(torch.split(fk_r_all, nr_list))
            self.fkp_rad = list(torch.split(fkp_r_all, nr_list))
            self.d12inv_rad = list(torch.split(d12inv_r_all, nr_list))
            self.fk_ang = list(torch.split(fk_a_all, na_list))
            self.fkp_ang = list(torch.split(fkp_a_all, na_list))
            self.d12inv_ang = list(torch.split(d12inv_a_all, na_list))
            self.blm = list(torch.split(blm_all, na_list))

        self.n_energy = sum(self.has_energy_flag)
        self.n_forces = sum(self.has_forces_flag)
        self.n_virial = sum(self.has_virial_flag)
        self.has_forces = self.n_forces > 0
        self.has_virial = self.n_virial > 0

    def collate(self, indices: List[int]) -> Dict:
        """Fast GPU-side batch collation. No CPU→GPU transfer."""
        offsets = [0]
        for i in indices:
            offsets.append(offsets[-1] + self.natoms[i])
        N_total = offsets[-1]
        B = len(indices)

        at_list = [self.atom_types[i] for i in indices]
        atom_types = torch.cat(at_list)

        struct_idx = torch.cat([
            torch.full((self.natoms[i],), k, dtype=torch.long,
                       device=self.device)
            for k, i in enumerate(indices)
        ])

        pi_r = torch.cat([self.pi_rad[i] + offsets[k]
                          for k, i in enumerate(indices)])
        pj_r = torch.cat([self.pj_rad[i] + offsets[k]
                          for k, i in enumerate(indices)])
        rij_r = torch.cat([self.rij_rad[i] for i in indices])
        pi_a = torch.cat([self.pi_ang[i] + offsets[k]
                          for k, i in enumerate(indices)])
        pj_a = torch.cat([self.pj_ang[i] + offsets[k]
                          for k, i in enumerate(indices)])
        rij_a = torch.cat([self.rij_ang[i] for i in indices])

        energy = torch.tensor([self.energy[i] for i in indices],
                              dtype=self.dtype, device=self.device)
        natoms = torch.tensor([self.natoms[i] for i in indices],
                              dtype=self.dtype, device=self.device)

        volumes = self.volumes[torch.as_tensor(indices, device=self.device,
                                                dtype=torch.long)]

        batch = {
            "N": N_total, "num_structures": B,
            "atom_types": atom_types, "struct_idx": struct_idx,
            "pair_i_rad": pi_r, "pair_j_rad": pj_r, "rij_rad": rij_r,
            "pair_i_ang": pi_a, "pair_j_ang": pj_a, "rij_ang": rij_a,
            "energy": energy, "natoms": natoms, "volumes": volumes,
        }

        batch["energy_mask"] = torch.tensor(
            [self.has_energy_flag[i] for i in indices],
            dtype=torch.bool, device=self.device)

        batch["forces"] = torch.cat([self.forces[i] for i in indices])
        force_flags = [self.has_forces_flag[i] for i in indices]
        batch["force_mask"] = torch.cat([
            torch.full((self.natoms[indices[k]],), force_flags[k],
                       dtype=torch.bool, device=self.device)
            for k in range(B)
        ])

        batch["virial"] = torch.stack([self.virial[i] for i in indices])
        batch["virial_mask"] = torch.tensor(
            [self.has_virial_flag[i] for i in indices],
            dtype=torch.bool, device=self.device)

        if self.has_cached_basis:
            batch["fk_rad"] = torch.cat([self.fk_rad[i] for i in indices])
            batch["fkp_rad"] = torch.cat([self.fkp_rad[i] for i in indices])
            batch["d12inv_rad"] = torch.cat([self.d12inv_rad[i] for i in indices])
            batch["fk_ang"] = torch.cat([self.fk_ang[i] for i in indices])
            batch["fkp_ang"] = torch.cat([self.fkp_ang[i] for i in indices])
            batch["d12inv_ang"] = torch.cat([self.d12inv_ang[i] for i in indices])
            batch["blm"] = torch.cat([self.blm[i] for i in indices])

        return batch


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess_one_frame(args):
    """Worker: build neighbor lists for a single frame. Picklable for mp.Pool."""
    frame, rc_rad, rc_ang, max_rc, type_names, dtype = args
    positions = frame["positions"].astype(dtype)
    cell = frame["cell"].astype(dtype)
    atom_types = np.array([type_names.index(s) for s in frame["species"]],
                          dtype=np.int64)
    pair_i, pair_j, rij = build_neighbor_list_np(positions, cell, max_rc)
    dij = np.linalg.norm(rij, axis=1)
    rad_mask = dij < rc_rad
    ang_mask = dij < rc_ang
    s = {
        "natoms": frame["natoms"],
        "atom_types": atom_types,
        "volume": float(abs(np.linalg.det(cell))),   # Å³, used for stress RMSE
        "pair_i_rad": pair_i[rad_mask], "pair_j_rad": pair_j[rad_mask],
        "rij_rad": rij[rad_mask].astype(dtype),
        "pair_i_ang": pair_i[ang_mask], "pair_j_ang": pair_j[ang_mask],
        "rij_ang": rij[ang_mask].astype(dtype),
    }
    if "energy" in frame:
        s["energy"] = frame["energy"]
    if "forces" in frame:
        s["forces"] = frame["forces"].astype(dtype)
    if "virial" in frame:
        s["virial"] = frame["virial"].astype(dtype)
    return s


def preprocess_structures(frames, config, dtype=np.float32, n_workers=None):
    """Build neighbor lists for all frames, parallelized across CPU cores.

    Per-frame work is embarrassingly parallel. Worker behavior:

    - **Worker count** defaults to ``cpu_count() // LOCAL_WORLD_SIZE`` so
      DDP ranks on the same node don't oversubscribe CPU cores. Override
      with ``TORCHNEP_PREPROC_WORKERS`` env var.

    - **Start method** defaults to ``fork`` (fastest). Workers do pure numpy
      (neighbor list + type lookup) and never touch CUDA, so fork is safe
      even after the parent has initialized CUDA — same pattern used by
      PyTorch's ``DataLoader(num_workers>0)``. Override with
      ``TORCHNEP_MP_START_METHOD=spawn`` on systems where fork-after-CUDA
      behaves pathologically (rare).

    Disable pooling entirely with ``n_workers=1`` (useful for debugging).
    """
    rc_rad = config["cutoff_radial"]
    rc_ang = config["cutoff_angular"]
    type_names = config["type_names"]
    max_rc = max(rc_rad, rc_ang)

    if n_workers is None:
        cpu_total = os.cpu_count() or 1
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        n_workers = max(1, cpu_total // local_world)
        n_workers = int(os.environ.get("TORCHNEP_PREPROC_WORKERS", n_workers))

    if n_workers <= 1 or len(frames) < 64:
        return [_preprocess_one_frame((f, rc_rad, rc_ang, max_rc, type_names, dtype))
                for f in frames]

    import multiprocessing as mp
    method = os.environ.get("TORCHNEP_MP_START_METHOD", "fork")
    try:
        ctx = mp.get_context(method)
    except ValueError:
        ctx = mp.get_context("spawn")

    args = [(f, rc_rad, rc_ang, max_rc, type_names, dtype) for f in frames]
    chunksize = max(1, len(frames) // (n_workers * 4))
    with ctx.Pool(n_workers) as pool:
        return pool.map(_preprocess_one_frame, args, chunksize=chunksize)


def compute_max_neighbors(structures):
    """Return (max_NN_radial, max_NN_angular) over all structures."""
    max_rad = max_ang = 0
    for s in structures:
        n = s["natoms"]
        if len(s["pair_i_rad"]) > 0:
            counts = np.bincount(s["pair_i_rad"], minlength=n)
            max_rad = max(max_rad, int(counts.max()))
        if len(s["pair_i_ang"]) > 0:
            counts = np.bincount(s["pair_i_ang"], minlength=n)
            max_ang = max(max_ang, int(counts.max()))
    return max_rad, max_ang


@torch.no_grad()
def compute_q_scaler(model, data_store, batch_size=1000, backend="loop"):
    """Compute descriptor min/max across training set.

    Uses the cached-basis path (reusing data_store's precomputed Chebyshev +
    angular basis) — orders of magnitude faster than recomputing from rij.

    ``batch_size`` is a q-scaler-only knob; independent from the training
    batch size because q-scaler has no backward and can tolerate much bigger
    batches (default 1000). Set smaller if GPU memory is tight.
    ``backend`` should match the training backend so the type-pair contraction
    order of operations (and hence the floating-point accumulation) is the
    same as what training will see.
    """
    model.eval()
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    q_min = torch.full((model.dim,), float("inf"), dtype=dtype, device=dev)
    q_max = torch.full((model.dim,), float("-inf"), dtype=dtype, device=dev)

    for start in range(0, data_store.n, batch_size):
        end = min(start + batch_size, data_store.n)
        batch = data_store.collate(list(range(start, end)))
        q = ops.compute_descriptors_cached(
            batch["fk_rad"], batch["fk_ang"], batch["blm"],
            batch["pair_i_rad"], batch["pair_j_rad"],
            batch["pair_i_ang"], batch["pair_j_ang"],
            batch["atom_types"], batch["N"],
            model.c_param_2, model.c_param_3,
            model.n_max_radial, model.n_max_angular,
            model.l_max_3b, model.l_max_4b, model.l_max_5b,
            model.num_lm, model._c3b, model._c4b, model._c5b,
            dtype, dev,
            backend=backend,
        )
        q_min = torch.min(q_min, q.min(0).values)
        q_max = torch.max(q_max, q.max(0).values)

    model.train()
    return q_min, q_max


# ---------------------------------------------------------------------------
# LR scheduler helpers
# ---------------------------------------------------------------------------

def _make_lr_scheduler(optimizer, mode, factor, patience, step_size, min_lr):
    """Build the LR scheduler — "plateau" (default) or "step".

    "plateau" → ReduceLROnPlateau(factor, patience, min_lr=min_lr).
    "step"    → StepLR(step_size, gamma=factor); min_lr enforced manually
                after each step() via _scheduler_step (StepLR has no min_lr).
    """
    if mode == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=factor)
    # default: plateau
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=factor,
        patience=patience, min_lr=min_lr)


def _scheduler_step(scheduler, avg_loss, mode, optimizer, min_lr):
    """Advance the scheduler; for 'step' mode, clamp LR at min_lr manually."""
    if mode == "step":
        scheduler.step()
        for pg in optimizer.param_groups:
            if pg["lr"] < min_lr:
                pg["lr"] = min_lr
    else:
        scheduler.step(avg_loss)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(path, model, optimizer, scheduler, epoch, best_loss,
                     loss_weights=None):
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    m = m.module if hasattr(m, "module") else m
    state = {
        "epoch": epoch,
        "best_loss": best_loss,
        "model_state": m.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    if loss_weights is not None:
        state["loss_weights"] = loss_weights
    torch.save(state, path)


def _load_checkpoint(path, model, optimizer, scheduler, device):
    """Load checkpoint.

    Returns (start_epoch, best_loss, saved_loss_weights). The third element is
    the dict of loss weights active when the checkpoint was saved (or None for
    pre-feature checkpoints). Callers compare against current nep.in weights to
    decide whether to reset best_loss.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    m = m.module if hasattr(m, "module") else m
    m.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        # Scheduler class may have changed between runs (user switched
        # lr_scheduler mode). Tolerate that by skipping the state load —
        # scheduler just starts fresh this run.
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except (KeyError, ValueError, TypeError):
            pass
    return ckpt["epoch"], ckpt["best_loss"], ckpt.get("loss_weights")


# ---------------------------------------------------------------------------
# Unified training entry point (single GPU or torchrun DDP)
# ---------------------------------------------------------------------------

def train_nep(
    config_file: str,
    data_file: str,
    output_dir: str = ".",
    device: str = None,
    precision: str = "float32",
    backend: str = "auto",
    use_autograd_forces: bool = False,
    use_swa: bool = False,
    use_compile: bool = False,
    print_interval: int = 10,
    restart: bool = True,
    checkpoint_interval: int = 100,
    prediction_interval: int = 20,
    finetune_from: str = None,
    reset_lr: float = None,
    slim_types: bool = False,
    energy_key: str = "energy",
):
    """Train a NEP model on a single device (GPU / CPU / MPS).

    Hyperparameters (epoch / batch / lr / lambda_e,f,v / stage2* / …) come
    from ``config_file`` only. See README for the full nep.in reference.

    Launch:  python run_train.py

    Parameters
    ----------
    config_file, data_file, output_dir : paths.
    device : "cuda" | "cpu" | "mps" — auto-detected if omitted.
    precision : "float32" (default) or "float64".
    backend : "auto" | "loop" | "bmm" — see torchnep.ops.resolve_backend.
    use_autograd_forces : True → autograd-through-rij forces (slower, gold
        standard); False (default) → analytical chain rule.
    use_swa : True → maintain an averaged model during stage 2 and save it
        as ``nep_average.txt`` / ``nep_average.pt`` at the end.
    use_compile : wrap model in torch.compile (~10 % extra speedup).
    print_interval : log a line to screen every N epochs (all epochs still
        land in output.log).
    restart : on fresh output_dir, write new log; otherwise resume from
        checkpoint.pt if present.
    checkpoint_interval : save checkpoint.pt every N epochs.
    prediction_interval : every N epochs, run predict_from_store with the
        current nep_best weights and overwrite {energy,force,virial}_predict.out
        in output_dir — lets you watch the parity-plot converge live.
        Set to 0 or a negative value to disable.
    finetune_from : path to an existing .pt or nep.txt to load weights from.
    reset_lr : override LR after resume/finetune.
    slim_types : drop element types not present in data_file before training.
    energy_key : name of the comment-line tag read as the reference energy
        (default ``"energy"``). Set to ``"atomization_energy"`` to train
        against atomization energies instead of totals.
    """
    # ---- Device ----------------------------------------------------------
    if device is None:
        device = _default_device()
    dev = torch.device(device)
    dtype = torch.float32 if precision == "float32" else torch.float64

    # ---- Logging ---------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    _out_log_file = open(os.path.join(output_dir, "output.log"),
                         "a" if restart else "w")

    def _log(msg=""):
        print(msg)
        _out_log_file.write(msg + "\n")
        _out_log_file.flush()

    # ---- Banner ----------------------------------------------------------
    total_t0 = time.time()
    _log(_BANNER.rstrip())
    _log(f"   torchnep  v{__version__}   author: {_AUTHOR}")
    _log(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _log("")
    for line in _backend_info(dev):
        _log(line)
    _log(f"Precision: {precision}")
    _log("")

    # ---- Config (all hyperparameters from nep.in) -----------------------
    orig_config = parse_nep_in(config_file)
    config = orig_config
    # Model regularisation coefficients
    lambda_1 = config["lambda_1"]
    lambda_2 = config["lambda_2"]
    # Training schedule + loss weights
    num_epochs         = config["num_epochs"]
    batch_size         = config["batch_size"]
    lr                 = config["lr"]
    stop_lr            = config["stop_lr"]
    scheduler_patience = config["scheduler_patience"]
    scheduler_factor   = config["scheduler_factor"]
    lr_scheduler_mode  = config["lr_scheduler"]     # "plateau" | "step"
    step_size          = config["step_size"]        # only used when mode=="step"
    max_grad_norm      = config["max_grad_norm"]
    pref_e             = config["lambda_e"]
    pref_f             = config["lambda_f"]
    pref_v             = config["lambda_v"]
    # Optional stage-2 block
    stage2             = config["stage2"]
    start_stage2       = config.get("start_stage2")  # may be None → auto 0.75·num_epochs
    stage2_lr          = config["stage2_lr"]
    stage2_pref_e      = config["stage2_pref_e"]
    stage2_pref_f      = config["stage2_pref_f"]
    stage2_pref_v      = config["stage2_pref_v"]

    # ---- Data ------------------------------------------------------------
    _log("Loading training data...")
    _log(f"  energy label: {energy_key}")
    frames = read_xyz(data_file, energy_key=energy_key)
    _log(f"  {len(frames)} structures")

    # Sort by natoms + bucket for stable batch compute (see _sort_and_shard).
    frames, boundaries, _global_idx, _dropped = _sort_and_shard(
        frames, rank=0, world_size=1, batch_size=batch_size)
    num_buckets = len(boundaries)
    _log(f"  {num_buckets} buckets (sorted by natoms, batch_size={batch_size})")

    # slim_types: detect which element types actually appear in the data and
    # narrow config before building neighbor lists / GPUDataStore / model.
    # This makes the entire training run faster, not just the output file.
    _slim_keep = None  # None = no slimming; list = types to keep
    if slim_types:
        seen_species = set(s for f in frames for s in f["species"])
        keep = [t for t in orig_config["type_names"] if t in seen_species]
        removed = [t for t in orig_config["type_names"] if t not in keep]
        if removed:
            _slim_keep = keep
            config = dict(orig_config)
            config["type_names"] = keep
            config["num_types"] = len(keep)
            _log(f"  slim_types: {orig_config['type_names']} → {keep} "
                 f"(removing: {removed})")
        else:
            _log("  slim_types: all types present in data, nothing to remove")

    _log("Building neighbor lists...")
    t0 = time.time()
    np_dtype = np.float64 if precision == "float64" else np.float32
    structures = preprocess_structures(frames, config, np_dtype)
    _log(f"  Done in {time.time() - t0:.1f}s")

    max_NN_rad, max_NN_ang = compute_max_neighbors(structures)

    _log(f"Pre-loading data to {dev} (with cached basis)...")
    t0 = time.time()
    data_store = GPUDataStore(structures, dev, dtype, config=config)
    del structures
    if dev.type == "cuda":
        torch.cuda.synchronize()
    _log(f"  Loaded ({time.time() - t0:.1f}s)")
    _log(f"  Data: {data_store.n} structures, "
         f"{data_store.n_energy} with energy, "
         f"{data_store.n_forces} with forces, "
         f"{data_store.n_virial} with virial")

    # ---- Model -----------------------------------------------------------
    model = NEPModel(config).to(dtype).to(dev)

    if finetune_from is not None:
        # Load pre-trained weights; skip random b1 init from mean_epa.
        # q_scaler will be recomputed below on the new dataset.
        ft_path = finetune_from

        def _load_weights(target, path):
            if path.endswith(".pt"):
                state = torch.load(path, map_location=dev, weights_only=False)
                if "model_state" in state:
                    state = state["model_state"]
                target.load_state_dict(state, strict=True)
            else:
                target.load_weights_from_nep_txt(path)

        if _slim_keep is not None:
            # Load full pre-trained model (orig arch), then slim to current config
            full_model = NEPModel(orig_config).to(dtype).to(dev)
            _load_weights(full_model, ft_path)
            slimmed = slim_model(full_model, config["type_names"])
            model.load_state_dict(slimmed.state_dict())
            del full_model, slimmed
            _log(f"Fine-tuning from: {ft_path}  "
                 f"[{orig_config['num_types']} → {config['num_types']} types]")
        else:
            _load_weights(model, ft_path)
            _log(f"Fine-tuning from: {ft_path}")
        _log(f"Model: {sum(p.numel() for p in model.parameters())} params, "
             f"dim={model.dim}, b1={model.b1.item():.4f}")
    else:
        mean_epa = np.mean([data_store.energy[i] / data_store.natoms[i]
                            for i in range(data_store.n)
                            if data_store.has_energy_flag[i]])
        with torch.no_grad():
            model.b1.fill_(-mean_epa)
        _log(f"Model: {sum(p.numel() for p in model.parameters())} params, "
             f"dim={model.dim}, b1 init={model.b1.item():.4f}")

    # Resolve "auto" backend now that we know num_types (and the CUDA kernel
    # load attempt above has updated availability).
    from .ops import resolve_backend as _resolve_backend
    backend = _resolve_backend(backend, num_types=model.num_types)
    _log(f"Compute backend: {backend}")

    _log("Computing q_scaler...")
    t0 = time.time()
    q_min, q_max = compute_q_scaler(model, data_store, backend=backend)
    model.set_q_scaler(q_min, q_max)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    _log(f"  Done in {time.time() - t0:.1f}s")

    if use_compile and hasattr(torch, "compile"):
        _log("Compiling model with torch.compile...")
        model = torch.compile(model)

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=lambda_2, amsgrad=True)

    if stage2 and start_stage2 is None:
        start_stage2 = max(1, int(num_epochs * 0.75))

    lr_scheduler = _make_lr_scheduler(
        optimizer, lr_scheduler_mode, scheduler_factor,
        scheduler_patience, step_size, stop_lr)

    def _loss_fn(pred, ref):
        return torch.mean((pred - ref) ** 2)

    swa_model = None
    stage2_scheduler = None
    if stage2:
        stage2_scheduler = _make_lr_scheduler(
            optimizer, lr_scheduler_mode, scheduler_factor,
            scheduler_patience, step_size, stop_lr)
        if use_swa:
            swa_model = AveragedModel(raw_model)

    # Snapshot of current loss weights — saved in checkpoint so a restart can
    # detect that the user edited nep.in and reset best_loss (otherwise the
    # old-scale best_loss would keep the new run from ever saving a new best).
    cur_loss_weights = {
        "lambda_e": pref_e, "lambda_f": pref_f, "lambda_v": pref_v,
        "stage2_pref_e": stage2_pref_e, "stage2_pref_f": stage2_pref_f,
        "stage2_pref_v": stage2_pref_v,
    }

    ckpt_path = os.path.join(output_dir, "checkpoint.pt")
    start_epoch = 1
    best_loss = float("inf")
    stage2_lr_applied = False  # tracks whether stage2 lr/reset has fired yet
    if restart and os.path.exists(ckpt_path) and finetune_from is None:
        start_epoch, best_loss, saved_loss_weights = _load_checkpoint(
            ckpt_path, model, optimizer, lr_scheduler, dev)
        start_epoch += 1
        _log(f"Resumed from checkpoint: epoch {start_epoch - 1}, "
             f"best_loss={best_loss:.4e}")
        if saved_loss_weights is not None and saved_loss_weights != cur_loss_weights:
            _log("Loss weights changed since checkpoint was saved — "
                 "resetting best_loss so the new scale can establish a new best.")
            _log(f"  saved:   {saved_loss_weights}")
            _log(f"  current: {cur_loss_weights}")
            best_loss = float("inf")

    # Override LR after checkpoint load (useful when resuming with a new lr)
    if reset_lr is not None:
        for pg in optimizer.param_groups:
            pg["lr"] = reset_lr
        _log(f"LR reset to {reset_lr:.2e}")

    n_structs = data_store.n
    # has_forces / has_virial are recomputed per-epoch inside the loop using
    # the CURRENT stage's weights — so a stage-1 weight of 0 can still enable
    # computation in stage 2 (and vice versa). Do NOT latch them here.

    loss_log_mode = "a" if (restart and start_epoch > 1) else "w"
    loss_log = open(os.path.join(output_dir, "loss.out"), loss_log_mode)
    if loss_log_mode == "w":
        loss_log.write("epoch  loss  rmse_e(eV/atom)  rmse_f(eV/A)  "
                       "rmse_v(eV/atom)  rmse_stress(GPa)  gnorm\n")

    backend_str = {
        "loop": "PyTorch type-pair loop",
        "bmm":  "PyTorch fancy-index + torch.bmm (batched GEMM)",
    }.get(backend, backend)
    force_str = ("autograd (create_graph)" if use_autograd_forces
                 else "analytical")
    clip_str = f"grad_clip={max_grad_norm}" if max_grad_norm > 0 else "no grad clip"
    _log(f"\nTraining: epochs {start_epoch}-{num_epochs}, "
         f"batch={batch_size}, dtype={precision}")
    _log(f"Backend: {backend_str} | forces: {force_str} | "
         f"{clip_str} | loss: MSE")
    if lr_scheduler_mode == "step":
        sched_desc = (f"StepLR(step_size={step_size}, gamma={scheduler_factor})"
                      f", stop_lr={stop_lr}")
    else:
        sched_desc = (f"ReduceLROnPlateau(patience={scheduler_patience}, "
                      f"factor={scheduler_factor}), stop_lr={stop_lr}")
    _log(f"LR: {lr}, {sched_desc}")
    _log(f"Loss weights: E={pref_e}  F={pref_f}  V={pref_v}")
    if stage2:
        _log(f"Stage 2: epoch {start_stage2}→{num_epochs}, "
             f"lr={stage2_lr}, {sched_desc}, "
             f"SWA={'ON' if use_swa else 'OFF'}")
        _log(f"Stage 2 weights: E={stage2_pref_e}  "
             f"F={stage2_pref_f}  V={stage2_pref_v}")
    _log("-" * 72)

    train_t0 = time.time()

    try:
        for epoch in range(start_epoch, num_epochs + 1):
            t_epoch = time.time()
            model.train()

            # Bucket batching: frames in data_store are sorted by natoms;
            # a "bucket" is a contiguous slice of ``batch_size`` frames and
            # corresponds to one DDP iteration. Shuffling only the bucket
            # ORDER keeps batches natoms-homogeneous while randomising the
            # per-epoch iteration sequence.
            g = torch.Generator()
            g.manual_seed(epoch)
            bucket_perm = torch.randperm(num_buckets, generator=g).tolist()

            # Loss / gnorm accumulators stay on GPU; .item() once at epoch end
            # to avoid per-step GPU↔CPU sync (a major source of GPU-util sawtooth).
            sum_le = torch.zeros((), dtype=torch.float64, device=dev)
            sum_lf = torch.zeros((), dtype=torch.float64, device=dev)
            sum_lv = torch.zeros((), dtype=torch.float64, device=dev)
            sum_ls = torch.zeros((), dtype=torch.float64, device=dev)
            sum_e_structs = torch.zeros((), dtype=torch.float64, device=dev)
            sum_f_atoms = torch.zeros((), dtype=torch.float64, device=dev)
            sum_v_structs = torch.zeros((), dtype=torch.float64, device=dev)
            max_gn_t = torch.zeros((), dtype=torch.float64, device=dev)

            in_stage2 = stage2 and epoch >= start_stage2
            if in_stage2:
                cur_pref_e, cur_pref_f, cur_pref_v = (
                    stage2_pref_e, stage2_pref_f, stage2_pref_v)
                # Apply stage 2 lr + reset the first time we hit stage 2 in
                # THIS run — covers both the natural transition and resuming
                # from a stage-1 checkpoint whose start_epoch has already
                # crossed into stage 2 (``epoch == start_stage2`` would miss
                # the second case, leaving the optimizer on stage-1 lr).
                if not stage2_lr_applied:
                    stage2_lr_applied = True
                    # Save an end-of-stage-1 snapshot BEFORE applying stage-2
                    # weights/lr, so the user can restart from this point with
                    # different stage-2 settings. Guard: only save if we
                    # actually trained through stage 1 in this run; if we
                    # resumed from a mid-stage-2 checkpoint, the current state
                    # isn't stage-1 state and we mustn't overwrite.
                    if start_epoch <= start_stage2:
                        raw_model.save_nep_txt(
                            os.path.join(output_dir, "nep_stage1.txt"),
                            max_NN_rad, max_NN_ang)
                        torch.save(raw_model.state_dict(),
                                   os.path.join(output_dir, "nep_stage1.pt"))
                        _log(f"\nSaved end-of-stage-1 snapshot: "
                             f"nep_stage1.pt / nep_stage1.txt")
                    for pg in optimizer.param_groups:
                        pg['lr'] = stage2_lr
                    _log(f"\n{'='*72}")
                    tag = ("Stage 2 started" if epoch == start_stage2
                           else f"Stage 2 resumed (from checkpoint)")
                    _log(f"{tag} at epoch {epoch}: "
                         f"E_w={cur_pref_e}, F_w={cur_pref_f}, "
                         f"V_w={cur_pref_v}, lr={stage2_lr:.2e}")
                    _log(f"{'='*72}")
                    # Stage 2 uses different loss weights — old best_loss is
                    # on a different scale, so reset it whichever way we
                    # entered.
                    best_loss = float("inf")
            else:
                cur_pref_e, cur_pref_f, cur_pref_v = pref_e, pref_f, pref_v

            # Per-epoch compute eligibility: a weight of 0 means "don't compute
            # this channel". This is recomputed every epoch so a stage-1 zero
            # weight doesn't block stage-2 computation (see stage transition
            # above) — and so pref_v=0 really skips virial compute/backward.
            has_forces = data_store.has_forces and cur_pref_f > 0
            has_virial = data_store.has_virial and cur_pref_v > 0

            for bi in bucket_perm:
                start, end = boundaries[bi]
                idx = list(range(start, end))
                batch = data_store.collate(idx)

                if use_autograd_forces:
                    result = raw_model.compute_properties(
                        batch["rij_rad"], batch["rij_ang"],
                        batch["pair_i_rad"], batch["pair_j_rad"],
                        batch["pair_i_ang"], batch["pair_j_ang"],
                        batch["atom_types"], batch["N"],
                        batch["struct_idx"], batch["num_structures"],
                        need_forces=has_forces, need_virial=has_virial,
                        backend=backend)
                else:
                    result = raw_model.compute_properties_cached(
                        batch, need_forces=has_forces, need_virial=has_virial,
                        backend=backend)

                e_pa_pred = result["Etot"] / batch["natoms"]
                e_pa_ref = batch["energy"] / batch["natoms"]
                e_mask = batch["energy_mask"]
                loss = torch.tensor(0.0, dtype=dtype, device=dev)
                # sum_l* accumulates per-batch MSE so the rmse_* columns in
                # the log are real RMSE. Optimizer sees _loss_fn (MSE) too.
                # Accumulators stay GPU-side (no .item() in the hot loop).
                # sum_le/lf/lv/ls accumulate ``mean_sq * count`` so the per-
                # sample MSE recovered as sum / total_count is identical to
                # before — just deferred-sync.
                if e_mask.any():
                    diff_e = e_pa_pred[e_mask] - e_pa_ref[e_mask]
                    loss_e = (diff_e ** 2).mean()
                    loss = loss + cur_pref_e * loss_e
                    sum_le += loss_e.detach().to(torch.float64) * e_mask.sum().to(torch.float64)

                if has_forces:
                    f_mask = batch["force_mask"]
                    if f_mask.any():
                        f_pred = result["forces"][f_mask]
                        f_ref = batch["forces"][f_mask]
                        loss_f = ((f_pred - f_ref) ** 2).mean()
                        loss = loss + cur_pref_f * loss_f
                        sum_lf += loss_f.detach().to(torch.float64) * f_mask.sum().to(torch.float64)

                if has_virial and "virial" in result:
                    v_mask = batch["virial_mask"]
                    if v_mask.any():
                        v_atom = result["virial"]
                        v_sys = torch.zeros(batch["num_structures"], 9,
                                            dtype=dtype, device=dev)
                        si = batch["struct_idx"].unsqueeze(-1).expand_as(v_atom)
                        v_sys.scatter_add_(0, si, v_atom)
                        v_ref = batch["virial"]
                        if v_ref.shape[1] == 9:
                            na = batch["natoms"][v_mask].unsqueeze(-1)
                            v_pred_pa = v_sys[v_mask] / na
                            v_ref_pa = v_ref[v_mask] / na
                            v_diff = v_pred_pa - v_ref_pa
                            loss_v = (v_diff ** 2).mean()
                            loss = loss + cur_pref_v * loss_v
                            sum_lv += loss_v.detach().to(torch.float64) * v_mask.sum().to(torch.float64)
                            # Stress RMSE (eV/Å³): convert the same diff using
                            # per-frame (natoms/volume). Sign cancels under MSE.
                            scale = (batch["natoms"][v_mask]
                                     / batch["volumes"][v_mask]).unsqueeze(-1)
                            s_diff = v_diff * scale
                            sum_ls += (s_diff ** 2).mean().detach().to(torch.float64) * v_mask.sum().to(torch.float64)

                if lambda_1 > 0:
                    l1 = sum(p.abs().sum() for p in model.parameters())
                    loss = loss + lambda_1 * l1

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # gn stays as a tensor — only `.item()` it when we need to
                # branch on isfinite (one sync per step instead of two).
                if max_grad_norm > 0:
                    gn_t = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm)
                else:
                    gn_t = torch.sqrt(sum(
                        p.grad.norm()**2 for p in raw_model.parameters()
                        if p.grad is not None))
                gn = gn_t.item()

                if not np.isfinite(gn):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()

                if in_stage2 and swa_model is not None:
                    swa_model.update_parameters(raw_model)

                sum_e_structs += batch["energy_mask"].sum().to(torch.float64)
                sum_f_atoms += batch["force_mask"].sum().to(torch.float64)
                sum_v_structs += batch["virial_mask"].sum().to(torch.float64)
                if gn_t.dtype != torch.float64:
                    gn_t = gn_t.to(torch.float64)
                max_gn_t = torch.maximum(max_gn_t, gn_t.detach())

            # Per-sample (not per-batch) averaging so avg_loss is self-
            # consistent with rmse_{e,f,v}: avg_loss == Σ pref_X · MSE_X
            # where each MSE_X aggregates over all samples in the epoch.
            from .constants import EV_PER_A3_TO_GPa
            # Single sync at epoch end: pull all GPU accumulators to host
            # in one go (one `.tolist()`) instead of 4–6 .item() per step.
            _vals = torch.stack([
                sum_le, sum_lf, sum_lv, sum_ls,
                sum_e_structs, sum_f_atoms, sum_v_structs, max_gn_t,
            ]).tolist()
            (sum_le, sum_lf, sum_lv, sum_ls,
             sum_e_structs, sum_f_atoms, sum_v_structs, max_gn) = _vals
            mse_e = sum_le / max(sum_e_structs, 1.0)
            mse_f = sum_lf / max(sum_f_atoms, 1.0) if sum_lf > 0 else 0.0
            mse_v = sum_lv / max(sum_v_structs, 1.0) if sum_lv > 0 else 0.0
            mse_s = sum_ls / max(sum_v_structs, 1.0) if sum_ls > 0 else 0.0
            avg_loss = (cur_pref_e * mse_e + cur_pref_f * mse_f
                        + cur_pref_v * mse_v)
            # Output units: eV/atom (E, V), eV/Å (F), GPa (stress).
            rmse_e = np.sqrt(mse_e)
            rmse_f = np.sqrt(mse_f)
            rmse_v = np.sqrt(mse_v)
            rmse_s_gpa = np.sqrt(mse_s) * EV_PER_A3_TO_GPa
            dt = time.time() - t_epoch

            if in_stage2 and stage2_scheduler is not None:
                _scheduler_step(stage2_scheduler, avg_loss,
                                lr_scheduler_mode, optimizer, stop_lr)
            elif not in_stage2:
                _scheduler_step(lr_scheduler, avg_loss,
                                lr_scheduler_mode, optimizer, stop_lr)

            loss_log.write(f"{epoch} {avg_loss:.6e} {rmse_e:.6f} "
                           f"{rmse_f:.6f} {rmse_v:.6f} {rmse_s_gpa:.4f} "
                           f"{max_gn:.2f}\n")
            loss_log.flush()

            stage_str = "[S2] " if in_stage2 else ""
            cur_lr = optimizer.param_groups[0]['lr']
            v_str = (f" | V {rmse_v:.5f} eV/atom | S {rmse_s_gpa:.3f} GPa"
                     if has_virial else "")
            line = (f"{stage_str}Epoch {epoch:4d} | loss {avg_loss:.4e} | "
                    f"E {rmse_e:.5f} eV/atom | F {rmse_f:.5f} eV/A"
                    f"{v_str} | gnorm {max_gn:.1f} | "
                    f"lr {cur_lr:.2e} | {dt:.1f}s")
            if epoch % print_interval == 0 or epoch == 1:
                _log(line)
            else:
                _out_log_file.write(line + "\n")
                _out_log_file.flush()

            if avg_loss < best_loss:
                best_loss = avg_loss
                raw_model.save_nep_txt(
                    os.path.join(output_dir, "nep_best.txt"),
                    max_NN_rad, max_NN_ang)
                torch.save(raw_model.state_dict(),
                           os.path.join(output_dir, "nep_best.pt"))

            if epoch % checkpoint_interval == 0 or epoch == num_epochs:
                _save_checkpoint(ckpt_path, model, optimizer,
                                 lr_scheduler, epoch, best_loss,
                                 loss_weights=cur_loss_weights)

            # Interim predict — overwrites the same output files, so users can
            # refresh the parity plot live. Runs on the CURRENT-epoch weights
            # (not nep_best) so the predict loss matches what was just logged
            # for this epoch: it should fall between this epoch's and the next
            # epoch's displayed loss (current weights = end-of-epoch, whereas
            # the screen average covers weights that were still improving
            # throughout the epoch).
            # Skip on the final epoch — the end-of-training predict (below)
            # immediately overwrites these files with the final-epoch result.
            if (prediction_interval > 0
                    and epoch % prediction_interval == 0
                    and epoch != num_epochs):
                # Silent interim predict — reuses data_store's preprocessed
                # neighbor lists + basis (no xyz re-read, no recompute).
                predict_from_store(raw_model, data_store, output_dir,
                                   batch_size=batch_size, backend=backend,
                                   verbose=False)
    finally:
        if loss_log is not None:
            loss_log.close()

    # Final-epoch model (what the current weights actually are).
    raw_model.save_nep_txt(os.path.join(output_dir, "nep_final.txt"),
                           max_NN_rad, max_NN_ang)
    # SWA-averaged model (only when user opted in and stage 2 ran).
    if swa_model is not None:
        swa_state = swa_model.module.state_dict()
        # Keep a copy of the final-epoch weights so we can restore them
        # after saving SWA — the end-of-training predict below must see
        # final weights, not SWA-averaged ones.
        final_state = {k: v.clone() for k, v in raw_model.state_dict().items()}
        raw_model.load_state_dict(swa_state)
        raw_model.save_nep_txt(os.path.join(output_dir, "nep_average.txt"),
                               max_NN_rad, max_NN_ang)
        torch.save(swa_state, os.path.join(output_dir, "nep_average.pt"))
        raw_model.load_state_dict(final_state)
        _log("SWA model saved to nep_average.txt / nep_average.pt")

    train_time = time.time() - train_t0
    h, rem = divmod(train_time, 3600)
    m_, s = divmod(rem, 60)
    _log(f"\nDone. Best loss: {best_loss:.6e}")
    _log(f"Training time: {int(h):02d}:{int(m_):02d}:{s:04.1f}")

    # End-of-training predict reuses the in-memory data_store (no xyz re-read)
    # and the final-epoch weights in raw_model (no model-file round-trip).
    _log("\nRunning prediction on training set (final-epoch model)...")
    pred_t0 = time.time()
    predict_from_store(raw_model, data_store, output_dir,
                       batch_size=batch_size, backend=backend,
                       verbose=False)
    _log(f"  Prediction time: {time.time() - pred_t0:.1f}s")

    total_time = time.time() - total_t0
    h, rem = divmod(total_time, 3600)
    m_, s = divmod(rem, 60)
    _log(f"\nTotal time (data + train + predict): "
         f"{int(h):02d}:{int(m_):02d}:{s:04.1f}")
    _log(f"Output: {output_dir}/")
    _out_log_file.close()
