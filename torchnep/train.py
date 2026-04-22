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
    """Sort frames by natoms and return this rank's local slice + bucket count.

    Parameters
    ----------
    frames       : list of frame dicts (as returned by read_xyz).
    rank         : int in [0, world_size).
    world_size   : int  ≥ 1.
    batch_size   : int  per-rank batch size.

    Returns
    -------
    local_frames : list of frames this rank should preprocess / store.
                   Length is ``num_buckets * batch_size`` for DDP, or
                   ``len(frames)`` for single-GPU (last bucket may be partial).
    num_buckets  : number of buckets == iterations per epoch on this rank.
    dropped      : number of frames dropped from the end of the dataset
                   (always 0 for single-GPU).
    """
    n = len(frames)
    B = batch_size
    W = world_size

    sorted_frames = sorted(frames, key=lambda f: f["natoms"])

    if W == 1:
        num_buckets = (n + B - 1) // B
        return sorted_frames, num_buckets, 0

    num_full_buckets = n // (W * B)
    if num_full_buckets == 0:
        raise RuntimeError(
            f"dataset too small for bucket batching with DDP: need at least "
            f"world_size × batch_size = {W * B} frames, got {n}. Reduce "
            f"batch_size or run on a single GPU.")
    dropped = n - num_full_buckets * W * B

    local = []
    for b in range(num_full_buckets):
        bucket_start = b * W * B
        rank_start = bucket_start + rank * B
        local.extend(sorted_frames[rank_start:rank_start + B])
    return local, num_full_buckets, dropped



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

        batch = {
            "N": N_total, "num_structures": B,
            "atom_types": atom_types, "struct_idx": struct_idx,
            "pair_i_rad": pi_r, "pair_j_rad": pj_r, "rij_rad": rij_r,
            "pair_i_ang": pi_a, "pair_j_ang": pj_a, "rij_ang": rij_a,
            "energy": energy, "natoms": natoms,
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
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(path, model, optimizer, scheduler, epoch, best_loss):
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
    torch.save(state, path)


def _load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    m = m.module if hasattr(m, "module") else m
    m.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt["best_loss"]


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
    frames, num_buckets, _dropped = _sort_and_shard(
        frames, rank=0, world_size=1, batch_size=batch_size)
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

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor,
        patience=scheduler_patience, min_lr=stop_lr)

    def _loss_fn(pred, ref):
        return torch.mean((pred - ref) ** 2)

    swa_model = None
    stage2_scheduler = None
    if stage2:
        stage2_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=scheduler_factor,
            patience=scheduler_patience, min_lr=stop_lr)
        if use_swa:
            swa_model = AveragedModel(raw_model)

    ckpt_path = os.path.join(output_dir, "checkpoint.pt")
    start_epoch = 1
    best_loss = float("inf")
    best_state = None  # cached state_dict of the best-loss model so far
    if restart and os.path.exists(ckpt_path) and finetune_from is None:
        start_epoch, best_loss = _load_checkpoint(
            ckpt_path, model, optimizer, lr_scheduler, dev)
        start_epoch += 1
        _log(f"Resumed from checkpoint: epoch {start_epoch - 1}, "
             f"best_loss={best_loss:.4e}")

    # Override LR after checkpoint load (useful when resuming with a new lr)
    if reset_lr is not None:
        for pg in optimizer.param_groups:
            pg["lr"] = reset_lr
        _log(f"LR reset to {reset_lr:.2e}")

    n_structs = data_store.n
    has_forces = data_store.has_forces and pref_f > 0
    has_virial = data_store.has_virial and pref_v > 0

    loss_log_mode = "a" if (restart and start_epoch > 1) else "w"
    loss_log = open(os.path.join(output_dir, "loss.out"), loss_log_mode)
    if loss_log_mode == "w":
        loss_log.write("epoch  loss  rmse_e(meV/atom)  rmse_f(eV/A)  "
                       "rmse_v(meV/atom)  gnorm\n")

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
    _log(f"LR: {lr}, ReduceLROnPlateau(patience={scheduler_patience}, "
         f"factor={scheduler_factor}), stop_lr={stop_lr}")
    _log(f"Loss weights: E={pref_e}  F={pref_f}  V={pref_v}")
    if stage2:
        _log(f"Stage 2: epoch {start_stage2}→{num_epochs}, "
             f"lr={stage2_lr}, ReduceLROnPlateau, "
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

            sum_le = sum_lf = sum_lv = 0.0
            sum_e_structs = sum_f_atoms = sum_v_structs = 0
            max_gn = 0.0

            in_stage2 = stage2 and epoch >= start_stage2
            if in_stage2:
                cur_pref_e, cur_pref_f, cur_pref_v = (
                    stage2_pref_e, stage2_pref_f, stage2_pref_v)
                if epoch == start_stage2:
                    for pg in optimizer.param_groups:
                        pg['lr'] = stage2_lr
                    _log(f"\n{'='*72}")
                    _log(f"Stage 2 started at epoch {epoch}: "
                         f"E_w={cur_pref_e}, F_w={cur_pref_f}, "
                         f"V_w={cur_pref_v}, lr={stage2_lr:.2e}")
                    _log(f"{'='*72}")
                    best_loss = float("inf")
            else:
                cur_pref_e, cur_pref_f, cur_pref_v = pref_e, pref_f, pref_v

            for bi in bucket_perm:
                start = bi * batch_size
                end = min(start + batch_size, n_structs)
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
                if e_mask.any():
                    diff_e = e_pa_pred[e_mask] - e_pa_ref[e_mask]
                    loss_e = _loss_fn(e_pa_pred[e_mask], e_pa_ref[e_mask])
                    loss = loss + cur_pref_e * loss_e
                    sum_le += (diff_e ** 2).mean().item() * e_mask.sum().item()

                if has_forces:
                    f_mask = batch["force_mask"]
                    if f_mask.any():
                        f_pred = result["forces"][f_mask]
                        f_ref = batch["forces"][f_mask]
                        loss_f = _loss_fn(f_pred, f_ref)
                        loss = loss + cur_pref_f * loss_f
                        sum_lf += ((f_pred - f_ref) ** 2).mean().item() * f_mask.sum().item()

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
                            loss_v = _loss_fn(v_pred_pa, v_ref_pa)
                            loss = loss + cur_pref_v * loss_v
                            sum_lv += ((v_pred_pa - v_ref_pa) ** 2).mean().item() * v_mask.sum().item()

                if lambda_1 > 0:
                    l1 = sum(p.abs().sum() for p in model.parameters())
                    loss = loss + lambda_1 * l1

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if max_grad_norm > 0:
                    gn = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm).item()
                else:
                    gn = torch.sqrt(sum(
                        p.grad.norm()**2 for p in raw_model.parameters()
                        if p.grad is not None)).item()

                if not np.isfinite(gn):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()

                if in_stage2 and swa_model is not None:
                    swa_model.update_parameters(raw_model)

                sum_e_structs += batch["energy_mask"].sum().item()
                sum_f_atoms += batch["force_mask"].sum().item()
                sum_v_structs += batch["virial_mask"].sum().item()
                max_gn = max(max_gn, gn)

            # Per-sample (not per-batch) averaging so avg_loss is self-
            # consistent with rmse_{e,f,v}: avg_loss == Σ pref_X · MSE_X
            # where each MSE_X aggregates over all samples in the epoch.
            mse_e = sum_le / max(sum_e_structs, 1)
            mse_f = sum_lf / max(sum_f_atoms, 1) if sum_lf > 0 else 0.0
            mse_v = sum_lv / max(sum_v_structs, 1) if sum_lv > 0 else 0.0
            avg_loss = (cur_pref_e * mse_e + cur_pref_f * mse_f
                        + cur_pref_v * mse_v)
            rmse_e = np.sqrt(mse_e) * 1000
            rmse_f = np.sqrt(mse_f)
            rmse_v = np.sqrt(mse_v) * 1000
            dt = time.time() - t_epoch

            if in_stage2 and stage2_scheduler is not None:
                stage2_scheduler.step(avg_loss)
            elif not in_stage2:
                lr_scheduler.step(avg_loss)

            loss_log.write(f"{epoch} {avg_loss:.6e} {rmse_e:.4f} "
                           f"{rmse_f:.4f} {rmse_v:.4f} {max_gn:.2f}\n")
            loss_log.flush()

            stage_str = "[S2] " if in_stage2 else ""
            cur_lr = optimizer.param_groups[0]['lr']
            v_str = f" | V {rmse_v:.1f} meV/atom" if has_virial else ""
            line = (f"{stage_str}Epoch {epoch:4d} | loss {avg_loss:.4e} | "
                    f"E {rmse_e:.1f} meV/atom | F {rmse_f:.4f} eV/A"
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
                # Cache the best state for interim predicts (no file round-trip).
                best_state = {k: v.detach().clone()
                              for k, v in raw_model.state_dict().items()}

            if epoch % checkpoint_interval == 0 or epoch == num_epochs:
                _save_checkpoint(ckpt_path, model, optimizer,
                                 lr_scheduler, epoch, best_loss)

            # Interim predict on the current nep_best weights — overwrites the
            # same output files, so users can refresh the parity plot live.
            # Skip on the final epoch — the end-of-training predict (below)
            # immediately overwrites these files with the final-epoch result.
            if (prediction_interval > 0
                    and epoch % prediction_interval == 0
                    and epoch != num_epochs
                    and best_state is not None):
                # Silent interim predict — reuses data_store's preprocessed
                # neighbor lists + basis (no xyz re-read, no recompute).
                current_state = {k: v.detach().clone()
                                 for k, v in raw_model.state_dict().items()}
                raw_model.load_state_dict(best_state)
                predict_from_store(raw_model, data_store, output_dir,
                                   batch_size=batch_size, backend=backend,
                                   verbose=False)
                raw_model.load_state_dict(current_state)
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
