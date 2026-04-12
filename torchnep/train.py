"""
NEP training with PyTorch — GPU-optimized.

Key performance features:
- Pre-loads ALL structure data to GPU memory (eliminates CPU→GPU transfer)
- GPU-side batch collation (fast concat of pre-loaded tensors)
- torch.compile for kernel fusion
- Mixed precision support
- Gradient accumulation for large effective batch sizes
"""

import os
import time
import torch
import numpy as np
from typing import List, Dict, Tuple

from .model import NEPModel
from .data import read_xyz, parse_nep_in
from . import ops


def _default_device() -> str:
    """Select best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Neighbor list construction (numpy, CPU, runs once at startup)
# ---------------------------------------------------------------------------

def build_neighbor_list_np(positions, cell, cutoff):
    """Build neighbor list using numpy (for preprocessing). Returns arrays."""
    N = positions.shape[0]
    inv_cell = np.linalg.inv(cell)

    n_rep = [int(np.ceil(cutoff / (1.0 / np.linalg.norm(inv_cell[i])))) for i in range(3)]

    a_r = np.arange(-n_rep[0], n_rep[0] + 1)
    b_r = np.arange(-n_rep[1], n_rep[1] + 1)
    c_r = np.arange(-n_rep[2], n_rep[2] + 1)
    shifts_frac = np.stack(np.meshgrid(a_r, b_r, c_r, indexing="ij"), axis=-1)
    shifts_frac = shifts_frac.reshape(-1, 3).astype(positions.dtype)
    shifts_cart = shifts_frac @ cell
    S = shifts_cart.shape[0]

    # Memory check: fully vectorized if feasible
    if N * N * S < 8_000_000:
        disp = (positions[None, :, None, :] + shifts_cart[None, None, :, :]
                - positions[:, None, None, :])
        dist = np.linalg.norm(disp, axis=-1)
        zero_shift = np.all(shifts_frac == 0, axis=1)
        self_mask = np.eye(N, dtype=bool)[:, :, None] & zero_shift[None, None, :]
        valid = (dist < cutoff) & (dist > 1e-10) & ~self_mask
        idx_i, idx_j, idx_s = np.where(valid)
        return idx_i.astype(np.int64), idx_j.astype(np.int64), disp[idx_i, idx_j, idx_s]
    else:
        # Per-shift vectorization: O(S * N^2) but each shift is vectorized
        zero_shift = np.all(shifts_frac == 0, axis=1)
        all_i, all_j, all_rij = [], [], []
        idx_j_base = np.arange(N)
        for si in range(S):
            shifted = positions + shifts_cart[si]  # (N, 3)
            # All pairwise displacements: disp[i, j] = shifted[j] - positions[i]
            disp = shifted[None, :, :] - positions[:, None, :]  # (N, N, 3)
            dist = np.linalg.norm(disp, axis=-1)  # (N, N)
            valid = (dist < cutoff) & (dist > 1e-10)
            if zero_shift[si]:
                np.fill_diagonal(valid, False)
            ii, jj = np.where(valid)
            if len(ii) > 0:
                all_i.append(ii)
                all_j.append(jj)
                all_rij.append(disp[ii, jj])
        if not all_i:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3), positions.dtype)
        return (np.concatenate(all_i).astype(np.int64),
                np.concatenate(all_j).astype(np.int64),
                np.concatenate(all_rij))


# ---------------------------------------------------------------------------
# GPU data store — all data pre-loaded to device
# ---------------------------------------------------------------------------

class GPUDataStore:
    """Pre-loads all structure data to GPU for zero-copy batch collation.

    When precompute_basis=True, also caches Chebyshev basis functions and
    angular basis on GPU (saves recomputing every iteration).
    """

    def __init__(self, structures: List[Dict], device: torch.device,
                 dtype: torch.dtype, config: dict = None):
        self.device = device
        self.dtype = dtype
        self.n = len(structures)
        self.has_cached_basis = config is not None

        # ------------------------------------------------------------------
        # Vectorized build: concat all per-frame numpy arrays once, transfer
        # to GPU in bulk, then split into per-frame views. This avoids both
        # the per-frame host→device launch overhead and the per-frame kernel
        # launch overhead for basis computation (N_frames → 1 launch).
        # ------------------------------------------------------------------
        # Per-frame sizes (needed for splitting)
        n_rad = np.array([len(s["pair_i_rad"]) for s in structures], dtype=np.int64)
        n_ang = np.array([len(s["pair_i_ang"]) for s in structures], dtype=np.int64)
        self.natoms = [int(s["natoms"]) for s in structures]

        def _concat_and_to(key, np_dtype, tdtype, default=None):
            arrs = []
            for s in structures:
                a = s[key] if key in s else default
                arrs.append(a if a is not None else np.empty(0, np_dtype))
            flat_np = np.concatenate([np.asarray(a).reshape(-1) for a in arrs])
            return torch.from_numpy(flat_np.astype(np_dtype, copy=False)).to(
                device=device, dtype=tdtype, non_blocking=True)

        # atoms (variable length per frame)
        at_cat = np.concatenate([np.asarray(s["atom_types"]).astype(np.int64)
                                 for s in structures])
        at_all = torch.from_numpy(at_cat).to(device=device, non_blocking=True)

        pi_r_cat = np.concatenate([np.asarray(s["pair_i_rad"]).astype(np.int64)
                                   for s in structures])
        pj_r_cat = np.concatenate([np.asarray(s["pair_j_rad"]).astype(np.int64)
                                   for s in structures])
        rij_r_cat = np.concatenate([np.asarray(s["rij_rad"])
                                    for s in structures]).reshape(-1, 3)

        pi_a_cat = np.concatenate([np.asarray(s["pair_i_ang"]).astype(np.int64)
                                   for s in structures])
        pj_a_cat = np.concatenate([np.asarray(s["pair_j_ang"]).astype(np.int64)
                                   for s in structures])
        rij_a_cat = np.concatenate([np.asarray(s["rij_ang"])
                                    for s in structures]).reshape(-1, 3)

        pi_r_all = torch.from_numpy(pi_r_cat).to(device=device, non_blocking=True)
        pj_r_all = torch.from_numpy(pj_r_cat).to(device=device, non_blocking=True)
        rij_r_all = torch.from_numpy(rij_r_cat).to(device=device, dtype=dtype,
                                                    non_blocking=True)
        pi_a_all = torch.from_numpy(pi_a_cat).to(device=device, non_blocking=True)
        pj_a_all = torch.from_numpy(pj_a_cat).to(device=device, non_blocking=True)
        rij_a_all = torch.from_numpy(rij_a_cat).to(device=device, dtype=dtype,
                                                    non_blocking=True)

        # Energy / forces / virial (variable length for forces)
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
                if v.shape[0] == 6:  # xx, yy, zz, xy, yz, zx → 3x3
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

        # Batched basis computation — one kernel launch for everything
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

        # Split concatenated tensors into per-frame views (no extra alloc)
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

        # Concatenate atom data
        at_list = [self.atom_types[i] for i in indices]
        atom_types = torch.cat(at_list)

        struct_idx = torch.cat([
            torch.full((self.natoms[i],), k, dtype=torch.long,
                       device=self.device)
            for k, i in enumerate(indices)
        ])

        # Pair data with offsets (all on GPU already)
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

        # Per-structure availability masks
        energy_mask = torch.tensor([self.has_energy_flag[i] for i in indices],
                                   dtype=torch.bool, device=self.device)
        batch["energy_mask"] = energy_mask

        batch["forces"] = torch.cat([self.forces[i] for i in indices])
        # Per-atom force mask: expand structure-level flag to atoms
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

def _preprocess_one(args):
    """Worker: build neighbor list + package one structure. Must be top-level
    for multiprocessing pickling."""
    frame, rc_rad, rc_ang, max_rc, type_names, dtype_str = args
    dtype = np.dtype(dtype_str)

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


def preprocess_structures(frames, config, dtype=np.float32, num_workers=None):
    """Build neighbor lists for all frames in parallel across CPU cores.

    Parameters
    ----------
    num_workers : int or None
        Number of worker processes. None → auto (os.cpu_count()).
        Set to 1 to disable multiprocessing (useful for debugging).
    """
    rc_rad = config["cutoff_radial"]
    rc_ang = config["cutoff_angular"]
    type_names = config["type_names"]
    max_rc = max(rc_rad, rc_ang)
    dtype_str = np.dtype(dtype).name

    if num_workers is None:
        num_workers = os.cpu_count() or 1
    num_workers = min(num_workers, len(frames))

    tasks = [(f, rc_rad, rc_ang, max_rc, type_names, dtype_str) for f in frames]

    # Small dataset or explicit serial: in-process loop
    if num_workers <= 1 or len(frames) < 16:
        structures = []
        for idx, a in enumerate(tasks):
            structures.append(_preprocess_one(a))
            if (idx + 1) % 500 == 0:
                print(f"  Preprocessed {idx + 1}/{len(frames)}")
        return structures

    # Parallel: chunksize to amortize IPC overhead
    from multiprocessing import Pool
    chunksize = max(1, len(frames) // (num_workers * 8))
    structures = [None] * len(frames)
    print(f"  Building neighbor lists with {num_workers} workers "
          f"(chunksize={chunksize})...")
    with Pool(num_workers) as pool:
        for idx, s in enumerate(pool.imap(_preprocess_one, tasks,
                                          chunksize=chunksize)):
            structures[idx] = s
            if (idx + 1) % 500 == 0:
                print(f"  Preprocessed {idx + 1}/{len(frames)}")
    return structures


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


def compute_energy_shift(structures, num_types):
    """Per-type energy shift via least squares (only uses frames with energy)."""
    rows_A = []
    rows_b = []
    for s in structures:
        if "energy" not in s:
            continue
        counts = np.bincount(s["atom_types"], minlength=num_types).astype(np.float64)
        rows_A.append(counts)
        rows_b.append(s["energy"])
    A = np.array(rows_A)
    b = np.array(rows_b)
    shift, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return shift


@torch.no_grad()
def compute_q_scaler(model, data_store, batch_size=64, pytorch_only=True):
    """Compute descriptor min/max across training set.

    pytorch_only must match the value used during training so that the
    descriptor statistics (and therefore q_scaler normalization) are
    consistent with the actual training-time descriptor values.
    """
    model.eval()
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    q_min = torch.full((model.dim,), float("inf"), dtype=dtype, device=dev)
    q_max = torch.full((model.dim,), float("-inf"), dtype=dtype, device=dev)

    for start in range(0, data_store.n, batch_size):
        end = min(start + batch_size, data_store.n)
        batch = data_store.collate(list(range(start, end)))
        q = model.compute_descriptors(
            batch["rij_rad"], batch["rij_ang"],
            batch["pair_i_rad"], batch["pair_j_rad"],
            batch["pair_i_ang"], batch["pair_j_ang"],
            batch["atom_types"], batch["N"],
            pytorch_only=pytorch_only,
        )
        q_min = torch.min(q_min, q.min(0).values)
        q_max = torch.max(q_max, q.max(0).values)

    model.train()
    return q_min, q_max


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def _save_checkpoint(path, model, optimizer, scheduler, epoch, best_loss):
    """Save training checkpoint (model + optimizer state)."""
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
    """Load checkpoint. Returns (start_epoch, best_loss)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    m.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt["best_loss"]


def train_nep(
    config_file: str,
    data_file: str,
    output_dir: str = ".",
    device: str = None,
    precision: str = "float32",
    num_epochs: int = None,
    batch_size: int = None,
    lr: float = None,
    print_interval: int = 10,
    use_compile: bool = False,
    restart: bool = True,
    checkpoint_interval: int = 100,
    pytorch_only: bool = True,
    use_autograd_forces: bool = False,
    max_grad_norm: float = None,
    # Stage 1 loss weights — read from nep.in lambda_e/f/v by default.
    pref_e: float = None,
    pref_f: float = None,
    pref_v: float = None,
    # LR schedule: ReduceLROnPlateau (MACE default).
    scheduler_patience: int = None,
    scheduler_factor: float = None,
    stop_lr: float = None,
    # Huber loss: if > 0, use Huber loss instead of MSE.
    huber_delta: float = None,
    # Stage 2 (MACE-inspired): energy-focused fine-tuning with SWA.
    stage2: bool = None,
    start_stage2: int = None,
    stage2_lr: float = None,
    stage2_pref_e: float = None,
    stage2_pref_f: float = None,
    stage2_pref_v: float = None,
    use_swa: bool = None,
):
    """Train a NEP model.

    Training strategy follows MACE (Batatia et al.):
    - Stage 1: Fixed loss weights + ReduceLROnPlateau (patience=50, factor=0.8).
    - Stage 2 (optional): Energy-focused fine-tuning with SWA model averaging.

    Parameters
    ----------
    config_file : str
        Path to nep.in.
    data_file : str
        Path to train.xyz.
    lr : float
        Initial learning rate.  Default 0.01.
    max_grad_norm : float
        Clip gradient norm.  Default 10.0.
    pref_e : float
        Energy loss weight.  Default 1.0.
    pref_f : float
        Force loss weight.  Default 100.0.
    pref_v : float
        Virial loss weight.  Default 1.0.
    scheduler_patience : int
        Epochs w/o improvement before LR reduction.  Default 50.
    scheduler_factor : float
        Factor to multiply LR by on plateau.  Default 0.8.
    stop_lr : float
        Minimum learning rate.  Default 1e-6.
    huber_delta : float
        If > 0, use Huber loss instead of MSE.  Default 0 (MSE).
    stage2 : bool
        Enable Stage 2 energy-focused fine-tuning.  Default False.
    start_stage2 : int
        Epoch to begin Stage 2.  Default: 75% of num_epochs.
    stage2_lr : float
        Fixed LR during Stage 2.  Default 1e-3.
    stage2_pref_e / stage2_pref_f / stage2_pref_v : float
        Loss weights during Stage 2.  Default 1000 / 100 / 10.
    use_swa : bool
        Use Stochastic Weight Averaging during Stage 2.  Default True.
    """
    os.makedirs(output_dir, exist_ok=True)
    if device is None:
        device = _default_device()
    dev = torch.device(device)
    dtype = torch.float32 if precision == "float32" else torch.float64

    # Setup logging — _log prints to screen AND output.log
    _log_mode = "a" if restart else "w"
    _out_log_file = open(os.path.join(output_dir, "output.log"), _log_mode)

    def _log(msg=""):
        print(msg)
        _out_log_file.write(msg + "\n")
        _out_log_file.flush()

    from datetime import datetime
    from . import __version__
    total_t0 = time.time()
    _log(f"torchnep v{__version__} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _log(f"Device: {device} | PyTorch {torch.__version__} | precision: {precision}")

    # 1. Config — nep.in provides defaults, function args override.
    _log("Parsing nep.in...")
    config = parse_nep_in(config_file)

    # Regularization (always from nep.in)
    lambda_1 = config.get("lambda_1", 0.0)
    lambda_2 = config.get("lambda_2", 0.0)

    # Training params: function arg > nep.in > hardcoded default
    def _cfg(arg_val, cfg_key, default):
        if arg_val is not None:
            return arg_val
        return config.get(cfg_key, default)

    num_epochs = _cfg(num_epochs, "num_epochs", 200)
    batch_size = _cfg(batch_size, "batch_size", 32)
    lr = _cfg(lr, "lr", 0.01)
    max_grad_norm = _cfg(max_grad_norm, "max_grad_norm", 10.0)
    # Stage 1 weights: lambda_e/f/v in nep.in → pref_e/f/v
    pref_e = _cfg(pref_e, "lambda_e", 1.0)
    pref_f = _cfg(pref_f, "lambda_f", 100.0)
    pref_v = _cfg(pref_v, "lambda_v", 1.0)
    scheduler_patience = _cfg(scheduler_patience, "scheduler_patience", 50)
    scheduler_factor = _cfg(scheduler_factor, "scheduler_factor", 0.8)
    stop_lr = _cfg(stop_lr, "stop_lr", 1e-6)
    huber_delta = _cfg(huber_delta, "huber_delta", 0.0)
    # Stage 2
    stage2 = _cfg(stage2, "stage2", False)
    start_stage2 = _cfg(start_stage2, "start_stage2", None)
    stage2_lr = _cfg(stage2_lr, "stage2_lr", 1e-3)
    stage2_pref_e = _cfg(stage2_pref_e, "stage2_pref_e", 1000.0)
    stage2_pref_f = _cfg(stage2_pref_f, "stage2_pref_f", 100.0)
    stage2_pref_v = _cfg(stage2_pref_v, "stage2_pref_v", 10.0)
    use_swa = _cfg(use_swa, "use_swa", True)

    # 2. Load data
    _log("Loading training data...")
    frames = read_xyz(data_file)
    _log(f"  {len(frames)} structures")

    # 3. Preprocess (CPU)
    _log("Building neighbor lists...")
    t0 = time.time()
    np_dtype = np.float64 if precision == "float64" else np.float32
    structures = preprocess_structures(frames, config, np_dtype)
    preprocess_time = time.time() - t0
    _log(f"  Done in {preprocess_time:.1f}s")

    # 4. Max neighbors (for nep.txt cutoff line)
    max_NN_rad, max_NN_ang = compute_max_neighbors(structures)

    # 5. Pre-load to GPU (with precomputed basis for analytical forces)
    _log(f"Pre-loading data to {device} (with cached basis)...")
    t0 = time.time()
    data_store = GPUDataStore(structures, dev, dtype, config=config)
    del structures  # free CPU memory
    data_load_time = time.time() - t0
    if dev.type == "cuda":
        mem = torch.cuda.memory_allocated() / 1e6
        _log(f"  GPU memory used: {mem:.0f} MB ({data_load_time:.1f}s)")
    elif dev.type == "mps":
        _log(f"  MPS data loaded ({data_load_time:.1f}s)")
    _log(f"  Data: {data_store.n} structures, "
         f"{data_store.n_energy} with energy, "
         f"{data_store.n_forces} with forces, "
         f"{data_store.n_virial} with virial")

    # 6. Model
    model = NEPModel(config).to(dtype).to(dev)
    # Initialize b1 to negative mean energy/atom so initial predictions
    # are centered near reference values (prevents gradient explosion).
    mean_epa = np.mean([data_store.energy[i] / data_store.natoms[i]
                        for i in range(data_store.n)
                        if data_store.has_energy_flag[i]])
    with torch.no_grad():
        model.b1.fill_(-mean_epa)
    _log(f"Model: {sum(p.numel() for p in model.parameters())} params, "
         f"dim={model.dim}, b1 init={model.b1.item():.4f}")

    # 7. q_scaler
    _log("Computing q_scaler...")
    q_min, q_max = compute_q_scaler(model, data_store, batch_size,
                                    pytorch_only=pytorch_only)
    model.set_q_scaler(q_min, q_max)

    # 8. Compile (PyTorch 2.0+)
    if use_compile and hasattr(torch, "compile"):
        _log("Compiling model with torch.compile...")
        model = torch.compile(model)

    # 9. Optimizer (Adam, MACE-style: amsgrad + small weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=lambda_2, amsgrad=True)

    # Stage 2 setup
    if stage2 and start_stage2 is None:
        start_stage2 = max(1, int(num_epochs * 0.75))

    # LR scheduler: ReduceLROnPlateau (MACE default)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor,
        patience=scheduler_patience, min_lr=stop_lr)

    # Huber loss helper
    use_huber = huber_delta > 0
    def _loss_fn(pred, ref):
        """MSE or Huber loss depending on huber_delta setting."""
        if use_huber:
            return torch.nn.functional.huber_loss(pred, ref, reduction="mean",
                                                  delta=huber_delta)
        return torch.mean((pred - ref) ** 2)

    # Stage 2 SWA setup (MACE-inspired)
    swa_model = None
    stage2_scheduler = None
    if stage2:
        # Stage 2 also uses ReduceLROnPlateau, starting from stage2_lr
        stage2_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=scheduler_factor,
            patience=scheduler_patience, min_lr=stop_lr)
        if use_swa:
            from torch.optim.swa_utils import AveragedModel
            m_ = model._orig_mod if hasattr(model, "_orig_mod") else model
            swa_model = AveragedModel(m_)

    ckpt_path = os.path.join(output_dir, "checkpoint.pt")
    start_epoch = 1
    best_loss = float("inf")
    if restart and os.path.exists(ckpt_path):
        start_epoch, best_loss = _load_checkpoint(ckpt_path, model, optimizer, lr_scheduler, dev)
        start_epoch += 1
        _log(f"Resumed from checkpoint: epoch {start_epoch - 1}, best_loss={best_loss:.4e}")

    # 10. Training
    n_structs = data_store.n
    has_forces = data_store.has_forces and pref_f > 0
    has_virial = data_store.has_virial and pref_v > 0

    # Open loss log: append if restarting, otherwise write fresh
    loss_log_mode = "a" if (restart and start_epoch > 1) else "w"
    loss_log = open(os.path.join(output_dir, "loss.out"), loss_log_mode)
    if loss_log_mode == "w":
        loss_log.write("epoch  loss  rmse_e(meV/atom)  rmse_f(eV/A)  rmse_v(meV/atom)\n")

    backend_str = ("pure-PyTorch" if pytorch_only else "CUDA-kernel accelerated")
    force_str = ("autograd (create_graph)" if use_autograd_forces
                 else "analytical")
    clip_str = f"grad_clip={max_grad_norm}" if max_grad_norm > 0 else "no grad clip"
    loss_type = f"Huber(delta={huber_delta})" if use_huber else "MSE"
    _log(f"\nTraining: epochs {start_epoch}-{num_epochs}, batch={batch_size}, "
         f"device={device}, dtype={precision}")
    _log(f"Backend: {backend_str} | forces: {force_str} | {clip_str} | loss: {loss_type}")
    _log(f"LR: {lr}, ReduceLROnPlateau(patience={scheduler_patience}, "
         f"factor={scheduler_factor}), stop_lr={stop_lr}")
    _log(f"Loss weights: E={pref_e}  F={pref_f}  V={pref_v}")
    if stage2:
        _log(f"Stage 2: epoch {start_stage2}→{num_epochs}, "
             f"lr={stage2_lr}, ReduceLROnPlateau, SWA={'ON' if use_swa else 'OFF'}")
        _log(f"Stage 2 weights: E={stage2_pref_e}  F={stage2_pref_f}  V={stage2_pref_v}")
    _log("-" * 72)

    train_t0 = time.time()
    try:
        for epoch in range(start_epoch, num_epochs + 1):
            t_epoch = time.time()
            model.train()
            perm = torch.randperm(n_structs)
            sum_loss = 0.0
            sum_le = 0.0
            sum_lf = 0.0
            max_gn = 0.0
            sum_lv = 0.0
            sum_e_structs = 0
            sum_f_atoms = 0
            sum_v_structs = 0
            sum_structs = 0
            n_batch = 0

            # Determine if we are in Stage 2
            in_stage2 = stage2 and epoch >= start_stage2

            if in_stage2:
                # Stage 2: energy-focused weights
                cur_pref_e = stage2_pref_e
                cur_pref_f = stage2_pref_f
                cur_pref_v = stage2_pref_v

                # Print stage transition once & set initial stage2 LR
                if epoch == start_stage2:
                    for pg in optimizer.param_groups:
                        pg['lr'] = stage2_lr
                    _log(f"\n{'='*72}")
                    _log(f"Stage 2 started at epoch {epoch}: "
                         f"E_w={cur_pref_e}, F_w={cur_pref_f}, V_w={cur_pref_v}, lr={stage2_lr:.2e}")
                    _log(f"{'='*72}")
                    best_loss = float("inf")
            else:
                # Stage 1: fixed weights, ReduceLROnPlateau handles LR
                cur_pref_e = pref_e
                cur_pref_f = pref_f
                cur_pref_v = pref_v

            for start in range(0, n_structs, batch_size):
                idx = perm[start:start + batch_size].tolist()
                batch = data_store.collate(idx)

                if use_autograd_forces:
                    result = model.compute_properties(
                        batch["rij_rad"], batch["rij_ang"],
                        batch["pair_i_rad"], batch["pair_j_rad"],
                        batch["pair_i_ang"], batch["pair_j_ang"],
                        batch["atom_types"], batch["N"],
                        batch["struct_idx"], batch["num_structures"],
                        need_forces=has_forces, need_virial=has_virial)
                else:
                    result = model.compute_properties_cached(
                        batch, need_forces=has_forces, need_virial=has_virial,
                        pytorch_only=pytorch_only)

                # Energy loss (per-atom)
                e_pa_pred = result["Etot"] / batch["natoms"]
                e_pa_ref = batch["energy"] / batch["natoms"]
                e_mask = batch["energy_mask"]
                loss = torch.tensor(0.0, dtype=dtype, device=dev)
                if e_mask.any():
                    loss_e = _loss_fn(e_pa_pred[e_mask], e_pa_ref[e_mask])
                    loss = loss + cur_pref_e * loss_e
                    sum_le += loss_e.item() * e_mask.sum().item()

                # Force loss
                if has_forces:
                    f_mask = batch["force_mask"]
                    if f_mask.any():
                        f_pred = result["forces"][f_mask]
                        f_ref = batch["forces"][f_mask]
                        loss_f = _loss_fn(f_pred, f_ref)
                        loss = loss + cur_pref_f * loss_f
                        sum_lf += loss_f.item() * f_mask.sum().item()

                # Virial loss (per-atom)
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
                            loss_v = _loss_fn(v_sys[v_mask] / na, v_ref[v_mask] / na)
                            loss = loss + cur_pref_v * loss_v
                            sum_lv += loss_v.item() * v_mask.sum().item()

                # L1
                if lambda_1 > 0:
                    l1 = sum(p.abs().sum() for p in model.parameters())
                    loss = loss + lambda_1 * l1

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # Gradient clipping (MatPL-style: clip norm, then step)
                m_ = model._orig_mod if hasattr(model, "_orig_mod") else model
                if max_grad_norm > 0:
                    gn = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm).item()
                else:
                    gn = torch.sqrt(sum(
                        p.grad.norm()**2 for p in m_.parameters()
                        if p.grad is not None)).item()

                # Skip only non-finite gradients
                if not np.isfinite(gn):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()

                # SWA: update averaged model parameters after each step
                if in_stage2 and swa_model is not None:
                    m_ = model._orig_mod if hasattr(model, "_orig_mod") else model
                    swa_model.update_parameters(m_)

                sum_loss += loss.item()
                sum_e_structs += batch["energy_mask"].sum().item()
                sum_f_atoms += batch["force_mask"].sum().item()
                sum_v_structs += batch["virial_mask"].sum().item()
                sum_structs += batch["num_structures"]
                n_batch += 1
                max_gn = max(max_gn, gn)

            dt = time.time() - t_epoch

            avg_loss = sum_loss / max(n_batch, 1)
            rmse_e = np.sqrt(sum_le / max(sum_e_structs, 1)) * 1000  # meV/atom
            rmse_f = np.sqrt(sum_lf / max(sum_f_atoms, 1)) if sum_lf > 0 else 0.0  # eV/Å
            rmse_v = np.sqrt(sum_lv / max(sum_v_structs, 1)) * 1000 if sum_lv > 0 else 0.0  # meV/atom

            loss_log.write(f"{epoch} {avg_loss:.6e} {rmse_e:.4f} {rmse_f:.4f} {rmse_v:.4f} {max_gn:.2f}\n")
            loss_log.flush()

            stage_str = "[S2] " if in_stage2 else ""
            epoch_line = (f"{stage_str}Epoch {epoch:4d} | loss {avg_loss:.4e} | "
                          f"E {rmse_e:.1f} meV/atom | F {rmse_f:.4f} eV/A")
            v_str = f" | V {rmse_v:.1f} meV/atom" if has_virial else ""
            cur_lr = optimizer.param_groups[0]['lr']
            epoch_line += f"{v_str} | gnorm {max_gn:.1f} | lr {cur_lr:.2e} | {dt:.1f}s"
            if epoch % print_interval == 0 or epoch == 1:
                _log(epoch_line)
            else:
                _out_log_file.write(epoch_line + "\n")
                _out_log_file.flush()

            # LR scheduler step — both stages use ReduceLROnPlateau
            if in_stage2 and stage2_scheduler is not None:
                stage2_scheduler.step(avg_loss)
            elif not in_stage2:
                lr_scheduler.step(avg_loss)

            m = model._orig_mod if hasattr(model, "_orig_mod") else model
            if avg_loss < best_loss:
                best_loss = avg_loss
                m.save_nep_txt(os.path.join(output_dir, "nep.txt"),
                               max_NN_rad, max_NN_ang)
                torch.save(m.state_dict(),
                           os.path.join(output_dir, "best_model.pt"))

            # Periodic checkpoint
            if epoch % checkpoint_interval == 0 or epoch == num_epochs:
                _save_checkpoint(ckpt_path, model, optimizer, lr_scheduler,
                                 epoch, best_loss)

    finally:
        loss_log.close()

    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    if hasattr(m, "module"):
        m = m.module
    m.save_nep_txt(os.path.join(output_dir, "nep_final.txt"),
                   max_NN_rad, max_NN_ang)

    # Save SWA averaged model if available
    if swa_model is not None:
        # Copy SWA averaged weights into a fresh model for export
        swa_state = swa_model.module.state_dict()
        m.load_state_dict(swa_state)
        m.save_nep_txt(os.path.join(output_dir, "nep_swa.txt"),
                       max_NN_rad, max_NN_ang)
        torch.save(swa_state, os.path.join(output_dir, "swa_model.pt"))
        _log(f"SWA model saved to nep_swa.txt")

    train_time = time.time() - train_t0
    h, rem = divmod(train_time, 3600)
    m_, s = divmod(rem, 60)
    _log(f"\nDone. Best loss: {best_loss:.6e}")
    _log(f"Training time: {int(h):02d}:{int(m_):02d}:{s:04.1f}")

    # Post-training prediction on training set
    nep_file = os.path.join(output_dir, "nep.txt")
    if os.path.exists(nep_file):
        _log("\nRunning prediction on training set...")
        from .predict import predict_dataset
        predict_dataset(nep_file, data_file, output_dir=output_dir,
                        dtype="float64", device=device)
        _log("Prediction done.")

    total_time = time.time() - total_t0
    h, rem = divmod(total_time, 3600)
    m_, s = divmod(rem, 60)
    _log(f"\nTotal time (data + train + predict): {int(h):02d}:{int(m_):02d}:{s:04.1f}")
    _log(f"Output: {output_dir}/")
    _out_log_file.close()


# ---------------------------------------------------------------------------
# DDP multi-GPU training
# ---------------------------------------------------------------------------

def _ddp_worker(rank, world_size, config_file, data_file, output_dir,
                precision="float32", num_epochs=None, batch_size=None,
                lr=None, print_interval=10, pytorch_only=True,
                max_grad_norm=None, pref_e=None, pref_f=None, pref_v=None,
                scheduler_patience=None, scheduler_factor=None, stop_lr=None,
                huber_delta=None, stage2=None, start_stage2=None,
                stage2_lr=None, stage2_pref_e=None, stage2_pref_f=None,
                stage2_pref_v=None, use_swa=None, checkpoint_interval=100):
    """Worker function for DDP training.

    Mirrors the single-GPU ``train_nep`` behaviour (logging, loss.out schema,
    checkpoints, Stage 2 / SWA) so multi-GPU runs produce the same artifacts.
    Training parameters default to the values in nep.in; explicit function
    arguments override the config values.

    Parameters
    ----------
    pytorch_only : bool
        If True (default), use pure-PyTorch backend.
        If False, use custom CUDA kernels for basis functions.
    """
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    dtype = torch.float32 if precision == "float32" else torch.float64
    is_main = rank == 0

    if is_main:
        os.makedirs(output_dir, exist_ok=True)
    dist.barrier()

    # ---- Logging (mirror single-GPU _log) --------------------------------
    _out_log_file = open(os.path.join(output_dir, "output.log"), "w") if is_main else None

    def _log(msg=""):
        if is_main:
            print(msg)
            _out_log_file.write(msg + "\n")
            _out_log_file.flush()

    from datetime import datetime
    from . import __version__
    total_t0 = time.time()
    _log(f"torchnep v{__version__} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _log(f"DDP training: {world_size} GPUs | PyTorch {torch.__version__} | "
         f"precision: {precision}")

    # ---- Config ----------------------------------------------------------
    _log("Parsing nep.in...")
    config = parse_nep_in(config_file)
    lambda_1 = config.get("lambda_1", 0.0)
    lambda_2 = config.get("lambda_2", 0.0)

    def _cfg(arg_val, cfg_key, default):
        if arg_val is not None:
            return arg_val
        return config.get(cfg_key, default)

    num_epochs = _cfg(num_epochs, "num_epochs", 200)
    batch_size = _cfg(batch_size, "batch_size", 32)
    lr = _cfg(lr, "lr", 0.01)
    max_grad_norm = _cfg(max_grad_norm, "max_grad_norm", 10.0)
    pref_e = _cfg(pref_e, "lambda_e", 1.0)
    pref_f = _cfg(pref_f, "lambda_f", 100.0)
    pref_v = _cfg(pref_v, "lambda_v", 1.0)
    scheduler_patience = _cfg(scheduler_patience, "scheduler_patience", 50)
    scheduler_factor = _cfg(scheduler_factor, "scheduler_factor", 0.8)
    stop_lr = _cfg(stop_lr, "stop_lr", 1e-6)
    huber_delta = _cfg(huber_delta, "huber_delta", 0.0)
    stage2 = _cfg(stage2, "stage2", False)
    start_stage2 = _cfg(start_stage2, "start_stage2", None)
    stage2_lr = _cfg(stage2_lr, "stage2_lr", 1e-3)
    stage2_pref_e = _cfg(stage2_pref_e, "stage2_pref_e", 1000.0)
    stage2_pref_f = _cfg(stage2_pref_f, "stage2_pref_f", 100.0)
    stage2_pref_v = _cfg(stage2_pref_v, "stage2_pref_v", 10.0)
    use_swa = _cfg(use_swa, "use_swa", True)

    # ---- Data ------------------------------------------------------------
    _log("Loading training data...")
    frames = read_xyz(data_file)
    _log(f"  {len(frames)} structures")

    _log("Building neighbor lists...")
    t0 = time.time()
    np_dtype = np.float64 if precision == "float64" else np.float32
    structures = preprocess_structures(frames, config, np_dtype)
    _log(f"  Done in {time.time() - t0:.1f}s")

    max_NN_rad, max_NN_ang = compute_max_neighbors(structures)

    _log(f"Pre-loading data to cuda:{rank} (with cached basis)...")
    t0 = time.time()
    data_store = GPUDataStore(structures, dev, dtype, config=config)
    del structures
    torch.cuda.synchronize()
    mem = torch.cuda.memory_allocated() / 1e6
    _log(f"  GPU memory used: {mem:.0f} MB ({time.time() - t0:.1f}s)")
    _log(f"  Data: {data_store.n} structures, "
         f"{data_store.n_energy} with energy, "
         f"{data_store.n_forces} with forces, "
         f"{data_store.n_virial} with virial")

    # ---- Model -----------------------------------------------------------
    model = NEPModel(config).to(dtype).to(dev)
    mean_epa = np.mean([data_store.energy[i] / data_store.natoms[i]
                        for i in range(data_store.n)
                        if data_store.has_energy_flag[i]])
    with torch.no_grad():
        model.b1.fill_(-mean_epa)
    _log(f"Model: {sum(p.numel() for p in model.parameters())} params, "
         f"dim={model.dim}, b1 init={model.b1.item():.4f}")

    # q_scaler: compute on rank 0, broadcast
    _log("Computing q_scaler...")
    if is_main:
        q_min, q_max = compute_q_scaler(model, data_store, batch_size,
                                        pytorch_only=pytorch_only)
    else:
        q_min = torch.zeros(model.dim, dtype=dtype, device=dev)
        q_max = torch.zeros(model.dim, dtype=dtype, device=dev)
    dist.broadcast(q_min, 0)
    dist.broadcast(q_max, 0)
    model.set_q_scaler(q_min, q_max)

    # Wrap in DDP — find_unused_parameters needed for per-type NN masking
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    raw_model = model.module

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=lambda_2, amsgrad=True)

    if stage2 and start_stage2 is None:
        start_stage2 = max(1, int(num_epochs * 0.75))

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor,
        patience=scheduler_patience, min_lr=stop_lr)

    use_huber = huber_delta > 0
    def _loss_fn(pred, ref):
        if use_huber:
            return torch.nn.functional.huber_loss(pred, ref, reduction="mean",
                                                  delta=huber_delta)
        return torch.mean((pred - ref) ** 2)

    swa_model = None
    stage2_scheduler = None
    if stage2:
        stage2_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=scheduler_factor,
            patience=scheduler_patience, min_lr=stop_lr)
        if use_swa and is_main:
            from torch.optim.swa_utils import AveragedModel
            swa_model = AveragedModel(raw_model)

    n_structs = data_store.n
    has_forces = data_store.has_forces and pref_f > 0
    has_virial = data_store.has_virial and pref_v > 0

    # Loss log (rank 0 only)
    loss_log = None
    if is_main:
        loss_log = open(os.path.join(output_dir, "loss.out"), "w")
        loss_log.write("epoch  loss  rmse_e(meV/atom)  rmse_f(eV/A)  "
                       "rmse_v(meV/atom)  gnorm\n")

    backend_str = ("pure-PyTorch" if pytorch_only else "CUDA-kernel accelerated")
    clip_str = f"grad_clip={max_grad_norm}" if max_grad_norm > 0 else "no grad clip"
    loss_type = f"Huber(delta={huber_delta})" if use_huber else "MSE"
    _log(f"\nTraining: epochs 1-{num_epochs}, batch={batch_size} per rank x "
         f"{world_size} ranks, dtype={precision}")
    _log(f"Backend: {backend_str} | {clip_str} | loss: {loss_type}")
    _log(f"LR: {lr}, ReduceLROnPlateau(patience={scheduler_patience}, "
         f"factor={scheduler_factor}), stop_lr={stop_lr}")
    _log(f"Loss weights: E={pref_e}  F={pref_f}  V={pref_v}")
    if stage2:
        _log(f"Stage 2: epoch {start_stage2}→{num_epochs}, "
             f"lr={stage2_lr}, ReduceLROnPlateau, SWA={'ON' if use_swa else 'OFF'}")
        _log(f"Stage 2 weights: E={stage2_pref_e}  F={stage2_pref_f}  V={stage2_pref_v}")
    _log("-" * 72)

    best_loss = float("inf")
    ckpt_path = os.path.join(output_dir, "checkpoint.pt")
    train_t0 = time.time()

    try:
        for epoch in range(1, num_epochs + 1):
            t_epoch = time.time()
            model.train()

            # Same permutation on all ranks; shard batches round-robin
            g = torch.Generator()
            g.manual_seed(epoch)
            perm = torch.randperm(n_structs, generator=g)

            sum_loss = sum_le = sum_lf = sum_lv = 0.0
            sum_e_structs = sum_f_atoms = sum_v_structs = 0
            n_batch = 0
            max_gn = 0.0

            in_stage2 = stage2 and epoch >= start_stage2
            if in_stage2:
                cur_pref_e, cur_pref_f, cur_pref_v = stage2_pref_e, stage2_pref_f, stage2_pref_v
                if epoch == start_stage2:
                    for pg in optimizer.param_groups:
                        pg['lr'] = stage2_lr
                    _log(f"\n{'='*72}")
                    _log(f"Stage 2 started at epoch {epoch}: "
                         f"E_w={cur_pref_e}, F_w={cur_pref_f}, V_w={cur_pref_v}, "
                         f"lr={stage2_lr:.2e}")
                    _log(f"{'='*72}")
                    best_loss = float("inf")
            else:
                cur_pref_e, cur_pref_f, cur_pref_v = pref_e, pref_f, pref_v

            batch_starts = list(range(0, n_structs, batch_size))
            for bi, start in enumerate(batch_starts):
                if bi % world_size != rank:
                    continue
                idx = perm[start:start + batch_size].tolist()
                batch = data_store.collate(idx)

                result = raw_model.compute_properties_cached(
                    batch, need_forces=has_forces, need_virial=has_virial,
                    pytorch_only=pytorch_only)

                e_pa = result["Etot"] / batch["natoms"]
                e_ref = batch["energy"] / batch["natoms"]
                e_mask = batch["energy_mask"]
                loss = torch.tensor(0.0, dtype=dtype, device=dev)
                if e_mask.any():
                    loss_e = _loss_fn(e_pa[e_mask], e_ref[e_mask])
                    loss = loss + cur_pref_e * loss_e
                    sum_le += loss_e.item() * e_mask.sum().item()

                if has_forces:
                    f_mask = batch["force_mask"]
                    if f_mask.any():
                        loss_f = _loss_fn(result["forces"][f_mask],
                                          batch["forces"][f_mask])
                        loss = loss + cur_pref_f * loss_f
                        sum_lf += loss_f.item() * f_mask.sum().item()

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
                            loss_v = _loss_fn(v_sys[v_mask] / na, v_ref[v_mask] / na)
                            loss = loss + cur_pref_v * loss_v
                            sum_lv += loss_v.item() * v_mask.sum().item()

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

                if in_stage2 and swa_model is not None and is_main:
                    swa_model.update_parameters(raw_model)

                sum_loss += loss.item()
                sum_e_structs += batch["energy_mask"].sum().item()
                sum_f_atoms += batch["force_mask"].sum().item()
                sum_v_structs += batch["virial_mask"].sum().item()
                n_batch += 1
                max_gn = max(max_gn, gn)

            # Aggregate metrics across ranks
            metrics = torch.tensor([sum_loss, sum_le, sum_lf, sum_lv,
                                    float(sum_e_structs), float(sum_f_atoms),
                                    float(sum_v_structs), float(n_batch),
                                    float(max_gn)], device=dev)
            dist.all_reduce(metrics)
            gn_tensor = torch.tensor(max_gn, device=dev)
            dist.all_reduce(gn_tensor, op=dist.ReduceOp.MAX)

            (t_loss, t_le, t_lf, t_lv, t_es, t_fa, t_vs,
             t_nb, _) = metrics.tolist()
            max_gn = gn_tensor.item()

            avg_loss = t_loss / max(t_nb, 1)
            rmse_e = np.sqrt(t_le / max(t_es, 1)) * 1000
            rmse_f = np.sqrt(t_lf / max(t_fa, 1)) if t_lf > 0 else 0.0
            rmse_v = np.sqrt(t_lv / max(t_vs, 1)) * 1000 if t_lv > 0 else 0.0
            dt = time.time() - t_epoch

            # LR scheduler step
            if in_stage2 and stage2_scheduler is not None:
                stage2_scheduler.step(avg_loss)
            elif not in_stage2:
                lr_scheduler.step(avg_loss)

            if is_main:
                loss_log.write(f"{epoch} {avg_loss:.6e} {rmse_e:.4f} "
                               f"{rmse_f:.4f} {rmse_v:.4f} {max_gn:.2f}\n")
                loss_log.flush()

                stage_str = "[S2] " if in_stage2 else ""
                cur_lr = optimizer.param_groups[0]['lr']
                v_str = f" | V {rmse_v:.1f} meV/atom" if has_virial else ""
                epoch_line = (f"{stage_str}Epoch {epoch:4d} | loss {avg_loss:.4e} | "
                              f"E {rmse_e:.1f} meV/atom | F {rmse_f:.4f} eV/A"
                              f"{v_str} | gnorm {max_gn:.1f} | lr {cur_lr:.2e} | {dt:.1f}s")
                if epoch % print_interval == 0 or epoch == 1:
                    _log(epoch_line)
                else:
                    _out_log_file.write(epoch_line + "\n")
                    _out_log_file.flush()

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    raw_model.save_nep_txt(os.path.join(output_dir, "nep.txt"),
                                           max_NN_rad, max_NN_ang)
                    torch.save(raw_model.state_dict(),
                               os.path.join(output_dir, "best_model.pt"))

                if epoch % checkpoint_interval == 0 or epoch == num_epochs:
                    _save_checkpoint(ckpt_path, model, optimizer, lr_scheduler,
                                     epoch, best_loss)
    finally:
        if is_main and loss_log is not None:
            loss_log.close()

    if is_main:
        raw_model.save_nep_txt(os.path.join(output_dir, "nep_final.txt"),
                               max_NN_rad, max_NN_ang)
        if swa_model is not None:
            swa_state = swa_model.module.state_dict()
            raw_model.load_state_dict(swa_state)
            raw_model.save_nep_txt(os.path.join(output_dir, "nep_swa.txt"),
                                   max_NN_rad, max_NN_ang)
            torch.save(swa_state, os.path.join(output_dir, "swa_model.pt"))
            _log("SWA model saved to nep_swa.txt")

        train_time = time.time() - train_t0
        h, rem = divmod(train_time, 3600)
        m_, s = divmod(rem, 60)
        _log(f"\nDone. Best loss: {best_loss:.6e}")
        _log(f"Training time: {int(h):02d}:{int(m_):02d}:{s:04.1f}")

        # Post-training prediction on training set (rank 0 only)
        nep_file = os.path.join(output_dir, "nep.txt")
        if os.path.exists(nep_file):
            _log("\nRunning prediction on training set...")
            from .predict import predict_dataset
            try:
                predict_dataset(nep_file, data_file, output_dir,
                                device=f"cuda:{rank}", precision=precision)
            except Exception as e:
                _log(f"  Prediction failed: {e}")

        total_time = time.time() - total_t0
        h, rem = divmod(total_time, 3600)
        m_, s = divmod(rem, 60)
        _log(f"\nTotal time (data + train + predict): "
             f"{int(h):02d}:{int(m_):02d}:{s:04.1f}")
        _log(f"Output: {output_dir}/")
        _out_log_file.close()

    if dist.is_initialized():
        dist.destroy_process_group()


def train_nep_ddp(
    config_file: str,
    data_file: str,
    output_dir: str = ".",
    precision: str = "float32",
    num_epochs: int = None,
    batch_size: int = None,
    lr: float = None,
    print_interval: int = 10,
    pytorch_only: bool = True,
    num_gpus: int = None,
):
    """Launch DDP multi-GPU training.

    Training parameters (num_epochs, batch_size, lr) are read from the config
    file (nep.in).  Explicit function arguments override the config values.

    Uses torchrun-compatible spawn. Can also be launched via:
        torchrun --nproc_per_node=2 -m torchnep.train_ddp ...

    Parameters
    ----------
    pytorch_only : bool
        If True (default), use pure-PyTorch backend.
        If False, use custom CUDA kernels for basis functions.
    num_gpus : int
        Number of GPUs. Default: all available.
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    if num_gpus <= 1:
        print("Only 1 GPU, falling back to single-GPU training")
        train_nep(config_file, data_file, output_dir, "cuda", precision,
                  num_epochs, batch_size, lr, print_interval,
                  pytorch_only=pytorch_only)
        return

    import torch.multiprocessing as mp
    mp.spawn(
        _ddp_worker,
        args=(num_gpus, config_file, data_file, output_dir,
              precision, num_epochs, batch_size, lr, print_interval,
              pytorch_only),
        nprocs=num_gpus,
        join=True,
    )
