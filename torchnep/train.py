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

        # Store per-structure GPU tensors
        self.atom_types = []
        self.pi_rad = []
        self.pj_rad = []
        self.rij_rad = []
        self.pi_ang = []
        self.pj_ang = []
        self.rij_ang = []
        self.natoms = []
        self.energy = []
        self.forces = []
        self.virial = []
        # Precomputed basis (if config provided)
        self.fk_rad = []
        self.fkp_rad = []
        self.d12inv_rad = []
        self.fk_ang = []
        self.fkp_ang = []
        self.d12inv_ang = []
        self.blm = []
        # Per-frame availability flags
        self.has_energy_flag = []
        self.has_forces_flag = []
        self.has_virial_flag = []

        for s in structures:
            self.natoms.append(s["natoms"])
            self.atom_types.append(
                torch.tensor(s["atom_types"], dtype=torch.long, device=device))
            self.pi_rad.append(
                torch.tensor(s["pair_i_rad"], dtype=torch.long, device=device))
            self.pj_rad.append(
                torch.tensor(s["pair_j_rad"], dtype=torch.long, device=device))
            rr = torch.tensor(s["rij_rad"], dtype=dtype, device=device)
            self.rij_rad.append(rr)
            self.pi_ang.append(
                torch.tensor(s["pair_i_ang"], dtype=torch.long, device=device))
            self.pj_ang.append(
                torch.tensor(s["pair_j_ang"], dtype=torch.long, device=device))
            ra = torch.tensor(s["rij_ang"], dtype=dtype, device=device)
            self.rij_ang.append(ra)

            if "energy" in s:
                self.energy.append(s["energy"])
                self.has_energy_flag.append(True)
            else:
                self.energy.append(0.0)
                self.has_energy_flag.append(False)

            if "forces" in s:
                self.forces.append(
                    torch.tensor(s["forces"], dtype=dtype, device=device))
                self.has_forces_flag.append(True)
            else:
                self.forces.append(
                    torch.zeros(s["natoms"], 3, dtype=dtype, device=device))
                self.has_forces_flag.append(False)

            if "virial" in s:
                self.virial.append(
                    torch.tensor(s["virial"], dtype=dtype, device=device))
                self.has_virial_flag.append(True)
            else:
                self.virial.append(
                    torch.zeros(9, dtype=dtype, device=device))
                self.has_virial_flag.append(False)

            # Precompute basis functions (fixed geometry)
            if config is not None:
                dr = torch.norm(rr, dim=-1)
                fk_r, fkp_r = ops.chebyshev_basis_and_deriv(
                    dr, config["cutoff_radial"], config["basis_size_radial"])
                self.fk_rad.append(fk_r)
                self.fkp_rad.append(fkp_r)
                self.d12inv_rad.append(1.0 / dr)

                if ra.shape[0] > 0:
                    da = torch.norm(ra, dim=-1)
                    fk_a, fkp_a = ops.chebyshev_basis_and_deriv(
                        da, config["cutoff_angular"], config["basis_size_angular"])
                    self.fk_ang.append(fk_a)
                    self.fkp_ang.append(fkp_a)
                    self.d12inv_ang.append(1.0 / da)
                    d12inv_a = 1.0 / da
                    blm = ops.angular_basis(
                        ra[:, 0]*d12inv_a, ra[:, 1]*d12inv_a,
                        ra[:, 2]*d12inv_a, config["l_max"][0])
                    self.blm.append(blm)
                else:
                    self.fk_ang.append(torch.zeros(0, config["basis_size_angular"]+1,
                                                   dtype=dtype, device=device))
                    self.fkp_ang.append(torch.zeros(0, config["basis_size_angular"]+1,
                                                    dtype=dtype, device=device))
                    self.d12inv_ang.append(torch.zeros(0, dtype=dtype, device=device))
                    num_lm = sum(2*ll+1 for ll in range(1, config["l_max"][0]+1))
                    self.blm.append(torch.zeros(0, num_lm, dtype=dtype, device=device))

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

def preprocess_structures(frames, config, dtype=np.float32):
    """Build neighbor lists for all frames (CPU)."""
    rc_rad = config["cutoff_radial"]
    rc_ang = config["cutoff_angular"]
    type_names = config["type_names"]
    max_rc = max(rc_rad, rc_ang)

    structures = []
    for idx, frame in enumerate(frames):
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
        structures.append(s)

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
    num_epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    print_interval: int = 10,
    use_compile: bool = False,
    restart: bool = True,
    checkpoint_interval: int = 100,
    pytorch_only: bool = True,
    use_autograd_forces: bool = False,
    max_grad_norm: float = 1.0,
    # Dynamic loss weighting (MatPL-style, lr-dependent).
    # pref = limit + (start - limit) * (cur_lr / init_lr)
    # At start: high force weight → descriptors learn atomic environments fast.
    # At end: balanced weights → energy/virial fine-tuning.
    start_pref_e: float = 0.02,
    limit_pref_e: float = 1.0,
    start_pref_f: float = 1000.0,
    limit_pref_f: float = 1.0,
    start_pref_v: float = 50.0,
    limit_pref_v: float = 0.1,
    # LR schedule: per-epoch exponential decay.
    stop_lr: float = 3.51e-8,
    lr_decay_interval: int = 1,
):
    """Train a NEP model.

    Parameters
    ----------
    config_file : str
        Path to nep.in.
    data_file : str
        Path to train.xyz.
    output_dir : str
        Directory for output files.
    device : str
        'cuda', 'mps', or 'cpu'. Default: auto-detect (CUDA → MPS → CPU).
    precision : str
        'float32' or 'float64'.
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Structures per batch.
    lr : float
        Initial learning rate.  Default 1e-3 (MatPL default).
    print_interval : int
        Print every N epochs.
    use_compile : bool
        Use torch.compile (PyTorch 2.0+).
    restart : bool
        If True and checkpoint.pt exists in output_dir, resume from it.
    checkpoint_interval : int
        Save checkpoint every N epochs (also saves on best loss).
    pytorch_only : bool
        If True (default), force pure-PyTorch path everywhere — no handwritten
        CUDA kernels.
    use_autograd_forces : bool
        If True, compute forces via torch.autograd.grad (create_graph=True)
        instead of analytical force formulas.  Slower and uses more memory,
        but is the gold-standard pure-autograd reference for verifying
        gradient correctness.  Default False (use analytical forces).
    max_grad_norm : float
        Clip gradient norm to this value.  0 means no clipping.  Default 1.0.
    start_pref_e / limit_pref_e : float
        Energy loss weight at start / end of training.  Default 0.02 → 1.0.
    start_pref_f / limit_pref_f : float
        Force loss weight at start / end of training.  Default 1000 → 1.0.
    start_pref_v / limit_pref_v : float
        Virial loss weight at start / end of training.  Default 50 → 0.1.
    stop_lr : float
        Minimum learning rate.  Default 3.51e-8.
    lr_decay_interval : int
        Decay lr every N epochs.  Default 1 (every epoch).
    """
    os.makedirs(output_dir, exist_ok=True)
    if device is None:
        device = _default_device()
    dev = torch.device(device)
    dtype = torch.float32 if precision == "float32" else torch.float64


    # 1. Config
    print("Parsing nep.in...")
    config = parse_nep_in(config_file)
    lambda_e = config.get("lambda_e", 1.0)
    lambda_f = config.get("lambda_f", 1.0)
    lambda_v = config.get("lambda_v", 0.1)
    lambda_1 = config.get("lambda_1", 0.0)
    lambda_2 = config.get("lambda_2", 0.0)

    # 2. Load data
    print("Loading training data...")
    frames = read_xyz(data_file)
    print(f"  {len(frames)} structures")

    # 3. Preprocess (CPU)
    print("Building neighbor lists...")
    t0 = time.time()
    np_dtype = np.float64 if precision == "float64" else np.float32
    structures = preprocess_structures(frames, config, np_dtype)
    print(f"  Done in {time.time() - t0:.1f}s")

    # 4. Max neighbors (for nep.txt cutoff line)
    max_NN_rad, max_NN_ang = compute_max_neighbors(structures)

    # 5. Pre-load to GPU (with precomputed basis for analytical forces)
    print(f"Pre-loading data to {device} (with cached basis)...")
    t0 = time.time()
    data_store = GPUDataStore(structures, dev, dtype, config=config)
    del structures  # free CPU memory
    if dev.type == "cuda":
        mem = torch.cuda.memory_allocated() / 1e6
        print(f"  GPU memory used: {mem:.0f} MB ({time.time()-t0:.1f}s)")
    elif dev.type == "mps":
        print(f"  MPS data loaded ({time.time()-t0:.1f}s)")
    print(f"  Data: {data_store.n} structures, "
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
    print(f"Model: {sum(p.numel() for p in model.parameters())} params, "
          f"dim={model.dim}, b1 init={model.b1.item():.4f}")

    # 7. q_scaler
    print("Computing q_scaler...")
    q_min, q_max = compute_q_scaler(model, data_store, batch_size,
                                    pytorch_only=pytorch_only)
    model.set_q_scaler(q_min, q_max)

    # 8. Compile (PyTorch 2.0+)
    if use_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # 9. Optimizer (Adam, MatPL-style)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_2)

    # LR schedule: exponential decay every lr_decay_interval epochs.
    # Total decay steps = num_epochs / lr_decay_interval.
    # decay_factor chosen so lr reaches stop_lr at num_epochs.
    n_decays = max(num_epochs // max(lr_decay_interval, 1), 1)
    if n_decays > 0 and stop_lr < lr:
        decay_factor = np.exp(np.log(stop_lr / lr) / n_decays)
    else:
        decay_factor = 1.0

    def _set_epoch_lr(epoch):
        """Set learning rate at epoch boundary (step decay)."""
        n_steps = (epoch - 1) // max(lr_decay_interval, 1)
        real_lr = lr * (decay_factor ** n_steps)
        real_lr = max(real_lr, stop_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = real_lr
        return real_lr

    # 9b. Restart from checkpoint if available
    ckpt_path = os.path.join(output_dir, "checkpoint.pt")
    start_epoch = 1
    best_loss = float("inf")
    if restart and os.path.exists(ckpt_path):
        start_epoch, best_loss = _load_checkpoint(ckpt_path, model, optimizer, None, dev)
        start_epoch += 1
        print(f"Resumed from checkpoint: epoch {start_epoch - 1}, best_loss={best_loss:.4e}")

    # 10. Training
    n_structs = data_store.n
    has_forces = data_store.has_forces and lambda_f > 0
    has_virial = data_store.has_virial and lambda_v > 0

    backend_str = ("pure-PyTorch" if pytorch_only else "CUDA-kernel accelerated")
    force_str = ("autograd (create_graph)" if use_autograd_forces
                 else "analytical")
    clip_str = f"grad_clip={max_grad_norm}" if max_grad_norm > 0 else "no grad clip"
    print(f"\nTraining: epochs {start_epoch}-{num_epochs}, batch={batch_size}, "
          f"device={device}, dtype={precision}")
    print(f"Backend: {backend_str} | forces: {force_str} | {clip_str}")
    print(f"LR: {lr} → {stop_lr} (decay every {lr_decay_interval} epochs, {n_decays} steps)")
    print(f"Loss weights (start→end): "
          f"E={start_pref_e}→{limit_pref_e}  "
          f"F={start_pref_f}→{limit_pref_f}  "
          f"V={start_pref_v}→{limit_pref_v}")
    print("-" * 72)

    # Open log files: append if restarting, otherwise write fresh
    loss_log_mode = "a" if (restart and start_epoch > 1) else "w"
    loss_log = open(os.path.join(output_dir, "loss.out"), loss_log_mode)
    grad_log = open(os.path.join(output_dir, "grad_spike.log"), loss_log_mode)
    if loss_log_mode == "w":
        loss_log.write("epoch  loss  rmse_e(meV/atom)  rmse_f(eV/A)  rmse_v(meV/atom)\n")

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

            # Set lr for this epoch (per-epoch exponential decay)
            real_lr = _set_epoch_lr(epoch)

            # MatPL-style dynamic loss weighting:
            # pref = limit + (start - limit) * (cur_lr / init_lr)
            lr_ratio = real_lr / lr
            pref_e = limit_pref_e + (start_pref_e - limit_pref_e) * lr_ratio
            pref_f = limit_pref_f + (start_pref_f - limit_pref_f) * lr_ratio
            pref_v = limit_pref_v + (start_pref_v - limit_pref_v) * lr_ratio

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

                # Energy loss (MSE, per-atom)
                e_pa_pred = result["Etot"] / batch["natoms"]
                e_pa_ref = batch["energy"] / batch["natoms"]
                e_mask = batch["energy_mask"]
                loss = torch.tensor(0.0, dtype=dtype, device=dev)
                if e_mask.any():
                    mse_e = torch.mean((e_pa_pred[e_mask] - e_pa_ref[e_mask]) ** 2)
                    loss = loss + pref_e * mse_e
                    sum_le += mse_e.item() * e_mask.sum().item()

                # Force loss (MSE)
                if has_forces:
                    f_mask = batch["force_mask"]
                    if f_mask.any():
                        f_diff = (result["forces"] - batch["forces"])[f_mask]
                        mse_f = torch.mean(f_diff ** 2)
                        loss = loss + pref_f * mse_f
                        sum_lf += mse_f.item() * f_mask.sum().item()

                # Virial loss (MSE, per-atom)
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
                            mse_v = torch.mean(((v_sys[v_mask] - v_ref[v_mask]) / na) ** 2)
                            loss = loss + pref_v * mse_v
                            sum_lv += mse_v.item() * v_mask.sum().item()

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

            epoch_line = (f"Epoch {epoch:4d} | loss {avg_loss:.4e} | "
                          f"E {rmse_e:.1f} meV/atom | F {rmse_f:.4f} eV/A")
            v_str = f" | V {rmse_v:.1f} meV/atom" if has_virial else ""
            cur_lr = optimizer.param_groups[0]['lr']
            epoch_line += f"{v_str} | gnorm {max_gn:.1f} | lr {cur_lr:.2e} | {dt:.1f}s"
            if epoch % print_interval == 0 or epoch == 1:
                print(epoch_line)
            grad_log.write(epoch_line + "\n")
            grad_log.flush()

            m = model._orig_mod if hasattr(model, "_orig_mod") else model
            if avg_loss < best_loss:
                best_loss = avg_loss
                m.save_nep_txt(os.path.join(output_dir, "nep.txt"),
                               max_NN_rad, max_NN_ang)
                torch.save(m.state_dict(),
                           os.path.join(output_dir, "best_model.pt"))

            # Periodic checkpoint
            if epoch % checkpoint_interval == 0 or epoch == num_epochs:
                _save_checkpoint(ckpt_path, model, optimizer, None,
                                 epoch, best_loss)

    finally:
        loss_log.close()
        grad_log.close()

    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    if hasattr(m, "module"):
        m = m.module
    m.save_nep_txt(os.path.join(output_dir, "nep_final.txt"),
                   max_NN_rad, max_NN_ang)
    print(f"\nDone. Best loss: {best_loss:.6e}")
    print(f"Output: {output_dir}/")

    # Post-training prediction on training set
    nep_file = os.path.join(output_dir, "nep.txt")
    if os.path.exists(nep_file):
        print("\nRunning prediction on training set...")
        from .predict import predict_dataset
        predict_dataset(nep_file, data_file, output_dir=output_dir,
                        dtype="float64", device=device)


# ---------------------------------------------------------------------------
# DDP multi-GPU training
# ---------------------------------------------------------------------------

def _ddp_worker(rank, world_size, config_file, data_file, output_dir,
                precision, num_epochs, batch_size, lr, print_interval):
    """Worker function for DDP training."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    dtype = torch.float32 if precision == "float32" else torch.float64
    is_main = rank == 0

    config = parse_nep_in(config_file)
    lambda_e = config.get("lambda_e", 1.0)
    lambda_f = config.get("lambda_f", 1.0)
    lambda_v = config.get("lambda_v", 0.1)
    lambda_2 = config.get("lambda_2", 0.0)

    if is_main:
        print(f"DDP training: {world_size} GPUs")
        os.makedirs(output_dir, exist_ok=True)

    frames = read_xyz(data_file)
    np_dtype = np.float64 if precision == "float64" else np.float32
    structures = preprocess_structures(frames, config, np_dtype)
    # Each GPU gets all data (small dataset) — shard batches via permutation
    data_store = GPUDataStore(structures, dev, dtype, config=config)
    del structures

    model = NEPModel(config).to(dtype).to(dev)

    # q_scaler: compute on rank 0, broadcast
    if is_main:
        q_min, q_max = compute_q_scaler(model, data_store, batch_size)
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
                                 weight_decay=lambda_2, eps=1e-4)
    def _lr_lambda(epoch):
        warmup = 10
        if epoch < warmup:
            return 0.1 + 0.9 * epoch / max(warmup, 1)
        progress = (epoch - warmup) / max(num_epochs - warmup, 1)
        return 0.01 + 0.99 * 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    n_structs = data_store.n
    has_forces = data_store.has_forces and lambda_f > 0
    has_virial = data_store.has_virial and lambda_v > 0

    if is_main:
        print(f"Model: {sum(p.numel() for p in raw_model.parameters())} params")
        loss_log = open(os.path.join(output_dir, "loss.out"), "w")
        loss_log.write("epoch  loss  rmse_e(meV/atom)  rmse_f(eV/A)\n")

    best_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        t_epoch = time.time()
        model.train()

        # Each rank gets a different permutation → different batches
        g = torch.Generator()
        g.manual_seed(epoch)
        perm = torch.randperm(n_structs, generator=g)

        # Shard: rank processes every world_size-th batch
        sum_loss = sum_le = sum_lf = 0.0
        sum_atoms = n_batch = 0

        batch_starts = list(range(0, n_structs, batch_size))
        for bi, start in enumerate(batch_starts):
            if bi % world_size != rank:
                continue
            idx = perm[start:start + batch_size].tolist()
            batch = data_store.collate(idx)

            if data_store.has_cached_basis:
                result = raw_model.compute_properties_cached(
                    batch, need_forces=has_forces, need_virial=has_virial,
                    pytorch_only=True)
            else:
                result = raw_model.compute_properties(
                    batch["rij_rad"], batch["rij_ang"],
                    batch["pair_i_rad"], batch["pair_j_rad"],
                    batch["pair_i_ang"], batch["pair_j_ang"],
                    batch["atom_types"], batch["N"],
                    batch["struct_idx"], batch["num_structures"],
                    need_forces=has_forces, need_virial=has_virial)

            e_pa = result["Etot"] / batch["natoms"]
            e_ref = batch["energy"] / batch["natoms"]
            e_mask = batch["energy_mask"]
            loss = torch.tensor(0.0, dtype=dtype, device=dev)
            if e_mask.any():
                loss_e = torch.mean((e_pa[e_mask] - e_ref[e_mask])**2)
                loss = loss + lambda_e * loss_e
                sum_le += loss_e.item() * e_mask.sum().item()

            if has_forces:
                f_mask = batch["force_mask"]
                if f_mask.any():
                    f_diff = (result["forces"] - batch["forces"])[f_mask]
                    loss_f = torch.mean(f_diff ** 2)
                    loss = loss + lambda_f * loss_f
                    sum_lf += loss_f.item() * f_mask.sum().item()

            if has_virial and "virial" in result:
                v_mask = batch["virial_mask"]
                if v_mask.any():
                    va = result["virial"]
                    vs = torch.zeros(batch["num_structures"], 9, dtype=dtype, device=dev)
                    si = batch["struct_idx"].unsqueeze(-1).expand_as(va)
                    vs.scatter_add_(0, si, va)
                    vr = batch["virial"]
                    if vr.shape[1] == 9:
                        na = batch["natoms"][v_mask].unsqueeze(-1)
                        loss = loss + lambda_v * torch.mean(((vs[v_mask]-vr[v_mask])/na)**2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            sum_atoms += batch["N"]
            n_batch += 1

        scheduler.step()

        # Aggregate metrics across ranks
        metrics = torch.tensor([sum_loss, sum_le, sum_lf, float(sum_atoms),
                                float(n_batch)], device=dev)
        dist.all_reduce(metrics)
        total_loss, total_le, total_lf, total_atoms, total_batch = metrics.tolist()

        if is_main:
            avg = total_loss / max(total_batch, 1)
            rmse_e = np.sqrt(total_le / n_structs) * 1000
            rmse_f = np.sqrt(total_lf / max(total_atoms, 1)) if total_lf > 0 else 0.0
            dt = time.time() - t_epoch

            loss_log.write(f"{epoch} {avg:.6e} {rmse_e:.4f} {rmse_f:.4f}\n")
            loss_log.flush()

            if epoch % print_interval == 0 or epoch == 1:
                print(f"Epoch {epoch:4d} | loss {avg:.4e} | "
                      f"E {rmse_e:.1f} meV/atom | F {rmse_f:.4f} eV/A | "
                      f"{dt:.1f}s")

            if avg < best_loss:
                best_loss = avg
                raw_model.save_nep_txt(os.path.join(output_dir, "nep.txt"))

    if is_main:
        loss_log.close()
        raw_model.save_nep_txt(os.path.join(output_dir, "nep_final.txt"))
        print(f"\nDone. Best loss: {best_loss:.6e}")

    dist.destroy_process_group()


def train_nep_ddp(
    config_file: str,
    data_file: str,
    output_dir: str = ".",
    precision: str = "float32",
    num_epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-2,
    print_interval: int = 10,
    num_gpus: int = None,
):
    """Launch DDP multi-GPU training.

    Uses torchrun-compatible spawn. Can also be launched via:
        torchrun --nproc_per_node=2 -m torchnep.train_ddp ...

    Parameters
    ----------
    num_gpus : int
        Number of GPUs. Default: all available.
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    if num_gpus <= 1:
        print("Only 1 GPU, falling back to single-GPU training")
        train_nep(config_file, data_file, output_dir, "cuda", precision,
                  num_epochs, batch_size, lr, print_interval)
        return

    import torch.multiprocessing as mp
    mp.spawn(
        _ddp_worker,
        args=(num_gpus, config_file, data_file, output_dir,
              precision, num_epochs, batch_size, lr, print_interval),
        nprocs=num_gpus,
        join=True,
    )
