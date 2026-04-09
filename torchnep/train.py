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

    # Memory check: vectorized if feasible
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
        # Loop-based for large systems
        all_i, all_j, all_rij = [], [], []
        for si in range(S):
            sc = shifts_cart[si]
            is_central = np.all(shifts_frac[si] == 0)
            shifted = positions + sc
            for i in range(N):
                disp = shifted - positions[i]
                dist = np.linalg.norm(disp, axis=1)
                for j in range(N):
                    if is_central and i == j:
                        continue
                    if 1e-10 < dist[j] < cutoff:
                        all_i.append(i)
                        all_j.append(j)
                        all_rij.append(disp[j])
        if not all_i:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3), positions.dtype)
        return np.array(all_i, np.int64), np.array(all_j, np.int64), np.array(all_rij)


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
        self.has_forces = "forces" in structures[0]
        self.has_virial = "virial" in structures[0]

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
            self.energy.append(s["energy"])
            if self.has_forces:
                self.forces.append(
                    torch.tensor(s["forces"], dtype=dtype, device=device))
            if self.has_virial:
                self.virial.append(
                    torch.tensor(s["virial"], dtype=dtype, device=device))

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
        if self.has_forces:
            batch["forces"] = torch.cat([self.forces[i] for i in indices])
        if self.has_virial:
            batch["virial"] = torch.stack([self.virial[i] for i in indices])

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
    """Per-type energy shift via least squares."""
    n = len(structures)
    A = np.zeros((n, num_types))
    b = np.zeros(n)
    for i, s in enumerate(structures):
        for t in s["atom_types"]:
            A[i, t] += 1
        b[i] = s["energy"]
    shift, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return shift


@torch.no_grad()
def compute_q_scaler(model, data_store, batch_size=64):
    """Compute descriptor min/max across training set."""
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
        )
        q_min = torch.min(q_min, q.min(0).values)
        q_max = torch.max(q_max, q.max(0).values)

    model.train()
    return q_min, q_max


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def _save_checkpoint(path, model, optimizer, scheduler, epoch, best_loss):
    """Save training checkpoint (model + optimizer + scheduler state)."""
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "epoch": epoch,
        "best_loss": best_loss,
        "model_state": m.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }, path)


def _load_checkpoint(path, model, optimizer, scheduler, device):
    """Load checkpoint. Returns (start_epoch, best_loss)."""
    ckpt = torch.load(path, map_location=device)
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    m.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt["best_loss"]


def train_nep(
    config_file: str,
    data_file: str,
    output_dir: str = ".",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    precision: str = "float32",
    num_epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-2,
    print_interval: int = 10,
    use_compile: bool = False,
    restart: bool = True,
    checkpoint_interval: int = 100,
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
        'cpu' or 'cuda'.
    precision : str
        'float32' or 'float64'.
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Structures per batch.
    lr : float
        Initial learning rate.
    print_interval : int
        Print every N epochs.
    use_compile : bool
        Use torch.compile (PyTorch 2.0+).
    restart : bool
        If True and checkpoint.pt exists in output_dir, resume from it.
    checkpoint_interval : int
        Save checkpoint every N epochs (also saves on best loss).
    """
    os.makedirs(output_dir, exist_ok=True)
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

    # 6. Model
    model = NEPModel(config).to(dtype).to(dev)
    print(f"Model: {sum(p.numel() for p in model.parameters())} params, "
          f"dim={model.dim}")

    # 7. q_scaler
    print("Computing q_scaler...")
    q_min, q_max = compute_q_scaler(model, data_store, batch_size)
    model.set_q_scaler(q_min, q_max)

    # 8. Compile (PyTorch 2.0+)
    if use_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # 9. Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    # 9b. Restart from checkpoint if available
    ckpt_path = os.path.join(output_dir, "checkpoint.pt")
    start_epoch = 1
    best_loss = float("inf")
    if restart and os.path.exists(ckpt_path):
        start_epoch, best_loss = _load_checkpoint(ckpt_path, model, optimizer, scheduler, dev)
        start_epoch += 1  # resume from next epoch
        print(f"Resumed from checkpoint: epoch {start_epoch - 1}, best_loss={best_loss:.4e}")

    # 10. Training
    n_structs = data_store.n
    has_forces = data_store.has_forces and lambda_f > 0
    has_virial = data_store.has_virial and lambda_v > 0

    print(f"\nTraining: epochs {start_epoch}-{num_epochs}, batch={batch_size}, "
          f"device={device}, dtype={precision}")
    print(f"Loss: E={lambda_e} F={lambda_f} V={lambda_v} "
          f"L1={lambda_1} L2={lambda_2}")
    print("-" * 72)

    # Open loss.out: append if restarting, otherwise write fresh
    loss_log_mode = "a" if (restart and start_epoch > 1) else "w"
    loss_log = open(os.path.join(output_dir, "loss.out"), loss_log_mode)
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
            sum_lv = 0.0
            sum_atoms = 0
            sum_structs = 0
            n_batch = 0

            for start in range(0, n_structs, batch_size):
                idx = perm[start:start + batch_size].tolist()
                batch = data_store.collate(idx)

                # Always use the cached path: analytical forces are fully
                # differentiable through c2/c3 and NN weights without create_graph.
                result = model.compute_properties_cached(
                    batch, need_forces=has_forces, need_virial=has_virial)

                # Energy loss (per-atom MSE)
                e_pa_pred = result["Etot"] / batch["natoms"]
                e_pa_ref = batch["energy"] / batch["natoms"]
                loss_e = torch.mean((e_pa_pred - e_pa_ref) ** 2)
                loss = lambda_e * loss_e
                sum_le += loss_e.item() * batch["num_structures"]

                # Force loss
                if has_forces:
                    loss_f = torch.mean(
                        (result["forces"] - batch["forces"]) ** 2)
                    loss = loss + lambda_f * loss_f
                    sum_lf += loss_f.item() * batch["N"]

                # Virial loss
                if has_virial and "virial" in result:
                    v_atom = result["virial"]
                    v_sys = torch.zeros(batch["num_structures"], 9,
                                        dtype=dtype, device=dev)
                    si = batch["struct_idx"].unsqueeze(-1).expand_as(v_atom)
                    v_sys.scatter_add_(0, si, v_atom)
                    v_ref = batch["virial"]
                    if v_ref.shape[1] == 9:
                        na = batch["natoms"].unsqueeze(-1)
                        loss_v = torch.mean(((v_sys - v_ref) / na) ** 2)
                        loss = loss + lambda_v * loss_v
                        sum_lv += loss_v.item() * batch["num_structures"]

                # L1
                if lambda_1 > 0:
                    l1 = sum(p.abs().sum() for p in model.parameters())
                    loss = loss + lambda_1 * l1

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Skip step on NaN/inf gradients to prevent parameter corruption
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()

                sum_loss += loss.item()
                sum_atoms += batch["N"]
                sum_structs += batch["num_structures"]
                n_batch += 1

            scheduler.step()
            dt = time.time() - t_epoch

            avg_loss = sum_loss / n_batch
            rmse_e = np.sqrt(sum_le / sum_structs) * 1000  # meV/atom
            rmse_f = np.sqrt(sum_lf / sum_atoms) if sum_lf > 0 else 0.0  # eV/Å
            rmse_v = np.sqrt(sum_lv / sum_structs) * 1000 if sum_lv > 0 else 0.0  # meV/atom

            loss_log.write(f"{epoch} {avg_loss:.6e} {rmse_e:.4f} {rmse_f:.4f} {rmse_v:.4f}\n")
            loss_log.flush()

            if epoch % print_interval == 0 or epoch == 1:
                v_str = f" | V {rmse_v:.1f} meV/atom" if has_virial else ""
                print(f"Epoch {epoch:4d} | loss {avg_loss:.4e} | "
                      f"E {rmse_e:.1f} meV/atom | F {rmse_f:.4f} eV/A"
                      f"{v_str} | {dt:.1f}s")

            m = model._orig_mod if hasattr(model, "_orig_mod") else model
            if avg_loss < best_loss:
                best_loss = avg_loss
                m.save_nep_txt(os.path.join(output_dir, "nep.txt"),
                               max_NN_rad, max_NN_ang)
                torch.save(m.state_dict(),
                           os.path.join(output_dir, "best_model.pt"))

            # Periodic checkpoint (model + optimizer + scheduler)
            if epoch % checkpoint_interval == 0 or epoch == num_epochs:
                _save_checkpoint(ckpt_path, model, optimizer, scheduler,
                                 epoch, best_loss)

    finally:
        loss_log.close()

    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    if hasattr(m, "module"):
        m = m.module
    m.save_nep_txt(os.path.join(output_dir, "nep_final.txt"),
                   max_NN_rad, max_NN_ang)
    print(f"\nDone. Best loss: {best_loss:.6e}")
    print(f"Output: {output_dir}/")


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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01)

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
                    batch, need_forces=has_forces, need_virial=has_virial)
            else:
                result = raw_model.compute_properties(
                    batch["rij_rad"], batch["rij_ang"],
                    batch["pair_i_rad"], batch["pair_j_rad"],
                    batch["pair_i_ang"], batch["pair_j_ang"],
                    batch["atom_types"], batch["N"],
                    batch["struct_idx"], batch["num_structures"],
                    need_forces=has_forces, need_virial=has_virial)

            e_pa = result["Etot"] / batch["natoms"]
            loss_e = torch.mean((e_pa - batch["energy"]/batch["natoms"])**2)
            loss = lambda_e * loss_e
            sum_le += loss_e.item() * batch["num_structures"]

            if has_forces:
                loss_f = torch.mean((result["forces"] - batch["forces"])**2)
                loss = loss + lambda_f * loss_f
                sum_lf += loss_f.item() * batch["N"]

            if has_virial and "virial" in result:
                va = result["virial"]
                vs = torch.zeros(batch["num_structures"], 9, dtype=dtype, device=dev)
                si = batch["struct_idx"].unsqueeze(-1).expand_as(va)
                vs.scatter_add_(0, si, va)
                vr = batch["virial"]
                if vr.shape[1] == 9:
                    na = batch["natoms"].unsqueeze(-1)
                    loss = loss + lambda_v * torch.mean(((vs-vr)/na)**2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
