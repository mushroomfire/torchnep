"""
NEP training with PyTorch.

Single-GPU and multi-GPU share a single entry point: ``train_nep``.
Multi-GPU runs are launched via ``torchrun``; the function auto-detects
the torchrun environment (``WORLD_SIZE``, ``RANK``, ``LOCAL_RANK``) and
switches into DDP mode. Parameters and outputs are identical in both cases.

    # single GPU
    python run_train.py

    # multi-GPU (N cards)
    torchrun --nproc_per_node=N run_train.py
"""

import os
import platform
import time
import torch
import torch.distributed as dist
import numpy as np
from datetime import datetime
from typing import List, Dict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel

from .model import NEPModel
from .data import read_xyz, parse_nep_in, build_neighbor_list_np
from . import ops
from . import __version__
from .predict import predict_dataset
from .cuda_ops import _load_kernels, _load_cached_kernels


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
    """Describe compute backend for the startup banner."""
    lines = []
    if dev.type == "cuda":
        n_visible = torch.cuda.device_count()
        if world_size > 1:
            lines.append(f"Backend  : CUDA (DDP, {world_size} processes)")
        else:
            lines.append("Backend  : CUDA")
        names = {torch.cuda.get_device_name(i) for i in range(n_visible)}
        name_str = ", ".join(sorted(names))
        total_gb = sum(torch.cuda.get_device_properties(i).total_memory
                       for i in range(n_visible)) / 1e9
        lines.append(f"Devices  : {n_visible} x {name_str}  ({total_gb:.1f} GB total)")
    elif dev.type == "mps":
        lines.append("Backend  : MPS (Apple Silicon)")
    else:
        lines.append(f"Backend  : CPU ({platform.processor() or platform.machine()})")
    lines.append(f"PyTorch  : {torch.__version__}")
    return lines


def _default_device() -> str:
    """Select best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"



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

def preprocess_structures(frames, config, dtype=np.float32):
    """Build neighbor lists for all frames (CPU, serial)."""
    rc_rad = config["cutoff_radial"]
    rc_ang = config["cutoff_angular"]
    type_names = config["type_names"]
    max_rc = max(rc_rad, rc_ang)

    structures = []
    for frame in frames:
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


@torch.no_grad()
def compute_q_scaler(model, data_store, batch_size=64, pytorch_only=True):
    """Compute descriptor min/max across training set.

    ``pytorch_only`` must match the value used during training so that the
    descriptor statistics are consistent with the actual training-time
    descriptor values.
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
    pref_e: float = None,
    pref_f: float = None,
    pref_v: float = None,
    scheduler_patience: int = None,
    scheduler_factor: float = None,
    stop_lr: float = None,
    huber_delta: float = None,
    stage2: bool = None,
    start_stage2: int = None,
    stage2_lr: float = None,
    stage2_pref_e: float = None,
    stage2_pref_f: float = None,
    stage2_pref_v: float = None,
    use_swa: bool = None,
    finetune_from: str = None,
    reset_lr: float = None,
):
    """Train a NEP model. Single-GPU or multi-GPU (via torchrun).

    Training strategy follows MACE (Batatia et al.):
    - Stage 1: Fixed loss weights + ReduceLROnPlateau (patience=50, factor=0.8).
    - Stage 2 (optional): Energy-focused fine-tuning with SWA model averaging.

    Launch:
        python run_train.py                               # single GPU / CPU
        torchrun --nproc_per_node=N run_train.py          # N GPUs, DDP
    """
    # ---- Distributed detection -------------------------------------------
    is_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if is_ddp:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        dev = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        if device is None:
            device = _default_device()
        dev = torch.device(device)

    is_main = rank == 0
    dtype = torch.float32 if precision == "float32" else torch.float64

    # ---- Logging (rank 0 only) -------------------------------------------
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
    if is_ddp:
        dist.barrier()

    _out_log_file = (open(os.path.join(output_dir, "output.log"),
                          "a" if restart else "w")
                     if is_main else None)

    def _log(msg=""):
        if not is_main:
            return
        print(msg)
        _out_log_file.write(msg + "\n")
        _out_log_file.flush()

    # ---- Banner ----------------------------------------------------------
    total_t0 = time.time()
    _log(_BANNER.rstrip())
    _log(f"   torchnep  v{__version__}   author: {_AUTHOR}")
    _log(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _log("")
    for line in _backend_info(dev, world_size):
        _log(line)
    _log(f"Precision: {precision}")
    _log("")

    # ---- CUDA kernels -------------------------------------------------------
    # Compile/load CUDA extension kernels before data loading so any
    # first-time compilation is logged clearly and not confused with
    # training time. Cached .so files load instantly on subsequent runs.
    if dev.type == "cuda" and not pytorch_only:
        for name, loader in [
            ("nep_kernels", _load_kernels),
            ("nep_cached", _load_cached_kernels),
            ("nep_cuda_ops", ops._load_cuda_ops),
        ]:
            t0_k = time.time()
            k = loader()
            dt_k = time.time() - t0_k
            status = "OK" if k is not None else "unavailable (PyTorch fallback)"
            if dt_k > 1.0:  # only log if compilation actually happened
                _log(f"  CUDA kernel {name}: compiled ({dt_k:.0f}s) → {status}")

    # ---- Config ----------------------------------------------------------
    config = parse_nep_in(config_file)
    lambda_1 = config.get("lambda_1", 0.0)
    lambda_2 = config.get("lambda_2", 0.0)

    def _cfg(arg_val, cfg_key, default):
        return arg_val if arg_val is not None else config.get(cfg_key, default)

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
        if ft_path.endswith(".pt"):
            state = torch.load(ft_path, map_location=dev, weights_only=False)
            # accept either a raw state dict or a checkpoint dict
            if "model_state" in state:
                state = state["model_state"]
            model.load_state_dict(state, strict=True)
        else:
            # assume nep.txt
            model.load_weights_from_nep_txt(ft_path)
        _log(f"Fine-tuning from: {ft_path}")
        _log(f"Model: {sum(p.numel() for p in model.parameters())} params, "
             f"dim={model.dim}, b1 loaded={model.b1.item():.4f}")
    else:
        mean_epa = np.mean([data_store.energy[i] / data_store.natoms[i]
                            for i in range(data_store.n)
                            if data_store.has_energy_flag[i]])
        with torch.no_grad():
            model.b1.fill_(-mean_epa)
        _log(f"Model: {sum(p.numel() for p in model.parameters())} params, "
             f"dim={model.dim}, b1 init={model.b1.item():.4f}")

    # q_scaler: compute on rank 0, broadcast in DDP
    _log("Computing q_scaler...")
    if is_ddp and not is_main:
        q_min = torch.zeros(model.dim, dtype=dtype, device=dev)
        q_max = torch.zeros(model.dim, dtype=dtype, device=dev)
    else:
        q_min, q_max = compute_q_scaler(model, data_store, batch_size,
                                        pytorch_only=pytorch_only)
    if is_ddp:
        dist.broadcast(q_min, 0)
        dist.broadcast(q_max, 0)
    model.set_q_scaler(q_min, q_max)

    if use_compile and hasattr(torch, "compile"):
        _log("Compiling model with torch.compile...")
        model = torch.compile(model)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = (model.module if is_ddp
                 else (model._orig_mod if hasattr(model, "_orig_mod") else model))

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
            swa_model = AveragedModel(raw_model)

    ckpt_path = os.path.join(output_dir, "checkpoint.pt")
    start_epoch = 1
    best_loss = float("inf")
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

    loss_log = None
    if is_main:
        loss_log_mode = "a" if (restart and start_epoch > 1) else "w"
        loss_log = open(os.path.join(output_dir, "loss.out"), loss_log_mode)
        if loss_log_mode == "w":
            loss_log.write("epoch  loss  rmse_e(meV/atom)  rmse_f(eV/A)  "
                           "rmse_v(meV/atom)  gnorm\n")

    backend_str = "pure-PyTorch" if pytorch_only else "CUDA-kernel accelerated"
    force_str = ("autograd (create_graph)" if use_autograd_forces
                 else "analytical")
    clip_str = f"grad_clip={max_grad_norm}" if max_grad_norm > 0 else "no grad clip"
    loss_type = f"Huber(delta={huber_delta})" if use_huber else "MSE"
    _log(f"\nTraining: epochs {start_epoch}-{num_epochs}, "
         f"batch={batch_size}, dtype={precision}")
    _log(f"Backend: {backend_str} | forces: {force_str} | "
         f"{clip_str} | loss: {loss_type}")
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

            # Deterministic permutation — same on all ranks in DDP
            g = torch.Generator()
            g.manual_seed(epoch)
            perm = torch.randperm(n_structs, generator=g)

            sum_loss = sum_le = sum_lf = sum_lv = 0.0
            sum_e_structs = sum_f_atoms = sum_v_structs = 0
            n_batch = 0
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

            batch_starts = list(range(0, n_structs, batch_size))
            for bi, start in enumerate(batch_starts):
                # DDP: round-robin shard batches across ranks
                if is_ddp and bi % world_size != rank:
                    continue

                idx = perm[start:start + batch_size].tolist()
                batch = data_store.collate(idx)

                if use_autograd_forces:
                    result = raw_model.compute_properties(
                        batch["rij_rad"], batch["rij_ang"],
                        batch["pair_i_rad"], batch["pair_j_rad"],
                        batch["pair_i_ang"], batch["pair_j_ang"],
                        batch["atom_types"], batch["N"],
                        batch["struct_idx"], batch["num_structures"],
                        need_forces=has_forces, need_virial=has_virial)
                else:
                    result = raw_model.compute_properties_cached(
                        batch, need_forces=has_forces, need_virial=has_virial,
                        pytorch_only=pytorch_only)

                e_pa_pred = result["Etot"] / batch["natoms"]
                e_pa_ref = batch["energy"] / batch["natoms"]
                e_mask = batch["energy_mask"]
                loss = torch.tensor(0.0, dtype=dtype, device=dev)
                if e_mask.any():
                    loss_e = _loss_fn(e_pa_pred[e_mask], e_pa_ref[e_mask])
                    loss = loss + cur_pref_e * loss_e
                    sum_le += loss_e.item() * e_mask.sum().item()

                if has_forces:
                    f_mask = batch["force_mask"]
                    if f_mask.any():
                        f_pred = result["forces"][f_mask]
                        f_ref = batch["forces"][f_mask]
                        loss_f = _loss_fn(f_pred, f_ref)
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
                            loss_v = _loss_fn(v_sys[v_mask] / na,
                                              v_ref[v_mask] / na)
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

            # DDP: aggregate metrics across ranks
            if is_ddp:
                metrics = torch.tensor(
                    [sum_loss, sum_le, sum_lf, sum_lv,
                     float(sum_e_structs), float(sum_f_atoms),
                     float(sum_v_structs), float(n_batch)],
                    device=dev)
                dist.all_reduce(metrics)
                gn_t = torch.tensor(max_gn, device=dev)
                dist.all_reduce(gn_t, op=dist.ReduceOp.MAX)
                (sum_loss, sum_le, sum_lf, sum_lv,
                 sum_e_structs, sum_f_atoms, sum_v_structs,
                 n_batch) = metrics.tolist()
                max_gn = gn_t.item()

            avg_loss = sum_loss / max(n_batch, 1)
            rmse_e = np.sqrt(sum_le / max(sum_e_structs, 1)) * 1000
            rmse_f = (np.sqrt(sum_lf / max(sum_f_atoms, 1))
                      if sum_lf > 0 else 0.0)
            rmse_v = (np.sqrt(sum_lv / max(sum_v_structs, 1)) * 1000
                      if sum_lv > 0 else 0.0)
            dt = time.time() - t_epoch

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
                        os.path.join(output_dir, "nep.txt"),
                        max_NN_rad, max_NN_ang)
                    torch.save(raw_model.state_dict(),
                               os.path.join(output_dir, "best_model.pt"))

                if epoch % checkpoint_interval == 0 or epoch == num_epochs:
                    _save_checkpoint(ckpt_path, model, optimizer,
                                     lr_scheduler, epoch, best_loss)
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

        nep_file = os.path.join(output_dir, "nep.txt")
        if os.path.exists(nep_file):
            _log("\nRunning prediction on training set...")
            pred_t0 = time.time()
            predict_dataset(nep_file, data_file, output_dir=output_dir,
                            dtype="float64", device=str(dev))
            _log(f"  Prediction time: {time.time() - pred_t0:.1f}s")

        total_time = time.time() - total_t0
        h, rem = divmod(total_time, 3600)
        m_, s = divmod(rem, 60)
        _log(f"\nTotal time (data + train + predict): "
             f"{int(h):02d}:{int(m_):02d}:{s:04.1f}")
        _log(f"Output: {output_dir}/")
        _out_log_file.close()

    if is_ddp:
        dist.destroy_process_group()
