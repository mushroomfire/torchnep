"""
Data-sharded distributed NEP training.

Each GPU rank loads only 1/N of the training structures, so the total
GPU memory scales as 1/N instead of being replicated.  Gradients are
still all-reduced by DDP; q_scaler statistics and per-epoch metrics are
all-reduced explicitly.

Usage (must be launched with torchrun):

    torchrun --nproc_per_node=N run_train.py

where run_train.py calls ``train_nep_sharded(...)`` instead of ``train_nep``.

Single-GPU (N=1) works but is identical to ``train_nep`` in that case;
the function enforces torchrun launch so that dist is always initialised.
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
from .cuda_ops import _load_cached_kernels
from .model import slim_model
from .train import (
    _BANNER, _AUTHOR,
    _backend_info, _default_device,
    GPUDataStore,
    preprocess_structures,
    _save_checkpoint, _load_checkpoint,
)


# ---------------------------------------------------------------------------
# Sharded q_scaler
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_q_scaler_sharded(model, data_store, batch_size=64,
                               pytorch_only=True):
    """Compute descriptor min/max over the local shard, then all-reduce.

    Returns (q_min, q_max) that are globally consistent across all ranks.
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

    # Aggregate across all ranks
    dist.all_reduce(q_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(q_max, op=dist.ReduceOp.MAX)

    model.train()
    return q_min, q_max


# ---------------------------------------------------------------------------
# Sharded training entry point
# ---------------------------------------------------------------------------

def train_nep_sharded(
    config_file: str,
    data_file: str,
    output_dir: str = ".",
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
    slim_types: bool = False,
):
    """Data-sharded NEP training.  Must be launched via torchrun.

    Each rank loads structures[rank::world_size] only, so total GPU memory
    for the data store scales as 1/world_size.  Gradients are all-reduced by
    DDP; q_scaler and epoch metrics are all-reduced explicitly.

        torchrun --nproc_per_node=N run_train.py
    """
    # ---- Distributed init ------------------------------------------------
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    is_main = rank == 0
    dtype = torch.float32 if precision == "float32" else torch.float64

    # ---- Logging (rank 0 only) -------------------------------------------
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
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
    _log(f"Mode     : data-sharded DDP ({world_size} ranks, "
         f"each holds 1/{world_size} of structures)")
    _log("")

    # ---- CUDA kernels ----------------------------------------------------
    # Rank 0 JIT-compiles (if cache is cold) while others wait. After the
    # barrier all ranks load the same cached .so — no torch-extensions race.
    if not pytorch_only:
        if is_main:
            _load_cached_kernels()
        dist.barrier()
        if not is_main:
            _load_cached_kernels()

    # ---- Config ----------------------------------------------------------
    orig_config = parse_nep_in(config_file)
    config = orig_config
    lambda_1 = config.get("lambda_1", 0.0)
    lambda_2 = config.get("lambda_2", 0.0)

    def _cfg(arg_val, cfg_key, default):
        return arg_val if arg_val is not None else config.get(cfg_key, default)

    num_epochs         = _cfg(num_epochs,          "num_epochs",          200)
    batch_size         = _cfg(batch_size,           "batch_size",          32)
    lr                 = _cfg(lr,                   "lr",                  0.01)
    max_grad_norm      = _cfg(max_grad_norm,         "max_grad_norm",       10.0)
    pref_e             = _cfg(pref_e,               "lambda_e",            1.0)
    pref_f             = _cfg(pref_f,               "lambda_f",            100.0)
    pref_v             = _cfg(pref_v,               "lambda_v",            1.0)
    scheduler_patience = _cfg(scheduler_patience,   "scheduler_patience",  50)
    scheduler_factor   = _cfg(scheduler_factor,     "scheduler_factor",    0.8)
    stop_lr            = _cfg(stop_lr,              "stop_lr",             1e-6)
    huber_delta        = _cfg(huber_delta,          "huber_delta",         0.0)
    stage2             = _cfg(stage2,               "stage2",              False)
    start_stage2       = _cfg(start_stage2,         "start_stage2",        None)
    stage2_lr          = _cfg(stage2_lr,            "stage2_lr",           1e-3)
    stage2_pref_e      = _cfg(stage2_pref_e,        "stage2_pref_e",       1000.0)
    stage2_pref_f      = _cfg(stage2_pref_f,        "stage2_pref_f",       100.0)
    stage2_pref_v      = _cfg(stage2_pref_v,        "stage2_pref_v",       10.0)
    use_swa            = _cfg(use_swa,              "use_swa",             True)

    # ---- Data: each rank loads 1/world_size of structures ----------------
    _log("Loading training data...")
    frames = read_xyz(data_file)
    n_total = len(frames)

    # slim_types: all ranks agree on which types to keep (deterministic scan)
    _slim_keep = None
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

    _log(f"  {n_total} structures total → "
         f"rank {rank} loads {len(frames[rank::world_size])}")

    _log("Building neighbor lists (local shard)...")
    t0 = time.time()
    np_dtype = np.float64 if precision == "float64" else np.float32
    local_frames = frames[rank::world_size]
    structures = preprocess_structures(local_frames, config, np_dtype)
    _log(f"  Done in {time.time() - t0:.1f}s")

    # max_NN: local max then all-reduce so rank-0 has the global value
    def _compute_max_neighbors_local(structures):
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

    local_max_rad, local_max_ang = _compute_max_neighbors_local(structures)
    nn_t = torch.tensor([local_max_rad, local_max_ang], dtype=torch.long,
                        device=dev)
    dist.all_reduce(nn_t, op=dist.ReduceOp.MAX)
    max_NN_rad, max_NN_ang = int(nn_t[0].item()), int(nn_t[1].item())

    _log(f"Pre-loading local shard to {dev} (with cached basis)...")
    t0 = time.time()
    data_store = GPUDataStore(structures, dev, dtype, config=config)
    del structures
    torch.cuda.synchronize()
    _log(f"  Loaded ({time.time() - t0:.1f}s)")

    # Aggregate data counts across all ranks for the banner
    counts_t = torch.tensor(
        [data_store.n, data_store.n_energy,
         data_store.n_forces, data_store.n_virial],
        dtype=torch.long, device=dev)
    dist.all_reduce(counts_t)
    g_n, g_ne, g_nf, g_nv = counts_t.tolist()
    _log(f"  Global data: {g_n} structures, {g_ne} with energy, "
         f"{g_nf} with forces, {g_nv} with virial")

    # ---- Model -----------------------------------------------------------
    model = NEPModel(config).to(dtype).to(dev)

    if finetune_from is not None:
        def _load_weights(target, path):
            if path.endswith(".pt"):
                state = torch.load(path, map_location=dev, weights_only=False)
                if "model_state" in state:
                    state = state["model_state"]
                target.load_state_dict(state, strict=True)
            else:
                target.load_weights_from_nep_txt(path)

        if _slim_keep is not None:
            full_model = NEPModel(orig_config).to(dtype).to(dev)
            _load_weights(full_model, finetune_from)
            slimmed = slim_model(full_model, config["type_names"])
            model.load_state_dict(slimmed.state_dict())
            del full_model, slimmed
            _log(f"Fine-tuning from: {finetune_from}  "
                 f"[{orig_config['num_types']} → {config['num_types']} types]")
        else:
            _load_weights(model, finetune_from)
            _log(f"Fine-tuning from: {finetune_from}")
        _log(f"Model: {sum(p.numel() for p in model.parameters())} params, "
             f"dim={model.dim}, b1={model.b1.item():.4f}")
    else:
        # mean_epa: weighted average across all ranks
        local_epa_sum = sum(
            data_store.energy[i] / data_store.natoms[i]
            for i in range(data_store.n) if data_store.has_energy_flag[i]
        )
        local_n_e = float(data_store.n_energy)
        epa_t = torch.tensor([local_epa_sum, local_n_e], device=dev, dtype=torch.float64)
        dist.all_reduce(epa_t)
        mean_epa = float(epa_t[0] / epa_t[1]) if epa_t[1] > 0 else 0.0
        with torch.no_grad():
            model.b1.fill_(-mean_epa)
        _log(f"Model: {sum(p.numel() for p in model.parameters())} params, "
             f"dim={model.dim}, b1 init={model.b1.item():.4f}")

    # has_forces / has_virial: OR across ranks
    flags_t = torch.tensor(
        [int(data_store.has_forces), int(data_store.has_virial)],
        dtype=torch.long, device=dev)
    dist.all_reduce(flags_t, op=dist.ReduceOp.MAX)
    global_has_forces = bool(flags_t[0].item())
    global_has_virial = bool(flags_t[1].item())

    # q_scaler: local shard → all_reduce
    _log("Computing q_scaler (all-reduce across shards)...")
    q_min, q_max = _compute_q_scaler_sharded(model, data_store, batch_size,
                                              pytorch_only=pytorch_only)
    model.set_q_scaler(q_min, q_max)

    if use_compile and hasattr(torch, "compile"):
        _log("Compiling model with torch.compile...")
        model = torch.compile(model)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = (model.module._orig_mod
                 if hasattr(model.module, "_orig_mod")
                 else model.module)

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
    if restart and os.path.exists(ckpt_path):
        start_epoch, best_loss = _load_checkpoint(
            ckpt_path, model, optimizer, lr_scheduler, dev)
        start_epoch += 1
        _log(f"Resumed from checkpoint: epoch {start_epoch - 1}, "
             f"best_loss={best_loss:.4e}")

    n_local = data_store.n
    has_forces = global_has_forces and pref_f > 0
    has_virial = global_has_virial and pref_v > 0

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

            # Each rank permutes its own local indices independently.
            # DDP gradient all-reduce ensures consistent parameter updates.
            g = torch.Generator()
            g.manual_seed(epoch * world_size + rank)
            perm = torch.randperm(n_local, generator=g)

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

            for start in range(0, n_local, batch_size):
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
                # sum_l* always accumulates the true MSE so rmse_* columns in the
                # log are real RMSE regardless of huber_delta. The optimizer sees
                # _loss_fn (Huber or MSE) as the gradient source.
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
                        p.grad.norm() ** 2 for p in raw_model.parameters()
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

            # Aggregate metrics across all ranks
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

    dist.destroy_process_group()
