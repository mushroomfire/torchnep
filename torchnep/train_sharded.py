# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Yongchao Wu.
# Part of the torchnep project — https://github.com/mushroomfire/torchnep.

"""
Data-sharded distributed NEP training.

Each rank loads only 1/N of the training structures, so data-store memory
scales as 1/N instead of being replicated.  Gradients are all-reduced by
DDP; q_scaler statistics and per-epoch metrics are all-reduced explicitly.

Usage (typical multi-GPU launch via torchrun):

    torchrun --nproc_per_node=N run_train.py

where run_train.py calls ``train_nep_sharded(...)`` instead of ``train_nep``.

Backend is chosen automatically: NCCL when every rank has its own CUDA
device, otherwise gloo (covers CPU-only testing and the GPU-sharing case).
Single-rank (N=1) runs but offers nothing over ``train_nep`` in that case.
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

import torch.nn as nn
from .model import NEPModel
from .data import read_xyz, parse_nep_in, build_neighbor_list_np
from . import ops
from . import __version__
from .predict import predict_dataset
from .model import slim_model


class _NEPDDPShim(nn.Module):
    """Thin wrapper whose forward calls compute_properties{_cached}.

    Why this exists: DDP's gradient all-reduce is armed inside
    ``DistributedDataParallel.forward`` (it calls
    ``reducer.prepare_for_backward`` on the output). Calling
    ``self.module.compute_properties_cached(...)`` directly — even though the
    parameters live in the module — bypasses DDP's forward and therefore the
    reducer. Each rank then keeps its local gradient; weights drift per-rank
    and only rank 0's overfit copy is saved. Putting the compute call inside
    this shim's ``forward`` puts it on the DDP path.
    """

    def __init__(self, model: NEPModel):
        super().__init__()
        self.model = model

    def forward(self, batch, use_autograd_forces: bool,
                need_forces: bool, need_virial: bool, backend: str):
        if use_autograd_forces:
            return self.model.compute_properties(
                batch["rij_rad"], batch["rij_ang"],
                batch["pair_i_rad"], batch["pair_j_rad"],
                batch["pair_i_ang"], batch["pair_j_ang"],
                batch["atom_types"], batch["N"],
                batch["struct_idx"], batch["num_structures"],
                need_forces=need_forces, need_virial=need_virial,
                backend=backend)
        return self.model.compute_properties_cached(
            batch, need_forces=need_forces, need_virial=need_virial,
            backend=backend)

from .train import (
    _BANNER, _AUTHOR,
    _backend_info, _default_device,
    GPUDataStore,
    preprocess_structures,
    _save_checkpoint, _load_checkpoint,
    _sort_and_shard,
)


# ---------------------------------------------------------------------------
# Sharded q_scaler
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_q_scaler_sharded(model, data_store, batch_size=1000,
                               backend="loop"):
    """Compute descriptor min/max over the local shard, then all-reduce.

    Uses cached basis from data_store (no Chebyshev recompute). batch_size
    here is q-scaler-only (independent from training batch), default 1000.
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
    """Data-sharded NEP training.  Launch via torchrun (or any launcher that
    sets RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT).

    All hyperparameters (epoch / batch / lr / lambda_e,f,v / stage2* / ...)
    come from ``config_file`` — see README for the nep.in reference.

    Each rank loads structures[rank::world_size] only, so data-store memory
    scales as 1/world_size.  Gradients are all-reduced by DDP; q_scaler and
    epoch metrics are all-reduced explicitly.

        torchrun --nproc_per_node=N run_train.py

    Parameters mirror ``train_nep``; the only runtime-exclusive differences
    are the DDP launch, CUDA/gloo backend auto-select, and distributed
    q_scaler/metric aggregation.
    """
    # ---- Distributed init ------------------------------------------------
    # Wrap local_rank around the number of visible GPUs — lets several
    # processes share one GPU (useful for locally simulating multi-rank DDP).
    # NCCL refuses to share a GPU across ranks, so fall back to gloo (slower
    # but correct) when world_size > available GPUs.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cuda_available = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if cuda_available else 0
    if cuda_available:
        gpu_id = local_rank % max(1, n_gpus)
        torch.cuda.set_device(gpu_id)
        dev = torch.device(f"cuda:{gpu_id}")
    else:
        gpu_id = None
        dev = torch.device("cpu")
    if not dist.is_initialized():
        world_size_env = int(os.environ.get("WORLD_SIZE", 1))
        ddp_backend = "nccl" if cuda_available and world_size_env <= n_gpus else "gloo"
        # Pass device_id so NCCL can bind the rank to its CUDA device
        # deterministically (silences "Guessing device ID based on global
        # rank" and the collective-context warnings, and prevents hangs on
        # heterogeneous rank→GPU mappings). gloo ignores device_id.
        dist.init_process_group(backend=ddp_backend,
                                device_id=dev if cuda_available else None)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

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

    # ---- Config (all hyperparameters from nep.in) -----------------------
    orig_config = parse_nep_in(config_file)
    config = orig_config
    lambda_1 = config["lambda_1"]
    lambda_2 = config["lambda_2"]
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
    stage2             = config["stage2"]
    start_stage2       = config.get("start_stage2")
    stage2_lr          = config["stage2_lr"]
    stage2_pref_e      = config["stage2_pref_e"]
    stage2_pref_f      = config["stage2_pref_f"]
    stage2_pref_v      = config["stage2_pref_v"]

    # ---- Data: each rank loads 1/world_size of structures ----------------
    _log("Loading training data...")
    _log(f"  energy label: {energy_key}")
    frames = read_xyz(data_file, energy_key=energy_key)
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

    # Bucket sharding: sort globally by natoms, assign rank r the r-th
    # batch_size-block inside every full bucket, then split any remaining
    # frames as evenly as possible across ranks (the last iteration of the
    # epoch may have a rank-dependent batch size; all ranks still do the
    # same number of iterations, which is all DDP requires). Iteration i
    # on every rank draws from the same natoms quantile, so per-batch
    # compute is matched across ranks.
    local_frames, boundaries, dropped = _sort_and_shard(
        frames, rank=rank, world_size=world_size,
        batch_size=config["batch_size"])
    n_local = len(local_frames)
    num_buckets = len(boundaries)
    drop_note = (f", {dropped} frames dropped (remainder < world_size)"
                 if dropped else "")
    _log(f"  {n_total} structures total → "
         f"rank {rank} gets {n_local} (bucket-sharded across {world_size} ranks, "
         f"{num_buckets} buckets × batch_size={config['batch_size']}"
         f"{drop_note})")

    _log("Building neighbor lists (local shard)...")
    t0 = time.time()
    np_dtype = np.float64 if precision == "float64" else np.float32
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
    if cuda_available:
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

    # Resolve "auto" backend now that we know num_types + kernel availability.
    from .ops import resolve_backend as _resolve_backend
    backend = _resolve_backend(backend, num_types=model.num_types)
    _log(f"Compute backend: {backend}")

    # q_scaler: local shard → all_reduce
    _log("Computing q_scaler (all-reduce across shards)...")
    t_qs = time.time()
    q_min, q_max = _compute_q_scaler_sharded(model, data_store, backend=backend)
    model.set_q_scaler(q_min, q_max)
    if cuda_available:
        torch.cuda.synchronize()
    _log(f"  Done in {time.time() - t_qs:.1f}s")

    if use_compile and hasattr(torch, "compile"):
        _log("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Wrap in a shim whose forward calls compute_properties{_cached} — this
    # keeps the force/virial compute on DDP's forward path so the reducer can
    # arm backward all-reduce. See _NEPDDPShim docstring.
    shim = _NEPDDPShim(model)
    # All per-type nets are always touched in compute_properties_cached (dummy
    # pass for types absent in a given batch) so DDP sees every parameter in
    # every step — no need for find_unused_parameters, and no implicit grad
    # dilution for rare types.
    model = DDP(shim,
                device_ids=[gpu_id] if cuda_available else None,
                find_unused_parameters=False)
    # raw_model: unwrap DDP → shim → (optional torch.compile) → NEPModel
    _shim = model.module
    inner = _shim.model
    raw_model = inner._orig_mod if hasattr(inner, "_orig_mod") else inner

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

    if reset_lr is not None:
        for pg in optimizer.param_groups:
            pg["lr"] = reset_lr
        _log(f"LR reset to {reset_lr:.2e}")

    n_local = data_store.n
    has_forces = global_has_forces and pref_f > 0
    has_virial = global_has_virial and pref_v > 0

    loss_log = None
    if is_main:
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

            # Bucket-order shuffle: same seed on every rank → identical
            # bucket_perm across ranks. Each rank processes its own slice of
            # each bucket, so iteration i on every rank draws from the same
            # natoms quantile — per-batch compute stays balanced.
            g = torch.Generator()
            g.manual_seed(epoch)
            bucket_perm = torch.randperm(num_buckets, generator=g).tolist()

            sum_le = sum_lf = sum_lv = sum_ls = 0.0  # sum_ls is in (eV/Å³)²
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
                start, end = boundaries[bi]
                idx = list(range(start, end))
                batch = data_store.collate(idx)

                # Go through DDP wrapper (not raw_model.compute_*) so the
                # reducer arms backward all-reduce for this step.
                result = model(batch, use_autograd_forces,
                               has_forces, has_virial, backend)

                e_pa_pred = result["Etot"] / batch["natoms"]
                e_pa_ref = batch["energy"] / batch["natoms"]
                e_mask = batch["energy_mask"]
                f_mask = batch["force_mask"] if has_forces else None
                v_mask = batch["virial_mask"] if has_virial else None

                # --- DDP-correct normalisation --------------------------
                # DDP averages gradients by world_size. If each rank used a
                # plain local .mean() loss, the implicit per-rank n_local in
                # the denominator leaks into the global gradient (only
                # equivalent to single-card global mean when all n_local are
                # equal — which is NOT the case here because frames have
                # different atom counts).
                # Fix: each rank computes SUM-of-squared-errors, and we divide
                # by the GLOBAL count (all-reduced per batch). The × world_size
                # factor cancels DDP's /world_size averaging — giving a true
                # global-mean loss regardless of how atoms are sharded.
                counts = torch.tensor([
                    float(e_mask.sum().item()),
                    float(f_mask.sum().item()) if f_mask is not None else 0.0,
                    float(v_mask.sum().item()) if v_mask is not None else 0.0,
                ], device=dev, dtype=torch.float64)
                dist.all_reduce(counts)
                n_e_g = max(counts[0].item(), 1.0)
                n_f_g = max(counts[1].item(), 1.0)
                n_v_g = max(counts[2].item(), 1.0)
                ws = float(world_size)

                loss = torch.tensor(0.0, dtype=dtype, device=dev)

                if e_mask.any():
                    diff_e = e_pa_pred[e_mask] - e_pa_ref[e_mask]
                    sum_sq_e = (diff_e ** 2).sum()
                    loss = loss + cur_pref_e * sum_sq_e * ws / n_e_g
                    sum_le += sum_sq_e.item()  # global sum-of-squared-errors

                if f_mask is not None and f_mask.any():
                    f_pred = result["forces"][f_mask]
                    f_ref = batch["forces"][f_mask]
                    sum_sq_f = ((f_pred - f_ref) ** 2).sum()
                    # 3 components per atom → divide by (3 × n_f_g)
                    loss = loss + cur_pref_f * sum_sq_f * ws / (3.0 * n_f_g)
                    sum_lf += (sum_sq_f.item() / 3.0)

                if v_mask is not None and v_mask.any() and "virial" in result:
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
                        sum_sq_v = (v_diff ** 2).sum()
                        # 9 components per frame → divide by (9 × n_v_g)
                        loss = loss + cur_pref_v * sum_sq_v * ws / (9.0 * n_v_g)
                        sum_lv += (sum_sq_v.item() / 9.0)
                        # Stress (eV/Å³) = -virial_total / V. Sign cancels in MSE.
                        scale = (batch["natoms"][v_mask]
                                 / batch["volumes"][v_mask]).unsqueeze(-1)
                        sum_sq_s = ((v_diff * scale) ** 2).sum()
                        sum_ls += (sum_sq_s.item() / 9.0)

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

                sum_e_structs += batch["energy_mask"].sum().item()
                sum_f_atoms += batch["force_mask"].sum().item()
                sum_v_structs += batch["virial_mask"].sum().item()
                max_gn = max(max_gn, gn)

            # Aggregate metrics across all ranks
            metrics = torch.tensor(
                [sum_le, sum_lf, sum_lv, sum_ls,
                 float(sum_e_structs), float(sum_f_atoms),
                 float(sum_v_structs)],
                device=dev)
            dist.all_reduce(metrics)
            gn_t = torch.tensor(max_gn, device=dev)
            dist.all_reduce(gn_t, op=dist.ReduceOp.MAX)
            (sum_le, sum_lf, sum_lv, sum_ls,
             sum_e_structs, sum_f_atoms, sum_v_structs) = metrics.tolist()
            max_gn = gn_t.item()

            # Per-sample (not per-batch) averaging so avg_loss is self-
            # consistent with rmse_{e,f,v}: avg_loss == Σ pref_X · MSE_X
            # where each MSE_X aggregates over all samples in the epoch.
            from .constants import EV_PER_A3_TO_GPa
            mse_e = sum_le / max(sum_e_structs, 1)
            mse_f = sum_lf / max(sum_f_atoms, 1) if sum_lf > 0 else 0.0
            mse_v = sum_lv / max(sum_v_structs, 1) if sum_lv > 0 else 0.0
            mse_s = sum_ls / max(sum_v_structs, 1) if sum_ls > 0 else 0.0
            avg_loss = (cur_pref_e * mse_e + cur_pref_f * mse_f
                        + cur_pref_v * mse_v)
            rmse_e = np.sqrt(mse_e)                           # eV/atom
            rmse_f = np.sqrt(mse_f)                           # eV/Å
            rmse_v = np.sqrt(mse_v)                           # eV/atom
            rmse_s_gpa = np.sqrt(mse_s) * EV_PER_A3_TO_GPa    # GPa
            dt = time.time() - t_epoch

            if in_stage2 and stage2_scheduler is not None:
                stage2_scheduler.step(avg_loss)
            elif not in_stage2:
                lr_scheduler.step(avg_loss)

            if is_main:
                loss_log.write(f"{epoch} {avg_loss:.6e} {rmse_e:.6f} "
                               f"{rmse_f:.6f} {rmse_v:.6f} "
                               f"{rmse_s_gpa:.4f} {max_gn:.2f}\n")
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
                                     lr_scheduler, epoch, best_loss)

                # Interim predict — rank 0 only, reads nep_best.txt from disk.
                # Each rank has only 1/N of the structures so the in-memory
                # data_store can't give a full dataset prediction; we fall back
                # to the file-based predict_dataset here. Skip on the last
                # epoch — the end-of-training predict runs right after.
                if (prediction_interval > 0
                        and epoch % prediction_interval == 0
                        and epoch != num_epochs
                        and os.path.exists(os.path.join(output_dir,
                                                         "nep_best.txt"))):
                    predict_dataset(
                        os.path.join(output_dir, "nep_best.txt"),
                        data_file, output_dir=output_dir,
                        dtype=precision, device=str(dev),
                        backend=backend, verbose=False)
    finally:
        if is_main and loss_log is not None:
            loss_log.close()

    # Free every rank's data_store BEFORE rank 0 runs end-of-training predict
    # (which re-reads full xyz). This matters for simulated multi-process-on-
    # single-GPU testing; on real multi-GPU it is free extra memory for the
    # predict pass.
    del data_store
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    if is_main:
        raw_model.save_nep_txt(os.path.join(output_dir, "nep_final.txt"),
                               max_NN_rad, max_NN_ang)
        if swa_model is not None:
            swa_state = swa_model.module.state_dict()
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

        # End-of-training predict: rank 0 only. In sharded mode each rank holds
        # only 1/N of the structures, so reusing the local store would miss
        # the other ranks' frames — simpler to re-read the full xyz on rank 0.
        _log("\nRunning prediction on training set (final-epoch model)...")
        pred_t0 = time.time()
        predict_dataset(os.path.join(output_dir, "nep_final.txt"),
                        data_file, output_dir=output_dir,
                        dtype=precision, device=str(dev),
                        backend=backend)
        _log(f"  Prediction time: {time.time() - pred_t0:.1f}s")

        total_time = time.time() - total_t0
        h, rem = divmod(total_time, 3600)
        m_, s = divmod(rem, 60)
        _log(f"\nTotal time (data + train + predict): "
             f"{int(h):02d}:{int(m_):02d}:{s:04.1f}")
        _log(f"Output: {output_dir}/")
        _out_log_file.close()

    dist.destroy_process_group()
