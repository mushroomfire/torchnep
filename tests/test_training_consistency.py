"""
Test: pure-PyTorch CPU vs GPU training consistency.

Runs 5 epochs on CPU and GPU with the same random seed and verifies that
per-epoch losses and final model parameters agree within float32 tolerance.

This is the baseline correctness check before enabling handwritten CUDA kernels.
To later compare pure-torch vs CUDA-kernel results, run the same test but
pass pytorch_only=False for the GPU run.

Usage:
    python tests/test_training_consistency.py
    # or with pytest:
    pytest tests/test_training_consistency.py -v
"""

import os
import sys

import numpy as np
import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, ".."))

from torchnep import ops
from torchnep.data import parse_nep_in, read_xyz
from torchnep.model import NEPModel
from torchnep.train import (
    GPUDataStore,
    compute_q_scaler,
    preprocess_structures,
)

CONFIG_FILE = os.path.join(_DIR, "input_files", "nep.in")
DATA_FILE = os.path.join(_DIR, "input_files", "train_small.xyz")

SEED = 42
NUM_EPOCHS = 5
BATCH_SIZE = 8
LR = 1e-2


# ---------------------------------------------------------------------------
# Core training loop (extracted from train_nep, no file I/O)
# ---------------------------------------------------------------------------

def run_epochs(device_str, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
               lr=LR, seed=SEED, pytorch_only=True):
    """Run training and return per-epoch metrics + final parameters.

    Returns
    -------
    dict with keys:
        'losses'   : list[float], total loss per epoch
        'rmse_e'   : list[float], energy RMSE (meV/atom)
        'rmse_f'   : list[float], force RMSE (eV/Å)
        'rmse_v'   : list[float], virial RMSE (meV/atom)
        'params'   : dict[str, np.ndarray], final model parameters
        'grads'    : dict[str, np.ndarray], parameter gradients at last batch
                     of last epoch
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dev = torch.device(device_str)
    dtype = torch.float32

    config = parse_nep_in(CONFIG_FILE)
    lambda_e = config.get("lambda_e", 1.0)
    lambda_f = config.get("lambda_f", 1.0)
    lambda_v = config.get("lambda_v", 0.1)

    frames = read_xyz(DATA_FILE)
    structures = preprocess_structures(frames, config, np.float32)

    data_store = GPUDataStore(structures, dev, dtype, config=config)
    has_forces = data_store.has_forces and lambda_f > 0
    has_virial = data_store.has_virial and lambda_v > 0

    model = NEPModel(config).to(dtype).to(dev)
    q_min, q_max = compute_q_scaler(model, data_store, batch_size,
                                    pytorch_only=pytorch_only)
    model.set_q_scaler(q_min, q_max)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    losses, rmse_e_list, rmse_f_list, rmse_v_list = [], [], [], []
    last_grads = {}

    n_structs = data_store.n

    for epoch in range(1, num_epochs + 1):
        model.train()
        perm = torch.randperm(n_structs, generator=torch.Generator().manual_seed(seed + epoch))
        sum_loss = sum_le = sum_lf = sum_lv = 0.0
        sum_atoms = sum_structs = n_batch = 0

        for start in range(0, n_structs, batch_size):
            idx = perm[start:start + batch_size].tolist()
            batch = data_store.collate(idx)

            result = model.compute_properties_cached(
                batch, need_forces=has_forces, need_virial=has_virial)

            e_pa_pred = result["Etot"] / batch["natoms"]
            e_pa_ref = batch["energy"] / batch["natoms"]
            loss_e = torch.mean((e_pa_pred - e_pa_ref) ** 2)
            loss = lambda_e * loss_e
            sum_le += loss_e.item() * batch["num_structures"]

            if has_forces:
                loss_f = torch.mean((result["forces"] - batch["forces"]) ** 2)
                loss = loss + lambda_f * loss_f
                sum_lf += loss_f.item() * batch["N"]

            if has_virial and "virial" in result:
                v_atom = result["virial"]
                v_sys = torch.zeros(batch["num_structures"], 9, dtype=dtype, device=dev)
                si = batch["struct_idx"].unsqueeze(-1).expand_as(v_atom)
                v_sys.scatter_add_(0, si, v_atom)
                v_ref = batch["virial"]
                if v_ref.shape[1] == 9:
                    na = batch["natoms"].unsqueeze(-1)
                    loss_v = torch.mean(((v_sys - v_ref) / na) ** 2)
                    loss = loss + lambda_v * loss_v
                    sum_lv += loss_v.item() * batch["num_structures"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                optimizer.step()

            sum_loss += loss.item()
            sum_atoms += batch["N"]
            sum_structs += batch["num_structures"]
            n_batch += 1

        # Capture gradients at last batch of last epoch
        if epoch == num_epochs:
            for name, p in model.named_parameters():
                if p.grad is not None:
                    last_grads[name] = p.grad.detach().cpu().numpy()

        scheduler.step()

        losses.append(sum_loss / n_batch)
        rmse_e_list.append(np.sqrt(sum_le / sum_structs) * 1000)
        rmse_f_list.append(np.sqrt(sum_lf / sum_atoms) if sum_lf > 0 else 0.0)
        rmse_v_list.append(np.sqrt(sum_lv / sum_structs) * 1000 if sum_lv > 0 else 0.0)

    params = {name: p.detach().cpu().numpy()
              for name, p in model.named_parameters()}

    return {
        "losses": losses,
        "rmse_e": rmse_e_list,
        "rmse_f": rmse_f_list,
        "rmse_v": rmse_v_list,
        "params": params,
        "grads": last_grads,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cpu_loss_decreases():
    """Sanity check: loss should decrease over 5 epochs on CPU."""
    res = run_epochs("cpu")
    losses = res["losses"]
    print(f"\nCPU losses: {[f'{v:.4e}' for v in losses]}")
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.4e} → {losses[-1]:.4e}")
    print("PASS: CPU loss decreases")


def test_gpu_loss_decreases():
    """Sanity check: loss should decrease over 5 epochs on GPU."""
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return
    res = run_epochs("cuda")
    losses = res["losses"]
    print(f"\nGPU losses: {[f'{v:.4e}' for v in losses]}")
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.4e} → {losses[-1]:.4e}")
    print("PASS: GPU loss decreases")


def test_cpu_gpu_consistency():
    """CPU and GPU (pytorch_only=True) must produce consistent results.

    Both use identical pure-PyTorch code paths.  Differences come only from
    floating-point execution order (GPU uses fused ops and different rounding).
    Expected tolerance for float32: ~1e-3 relative on losses, ~1e-4 on params.
    """
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return

    print("\nRunning CPU training (pytorch_only=True)...")
    cpu_res = run_epochs("cpu", pytorch_only=True)

    print("Running GPU training (pytorch_only=True)...")
    gpu_res = run_epochs("cuda", pytorch_only=True)

    # --- Per-epoch loss comparison ---
    print("\nPer-epoch loss comparison:")
    print(f"  {'Epoch':>5}  {'CPU loss':>12}  {'GPU loss':>12}  {'rel diff':>10}")
    for i, (lc, lg) in enumerate(zip(cpu_res["losses"], gpu_res["losses"]), 1):
        rel = abs(lc - lg) / (abs(lc) + 1e-12)
        print(f"  {i:5d}  {lc:12.6e}  {lg:12.6e}  {rel:10.2e}")

    # Tolerance: float32 CPU vs GPU can differ ~1e-3 on accumulated sums
    LOSS_RTOL = 5e-3
    for i, (lc, lg) in enumerate(zip(cpu_res["losses"], gpu_res["losses"]), 1):
        rel = abs(lc - lg) / (abs(lc) + 1e-12)
        assert rel < LOSS_RTOL, (
            f"Epoch {i} loss CPU={lc:.6e} GPU={lg:.6e} rel_diff={rel:.2e} "
            f"> tolerance {LOSS_RTOL}")

    print(f"\nAll {NUM_EPOCHS} epoch losses agree within rel_tol={LOSS_RTOL}")

    # --- Per-epoch RMSE comparison ---
    print("\nEnergy RMSE (meV/atom):")
    for i, (ec, eg) in enumerate(zip(cpu_res["rmse_e"], gpu_res["rmse_e"]), 1):
        print(f"  Epoch {i}: CPU={ec:.3f}  GPU={eg:.3f}")

    print("\nForce RMSE (eV/Å):")
    for i, (fc, fg) in enumerate(zip(cpu_res["rmse_f"], gpu_res["rmse_f"]), 1):
        print(f"  Epoch {i}: CPU={fc:.4f}  GPU={fg:.4f}")

    # --- Final parameter comparison ---
    print("\nFinal parameter max absolute differences:")
    PARAM_ATOL = 1e-4
    for name in cpu_res["params"]:
        pc = cpu_res["params"][name]
        pg = gpu_res["params"][name]
        diff = np.max(np.abs(pc - pg))
        print(f"  {name:40s}  max_diff={diff:.2e}")
        assert diff < PARAM_ATOL, (
            f"Parameter '{name}' differs too much: max_diff={diff:.2e} "
            f"> tolerance {PARAM_ATOL}")

    print(f"\nAll parameters agree within atol={PARAM_ATOL}")
    print("\nPASS: CPU and GPU (pure-torch) are consistent")


def test_no_nan_in_gradients():
    """Gradients must be finite (no NaN/Inf) for both CPU and GPU."""
    for dev in (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])):
        res = run_epochs(dev)
        for name, g in res["grads"].items():
            assert np.all(np.isfinite(g)), (
                f"[{dev}] NaN/Inf in gradient of '{name}'")
        print(f"PASS [{dev}]: all gradients finite after {NUM_EPOCHS} epochs")


def _build_descriptor_inputs(dev, dtype):
    """Build a batch and model for descriptor comparison tests."""
    config = parse_nep_in(CONFIG_FILE)
    frames = read_xyz(DATA_FILE)
    structures = preprocess_structures(frames, config, np.float32)
    data_store = GPUDataStore(structures, dev, dtype, config=None)
    batch = data_store.collate(list(range(min(8, data_store.n))))

    torch.manual_seed(SEED)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model = NEPModel(config).to(dtype).to(dev)
    return model, batch, config


def test_cuda_kernel_descriptors_forward():
    """CUDA descriptor forward values must match pure-PyTorch within float32 tolerance.

    The CUDA kernel computes the same Chebyshev + angular basis in a custom kernel.
    Small differences (~1e-6) are expected from different floating-point ordering.
    """
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return

    dev = torch.device("cuda")
    dtype = torch.float32
    model, batch, _ = _build_descriptor_inputs(dev, dtype)

    common = dict(
        rij_rad=batch["rij_rad"], rij_ang=batch["rij_ang"],
        pi_rad=batch["pair_i_rad"], pj_rad=batch["pair_j_rad"],
        pi_ang=batch["pair_i_ang"], pj_ang=batch["pair_j_ang"],
        atom_types=batch["atom_types"], N=batch["N"],
        rc_radial=model.rc_radial, rc_angular=model.rc_angular,
        basis_size_radial=model.basis_size_radial,
        basis_size_angular=model.basis_size_angular,
        n_max_radial=model.n_max_radial, n_max_angular=model.n_max_angular,
        l_max_3b=model.l_max_3b, l_max_4b=model.l_max_4b, l_max_5b=model.l_max_5b,
        num_lm=model.num_lm, c3b_coeffs=model._c3b,
        c4b_coeffs=model._c4b, c5b_coeffs=model._c5b,
        dtype=dtype, device=dev,
        c2=model.c_param_2, c3=model.c_param_3,
    )

    with torch.no_grad():
        q_pt = ops.compute_descriptors(**common, pytorch_only=True)
        q_cu = ops.compute_descriptors(**common, pytorch_only=False)

    diff = (q_pt - q_cu).abs()
    max_diff = diff.max().item()
    rel_diff = (diff / (q_pt.abs() + 1e-8)).max().item()
    print(f"\nDescriptor forward: max_abs_diff={max_diff:.2e}  max_rel_diff={rel_diff:.2e}")

    ATOL = 1e-4
    assert max_diff < ATOL, (
        f"CUDA descriptor forward mismatch: max_abs_diff={max_diff:.2e} > {ATOL}")
    print("PASS: CUDA descriptor forward values agree with PyTorch")


def test_cuda_kernel_descriptors_backward():
    """Check c2/c3 gradients from CUDA path match pure-PyTorch autograd.

    The CUDA path uses:
      - CUDA kernel for radial forward (fused scatter-add)
      - PyTorch backward for grad_c2 (recomputes fk; avoids --use_fast_math trig error)
      - Pure PyTorch for the full angular path (c3→gn→sum_fxyz→q_ang) so c3.grad flows

    Expected: both c2 and c3 gradients agree within float32 tolerance.
    """
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return

    dev = torch.device("cuda")
    dtype = torch.float32
    model, batch, _ = _build_descriptor_inputs(dev, dtype)

    c2_pt = model.c_param_2.detach().clone().requires_grad_(True)
    c3_pt = model.c_param_3.detach().clone().requires_grad_(True) \
        if model.c_param_3 is not None else None
    c2_cu = model.c_param_2.detach().clone().requires_grad_(True)
    c3_cu = model.c_param_3.detach().clone().requires_grad_(True) \
        if model.c_param_3 is not None else None

    common = dict(
        rij_rad=batch["rij_rad"], rij_ang=batch["rij_ang"],
        pi_rad=batch["pair_i_rad"], pj_rad=batch["pair_j_rad"],
        pi_ang=batch["pair_i_ang"], pj_ang=batch["pair_j_ang"],
        atom_types=batch["atom_types"], N=batch["N"],
        rc_radial=model.rc_radial, rc_angular=model.rc_angular,
        basis_size_radial=model.basis_size_radial,
        basis_size_angular=model.basis_size_angular,
        n_max_radial=model.n_max_radial, n_max_angular=model.n_max_angular,
        l_max_3b=model.l_max_3b, l_max_4b=model.l_max_4b, l_max_5b=model.l_max_5b,
        num_lm=model.num_lm, c3b_coeffs=model._c3b,
        c4b_coeffs=model._c4b, c5b_coeffs=model._c5b,
        dtype=dtype, device=dev,
    )

    q_pt = ops.compute_descriptors(**common, c2=c2_pt, c3=c3_pt, pytorch_only=True)
    q_cu = ops.compute_descriptors(**common, c2=c2_cu, c3=c3_cu, pytorch_only=False)
    q_pt.sum().backward()
    q_cu.sum().backward()

    print("\nCUDA vs PyTorch gradient comparison (known CUDA backward bug):")
    all_ok = True
    ATOL = 1e-4
    for name, g_pt, g_cu in [("c2", c2_pt.grad, c2_cu.grad),
                               ("c3", c3_pt.grad if c3_pt is not None else None,
                                c3_cu.grad if c3_cu is not None else None)]:
        if g_pt is None and g_cu is None:
            continue
        if g_cu is None:
            print(f"  [BUG] grad/{name}: CUDA returned no gradient (None)")
            all_ok = False
            continue
        if g_pt is None:
            print(f"  [BUG] grad/{name}: PyTorch returned no gradient (None)")
            all_ok = False
            continue
        gdiff = (g_pt - g_cu).abs().max().item()
        status = "OK" if gdiff < ATOL else "BUG"
        print(f"  [{status}] grad/{name} max_abs_diff={gdiff:.2e}")
        if gdiff >= ATOL:
            all_ok = False

    if all_ok:
        print("PASS: CUDA backward agrees with PyTorch (bug fixed!)")
    else:
        print("\n*** CUDA backward has known gradient bug ***")
        print("    Forward pass is correct; training uses pytorch_only=True, so")
        print("    this bug does NOT affect training correctness.")
        raise AssertionError(
            "CUDA c2/c3 gradients differ from PyTorch autograd. "
            "Use pytorch_only=True (default) for training until this is fixed."
        )


# ---------------------------------------------------------------------------
# Efficiency benchmark
# ---------------------------------------------------------------------------

def benchmark_pytorch_vs_cuda(n_warmup=3, n_repeat=10):
    """Compare per-epoch throughput: pure-PyTorch vs CUDA-kernel path.

    Both paths use compute_properties_cached (analytical forces) for training.
    The difference is only in compute_q_scaler: pytorch_only=True uses PyTorch
    descriptors; pytorch_only=False uses the CUDA radial forward kernel.

    Also benchmarks descriptor-only forward (compute_descriptors) to isolate
    the CUDA kernel speedup on the scatter-add bottleneck.
    """
    import time

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return

    dev = torch.device("cuda")
    dtype = torch.float32
    config = parse_nep_in(CONFIG_FILE)
    frames = read_xyz(DATA_FILE)
    structures = preprocess_structures(frames, config, np.float32)
    data_store = GPUDataStore(structures, dev, dtype, config=config)

    # Full batch (all structures)
    batch = data_store.collate(list(range(data_store.n)))

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = NEPModel(config).to(dtype).to(dev)
    q_min, q_max = compute_q_scaler(model, data_store, BATCH_SIZE, pytorch_only=True)
    model.set_q_scaler(q_min, q_max)
    model.train()

    def time_fn(fn, label):
        # Warmup
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            fn()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / n_repeat * 1000
        print(f"  {label:50s}  {elapsed:8.2f} ms")
        return elapsed

    print(f"\nBenchmark on {data_store.n} structures "
          f"({batch['N']} atoms, {batch['rij_rad'].shape[0]} rad pairs, "
          f"{batch['rij_ang'].shape[0]} ang pairs), {n_repeat} repeats:")

    # 1. compute_descriptors: pytorch_only=True vs False
    common_desc = dict(
        rij_rad=batch["rij_rad"], rij_ang=batch["rij_ang"],
        pi_rad=batch["pair_i_rad"], pj_rad=batch["pair_j_rad"],
        pi_ang=batch["pair_i_ang"], pj_ang=batch["pair_j_ang"],
        atom_types=batch["atom_types"], N=batch["N"],
        rc_radial=model.rc_radial, rc_angular=model.rc_angular,
        basis_size_radial=model.basis_size_radial,
        basis_size_angular=model.basis_size_angular,
        n_max_radial=model.n_max_radial, n_max_angular=model.n_max_angular,
        l_max_3b=model.l_max_3b, l_max_4b=model.l_max_4b, l_max_5b=model.l_max_5b,
        num_lm=model.num_lm, c3b_coeffs=model._c3b,
        c4b_coeffs=model._c4b, c5b_coeffs=model._c5b,
        dtype=dtype, device=dev,
        c2=model.c_param_2, c3=model.c_param_3,
    )

    print("\n[Descriptor forward only, no gradient]")
    with torch.no_grad():
        t_pt = time_fn(
            lambda: ops.compute_descriptors(**common_desc, pytorch_only=True),
            "compute_descriptors  pytorch_only=True")
        t_cu = time_fn(
            lambda: ops.compute_descriptors(**common_desc, pytorch_only=False),
            "compute_descriptors  pytorch_only=False (CUDA rad fwd)")
    if t_pt > 0:
        print(f"  Speedup (CUDA fwd / PyTorch): {t_pt / t_cu:.2f}x")

    # 2. compute_properties_cached: training forward+backward (always pure PyTorch)
    print("\n[Training step: forward + backward, compute_properties_cached]")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    def training_step():
        result = model.compute_properties_cached(
            batch, need_forces=True, need_virial=False)
        e_pa_pred = result["Etot"] / batch["natoms"]
        e_pa_ref = batch["energy"] / batch["natoms"]
        loss = torch.mean((e_pa_pred - e_pa_ref) ** 2)
        if "forces" in result:
            loss = loss + torch.mean((result["forces"] - batch["forces"]) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    t_train = time_fn(training_step, "compute_properties_cached (training step)")
    structs_per_sec = data_store.n / (t_train / 1000)
    print(f"  Throughput: {structs_per_sec:.0f} structures/s")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: CPU loss decreases")
    print("=" * 60)
    test_cpu_loss_decreases()

    print("\n" + "=" * 60)
    print("Test 2: GPU loss decreases")
    print("=" * 60)
    test_gpu_loss_decreases()

    print("\n" + "=" * 60)
    print("Test 3: CPU vs GPU consistency (pytorch_only=True)")
    print("=" * 60)
    test_cpu_gpu_consistency()

    print("\n" + "=" * 60)
    print("Test 4: No NaN/Inf in gradients")
    print("=" * 60)
    test_no_nan_in_gradients()

    print("\n" + "=" * 60)
    print("Test 5: CUDA descriptor forward values vs PyTorch")
    print("=" * 60)
    test_cuda_kernel_descriptors_forward()

    print("\n" + "=" * 60)
    print("Test 6: CUDA descriptor backward gradients")
    print("=" * 60)
    test_cuda_kernel_descriptors_backward()

    print("\n" + "=" * 60)
    print("Benchmark: pure-PyTorch vs CUDA-kernel throughput")
    print("=" * 60)
    benchmark_pytorch_vs_cuda()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
