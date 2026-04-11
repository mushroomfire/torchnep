"""
Test CUDA integration into training backward pass.

Compares:
1. Pure PyTorch (type-pair loop) vs CUDA kernel: descriptor forward values
2. Pure PyTorch vs CUDA kernel: c2/c3 gradients through training step
3. Full training: CUDA vs PyTorch produce same loss trajectory
4. Performance benchmark

Usage:
    python tests/test_cuda_integration.py
"""

import os
import sys
import time
import numpy as np
import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, ".."))

from torchnep import ops
from torchnep.data import parse_nep_in, read_xyz
from torchnep.model import NEPModel
from torchnep.train import GPUDataStore, compute_q_scaler, preprocess_structures

CONFIG_FILE = os.path.join(_DIR, "input_files", "nep.in")
DATA_FILE = os.path.join(_DIR, "input_files", "train_small.xyz")
SEED = 42


def _build_test_data(dev, dtype=torch.float32):
    """Build model and batch for testing."""
    config = parse_nep_in(CONFIG_FILE)
    frames = read_xyz(DATA_FILE)
    structures = preprocess_structures(frames, config, np.float32)
    data_store = GPUDataStore(structures, dev, dtype, config=config)
    batch = data_store.collate(list(range(min(8, data_store.n))))
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = NEPModel(config).to(dtype).to(dev)
    q_min, q_max = compute_q_scaler(model, data_store, 8, pytorch_only=True)
    model.set_q_scaler(q_min, q_max)
    return model, batch, data_store, config


# ==========================================================================
# Test 1: Forward values match (descriptors)
# ==========================================================================

def test_descriptor_forward():
    """CUDA cached descriptors must match PyTorch loop-based descriptors."""
    print("\n=== Test 1: Descriptor forward values ===")
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return

    dev = torch.device("cuda")
    model, batch, _, _ = _build_test_data(dev)

    # Reference: pure PyTorch with type-pair loops
    c2_ref = model.c_param_2.detach().clone()
    c3_ref = model.c_param_3.detach().clone() if model.c_param_3 is not None else None

    dtype, device = model.q_scaler.dtype, model.q_scaler.device
    N = batch["N"]
    ntypes = c2_ref.shape[0]

    # ---- Pure PyTorch (loop-based) ----
    t1r = batch["atom_types"][batch["pair_i_rad"]]
    t2r = batch["atom_types"][batch["pair_j_rad"]]
    q_rad_pt = torch.zeros(N, model.n_max_radial + 1, dtype=dtype, device=device)
    for _t1 in range(ntypes):
        for _t2 in range(ntypes):
            _m = (t1r == _t1) & (t2r == _t2)
            if _m.any():
                _gn = batch["fk_rad"][_m] @ c2_ref[_t1, _t2].T
                q_rad_pt.scatter_add_(0,
                    batch["pair_i_rad"][_m].unsqueeze(-1).expand_as(_gn), _gn)

    # ---- CUDA (scatter_contraction) ----
    from torchnep.cuda_ops import scatter_contraction
    q_rad_cu = scatter_contraction(
        batch["fk_rad"], batch["pair_i_rad"], batch["pair_j_rad"],
        batch["atom_types"], c2_ref, N)

    diff = (q_rad_pt - q_rad_cu).abs()
    max_diff = diff.max().item()
    print(f"  Radial q: max_abs_diff = {max_diff:.2e}")
    assert max_diff < 1e-5, f"Radial descriptor mismatch: {max_diff:.2e}"

    # Angular gn contraction
    if c3_ref is not None:
        from torchnep.cuda_ops import type_contraction
        t1a = batch["atom_types"][batch["pair_i_ang"]]
        t2a = batch["atom_types"][batch["pair_j_ang"]]
        n_ap1 = model.n_max_angular + 1

        gn_pt = torch.zeros(batch["fk_ang"].shape[0], n_ap1, dtype=dtype, device=device)
        for _t1 in range(ntypes):
            for _t2 in range(ntypes):
                _m = (t1a == _t1) & (t2a == _t2)
                if _m.any():
                    gn_pt[_m] = batch["fk_ang"][_m] @ c3_ref[_t1, _t2].T

        gn_cu = type_contraction(
            batch["fk_ang"], batch["pair_i_ang"], batch["pair_j_ang"],
            batch["atom_types"], c3_ref)

        diff = (gn_pt - gn_cu).abs()
        max_diff = diff.max().item()
        print(f"  Angular gn: max_abs_diff = {max_diff:.2e}")
        assert max_diff < 1e-5, f"Angular gn mismatch: {max_diff:.2e}"

    print("  PASS: CUDA forward values match PyTorch")


# ==========================================================================
# Test 2: Backward gradients match (c2, c3)
# ==========================================================================

def test_descriptor_backward():
    """CUDA backward for c2/c3 must match PyTorch autograd."""
    print("\n=== Test 2: Descriptor backward gradients ===")
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return

    dev = torch.device("cuda")
    model, batch, _, _ = _build_test_data(dev)

    dtype, device = model.q_scaler.dtype, model.q_scaler.device
    N = batch["N"]
    ntypes = model.c_param_2.shape[0]

    # ---- PyTorch reference (loop-based) ----
    c2_pt = model.c_param_2.detach().clone().requires_grad_(True)
    c3_pt = model.c_param_3.detach().clone().requires_grad_(True) \
        if model.c_param_3 is not None else None

    t1r = batch["atom_types"][batch["pair_i_rad"]]
    t2r = batch["atom_types"][batch["pair_j_rad"]]
    q_pt = torch.zeros(N, model.n_max_radial + 1, dtype=dtype, device=device)
    for _t1 in range(ntypes):
        for _t2 in range(ntypes):
            _m = (t1r == _t1) & (t2r == _t2)
            if _m.any():
                _gn = batch["fk_rad"][_m] @ c2_pt[_t1, _t2].T
                q_pt.scatter_add_(0,
                    batch["pair_i_rad"][_m].unsqueeze(-1).expand_as(_gn), _gn)
    q_pt.sum().backward()

    # ---- CUDA ----
    c2_cu = model.c_param_2.detach().clone().requires_grad_(True)
    from torchnep.cuda_ops import scatter_contraction
    q_cu = scatter_contraction(
        batch["fk_rad"], batch["pair_i_rad"], batch["pair_j_rad"],
        batch["atom_types"], c2_cu, N)
    q_cu.sum().backward()

    abs_diff = (c2_pt.grad - c2_cu.grad).abs().max().item()
    rel_diff = abs_diff / (c2_pt.grad.abs().max().item() + 1e-12)
    print(f"  grad_c2: max_abs_diff = {abs_diff:.2e}, max_rel_diff = {rel_diff:.2e}")
    assert rel_diff < 1e-4, f"c2 gradient mismatch: rel={rel_diff:.2e}, abs={abs_diff:.2e}"

    # Angular c3 gradient
    if c3_pt is not None:
        from torchnep.cuda_ops import type_contraction
        n_ap1 = model.n_max_angular + 1

        gn_pt = torch.zeros(batch["fk_ang"].shape[0], n_ap1, dtype=dtype, device=device)
        for _t1 in range(ntypes):
            for _t2 in range(ntypes):
                _m = (batch["atom_types"][batch["pair_i_ang"]] == _t1) & \
                     (batch["atom_types"][batch["pair_j_ang"]] == _t2)
                if _m.any():
                    gn_pt[_m] = batch["fk_ang"][_m] @ c3_pt[_t1, _t2].T
        gn_pt.sum().backward()

        c3_cu = model.c_param_3.detach().clone().requires_grad_(True)
        gn_cu = type_contraction(
            batch["fk_ang"], batch["pair_i_ang"], batch["pair_j_ang"],
            batch["atom_types"], c3_cu)
        gn_cu.sum().backward()

        abs_diff = (c3_pt.grad - c3_cu.grad).abs().max().item()
        rel_diff = abs_diff / (c3_pt.grad.abs().max().item() + 1e-12)
        print(f"  grad_c3: max_abs_diff = {abs_diff:.2e}, max_rel_diff = {rel_diff:.2e}")
        assert rel_diff < 1e-4, f"c3 gradient mismatch: rel={rel_diff:.2e}, abs={abs_diff:.2e}"

    print("  PASS: CUDA backward gradients match PyTorch")


# ==========================================================================
# Test 3: Full training step consistency
# ==========================================================================

def test_training_step_consistency():
    """Full training step (forward + backward + optimizer) must match."""
    print("\n=== Test 3: Full training step consistency ===")
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return

    dev = torch.device("cuda")
    dtype = torch.float32
    config = parse_nep_in(CONFIG_FILE)
    frames = read_xyz(DATA_FILE)
    structures = preprocess_structures(frames, config, np.float32)

    lambda_e = config.get("lambda_e", 1.0)
    lambda_f = config.get("lambda_f", 1.0)

    def run_steps(use_cuda, n_steps=3):
        """Run n training steps and return losses + final params."""
        os.environ["TORCHNEP_NO_CUDA_KERNELS"] = "0" if use_cuda else "1"
        # Force reload of cached kernels
        import torchnep.cuda_ops as co
        co._cached_kernels = None

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        data_store = GPUDataStore(structures, dev, dtype, config=config)
        model = NEPModel(config).to(dtype).to(dev)
        q_min, q_max = compute_q_scaler(model, data_store, 8, pytorch_only=True)
        model.set_q_scaler(q_min, q_max)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        losses = []
        for step in range(n_steps):
            batch = data_store.collate(list(range(min(8, data_store.n))))
            result = model.compute_properties_cached(
                batch, need_forces=True, need_virial=False)
            e_pred = result["Etot"] / batch["natoms"]
            e_ref = batch["energy"] / batch["natoms"]
            loss = lambda_e * torch.mean((e_pred - e_ref) ** 2)
            if "forces" in result and data_store.has_forces:
                loss = loss + lambda_f * torch.mean(
                    (result["forces"] - batch["forces"]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        params = {n: p.detach().cpu().numpy()
                  for n, p in model.named_parameters()}
        grads = {n: p.grad.detach().cpu().numpy()
                 for n, p in model.named_parameters() if p.grad is not None}
        return losses, params, grads

    print("  Running PyTorch-only training steps...")
    losses_pt, params_pt, grads_pt = run_steps(use_cuda=False)

    print("  Running CUDA-accelerated training steps...")
    losses_cu, params_cu, grads_cu = run_steps(use_cuda=True)

    # Reset env
    os.environ.pop("TORCHNEP_NO_CUDA_KERNELS", None)

    print(f"\n  {'Step':>5}  {'PyTorch loss':>14}  {'CUDA loss':>14}  {'rel diff':>10}")
    for i, (lp, lc) in enumerate(zip(losses_pt, losses_cu)):
        rd = abs(lp - lc) / (abs(lp) + 1e-12)
        print(f"  {i+1:5d}  {lp:14.8e}  {lc:14.8e}  {rd:10.2e}")

    LOSS_RTOL = 1e-3  # atomicAdd float32 noise accumulates over steps
    for i, (lp, lc) in enumerate(zip(losses_pt, losses_cu)):
        rd = abs(lp - lc) / (abs(lp) + 1e-12)
        assert rd < LOSS_RTOL, f"Step {i+1} loss mismatch: rel_diff={rd:.2e}"

    print("\n  Final parameter differences:")
    for name in params_pt:
        diff = np.max(np.abs(params_pt[name] - params_cu[name]))
        scale = np.max(np.abs(params_pt[name])) + 1e-12
        rel = diff / scale
        print(f"    {name:40s}  max_diff={diff:.2e}  rel={rel:.2e}")
        assert rel < 1e-3, f"Parameter '{name}' mismatch: rel={rel:.2e}"

    print("\n  Gradient differences (last step):")
    for name in grads_pt:
        if name in grads_cu:
            diff = np.max(np.abs(grads_pt[name] - grads_cu[name]))
            scale = np.max(np.abs(grads_pt[name])) + 1e-12
            rel = diff / scale
            print(f"    {name:40s}  max_diff={diff:.2e}  rel={rel:.2e}")
            assert rel < 1e-3, f"Gradient '{name}' mismatch: rel={rel:.2e}"

    print("\n  PASS: CUDA training steps match PyTorch (within float32 tolerance)")


# ==========================================================================
# Benchmark
# ==========================================================================

def benchmark():
    """Compare performance: PyTorch loop vs CUDA kernel."""
    print("\n=== Benchmark: PyTorch loop vs CUDA kernel ===")
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return

    dev = torch.device("cuda")
    model, batch, data_store, config = _build_test_data(dev)
    model.train()

    lambda_e = config.get("lambda_e", 1.0)
    lambda_f = config.get("lambda_f", 1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    n_warmup = 5
    n_repeat = 20

    def training_step():
        result = model.compute_properties_cached(
            batch, need_forces=True, need_virial=False)
        e_pred = result["Etot"] / batch["natoms"]
        e_ref = batch["energy"] / batch["natoms"]
        loss = lambda_e * torch.mean((e_pred - e_ref) ** 2)
        if "forces" in result and data_store.has_forces:
            loss = loss + lambda_f * torch.mean(
                (result["forces"] - batch["forces"]) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    import torchnep.cuda_ops as co

    print(f"\n  Benchmark: {batch['N']} atoms, "
          f"{batch['rij_rad'].shape[0]} rad pairs, "
          f"{batch['rij_ang'].shape[0]} ang pairs, "
          f"{n_repeat} repeats")

    # --- PyTorch only ---
    os.environ["TORCHNEP_NO_CUDA_KERNELS"] = "1"
    co._cached_kernels = None
    for _ in range(n_warmup):
        training_step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        training_step()
    torch.cuda.synchronize()
    t_pt = (time.perf_counter() - t0) / n_repeat * 1000

    # --- CUDA kernels ---
    os.environ["TORCHNEP_NO_CUDA_KERNELS"] = "0"
    co._cached_kernels = None
    for _ in range(n_warmup):
        training_step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        training_step()
    torch.cuda.synchronize()
    t_cu = (time.perf_counter() - t0) / n_repeat * 1000

    os.environ.pop("TORCHNEP_NO_CUDA_KERNELS", None)

    print(f"\n  PyTorch loop:  {t_pt:8.2f} ms/step")
    print(f"  CUDA kernel:   {t_cu:8.2f} ms/step")
    print(f"  Speedup:       {t_pt/t_cu:.2f}x")
    print(f"  Throughput:    PyTorch {data_store.n/(t_pt/1000):.0f} "
          f"vs CUDA {data_store.n/(t_cu/1000):.0f} structures/s")


if __name__ == "__main__":
    test_descriptor_forward()
    test_descriptor_backward()
    test_training_step_consistency()
    benchmark()
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
