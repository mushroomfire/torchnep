"""Correctness tests for extended angular basis (L up to 8).

Checks three things:
  1. New data-driven ``angular_basis`` is bit-identical to the old hand-coded
     formula for L = 1..4 (regression).
  2. ``_compute_dblm_dhat`` matches PyTorch autograd for L = 1..8.
  3. ``_compute_dblm_dhat`` matches numerical finite differences for L = 1..8
     (independent cross-check, catches any drift in autograd trust).
"""

import torch

from torchnep.ops import angular_basis, _compute_dblm_dhat


def _hand_coded_lmax4(x, y, z):
    """Old implementation, for regression check (L=1..4 only)."""
    x2, y2, z2 = x * x, y * y, z * z
    x2my2 = x2 - y2
    blm = []
    blm.extend([z, x, y])
    blm.extend([3.0*z2 - 1.0, x*z, y*z, x2my2, 2.0*x*y])
    blm.extend([
        (5.0*z2 - 3.0)*z, (5.0*z2 - 1.0)*x, (5.0*z2 - 1.0)*y,
        x2my2*z, 2.0*x*y*z, x*(x2 - 3.0*y2), y*(3.0*x2 - y2),
    ])
    blm.extend([
        (35.0*z2 - 30.0)*z2 + 3.0, (7.0*z2 - 3.0)*x*z,
        (7.0*z2 - 3.0)*y*z, (7.0*z2 - 1.0)*x2my2,
        (7.0*z2 - 1.0)*2.0*x*y, z*x*(x2 - 3.0*y2),
        z*y*(3.0*x2 - y2), x2my2*x2my2 - 4.0*x2*y2,
        4.0*x*y*x2my2,
    ])
    return torch.stack(blm, dim=-1)


def test_regression_lmax4():
    """New data-driven angular_basis matches old hand-coded formula for L ≤ 4."""
    torch.manual_seed(0)
    x = torch.randn(50, dtype=torch.float64)
    y = torch.randn(50, dtype=torch.float64)
    z = torch.randn(50, dtype=torch.float64)
    for L in [1, 2, 3, 4]:
        new = angular_basis(x, y, z, L)
        ref = _hand_coded_lmax4(x, y, z)
        # ref is always full L=4; slice to matching num_lm for lower L
        num_lm_L = sum(2 * ll + 1 for ll in range(1, L + 1))
        diff = (new - ref[:, :num_lm_L]).abs().max().item()
        # Different summation order → FP roundoff at the last few bits; tolerate.
        assert diff < 1e-12, f"L={L}: new != hand-coded, max diff = {diff:.3e}"


def test_dblm_matches_autograd():
    """_compute_dblm_dhat matches PyTorch autograd on angular_basis, for L = 1..8."""
    for L_max in range(1, 9):
        torch.manual_seed(100 + L_max)
        n = 30
        x = torch.randn(n, dtype=torch.float64, requires_grad=True)
        y = torch.randn(n, dtype=torch.float64, requires_grad=True)
        z = torch.randn(n, dtype=torch.float64, requires_grad=True)

        blm = angular_basis(x, y, z, L_max)
        num_lm = blm.shape[1]
        ref = torch.zeros(n, num_lm, 3, dtype=torch.float64)
        for lm in range(num_lm):
            gx, gy, gz = torch.autograd.grad(
                blm[:, lm].sum(), [x, y, z], retain_graph=True)
            ref[:, lm, 0] = gx
            ref[:, lm, 1] = gy
            ref[:, lm, 2] = gz

        with torch.no_grad():
            got = _compute_dblm_dhat(x.detach(), y.detach(), z.detach(), L_max)

        diff = (got - ref).abs().max().item()
        rel = diff / max(ref.abs().max().item(), 1e-12)
        assert diff < 1e-10 or rel < 1e-10, (
            f"L_max={L_max}: dblm_dhat mismatch. max abs diff = {diff:.3e}, "
            f"rel = {rel:.3e}")


def test_dblm_matches_finite_diff():
    """Sanity: numerical FD check (independent of autograd) for L = 1..8."""
    eps = 1e-6
    for L_max in range(1, 9):
        torch.manual_seed(200 + L_max)
        n = 8
        x = torch.randn(n, dtype=torch.float64) * 0.8
        y = torch.randn(n, dtype=torch.float64) * 0.8
        z = torch.randn(n, dtype=torch.float64) * 0.8

        analytical = _compute_dblm_dhat(x, y, z, L_max)  # (n, num_lm, 3)

        # FD for d/dx
        fd = torch.zeros_like(analytical)
        for axis, var in enumerate((x, y, z)):
            vp = var + eps
            vm = var - eps
            args_p = [vp if i == axis else v for i, v in enumerate((x, y, z))]
            args_m = [vm if i == axis else v for i, v in enumerate((x, y, z))]
            blm_p = angular_basis(*args_p, L_max)
            blm_m = angular_basis(*args_m, L_max)
            fd[:, :, axis] = (blm_p - blm_m) / (2 * eps)

        diff = (analytical - fd).abs().max().item()
        rel = diff / max(fd.abs().max().item(), 1e-12)
        # FD is noisier (~eps² + rounding); allow 1e-6
        assert diff < 1e-6 or rel < 1e-6, (
            f"L_max={L_max}: FD disagrees with analytical dblm. "
            f"max abs diff = {diff:.3e}, rel = {rel:.3e}")


def test_num_lm_shape():
    """Output width follows num_lm = Σ(2L+1) for L=1..L_max."""
    for L_max in range(1, 9):
        x = torch.randn(5, dtype=torch.float64)
        y = torch.randn(5, dtype=torch.float64)
        z = torch.randn(5, dtype=torch.float64)
        expected = sum(2 * L + 1 for L in range(1, L_max + 1))
        assert angular_basis(x, y, z, L_max).shape == (5, expected)
        assert _compute_dblm_dhat(x, y, z, L_max).shape == (5, expected, 3)


def test_rejects_oob():
    """l_max_3b > 8 must raise."""
    x = torch.randn(3, dtype=torch.float64)
    try:
        angular_basis(x, x, x, 9)
    except ValueError:
        pass
    else:
        raise AssertionError("angular_basis(L=9) should have raised")


if __name__ == "__main__":
    test_regression_lmax4()
    print("regression L=1..4 OK")
    test_num_lm_shape()
    print("num_lm shapes OK")
    test_dblm_matches_autograd()
    print("dblm_dhat == autograd, L=1..8 OK")
    test_dblm_matches_finite_diff()
    print("dblm_dhat == finite-diff, L=1..8 OK")
    test_rejects_oob()
    print("oob rejection OK")
    print("ALL PASS")
