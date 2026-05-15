# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Yongchao Wu.
# Part of the torchnep project — https://github.com/mushroomfire/torchnep.

"""Direct gradient checks for the mixed-body descriptor blocks.

These tests target ``ops._angular_weight`` — the closed-form dEi/d(sum_fxyz)
expression that the analytical force path relies on. We compare its output
against PyTorch's autograd on the q-vs-s polynomial that
``compute_descriptors_cached`` builds.

Three checks:

  1. ``test_q112_polynomial_gradient``  : full chain rule against autograd.
  2. ``test_q1122_polynomial_gradient`` : full chain rule against autograd.
  3. ``test_full_angular_weight``        : ``_angular_weight`` matches
     ``torch.autograd.grad`` for a random Fp / s combo with all flags on.

The first two pin down the specific polynomial coefficients I hand-derived
when wiring the new descriptors. The third confirms the full multi-block
combination (3-body + q_222 + q_1111 + q_112 + q_1122) sums correctly.
"""
import sys
from pathlib import Path

import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from torchnep.constants import C3B, C4B, C5B, C4B2, C5B2
from torchnep.ops import _angular_weight


# Gradient checks run in float64 — autograd round-off should be < 1e-12.
DTYPE = torch.float64
TOL = 1e-12


def _q_112_from_s(s_lm, c4b2):
    """Per-atom q_112 from the L=1, L=2 entries of s. Mirrors the body of
    ``compute_descriptors_cached``. Returns shape (n_ap1,).
    """
    s10, s11r, s11i = s_lm[:, 0], s_lm[:, 1], s_lm[:, 2]
    s20, s21r, s21i = s_lm[:, 3], s_lm[:, 4], s_lm[:, 5]
    s22r, s22i = s_lm[:, 6], s_lm[:, 7]
    cb = c4b2
    return (cb[0]*s10*s10*s20
            + cb[1]*s10*(s11r*s21r + s11i*s21i)
            + cb[2]*s20*(s11r*s11r + s11i*s11i)
            + cb[3]*s22r*(s11r*s11r - s11i*s11i)
            + cb[4]*s11r*s11i*s22i)


def _q_1122_from_s(s_lm, c5b2):
    s10, s11r, s11i = s_lm[:, 0], s_lm[:, 1], s_lm[:, 2]
    s20, s21r, s21i = s_lm[:, 3], s_lm[:, 4], s_lm[:, 5]
    s22r, s22i = s_lm[:, 6], s_lm[:, 7]
    cb = c5b2
    a2 = s10*s10; b2 = s11r*s11r; c2_ = s11i*s11i
    d2 = s20*s20; e2 = s21r*s21r; f2 = s21i*s21i
    g2 = s22r*s22r; h2 = s22i*s22i
    return (cb[0]*a2*d2
            + cb[1]*(a2*e2 + a2*f2 + b2*e2 + c2_*f2)
            + cb[2]*(b2*g2 + b2*h2 + c2_*g2 + c2_*h2)
            + cb[3]*(a2*g2 + a2*h2)
            + cb[4]*(b2*f2 + c2_*e2)
            + cb[5]*(b2*d2 + c2_*d2)
            + cb[6]*(c2_*s20*s22r - b2*s20*s22r)
            + cb[7]*(s10*s11r*s20*s21r
                     + s10*s11i*s20*s21i
                     - s11r*s11i*s20*s22i)
            + cb[8]*(s10*s11r*s21r*s22r
                     + s10*s11r*s21i*s22i
                     + s10*s11i*s21r*s22i
                     - s10*s11i*s21i*s22r)
            + cb[9]*(s11r*s11i*s21r*s21i))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_q112_polynomial_gradient(seed):
    """Hand-derived q_112 gradient matches torch.autograd. Tests every
    (Fp, s_lm) entry — only the 8 relevant lm indices get non-zero weight.
    """
    torch.manual_seed(seed)
    n_ap1 = 5            # n_max_angular = 4 -> 5 channels
    num_lm = 24          # L=1..4 -> 3+5+7+9 = 24
    s = torch.randn(1, n_ap1, num_lm, dtype=DTYPE, requires_grad=True)
    Fp_block = torch.randn(1, n_ap1, dtype=DTYPE)
    c4b2 = torch.tensor(C4B2, dtype=DTYPE)

    q = _q_112_from_s(s[0], c4b2)               # (n_ap1,)
    energy = (Fp_block[0] * q).sum()
    grad_auto, = torch.autograd.grad(energy, s)

    # Now compute the same gradient via _angular_weight.
    dim_r = 5  # arbitrary radial size — doesn't enter q_112 derivative
    dim = dim_r + 4 * n_ap1 + n_ap1   # 3b uses 4 L's; only q_112 block follows
    Fp_full = torch.zeros(1, dim, dtype=DTYPE)
    off = dim_r + 4 * n_ap1   # offset of q_112 block (no q_222 / q_1111)
    Fp_full[:, off:off + n_ap1] = Fp_block

    w = _angular_weight(
        Fp_full, s.detach(), dim_r, n_ap1, l_max_3b=4,
        has_q_222=0, has_q_1111=0, has_q_112=1, has_q_1122=0,
        c3b_coeffs=torch.tensor(C3B[:num_lm], dtype=DTYPE),
        c4b_coeffs=torch.tensor(C4B, dtype=DTYPE),
        c5b_coeffs=torch.tensor(C5B, dtype=DTYPE),
        c4b2_coeffs=c4b2,
        c5b2_coeffs=torch.tensor(C5B2, dtype=DTYPE))

    # q is built from s[n,lm] but only lm=0..7 contribute. dE/ds = Fp * dq/ds.
    # _angular_weight currently emits 2 * (Fp * dq/ds) in the 3-body slot
    # (factor of 2 reflects the s**2*c sum / dq_3b convention), and 1 * elsewhere.
    # Our test only enables q_112, so the 3-body contribution is zero and the
    # output equals dE/ds exactly.
    assert torch.allclose(w[:, :, :8], grad_auto[:, :, :8], atol=TOL)
    # Indices 8..23 should remain zero (only 3-body would touch them; here off)
    assert torch.allclose(w[:, :, 8:], torch.zeros_like(w[:, :, 8:]), atol=TOL)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_q1122_polynomial_gradient(seed):
    """Same test for q_1122 — verifies the 10-coefficient combination."""
    torch.manual_seed(seed)
    n_ap1 = 5
    num_lm = 24
    s = torch.randn(1, n_ap1, num_lm, dtype=DTYPE, requires_grad=True)
    Fp_block = torch.randn(1, n_ap1, dtype=DTYPE)
    c5b2 = torch.tensor(C5B2, dtype=DTYPE)

    q = _q_1122_from_s(s[0], c5b2)
    energy = (Fp_block[0] * q).sum()
    grad_auto, = torch.autograd.grad(energy, s)

    dim_r = 5
    dim = dim_r + 4 * n_ap1 + n_ap1
    Fp_full = torch.zeros(1, dim, dtype=DTYPE)
    off = dim_r + 4 * n_ap1
    Fp_full[:, off:off + n_ap1] = Fp_block

    w = _angular_weight(
        Fp_full, s.detach(), dim_r, n_ap1, l_max_3b=4,
        has_q_222=0, has_q_1111=0, has_q_112=0, has_q_1122=1,
        c3b_coeffs=torch.tensor(C3B[:num_lm], dtype=DTYPE),
        c4b_coeffs=torch.tensor(C4B, dtype=DTYPE),
        c5b_coeffs=torch.tensor(C5B, dtype=DTYPE),
        c4b2_coeffs=torch.tensor(C4B2, dtype=DTYPE),
        c5b2_coeffs=c5b2)

    assert torch.allclose(w[:, :, :8], grad_auto[:, :, :8], atol=TOL)
    assert torch.allclose(w[:, :, 8:], torch.zeros_like(w[:, :, 8:]), atol=TOL)


@pytest.mark.parametrize("flag_tuple", [
    (1, 0, 0, 0),     # q_222 only (baseline path)
    (0, 0, 1, 0),     # q_112 in isolation
    (0, 0, 0, 1),     # q_1122 in isolation
    (1, 1, 1, 1),     # everything on
])
def test_full_angular_weight(flag_tuple):
    """Full multi-block ``_angular_weight`` matches autograd on the explicit
    polynomial expression that ``compute_descriptors_cached`` builds.

    Catches off-by-one offset bugs (each enabled flag adds one n_ap1 slot)
    and ensures the gradient contributions of different blocks accumulate
    correctly.
    """
    torch.manual_seed(7 * sum(flag_tuple) + 1)
    has_222, has_1111, has_112, has_1122 = flag_tuple
    n_ap1 = 5
    l_max_3b = 4
    num_lm = sum(2 * L + 1 for L in range(1, l_max_3b + 1))  # 24

    s = torch.randn(2, n_ap1, num_lm, dtype=DTYPE, requires_grad=True)

    dim_r = 7  # arbitrary
    dim = (dim_r + l_max_3b * n_ap1
           + (has_222 + has_1111 + has_112 + has_1122) * n_ap1)
    Fp = torch.randn(2, dim, dtype=DTYPE)

    c3b  = torch.tensor(C3B[:num_lm], dtype=DTYPE)
    c4b  = torch.tensor(C4B,  dtype=DTYPE)
    c5b  = torch.tensor(C5B,  dtype=DTYPE)
    c4b2 = torch.tensor(C4B2, dtype=DTYPE)
    c5b2 = torch.tensor(C5B2, dtype=DTYPE)

    # Build the full energy as exec'd by compute_descriptors_cached, then
    # autograd through s.
    parts = []
    for li in range(l_max_3b):
        L = li + 1
        nt = 2 * L + 1
        st = L * L - 1
        c = c3b[st:st + nt]
        sb2 = s[:, :, st:st + nt] ** 2
        ql = c[0] * sb2[:, :, 0]
        if nt > 1:
            ql = ql + 2.0 * (c[1:] * sb2[:, :, 1:]).sum(-1)
        parts.append(ql)
    q_3b = torch.stack(parts, dim=-1).transpose(1, 2).reshape(s.shape[0], -1)
    q_list = [q_3b]

    s10, s11r, s11i = s[:, :, 0], s[:, :, 1], s[:, :, 2]
    s20, s21r, s21i = s[:, :, 3], s[:, :, 4], s[:, :, 5]
    s22r, s22i = s[:, :, 6], s[:, :, 7]

    if has_222:
        q4 = (c4b[0]*s20**3
              + c4b[1]*s20*(s21r**2 + s21i**2)
              + c4b[2]*s20*(s22r**2 + s22i**2)
              + c4b[3]*s22r*(s21i**2 - s21r**2)
              + c4b[4]*s21r*s21i*s22i)
        q_list.append(q4)
    if has_1111:
        s0sq = s10**2
        s1sq = s11r**2 + s11i**2
        q5 = c5b[0]*s0sq**2 + c5b[1]*s0sq*s1sq + c5b[2]*s1sq**2
        q_list.append(q5)
    if has_112:
        q112 = (c4b2[0]*s10*s10*s20
                + c4b2[1]*s10*(s11r*s21r + s11i*s21i)
                + c4b2[2]*s20*(s11r*s11r + s11i*s11i)
                + c4b2[3]*s22r*(s11r*s11r - s11i*s11i)
                + c4b2[4]*s11r*s11i*s22i)
        q_list.append(q112)
    if has_1122:
        a2 = s10**2; b2 = s11r**2; c2_ = s11i**2
        d2 = s20**2; e2 = s21r**2; f2 = s21i**2
        g2 = s22r**2; h2 = s22i**2
        q1122 = (c5b2[0]*a2*d2
                 + c5b2[1]*(a2*e2 + a2*f2 + b2*e2 + c2_*f2)
                 + c5b2[2]*(b2*g2 + b2*h2 + c2_*g2 + c2_*h2)
                 + c5b2[3]*(a2*g2 + a2*h2)
                 + c5b2[4]*(b2*f2 + c2_*e2)
                 + c5b2[5]*(b2*d2 + c2_*d2)
                 + c5b2[6]*(c2_*s20*s22r - b2*s20*s22r)
                 + c5b2[7]*(s10*s11r*s20*s21r + s10*s11i*s20*s21i
                            - s11r*s11i*s20*s22i)
                 + c5b2[8]*(s10*s11r*s21r*s22r + s10*s11r*s21i*s22i
                            + s10*s11i*s21r*s22i - s10*s11i*s21i*s22r)
                 + c5b2[9]*(s11r*s11i*s21r*s21i))
        q_list.append(q1122)

    q = torch.cat(q_list, dim=-1)                                # (N, dim)
    # Pad with zero columns to match Fp's full ``dim`` (the radial block
    # doesn't depend on s, so dE/ds has no radial contribution).
    pad = torch.zeros(s.shape[0], dim_r, dtype=DTYPE)
    q_full = torch.cat([pad, q], dim=-1)
    energy = (Fp * q_full).sum()
    grad_auto, = torch.autograd.grad(energy, s)

    w = _angular_weight(
        Fp, s.detach(), dim_r, n_ap1, l_max_3b,
        has_222, has_1111, has_112, has_1122,
        c3b, c4b, c5b, c4b2, c5b2)

    assert torch.allclose(w, grad_auto, atol=TOL), (
        f"flags={flag_tuple}  "
        f"max diff = {(w - grad_auto).abs().max().item():.3e}")
