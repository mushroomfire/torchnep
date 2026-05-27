# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Yongchao Wu.
# Part of the torchnep project — https://github.com/mushroomfire/torchnep.

"""Tests for the q_123 / q_233 higher-L 4-body bispectrum channels.

These are GPUMD PR #1517's ``has_q_123`` / ``has_q_233`` (l_max flags 6 / 7).
torchnep uses GPUMD's exact coefficients, so the descriptor values are
bit-identical and a trained nep.txt is interoperable with GPUMD.

Checks:
  1. test_rotational_invariance     — each channel is unchanged under a global
     rotation of the neighbours (the defining property of an angular invariant).
  2. test_matches_gpumd_polynomial  — torchnep's term tables reproduce GPUMD's
     find_q polynomial bit-for-bit.
  3. test_analytical_force_vs_autograd — a model with both channels on produces
     analytical forces/virial that match autograd-on-rij.
  4. test_nep_txt_round_trip        — save_nep_txt writes the 7-field l_max line
     and NEPCalculator reloads to the same descriptor dim / finite output.
  5. test_lmax_guard / test_off_by_default — l_max>=3 requirement and that the
     channels are off unless requested.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torchnep.ops as ops
from torchnep.constants import (Q123_TERMS, Q233_TERMS, C4B_123, C4B_233)
from torchnep.data import build_neighbor_list_np
from torchnep.model import NEPModel
from torchnep.nep import NEPCalculator

DTYPE = torch.float64
CHANNELS = {"q_123": Q123_TERMS, "q_233": Q233_TERMS}


def _rand_rotation(rng):
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _moments(dirs, weights, l_max=3):
    x, y, z = (torch.tensor(dirs[:, i]) for i in range(3))
    blm = ops.angular_basis(x, y, z, l_max).numpy()
    return weights @ blm


def _eval(s, terms):
    return float(ops._eval_extra(torch.tensor(s).view(1, 1, -1), terms)[0, 0])


# --- GPUMD PR #1517 reference polynomials (verbatim from find_q) -------------

def _gpumd_q123(s):
    C = C4B_123
    return (C[6]*(s[12]*s[2]*s[4] - s[11]*s[2]*s[5] + s[1]*s[11]*s[4] + s[1]*s[12]*s[5])
            + C[5]*(s[0]*s[11]*s[6] + s[0]*s[12]*s[7])
            + C[3]*(s[14]*s[2]*s[6] - s[13]*s[2]*s[7] + s[1]*s[13]*s[6] + s[1]*s[14]*s[7])
            + C[4]*(s[10]*s[0]*s[5] + s[0]*s[4]*s[9])
            + C[1]*(s[10]*s[2]*s[3] + s[0]*s[3]*s[8] + s[1]*s[3]*s[9])
            + C[0]*(s[10]*s[2]*s[6] - s[10]*s[1]*s[7] - s[2]*s[7]*s[9] - s[1]*s[6]*s[9])
            + C[2]*(-s[2]*s[5]*s[8] - s[1]*s[4]*s[8]))


def _gpumd_q233(s):
    C = C4B_233
    return (C[0]*(s[3]*s[8]*s[8])
            + C[1]*(s[10]*s[10]*s[3] + s[3]*s[9]*s[9])
            + C[2]*(-s[10]*s[10]*s[6] + s[6]*s[9]*s[9])
            + C[3]*(s[4]*s[8]*s[9] + s[10]*s[5]*s[8])
            + C[4]*(-s[13]*s[13]*s[3] - s[14]*s[14]*s[3])
            + C[5]*(-s[14]*s[7]*s[9] - s[13]*s[6]*s[9] - s[10]*s[14]*s[6] + s[10]*s[13]*s[7])
            + C[6]*(s[10]*s[7]*s[9])
            + C[7]*(-s[11]*s[6]*s[8] - s[12]*s[7]*s[8])
            + C[8]*(s[11]*s[4]*s[9] + s[12]*s[5]*s[9] + s[10]*s[12]*s[4] - s[10]*s[11]*s[5])
            + C[9]*(s[12]*s[14]*s[4] + s[11]*s[14]*s[5] + s[13]*s[11]*s[4] - s[13]*s[12]*s[5]))


@pytest.mark.parametrize("name", list(CHANNELS))
def test_rotational_invariance(name):
    terms = CHANNELS[name]
    rng = np.random.default_rng(2026)
    worst = 0.0
    for _ in range(150):
        dirs = rng.standard_normal((10, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        w = rng.standard_normal(10)
        R = _rand_rotation(rng)
        q0 = _eval(_moments(dirs, w), terms)
        q1 = _eval(_moments(dirs @ R.T, w), terms)
        worst = max(worst, abs(q1 - q0) / (abs(q0) + 1e-12))
    assert worst < 1e-10, f"{name}: rotational variance {worst:.2e}"


@pytest.mark.parametrize("name,ref", [("q_123", _gpumd_q123), ("q_233", _gpumd_q233)])
def test_matches_gpumd_polynomial(name, ref):
    """torchnep term tables are bit-identical to GPUMD's find_q polynomial."""
    terms = CHANNELS[name]
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(2000):
        s = rng.standard_normal(15)
        worst = max(worst, abs(_eval(s, terms) - ref(s)))
    assert worst < 1e-12, f"{name}: max |torchnep - GPUMD| = {worst:.2e}"


@pytest.mark.parametrize("name", list(CHANNELS))
def test_extra_grad_matches_autograd(name):
    terms = CHANNELS[name]
    torch.manual_seed(7)
    s = torch.randn(3, 4, 15, dtype=DTYPE, requires_grad=True)
    q = ops._eval_extra(s, terms)
    grad_auto, = torch.autograd.grad(q.sum(), s)
    grad_ana = ops._extra_grad(s.detach(), terms)
    assert torch.allclose(grad_ana, grad_auto, atol=1e-12)


def _build_model(lmax, seed=0):
    torch.manual_seed(seed)
    cfg = {"num_types": 2, "type_names": ["Si", "O"],
           "cutoff_radial": 6.0, "cutoff_angular": 5.0,
           "n_max_radial": 3, "n_max_angular": 3,
           "basis_size_radial": 6, "basis_size_angular": 6,
           "l_max": lmax, "neuron": 16}
    m = NEPModel(cfg).to(DTYPE)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.1)
        m.q_scaler.uniform_(0.5, 1.5)
    return m


def _random_batch(N=40, seed=0):
    rng = np.random.default_rng(seed)
    cell = np.eye(3) * 9.0
    pos = rng.random((N, 3)) * 9.0
    species = rng.integers(0, 2, N)
    pi, pj, rij = build_neighbor_list_np(pos, cell, 6.0)
    dij = np.linalg.norm(rij, axis=1)
    at = torch.tensor(species, dtype=torch.long)
    rc_r, rc_a, br, ba = 6.0, 5.0, 6, 6
    rm, am = dij < rc_r, dij < rc_a
    pir, pjr, rr = (torch.tensor(pi[rm]), torch.tensor(pj[rm]),
                    torch.tensor(rij[rm], dtype=DTYPE))
    pia, pja, ra = (torch.tensor(pi[am]), torch.tensor(pj[am]),
                    torch.tensor(rij[am], dtype=DTYPE))
    dr, da = torch.norm(rr, dim=-1), torch.norm(ra, dim=-1)
    fkr, fkpr = ops.chebyshev_basis_and_deriv(dr, rc_r, br)
    fka, fkpa = ops.chebyshev_basis_and_deriv(da, rc_a, ba)
    d12r, d12a = 1.0 / dr.clamp(min=1e-10), 1.0 / da.clamp(min=1e-10)
    blm = ops.angular_basis(ra[:, 0] * d12a, ra[:, 1] * d12a, ra[:, 2] * d12a, 4)
    batch = {"N": N, "num_structures": 1, "atom_types": at,
             "struct_idx": torch.zeros(N, dtype=torch.long),
             "pair_i_rad": pir, "pair_j_rad": pjr, "rij_rad": rr,
             "fk_rad": fkr, "fkp_rad": fkpr, "d12inv_rad": d12r,
             "pair_i_ang": pia, "pair_j_ang": pja, "rij_ang": ra,
             "fk_ang": fka, "fkp_ang": fkpa, "d12inv_ang": d12a, "blm": blm}
    return batch, (pi, pj, rij, dij, at)


def test_analytical_force_vs_autograd():
    """Both channels on: analytical force/virial == autograd-on-rij."""
    # 6 fields after PR #1519: L_3b, q_222, q_1111, q_112, q_123, q_233
    m = _build_model([4, 1, 0, 1, 1, 1])
    batch, (pi, pj, rij, dij, at) = _random_batch()
    N = batch["N"]
    with torch.enable_grad():
        r = m.compute_properties_cached(batch, need_forces=True,
                                        need_virial=True, backend="loop")
    f_ana = r["forces"].detach().numpy()
    v_ana = r["virial"].detach().numpy()

    rm, am = dij < 6.0, dij < 5.0
    rr = torch.tensor(rij[rm], dtype=DTYPE, requires_grad=True)
    ra = torch.tensor(rij[am], dtype=DTYPE, requires_grad=True)
    pir, pjr = torch.tensor(pi[rm]), torch.tensor(pj[rm])
    pia, pja = torch.tensor(pi[am]), torch.tensor(pj[am])
    Ei = m.forward(rr, ra, pir, pjr, pia, pja, at, N, backend="loop")
    g = torch.autograd.grad(Ei.sum(), [rr, ra])
    f_auto, v_auto = ops.accumulate_forces_virial(
        N, pir, pjr, rr.detach(), g[0], pia, pja, ra.detach(), g[1],
        DTYPE, torch.device("cpu"))
    assert np.abs(f_ana - f_auto.numpy()).max() < 1e-9
    assert np.abs(v_ana - v_auto.numpy()).max() < 1e-9


def test_nep_txt_round_trip(tmp_path):
    m = _build_model([4, 1, 0, 1, 1, 1])
    p = tmp_path / "nep.txt"
    m.save_nep_txt(str(p), max_NN_radial=100, max_NN_angular=60)
    text = p.read_text()
    # 6-field GPUMD form (PR #1519). Field 2 uses the legacy
    # ``has_q_222 ? 2 : 0`` encoding so older GPUMD builds still load it.
    assert "l_max 4 2 0 1 1 1" in text
    calc = NEPCalculator(str(p), dtype=DTYPE)
    assert (calc.has_q_123, calc.has_q_233) == (1, 1)
    assert calc.dim == m.dim
    rng = np.random.default_rng(1)
    pos = rng.random((12, 3)) * 8.0
    species = ["Si" if i % 2 else "O" for i in range(12)]
    res = calc.compute(species=species, positions=pos, cell=np.eye(3) * 8.0)
    assert np.isfinite(res["energy"].numpy()).all()
    assert np.isfinite(res["forces"].numpy()).all()


def test_lmax_guard():
    """q_123 / q_233 need l_max_3b >= 3 (they use L=3 moments)."""
    with pytest.raises(ValueError):
        # 6-field PR #1519 layout: q_123 at field 5
        _build_model([2, 1, 0, 0, 1, 0])  # l_max_3b=2, q_123 on


def test_off_by_default():
    # 4-field GPUMD-core only — q_123/q_233 default to 0
    m = _build_model([4, 1, 0, 1])
    assert (m.has_q_123, m.has_q_233) == (0, 0)
    assert m.dim_angular_123 == 0 and m.dim_angular_233 == 0


def test_q1111_redundancy_warns():
    """has_q_1111=1 still works (backward compat) but warns it's redundant."""
    with pytest.warns(UserWarning, match="has_q_1111.*redundant"):
        _build_model([4, 1, 1, 0])  # q_1111 on
    # off -> no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _build_model([4, 1, 0, 0])
