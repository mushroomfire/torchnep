# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project.
# TorchNEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# TorchNEP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with TorchNEP.  If not, see <http://www.gnu.org/licenses/>.

"""Extrapolation grade: parameter gradients, MaxVol, build / grade / select."""

import numpy as np
import pytest
import torch

from _common import DATA_DIR
from torchnep.data import read_xyz
from torchnep.extrapolation import (
    ActiveSet, _Element, b_vectors, build_active_set, compute_gamma, select_structures)
from torchnep.model import NEPModel
from torchnep.train import compute_max_neighbors, preprocess_structures

TRAIN = str(DATA_DIR / "CrCoNi_train.xyz")


def _small_model(path, seed=0):
    """A small random NEP4 model (K = 6 x (21 + 2) = 138) for CrCoNi."""
    torch.manual_seed(seed)
    cfg = {"type_names": ["Cr", "Co", "Ni"], "num_types": 3,
           "cutoff_radial": 5.0, "cutoff_angular": 4.0,
           "n_max_radial": 4, "n_max_angular": 3,
           "basis_size_radial": 6, "basis_size_angular": 6,
           "l_max": [4, 0, 0, 0], "neuron": 6}
    m = NEPModel(cfg).double()
    frames = read_xyz(TRAIN)
    nn_r, nn_a = compute_max_neighbors(preprocess_structures(frames, cfg, np.float64))
    m.save_nep_txt(str(path), nn_r + 8, nn_a + 8)
    return str(path)


def _write_xyz(path, frames):
    with open(path, "w") as f:
        for fr in frames:
            cell = " ".join(f"{v:.10f}" for v in np.asarray(fr["cell"]).reshape(-1))
            f.write(f"{fr['natoms']}\nLattice=\"{cell}\" Properties=species:S:1:pos:R:3 "
                    f"pbc=\"T T T\"\n")
            for s, p in zip(fr["species"], fr["positions"]):
                f.write(f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")


def test_b_vectors_match_autograd():
    torch.manual_seed(0)
    n, D, H = 7, 5, 4
    q = torch.randn(n, D, dtype=torch.float64)
    w0 = torch.randn(H, D, dtype=torch.float64, requires_grad=True)
    b0 = torch.randn(H, dtype=torch.float64, requires_grad=True)
    w1 = torch.randn(H, dtype=torch.float64, requires_grad=True)
    B = b_vectors(q, w0.detach(), b0.detach(), w1.detach())
    for i in range(n):
        E = torch.tanh(q[i] @ w0.T - b0) @ w1
        gw0, gb0, gw1 = torch.autograd.grad(E, [w0, b0, w1])
        ref = torch.cat([gw0, gb0.unsqueeze(1), gw1.unsqueeze(1)], 1).reshape(-1)
        assert torch.allclose(B[i], ref, atol=1e-12)


def _identity_element(r, dtype=torch.float64):
    e = _Element("X", None, None, None)
    e.V = torch.eye(r, dtype=dtype)
    e.scale = torch.ones(r, dtype=dtype)
    return e


def test_maxvol_gives_dominant_rows():
    torch.manual_seed(1)
    n, r, tol = 400, 12, 1.01
    x = torch.randn(n, r, dtype=torch.float64) * torch.logspace(0, -2, r, dtype=torch.float64)
    q = torch.randn(n, 3, dtype=torch.float64)
    src = torch.stack([torch.arange(n), torch.zeros(n, dtype=torch.long)], 1)
    e = _identity_element(r)
    e.init_from_rows(x[:4 * r].clone(), q[:4 * r], src[:4 * r])
    vol0 = abs(float(torch.linalg.det(e.X)))
    xs = x.clone()
    e.maxvol(xs, q.clone(), src.clone(), tol)
    C = x @ torch.linalg.inv(e.X)
    assert float(C.abs().max()) <= tol + 1e-9          # no single swap gains more than tol
    assert abs(float(torch.linalg.det(e.X))) >= vol0
    # the active rows are rows of x, recorded with their origin
    for k in range(r):
        assert torch.allclose(e.X[k], x[int(e.src[k, 0])], atol=1e-12)
    assert torch.allclose(e.Ainv, torch.linalg.inv(e.X), atol=1e-9)


def test_build_grade_reload(tmp_path):
    model = _small_model(tmp_path / "nep.txt")
    out = tmp_path / "as.pt"
    aset = build_active_set(model, TRAIN, str(out), device="cpu", verbose=False)
    for e in aset.elements:
        assert 0 < e.r <= aset.K
    g = compute_gamma(model, str(out), TRAIN, per_atom=True, device="cpu", verbose=False)
    assert g["gamma"].shape == (24,) and g["gamma_atoms"].shape == (24 * 108,)
    assert g["gamma"].max() <= aset.tol + 1e-6          # converged MaxVol on its training set
    assert np.all(g["gamma_res"] <= 1 + 1e-9)
    per_frame = g["gamma_atoms"].reshape(24, 108)
    assert np.allclose(per_frame.max(1), g["gamma"])
    assert np.all(per_frame[np.arange(24), g["atom"]] == g["gamma"])
    # float32 descriptors grade the same structures almost identically
    g32 = compute_gamma(model, str(out), TRAIN, device="cpu", dtype="float32", verbose=False)
    assert np.allclose(g32["gamma"], g["gamma"], rtol=1e-3)
    # an active set only loads with its own model
    other = _small_model(tmp_path / "other.txt", seed=1)
    with pytest.raises(ValueError):
        ActiveSet.load(str(out), other, device="cpu")


def test_select_skips_near_duplicates(tmp_path):
    model = _small_model(tmp_path / "nep.txt")
    aset = build_active_set(model, TRAIN, str(tmp_path / "as.pt"), device="cpu", verbose=False)
    rng = np.random.default_rng(0)
    frames = read_xyz(TRAIN)[:6]
    cand = []
    for fr in frames:                           # strongly rattled frame + an exact copy
        fr = dict(fr)
        fr["positions"] = fr["positions"] + rng.normal(0, 0.25, fr["positions"].shape)
        cand += [fr, fr]
    path = tmp_path / "cand.xyz"
    _write_xyz(path, cand)
    res = select_structures(model, str(tmp_path / "as.pt"), str(path),
                            output_xyz=str(tmp_path / "chosen.xyz"),
                            output_active_set=str(tmp_path / "as2.pt"),
                            device="cpu", verbose=False)
    chosen = res["index"]
    assert len(chosen) > 0
    assert len(set(i // 2 for i in chosen)) == len(chosen)        # never both copies
    assert np.all(res["gamma_at_choice"] > aset.tol)
    assert len(read_xyz(str(tmp_path / "chosen.xyz"))) == len(chosen)
    # against the extended active set nothing chosen extrapolates any more
    g2 = compute_gamma(model, str(tmp_path / "as2.pt"), str(tmp_path / "chosen.xyz"),
                       device="cpu", verbose=False)
    assert g2["gamma"].max() <= aset.tol + 1e-6
    # a budget stops the choice early
    res1 = select_structures(model, str(tmp_path / "as.pt"), str(path), max_frames=1,
                             device="cpu", verbose=False)
    assert len(res1["index"]) == 1 and res1["index"][0] == chosen[0]
