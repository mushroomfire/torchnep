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

"""Per-species cutoffs: ``cutoff rR1 rA1 rR2 rA2 ...`` in nep.in (GPUMD).

The cutoff of an element pair is the mean of the two species' values; it
enters the cutoff function and the Chebyshev argument of that pair and
decides which pairs are neighbors at all.

- parsing / validation and the (T, T) pair table;
- per-species values that are all equal reproduce the uniform cutoff exactly
  on all four compute paths (autograd / analytical x eager / compile);
- distinct values: the four paths agree with each other, all three neighbor
  modes (cached / compact / on_the_fly) build the same pairs, every kept pair
  is inside its own cutoff, and the result differs from the uniform model;
- nep.txt round trip through NEPCalculator (header with 2 * T + 3 tokens).
GPUMD parity of the whole thing: the CrCoNi_multicut fixture in
test_gpumd_parity.py.
"""
import numpy as np
import pytest
import torch

from torchnep.data import parse_nep_in, cutoff_pair_table, pair_cutoff_np
from torchnep.model import NEPModel
from torchnep.nep import NEPCalculator
from torchnep.train import preprocess_structures, StreamDataStore, compute_max_neighbors
from test_zbl_flexible import PATHS, _efv, _frames

BASE = "type 3 Cr Co Ni\nn_max 4 4\nbasis_size 6 6\nl_max 4 2 1\nneuron 30\n"
UNI = "cutoff 6 4\n"
SAME = "cutoff 6 4 6 4 6 4\n"
PER = "cutoff 6 4 5 3.5 4.5 3\n"          # Cr 6/4, Co 5/3.5, Ni 4.5/3


def _cfg(tmp_path, cutoff_line):
    p = tmp_path / "nep.in"
    p.write_text(BASE + cutoff_line)
    return parse_nep_in(str(p))


def _model_and_store(tmp_path, cutoff_line, frames, mode="cached"):
    cfg = _cfg(tmp_path, cutoff_line)
    torch.manual_seed(0)
    model = NEPModel(cfg).double()
    structs = preprocess_structures(frames, cfg, np.float64, mode=mode)
    store = StreamDataStore(structs, torch.device("cpu"), torch.float64,
                            config=cfg, neighbor_mode=mode)
    return cfg, model, store


def _same_weights(dst, src):
    dst.load_state_dict(src.state_dict(), strict=True)


def test_parse_and_table(tmp_path):
    cfg = _cfg(tmp_path, PER)
    assert cfg["cutoff_radial_per_type"] == [6.0, 5.0, 4.5]
    assert cfg["cutoff_angular_per_type"] == [4.0, 3.5, 3.0]
    assert cfg["cutoff_radial"] == 6.0 and cfg["cutoff_angular"] == 4.0
    tab = cutoff_pair_table(cfg["cutoff_radial_per_type"])
    assert tab.shape == (3, 3) and tab[0, 1] == 5.5 and tab[1, 2] == 4.75 and tab[2, 2] == 4.5
    assert np.allclose(tab, tab.T)
    at = np.array([0, 2, 1]); pi = np.array([0, 1, 2]); pj = np.array([1, 2, 0])
    assert np.allclose(pair_cutoff_np(tab, at, pi, pj), [5.25, 4.75, 5.5])
    assert pair_cutoff_np(6.0, at, pi, pj) == 6.0
    uni = _cfg(tmp_path, UNI)
    assert uni["cutoff_radial_per_type"] is None and uni["cutoff_radial"] == 6.0
    with pytest.raises(ValueError):                       # 4 values for 3 types
        _cfg(tmp_path, "cutoff 6 4 5 3.5\n")
    with pytest.raises(ValueError):                       # angular > radial
        _cfg(tmp_path, "cutoff 6 4 5 5.5 4.5 3\n")
    with pytest.raises(ValueError):                       # per-species before 'type'
        p = tmp_path / "bad.in"
        p.write_text("cutoff 6 4 5 3.5 4.5 3\n" + BASE)
        parse_nep_in(str(p))


@pytest.mark.parametrize("path", PATHS)
def test_equal_values_reproduce_uniform(tmp_path, path):
    """'cutoff 6 4 6 4 6 4' must be the uniform 'cutoff 6 4' model."""
    frames = _frames(n=4)
    _, m_u, s_u = _model_and_store(tmp_path, UNI, frames)
    _, m_s, s_s = _model_and_store(tmp_path, SAME, frames)
    _same_weights(m_s, m_u)
    assert m_u.rc_radial_per_type is None and m_s.rc_radial_per_type == [6.0] * 3
    b_u = s_u.collate(list(range(len(frames)))); b_s = s_s.collate(list(range(len(frames))))
    assert torch.equal(b_u["pair_i_rad"], b_s["pair_i_rad"])
    assert torch.equal(b_u["pair_i_ang"], b_s["pair_i_ang"])
    e_u, f_u, v_u = _efv(m_u, b_u, path)
    e_s, f_s, v_s = _efv(m_s, b_s, path)
    assert torch.allclose(e_u, e_s, atol=1e-10, rtol=0)
    assert torch.allclose(f_u, f_s, atol=1e-10, rtol=0)
    assert torch.allclose(v_u, v_s, atol=1e-10, rtol=0)


@pytest.mark.parametrize("path", PATHS[1:])
def test_paths_agree_and_differ_from_uniform(tmp_path, path):
    frames = _frames(n=4)
    _, m_p, s_p = _model_and_store(tmp_path, PER, frames)
    _, m_u, s_u = _model_and_store(tmp_path, UNI, frames)
    _same_weights(m_u, m_p)
    b_p = s_p.collate(list(range(len(frames))))
    e_ref, f_ref, v_ref = _efv(m_p, b_p, "autograd")
    e, f, v = _efv(m_p, b_p, path)
    assert torch.allclose(e, e_ref, atol=1e-9) and torch.allclose(f, f_ref, atol=1e-8) \
        and torch.allclose(v, v_ref, atol=1e-8)
    e_u, f_u, _ = _efv(m_u, s_u.collate(list(range(len(frames)))), path)
    assert (f - f_u).abs().max() > 1e-2 and (e - e_u).abs().max() > 1e-3


def test_pairs_inside_their_own_cutoff(tmp_path):
    frames = _frames(n=4)
    cfg, _, store = _model_and_store(tmp_path, PER, frames)
    b = store.collate(list(range(len(frames))))
    tab_r = cutoff_pair_table(cfg["cutoff_radial_per_type"])
    tab_a = cutoff_pair_table(cfg["cutoff_angular_per_type"])
    at = b["atom_types"].numpy()
    for key, tab in (("rad", tab_r), ("ang", tab_a)):
        pi, pj = b[f"pair_i_{key}"].numpy(), b[f"pair_j_{key}"].numpy()
        d = b[f"rij_{key}"].norm(dim=1).numpy()
        rc = pair_cutoff_np(tab, at, pi, pj)
        assert np.all(d < rc)
        # the per-pair filter is active: some pairs lie between the smallest
        # and the largest pair cutoff, so a uniform max-cutoff list is longer
        _, _, s_max = _model_and_store(tmp_path, UNI, frames)
        _, _, s_min = _model_and_store(tmp_path, "cutoff 4.5 3\n", frames)
        n_max = len(s_max.collate(list(range(len(frames))))[f"pair_i_{key}"])
        n_min = len(s_min.collate(list(range(len(frames))))[f"pair_i_{key}"])
        assert n_min < len(pi) < n_max


@pytest.mark.parametrize("mode", ["compact", "on_the_fly"])
def test_neighbor_modes_match_cached(tmp_path, mode):
    frames = _frames(n=4)
    _, m, s_c = _model_and_store(tmp_path, PER, frames)
    _, m2, s_m = _model_and_store(tmp_path, PER, frames, mode=mode)
    _same_weights(m2, m)
    idx = list(range(len(frames)))
    b_c, b_m = s_c.collate(idx), s_m.collate(idx)
    assert len(b_c["pair_i_rad"]) == len(b_m["pair_i_rad"])
    assert len(b_c["pair_i_ang"]) == len(b_m["pair_i_ang"])
    e_c, f_c, v_c = _efv(m, b_c, "analytical")
    e_m, f_m, v_m = _efv(m2, b_m, "analytical")
    assert torch.allclose(e_c, e_m, atol=1e-9) and torch.allclose(f_c, f_m, atol=1e-8) \
        and torch.allclose(v_c, v_m, atol=1e-8)
    if mode == "on_the_fly":
        assert s_m.scan_max_neighbors() == compute_max_neighbors(
            preprocess_structures(frames, _cfg(tmp_path, PER), np.float64))


def test_nep_txt_round_trip(tmp_path):
    frames = _frames(n=3)
    cfg, m, store = _model_and_store(tmp_path, PER, frames)
    structs = preprocess_structures(frames, cfg, np.float64)
    nn_r, nn_a = compute_max_neighbors(structs)
    path = tmp_path / "nep.txt"
    m.save_nep_txt(str(path), nn_r, nn_a)
    lines = path.read_text().splitlines()
    cut = [ln for ln in lines[:8] if ln.startswith("cutoff")][0].split()
    assert cut == ["cutoff", "6", "4", "5", "3.5", "4.5", "3", str(nn_r), str(nn_a)]

    calc = NEPCalculator(str(path), dtype=torch.float64)
    assert calc.rc_radial_per_type == [6.0, 5.0, 4.5]
    assert calc.rc_angular_per_type == [4.0, 3.5, 3.0]
    assert calc.rc_radial == 6.0 and calc.rc_angular == 4.0
    assert torch.allclose(calc.rc_radial_pair.double(),
                          torch.tensor(cutoff_pair_table([6.0, 5.0, 4.5])))
    b = store.collate(list(range(len(frames))))
    e_m, f_m, _ = _efv(m, b, "analytical")
    off = 0
    for k, fr in enumerate(frames):
        r = calc.compute(fr["species"], fr["positions"], fr["cell"])
        n = fr["natoms"]
        assert abs(float(r["energy"].sum()) - float(e_m[k])) < 1e-6 * max(1.0, abs(float(e_m[k])))
        assert np.allclose(r["forces"].numpy(), f_m[off:off + n].numpy(), rtol=1e-7, atol=1e-6)
        off += n

    # uniform models keep the classic 5-token line
    _, m_u, _ = _model_and_store(tmp_path, UNI, frames)
    m_u.save_nep_txt(str(tmp_path / "uni.txt"), nn_r, nn_a)
    cut = [ln for ln in (tmp_path / "uni.txt").read_text().splitlines()[:8]
           if ln.startswith("cutoff")][0].split()
    assert cut == ["cutoff", "6", "4", str(nn_r), str(nn_a)]
