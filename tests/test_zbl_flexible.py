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

"""Flexible ZBL (GPUMD ``zbl.in``): ``zbl <file>`` in nep.in.

- a zbl.in that spells out the universal parameters must give the SAME
  energies / forces as the universal path;
- arbitrary per-pair parameters must match an independent numpy evaluation
  of E = K Z_i Z_j / d * phi(d * a_inv) * fc(d; rc_inner_ij, rc_outer_ij);
- nep.txt round trip: the table is written after the q_scaler under a
  ``zbl 0 0`` header and read back by the calculator and the trainer.
Every ZBL check runs on all four training compute paths (``PATHS``):
autograd forces and analytical forces, each eager and under torch.compile.
The compiled variants use the ``eager`` dynamo backend so they run on the
CPU-only CI: the point is that the traced SOURCE (CompiledAutogradForce._raw,
NEPModel._cached_core) carries the per-pair table, not Inductor codegen.
The GPUMD parity of the whole thing is covered by the CrCoNi_flexzbl fixture
in test_gpumd_parity.py; the Inductor pipeline itself by
test_compiled_autograd.py (CUDA).
"""
import numpy as np
import pytest
import torch

from torchnep.constants import ZBL_PARA, ELEMENTS, K_C_SP
from torchnep.data import read_xyz, parse_nep_in, read_zbl_in, zbl_pair_index
from torchnep.model import NEPModel
from torchnep.nep import NEPCalculator
from torchnep.train import preprocess_structures, StreamDataStore, compute_max_neighbors
from _common import DATA_DIR

XYZ = DATA_DIR / "CrCoNi_train.xyz"
NEP_IN = ("type 3 Cr Co Ni\ncutoff 6 4\nn_max 4 4\nbasis_size 6 6\nl_max 4 2 1\nneuron 30\n")


def _frames(n=6, scale=0.8):
    """CrCoNi frames plus compressed copies (NN ~1.2 A) so the ZBL switching
    windows are actually sampled."""
    fr = read_xyz(str(XYZ))[:n]
    out = []
    for f in fr:
        out.append(f)
        g = dict(f); g["positions"] = f["positions"] * scale; g["cell"] = f["cell"] * scale
        out.append(g)
    return out


def _write_zbl_in(path, rows):
    path.write_text("\n".join(" ".join(f"{v:.17g}" for v in r) for r in rows) + "\n")


def _universal_rows(num_types, rc_outer):
    n = num_types * (num_types + 1) // 2
    return [[rc_outer / 2.0, rc_outer] + list(ZBL_PARA) for _ in range(n)]


def _custom_rows(num_types, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(num_types * (num_types + 1) // 2):
        rc_o = float(rng.uniform(1.8, 2.8)); rc_i = float(rng.uniform(0.6, 0.9 * rc_o))
        coef = [float(c * rng.uniform(0.8, 1.2)) for c in ZBL_PARA]
        rows.append([rc_i, rc_o] + coef)
    return rows


def _model_and_batch(tmp_path, zbl_line, frames):
    p = tmp_path / "nep.in"; p.write_text(NEP_IN + zbl_line + "\n")
    cfg = parse_nep_in(str(p))
    torch.manual_seed(0)
    model = NEPModel(cfg).double()
    structs = preprocess_structures(frames, cfg, np.float64)
    store = StreamDataStore(structs, torch.device("cpu"), torch.float64, config=cfg)
    return cfg, model, store.collate(list(range(len(frames))))


PATHS = ["autograd", "autograd+compile", "analytical", "analytical+compile"]


def _efv(model, batch, path):
    """Energy / forces / virial through one of the four training compute paths
    (the same objects train_nep wires up for use_autograd_forces x use_compile):
    ``autograd``            NEPModel.compute_properties  -> ops.compute_zbl
    ``autograd+compile``    CompiledAutogradForce (make_fx graph) -> ops.compute_zbl_pair
    ``analytical``          NEPModel.compute_properties_cached -> ops.compute_zbl_pair
    ``analytical+compile``  the same with torch.compile(model._cached_core)"""
    args = (batch["rij_rad"], batch["rij_ang"], batch["pair_i_rad"], batch["pair_j_rad"],
            batch["pair_i_ang"], batch["pair_j_ang"], batch["atom_types"], batch["N"],
            batch["struct_idx"], batch["num_structures"])
    with torch.enable_grad():
        if path == "autograd":
            out = model.compute_properties(*args, need_forces=True, need_virial=True, backend="loop")
        elif path == "autograd+compile":
            from torchnep.compiled_autograd import CompiledAutogradForce
            out = CompiledAutogradForce(model, backend="eager").compute_properties(
                *args, need_forces=True, need_virial=True, backend="bmm")
        elif path == "analytical":
            out = model.compute_properties_cached(batch, need_forces=True, need_virial=True,
                                                  backend="loop")
        else:
            core = torch.compile(model._cached_core, dynamic=True, backend="eager")
            out = model.compute_properties_cached(batch, need_forces=True, need_virial=True,
                                                  backend="bmm", core_fn=core)
    return out["Etot"].detach(), out["forces"].detach(), out["virial"].detach()


def _energy_forces(model, batch, path):
    e, f, _ = _efv(model, batch, path)
    return e, f


def test_parse_zbl_file_and_table(tmp_path):
    rows = _custom_rows(3, seed=1)
    _write_zbl_in(tmp_path / "my_zbl.in", rows)
    (tmp_path / "nep.in").write_text(NEP_IN + "zbl my_zbl.in\n")
    cfg = parse_nep_in(str(tmp_path / "nep.in"))
    assert np.allclose(np.array(cfg["zbl_flexible"]), np.array(rows))
    assert cfg["zbl"] == pytest.approx(max(r[1] for r in rows))
    assert zbl_pair_index(1, 0, 3) == zbl_pair_index(0, 1, 3) == 1 and zbl_pair_index(2, 1, 3) == 4 and zbl_pair_index(2, 2, 3) == 5
    with pytest.raises(ValueError):
        read_zbl_in(str(tmp_path / "my_zbl.in"), 4)      # wrong number of rows


@pytest.mark.parametrize("path", PATHS)
def test_universal_file_matches_universal(tmp_path, path):
    frames = _frames()
    cfg_u, m_u, b_u = _model_and_batch(tmp_path, "zbl 2.5", frames)
    _write_zbl_in(tmp_path / "zbl.in", _universal_rows(3, 2.5))
    cfg_f, m_f, b_f = _model_and_batch(tmp_path, "zbl zbl.in", frames)
    m_f.load_state_dict({k: v for k, v in m_u.state_dict().items() if k != "zbl_flexible"}, strict=False)
    assert m_f.zbl_flexible is not None and m_u.zbl_flexible is None
    for m in (m_u, m_f):
        assert torch.allclose(m.zbl_phi_pair, torch.tensor(ZBL_PARA, dtype=torch.float64).expand(3, 3, 8))
    e_u, f_u = _energy_forces(m_u, b_u, path)
    e_f, f_f = _energy_forces(m_f, b_f, path)
    assert torch.equal(e_u, e_f) and torch.equal(f_u, f_f)


def _numpy_zbl(batch, cfg, rows, atom_numbers):
    """Independent evaluation of the flexible ZBL total energy per frame."""
    T = cfg["num_types"]
    at = batch["atom_types"].numpy(); pi = batch["pair_i_ang"].numpy(); pj = batch["pair_j_ang"].numpy()
    rij = batch["rij_ang"].numpy(); sidx = batch["struct_idx"].numpy()
    d = np.linalg.norm(rij, axis=1)
    E = np.zeros(int(sidx.max()) + 1)
    for p in range(len(pi)):
        t1, t2 = at[pi[p]], at[pj[p]]
        row = rows[zbl_pair_index(int(t1), int(t2), T)]
        rc_i, rc_o, a = row[0], row[1], row[2:]
        if d[p] >= rc_o:
            continue
        zi, zj = atom_numbers[t1], atom_numbers[t2]
        a_inv = (zi ** 0.23 + zj ** 0.23) * 2.134563
        x = d[p] * a_inv
        phi = sum(a[2 * k] * np.exp(-a[2 * k + 1] * x) for k in range(4))
        fc = 1.0 if d[p] < rc_i else 0.5 * np.cos(np.pi * (d[p] - rc_i) / (rc_o - rc_i)) + 0.5
        E[sidx[pi[p]]] += 0.5 * K_C_SP * zi * zj * phi / d[p] * fc
    return E


@pytest.mark.parametrize("path", PATHS)
def test_custom_table_matches_numpy(tmp_path, path):
    frames = _frames(n=4)
    rows = _custom_rows(3, seed=3)
    _write_zbl_in(tmp_path / "zbl.in", rows)
    cfg, m, b = _model_and_batch(tmp_path, "zbl zbl.in", frames)
    # zero the NN contribution: compare the ZBL part only via (with - without)
    cfg0, m0, b0 = _model_and_batch(tmp_path, "zbl 2.5", frames)
    m0.load_state_dict({k: v for k, v in m.state_dict().items() if k != "zbl_flexible"}, strict=False)
    # universal part evaluated by numpy for the same pairs, then flexible
    e_flex, _ = _energy_forces(m, b, path)
    e_uni, _ = _energy_forces(m0, b0, path)
    Z = [ELEMENTS.index(n) + 1 for n in cfg["type_names"]]
    ref_flex = _numpy_zbl(b, cfg, rows, Z)
    ref_uni = _numpy_zbl(b0, cfg0, _universal_rows(3, 2.5), Z)
    assert np.allclose((e_flex - e_uni).numpy(), ref_flex - ref_uni, atol=1e-7, rtol=1e-9)
    assert np.abs(ref_flex - ref_uni).max() > 1e-3      # the test actually exercised ZBL


@pytest.mark.parametrize("path", PATHS[1:])
def test_custom_table_paths_agree(tmp_path, path):
    """Forces and virial from a custom table must agree across the four paths
    (the autograd path is the reference: it gathers the table per pair and is
    pinned to numpy above). A path that silently falls back to the universal
    coefficients (CompiledAutogradForce used to drop the table when tracing)
    reproduces the universal model instead, which the second assertion rules
    out."""
    frames = _frames(n=4)
    rows = _custom_rows(3, seed=5)
    _write_zbl_in(tmp_path / "zbl.in", rows)
    _, m, b = _model_and_batch(tmp_path, "zbl zbl.in", frames)
    _, m0, b0 = _model_and_batch(tmp_path, "zbl 2.5", frames)
    m0.load_state_dict({k: v for k, v in m.state_dict().items() if k != "zbl_flexible"}, strict=False)
    e_ref, f_ref, v_ref = _efv(m, b, "autograd")
    e, f, v = _efv(m, b, path)
    assert torch.allclose(e, e_ref, atol=1e-9) and torch.allclose(f, f_ref, atol=1e-8) \
        and torch.allclose(v, v_ref, atol=1e-8)
    e_uni, f_uni, _ = _efv(m0, b0, path)
    assert not torch.allclose(f, f_uni, atol=1e-3) and not torch.allclose(e, e_uni, atol=1e-3)


def test_nep_txt_round_trip(tmp_path):
    frames = _frames(n=4)
    rows = _custom_rows(3, seed=5)
    _write_zbl_in(tmp_path / "zbl.in", rows)
    cfg, m, b = _model_and_batch(tmp_path, "zbl zbl.in", frames)
    structs = preprocess_structures(frames, cfg, np.float64)
    nn_r, nn_a = compute_max_neighbors(structs)
    path = tmp_path / "nep.txt"
    m.save_nep_txt(str(path), nn_r, nn_a)
    lines = path.read_text().splitlines()
    assert lines[1].split() == ["zbl", "0", "0"]
    tail = np.array([float(x) for x in lines[-60:]]).reshape(6, 10)
    assert np.allclose(tail, np.array(rows))

    # calculator: same energies / forces as the training model
    calc = NEPCalculator(str(path), dtype=torch.float64)
    assert calc.zbl_flexible
    e_m, f_m = _energy_forces(m, b, "analytical")
    off = 0
    for k, fr in enumerate(frames):
        r = calc.compute(fr["species"], fr["positions"], fr["cell"])
        n = fr["natoms"]
        assert abs(float(r["energy"].sum()) - float(e_m[k])) < 1e-6 * max(1.0, abs(float(e_m[k])))
        assert np.allclose(r["forces"].numpy(), f_m[off:off + n].numpy(), rtol=1e-7, atol=1e-6)
        off += n

    # trainer-side load: a universal model picks the table up from the file
    cfg_u, m_u, _ = _model_and_batch(tmp_path, "zbl 2.5", frames)
    m_u.load_weights_from_nep_txt(str(path))
    assert m_u.zbl_flexible is not None
    assert torch.allclose(m_u.zbl_flexible.double(), torch.tensor(rows, dtype=torch.float64))
    e_l, f_l = _energy_forces(m_u, b, "analytical")
    assert torch.allclose(e_l, e_m) and torch.allclose(f_l, f_m)
