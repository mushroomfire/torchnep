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
"""Every ZBL variant on every compute path, against NEP_CPU.

The cases (zbl_cases.py): universal, typewise with the cap active for some,
none and almost all pairs, zbl.in, zbl.in plus a typewise line, each
ZBL mode with per-species cutoffs, a single type, and CrCoNi with ``zbl 2``
(Cr-Cr capped). The structures put dimers of every element pair on both
sides of every cutoff, plus dense, sheared and thinner-than-cutoff cells.

References, all independent of torchnep's ZBL code:
  data/zbl/<case>.nepcpu.npz  NEP_CPU (the LAMMPS / C++ implementation) on
      the case's nep.txt in double precision: the full model and, with the
      network output switched off, the ZBL term alone
  zbl_oracle.py  a numpy port of GPUMD's ZBL that reads the nep.in TEXT;
      it pins the nep.txt header and the models built at test time
      (slimmed models), and is itself checked against NEP_CPU here

Paths: the four training paths (autograd / analytical forces, eager /
compiled; on CUDA the compiled ones run Inductor), the model built the way
the trainer builds it (config from nep.in, weights from nep.txt); the
calculator's compute, compute_batch and compute_tiled, the ASE calculator,
predict_dataset; nep.txt round trip, checkpoint round trip, slim_model, and
a training run whose end-of-run predictions (*_train.out) must equal the
calculator on the nep.txt it wrote.

float64 is the strict check (agreement ~1e-14; tolerance rtol 1e-10). In
float32 a single-element comparison is meaningful only where each atom has
one partner, so float32 checks the ZBL forces of the dimer frames.

The fixtures are rebuilt with make_zbl_fixtures.py.
"""
import functools
import itertools
import os
import warnings

import numpy as np
import pytest
import torch

from torchnep import constants, predict_dataset, train_nep
from torchnep.data import parse_nep_in, read_xyz
from torchnep.model import NEPModel, slim_config, slim_model
from torchnep.nep import NEPCalculator
from torchnep.train import StreamDataStore, compute_max_neighbors, preprocess_structures
from _common import devices
import zbl_oracle
from zbl_cases import (CASES, SYSTEMS, ZBL_DIR, dimer_frames, nep_txt, nepcpu_ref,
                       random_frame, spec, xyz, zbl_in_text)

NAMES = list(CASES)
PATHS = ["autograd", "autograd+compile", "analytical", "analytical+compile"]
RTOL64, ATOL64 = 1e-10, 1e-8
DEVICES = [d for d in devices() if d != "mps"]


# --------------------------------------------------------------------------
# helpers

@functools.lru_cache(maxsize=None)
def _frames(system):
    return tuple(read_xyz(str(xyz(system))))


@functools.lru_cache(maxsize=None)
def _ref(name):
    return dict(np.load(nepcpu_ref(name)))


def _offsets(frames):
    return np.cumsum([0] + [len(f["species"]) for f in frames])


def _n_dimer_frames(system):
    """The dimer frames come first, in 21 A cubic cells (zbl_cases.dimer_frames)."""
    return sum(1 for f in _frames(system) if np.array_equal(f["cell"], np.diag([21.0] * 3)))


def _config(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_nep_in(str(ZBL_DIR / f"{name}.nep.in"))


def _model(name, dtype=torch.float64, device="cpu", zbl=True):
    """The model the trainer builds (config from nep.in) with the case's
    weights; ``zbl=False``: the same model without the ZBL term."""
    cfg = _config(name)
    if not zbl:
        cfg = {k: v for k, v in cfg.items()
               if k not in ("zbl", "typewise_cutoff_zbl_factor", "zbl_flexible", "zbl_file")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = NEPModel(cfg).to(dtype).to(device)
    model.load_weights_from_nep_txt(str(nep_txt(name)))
    return model


def _batch(name, frames, dtype=torch.float64, device="cpu"):
    cfg = _config(name)
    np_dtype = np.float64 if dtype == torch.float64 else np.float32
    structs = preprocess_structures(list(frames), cfg, np_dtype)
    store = StreamDataStore(structs, torch.device(device), dtype, config=cfg)
    return store.collate(list(range(len(frames))))


def _efv(model, batch, path, device="cpu"):
    """Per-frame energy, per-atom forces, per-frame virial (3 x 3) through one
    of the four training paths (the objects train_nep wires up for
    use_autograd_forces x use_compile). On CPU the compiled paths run the
    traced graph without code generation, on CUDA through Inductor."""
    args = (batch["rij_rad"], batch["rij_ang"], batch["pair_i_rad"], batch["pair_j_rad"],
            batch["pair_i_ang"], batch["pair_j_ang"], batch["atom_types"], batch["N"],
            batch["struct_idx"], batch["num_structures"])
    cuda = device == "cuda"
    with torch.enable_grad():
        if path == "autograd":
            out = model.compute_properties(*args, need_forces=True, need_virial=True, backend="loop")
        elif path == "autograd+compile":
            from torchnep.compiled_autograd import CompiledAutogradForce
            out = CompiledAutogradForce(model, backend="inductor" if cuda else "fx").compute_properties(
                *args, need_forces=True, need_virial=True, backend="bmm")
        elif path == "analytical":
            out = model.compute_properties_cached(batch, need_forces=True, need_virial=True,
                                                  backend="loop")
        else:
            core = torch.compile(model._cached_core, dynamic=True,
                                 backend="inductor" if cuda else "eager")
            out = model.compute_properties_cached(batch, need_forces=True, need_virial=True,
                                                  backend="bmm", core_fn=core)
    S = int(batch["num_structures"])
    v = torch.zeros(S, 9, dtype=torch.float64, device=out["virial"].device)
    v.index_add_(0, batch["struct_idx"], out["virial"].detach().double())
    return (out["Etot"].detach().double().cpu().numpy(), out["forces"].detach().double().cpu().numpy(),
            v.reshape(S, 3, 3).cpu().numpy())


def _close(actual, desired, what, rtol=RTOL64, atol=ATOL64):
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, err_msg=what)


def _frame_energies(e_atom, frames):
    return np.add.reduceat(e_atom, _offsets(frames)[:-1])


# --------------------------------------------------------------------------
# the references and the test design itself

def test_constants_are_gpumds():
    """torchnep's ZBL constants equal the GPUMD source (copied into the oracle)."""
    assert list(constants.COVALENT_RADIUS[:94]) == list(zbl_oracle.COVALENT_RADIUS)
    assert list(constants.ZBL_PARA) == list(zbl_oracle.ZBL_UNIVERSAL)
    assert constants.K_C_SP == zbl_oracle.K_C_SP
    assert list(constants.ELEMENTS[:94]) == list(zbl_oracle.SYMBOLS)


@pytest.mark.parametrize("name", NAMES)
def test_oracle_matches_nepcpu(name):
    """The numpy oracle (from the nep.in text) and NEP_CPU (from the nep.txt
    torchnep wrote) give the same ZBL term: the oracle is trustworthy, and the
    nep.txt carries the ZBL settings of the nep.in."""
    frames, ref = _frames(CASES[name]["system"]), _ref(name)
    off = _offsets(frames)
    for k, fr in enumerate(frames):
        pe, F, W = zbl_oracle.zbl_reference(spec(name), fr["species"], fr["positions"], fr["cell"])
        _close(pe, ref["e_atom_zbl"][off[k]:off[k + 1]], f"{name} frame {k} energy")
        _close(F, ref["forces_zbl"][off[k]:off[k + 1]], f"{name} frame {k} forces")
        _close(W, ref["virial_zbl"][k], f"{name} frame {k} virial")


@pytest.mark.parametrize("name", NAMES)
def test_gpumd_agrees(name):
    """GPUMD's nep (prediction mode, float32, ``%g`` output) reading the ZBL
    settings from the nep.in — the training-side rules, the setting of
    PR #29 — agrees with NEP_CPU on the nep.txt (and so with every path
    above). The force error of each atom of the dimer frames, where each
    atom has a single partner, is bounded by its own force (observed:
    1e-3 eV/A + 8e-6 |F|); elsewhere by the frame's largest force."""
    system = CASES[name]["system"]
    frames, ref = _frames(system), _ref(name)
    g = np.load(ZBL_DIR / f"{name}.gpumd.npz")
    off, nat = _offsets(frames), np.diff(_offsets(frames))
    _close(g["E_per_atom"], _frame_energies(ref["e_atom"], frames) / nat, f"{name} GPUMD energy",
           rtol=2e-5, atol=1e-4)
    nd = off[_n_dimer_frames(system)]
    dF = np.linalg.norm(g["F"][:nd] - ref["forces"][:nd], axis=1)
    bound = 2e-3 + 3e-5 * np.linalg.norm(ref["forces"][:nd], axis=1)
    assert (dF <= bound).all(), f"{name} GPUMD dimer forces: worst |dF| / bound {(dF / bound).max():.2f}"
    for k in range(_n_dimer_frames(system), len(frames)):
        sl = slice(off[k], off[k + 1])
        scale = np.abs(ref["forces"][sl]).max()
        _close(g["F"][sl], ref["forces"][sl], f"{name} GPUMD forces frame {k}", rtol=0, atol=5e-5 * scale)
    w = ref["virial"].reshape(-1, 9)[:, [0, 4, 8, 1, 5, 6]] / nat[:, None]
    _close(g["V_per_atom"], w, f"{name} GPUMD virial", rtol=2e-5, atol=2e-5 * np.abs(w).max())


@pytest.mark.parametrize("system", list(SYSTEMS))
def test_structures_sample_every_region(system):
    """Guard on the test design (the PR #29 bug survived because no test pair
    ever reached the typewise cap): for every case and element pair the
    dimers sample d < rc_inner (when > 0), the switching window, beyond
    rc_outer, and for capped typewise pairs the window between the cap and
    the uncapped typewise cutoff, where the ZBL must be exactly zero."""
    types = SYSTEMS[system]
    dimers = {}
    for fr in _frames(system)[:_n_dimer_frames(system)]:
        p = np.asarray(fr["positions"])
        for i in range(0, len(p), 2):
            key = tuple(sorted((types.index(fr["species"][i]), types.index(fr["species"][i + 1]))))
            dimers.setdefault(key, []).append(np.linalg.norm(p[i + 1] - p[i]))
    for name, case in CASES.items():
        if case["system"] != system:
            continue
        s = spec(name)
        for a, b in itertools.combinations_with_replacement(range(len(types)), 2):
            d = np.array(dimers[(a, b)])
            rc_i, rc_o, _ = s.pair(a, b)
            assert (d > rc_o).sum() >= 2, (name, a, b)
            assert ((d > rc_i) & (d < rc_o)).sum() >= 3, (name, a, b)
            if rc_i > 0:
                assert (d < rc_i).sum() >= 1, (name, a, b)
            if s.factor is not None and s.table is None:
                uncapped = s.factor * (zbl_oracle.COVALENT_RADIUS[s.Z[a] - 1]
                                       + zbl_oracle.COVALENT_RADIUS[s.Z[b] - 1])
                if uncapped > rc_o:
                    assert ((d > rc_o) & (d < uncapped)).sum() >= 2, (name, a, b)
    # and every system has a typewise case with a capped pair
    capped = [n for n, c in CASES.items() if c["system"] == system and spec(n).factor
              and spec(n).table is None
              and any(spec(n).factor * (zbl_oracle.COVALENT_RADIUS[spec(n).Z[a] - 1]
                                        + zbl_oracle.COVALENT_RADIUS[spec(n).Z[b] - 1])
                      > spec(n).zbl for a in range(len(types)) for b in range(len(types)))]
    assert capped, system


def test_typewise_line_with_zbl_in_warns():
    with pytest.warns(UserWarning, match="use_typewise_cutoff_zbl is ignored"):
        NEPModel(parse_nep_in(str(ZBL_DIR / "PbHCsI_flextw.nep.in")))


# --------------------------------------------------------------------------
# training paths

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("name", NAMES)
def test_training_paths_float64(name, path, device):
    """Energy, forces, virial of the whole model and of the ZBL term alone
    (model minus the same model without ZBL) equal NEP_CPU."""
    frames, ref = _frames(CASES[name]["system"]), _ref(name)
    batch = _batch(name, frames, device=device)
    e, f, v = _efv(_model(name, device=device), batch, path, device)
    e0, f0, v0 = _efv(_model(name, device=device, zbl=False), batch, path, device)
    tol = {} if device == "cpu" else {"rtol": 1e-9, "atol": 1e-7}
    _close(e, _frame_energies(ref["e_atom"], frames), f"{name} {path} energy", **tol)
    _close(f, ref["forces"], f"{name} {path} forces", **tol)
    _close(v, ref["virial"], f"{name} {path} virial", **tol)
    _close(e - e0, _frame_energies(ref["e_atom_zbl"], frames), f"{name} {path} ZBL energy", **tol)
    _close(f - f0, ref["forces_zbl"], f"{name} {path} ZBL forces", **tol)
    _close(v - v0, ref["virial_zbl"], f"{name} {path} ZBL virial", **tol)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("name", NAMES)
def test_training_paths_float32(name, path, device):
    """float32: ZBL forces of the dimer frames, atom by atom (each atom has a
    single partner, so the relative error is meaningful down to the smallest
    repulsion near a cutoff)."""
    system = CASES[name]["system"]
    nd = _n_dimer_frames(system)
    frames, ref = _frames(system)[:nd], _ref(name)
    n_atoms = _offsets(frames)[-1]
    batch = _batch(name, frames, torch.float32, device)
    _, f, _ = _efv(_model(name, torch.float32, device), batch, path, device)
    _, f0, _ = _efv(_model(name, torch.float32, device, zbl=False), batch, path, device)
    _close(f - f0, ref["forces_zbl"][:n_atoms], f"{name} {path} float32 ZBL forces",
           rtol=2e-4, atol=2e-3)


# --------------------------------------------------------------------------
# calculator paths

def _calc(name, device="cpu", dtype=torch.float64):
    return NEPCalculator(str(nep_txt(name)), dtype=dtype, device=device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", NAMES)
def test_calculator_compute(name, device):
    """NEPCalculator.compute: per-atom energies, forces, virial, and the ZBL
    component it reports."""
    frames, ref = _frames(CASES[name]["system"]), _ref(name)
    calc, off = _calc(name, device), _offsets(frames)
    for k, fr in enumerate(frames):
        r = calc.compute(fr["species"], fr["positions"], fr["cell"], return_components=True)
        sl = slice(off[k], off[k + 1])
        n = off[k + 1] - off[k]
        for key, rk in (("energy", "e_atom"), ("energy_zbl", "e_atom_zbl")):
            _close(r[key].cpu().numpy(), ref[rk][sl], f"{name} frame {k} {key}")
        for key, rk in (("forces", "forces"), ("forces_zbl", "forces_zbl")):
            _close(r[key].cpu().numpy(), ref[rk][sl], f"{name} frame {k} {key}")
        for key, rk in (("virial", "virial"), ("virial_zbl", "virial_zbl")):
            _close(r[key].cpu().numpy().reshape(n, 9).sum(0).reshape(3, 3), ref[rk][k],
                   f"{name} frame {k} {key}")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", NAMES)
def test_calculator_batch_and_tiled(name, device):
    """compute_batch (the predict path) on all frames at once, and
    compute_tiled with a small block so ZBL pairs straddle blocks (the dimer
    cells are 21 A, large enough for the linked cells)."""
    from test_gpumd_parity import _build_batch, _preprocess_for_prediction
    frames, ref = _frames(CASES[name]["system"]), _ref(name)
    calc, off = _calc(name, device), _offsets(frames)
    fr_n = [dict(f, natoms=len(f["species"])) for f in frames]
    batch = _build_batch(_preprocess_for_prediction(fr_n, calc, np.float64),
                         list(range(len(frames))), calc, torch.float64, torch.device(device))
    with torch.enable_grad():
        r = calc.compute_batch(batch)
    _close(r["Ei"].detach().cpu().numpy(), ref["e_atom"], f"{name} batch energy")
    _close(r["forces"].detach().cpu().numpy(), ref["forces"], f"{name} batch forces")
    for k, fr in enumerate(frames[:_n_dimer_frames(CASES[name]["system"])]):
        t = calc.compute_tiled(fr["species"], fr["positions"], fr["cell"], block_size=7)
        sl = slice(off[k], off[k + 1])
        _close(t["energy"].cpu().numpy(), ref["e_atom"][sl], f"{name} tiled frame {k} energy")
        _close(t["forces"].cpu().numpy(), ref["forces"][sl], f"{name} tiled frame {k} forces")
        _close(t["virial"].cpu().numpy().sum(0).reshape(3, 3), ref["virial"][k],
               f"{name} tiled frame {k} virial")


@pytest.mark.parametrize("name", NAMES)
def test_ase_calculator(name):
    pytest.importorskip("ase")
    from ase import Atoms
    from torchnep.ase_calculator import NEP
    frames, ref = _frames(CASES[name]["system"]), _ref(name)
    calc, off = NEP(str(nep_txt(name)), dtype="float64"), _offsets(frames)
    for k in (0, len(frames) - 1):
        fr = frames[k]
        atoms = Atoms(fr["species"], positions=fr["positions"], cell=fr["cell"], pbc=True)
        comp = calc.get_components(atoms)
        sl = slice(off[k], off[k + 1])
        vol = abs(np.linalg.det(fr["cell"]))
        for part, sfx in (("total", ""), ("zbl", "_zbl")):
            _close(comp[part]["energy"], ref["e_atom" + sfx][sl].sum(), f"{name} ASE {part} energy")
            _close(comp[part]["forces"], ref["forces" + sfx][sl], f"{name} ASE {part} forces")
            w = ref["virial" + sfx][k]
            voigt = -np.array([w[0, 0], w[1, 1], w[2, 2], w[1, 2], w[0, 2], w[0, 1]]) / vol
            _close(comp[part]["stress"], voigt, f"{name} ASE {part} stress")


@pytest.mark.parametrize("name", ["PbHCsI_tw25", "PbHCsI_flex", "PbHCsI_pc_tw25", "CrCoNi_tw20"])
def test_predict_dataset(name, tmp_path):
    """predict_dataset's energy/force/virial_train.out (%.10g) equal NEP_CPU."""
    system = CASES[name]["system"]
    frames, ref = _frames(system), _ref(name)
    predict_dataset(str(nep_txt(name)), str(xyz(system)), str(tmp_path), dtype="float64",
                    device="cpu", verbose=False)
    nat = np.diff(_offsets(frames))
    e = np.loadtxt(tmp_path / "energy_train.out")[:, 0]
    f = np.loadtxt(tmp_path / "force_train.out")[:, :3]
    v = np.loadtxt(tmp_path / "virial_train.out")[:, :6]
    w = ref["virial"].reshape(-1, 9)[:, [0, 4, 8, 1, 5, 6]]
    _close(e, _frame_energies(ref["e_atom"], frames) / nat, f"{name} energy_train.out", rtol=1e-9)
    _close(f, ref["forces"], f"{name} force_train.out", rtol=1e-9)
    _close(v, w / nat[:, None], f"{name} virial_train.out", rtol=1e-9)


# --------------------------------------------------------------------------
# nep.txt, checkpoints, slimming

def _numbers(path):
    out = []
    for ln in path.read_text().splitlines():
        try:
            out.append(float(ln.split()[0]))
        except (ValueError, IndexError):
            pass
    return np.array(out)


@pytest.mark.parametrize("name", NAMES)
def test_nep_txt_header_and_round_trip(name, tmp_path):
    """The model the trainer builds writes the ZBL line GPUMD writes for the
    same nep.in (fitness.cu: ``zbl rc_inner rc_outer [factor]``, ``zbl 0 0``
    plus the zbl.in table after the q_scaler), every number survives the
    round trip, and the calculator reads the settings back."""
    s = spec(name)
    out = tmp_path / "nep.txt"
    cut = next(ln for ln in nep_txt(name).read_text().splitlines() if ln.startswith("cutoff"))
    _model(name).save_nep_txt(str(out), *(int(x) for x in cut.split()[-2:]))
    lines = out.read_text().splitlines()
    assert lines[0].split()[0] == "nep4_zbl"
    zbl = [float(x) for x in lines[1].split()[1:]]
    if s.table is not None:
        assert zbl == [0.0, 0.0]
        n = len(s.table)
        tail = np.array([float(x) for x in lines[-10 * n:]]).reshape(n, 10)
        np.testing.assert_array_equal(tail, s.table)
    elif s.factor is not None:
        assert zbl == [s.zbl / 2, s.zbl, s.factor]
    else:
        assert zbl == [s.zbl / 2, s.zbl]
    np.testing.assert_array_equal(_numbers(out), _numbers(nep_txt(name)))
    calc = NEPCalculator(str(out), dtype=torch.float64)
    if s.table is None:
        assert (calc.zbl_rc_outer, calc.zbl_typewise_factor) == (s.zbl, s.factor)
    else:
        assert calc.zbl_flexible


@pytest.mark.parametrize("name", ["PbHCsI_tw25", "PbHCsI_flex", "PbHCsI_pc_tw25", "Cs_tw25"])
def test_checkpoint_round_trip(name):
    """A state dict loaded into a model built from the config reproduces the
    model; one written while the constant tables were float32 (<= 1.0.7a2)
    has its rounding removed on load."""
    frames = _frames(CASES[name]["system"])
    batch = _batch(name, frames)
    m = _model(name)
    state = {k: v.clone() for k, v in m.state_dict().items()}
    for key in ("_c3b", "_c4b", "_c5b", "_c4b2", "zbl_rc_inner_per_type", "zbl_rc_outer_per_type"):
        if key in state:
            state[key] = state[key].float().double()          # an old checkpoint
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m2 = NEPModel(_config(name)).double()
    m2.load_state_dict(state)
    for key in ("_c3b", "_c4b", "_c5b", "_c4b2", "zbl_rc_inner_per_type", "zbl_rc_outer_per_type"):
        if hasattr(m, key):
            assert torch.equal(getattr(m2, key), getattr(m, key)), key
    for path in ("autograd", "analytical"):
        for a, b in zip(_efv(m2, batch, path), _efv(m, batch, path)):
            np.testing.assert_array_equal(a, b)


def _subset_frames(types, seed=0):
    """Dimers of every pair of `types` at all distances the parent system
    probes, plus two dense cells, built at test time."""
    rng = np.random.default_rng(seed)
    frames = []
    for fr in dimer_frames("PbHCsI", rng):
        keep = [i for i in range(0, len(fr["species"]), 2)
                if fr["species"][i] in types and fr["species"][i + 1] in types]
        idx = [j for i in keep for j in (i, i + 1)]
        if idx:
            frames.append({"species": [fr["species"][j] for j in idx],
                           "positions": fr["positions"][idx], "cell": fr["cell"]})
    for _ in range(2):
        cell = np.diag(rng.uniform(5.0, 6.0, 3)) + rng.uniform(-0.6, 0.6, (3, 3))
        frames.append(random_frame(types, rng, cell, 12))
    for f in frames:
        f["natoms"] = len(f["species"])
    return frames


@pytest.mark.parametrize("keep", [["I", "Cs"], ["Cs"]], ids=["I-Cs", "Cs"])
@pytest.mark.parametrize("name", ["PbHCsI_tw25", "PbHCsI_uni25", "PbHCsI_flex", "PbHCsI_pc_tw25"])
def test_slim_model(name, keep, tmp_path):
    """slim_model to a subset (in another order): the slim model, and the
    model the trainer builds from slim_config with the slim weights, equal the
    full model on structures of the subset; the nep.txt of the slim model
    gives the same through the calculator, and its ZBL term equals the oracle."""
    frames = _subset_frames(keep)
    full = _model(name)
    slim = slim_model(full, keep)
    cfg = slim_config(_config(name), keep)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer_model = NEPModel(cfg).double()
    trainer_model.load_state_dict(slim.state_dict())
    e_f, f_f, v_f = _efv(full, _batch(name, frames), "analytical")
    structs = preprocess_structures(frames, cfg, np.float64)
    nn_r, nn_a = compute_max_neighbors(structs)
    sbatch = StreamDataStore(structs, torch.device("cpu"), torch.float64, config=cfg).collate(
        list(range(len(frames))))
    for m in (slim, trainer_model):
        for path in ("autograd", "analytical"):
            for a, b, what in zip(_efv(m, sbatch, path), (e_f, f_f, v_f), ("energy", "forces", "virial")):
                _close(a, b, f"{name} slim {keep} {path} {what}", rtol=1e-9)
    structs = preprocess_structures(frames, cfg, np.float64)
    slim.save_nep_txt(str(tmp_path / "slim.txt"), nn_r, nn_a)
    calc = NEPCalculator(str(tmp_path / "slim.txt"), dtype=torch.float64)
    s = spec(name)
    off = _offsets(frames)
    for k, fr in enumerate(frames):
        r = calc.compute(fr["species"], fr["positions"], fr["cell"], return_components=True)
        pe, F, W = zbl_oracle.zbl_reference(s, fr["species"], fr["positions"], fr["cell"])
        n = len(fr["species"])
        _close(r["energy_zbl"].numpy(), pe, f"{name} slim {keep} frame {k} ZBL energy")
        _close(r["forces_zbl"].numpy(), F, f"{name} slim {keep} frame {k} ZBL forces")
        _close(r["virial_zbl"].numpy().reshape(n, 9).sum(0).reshape(3, 3), W, f"{name} slim ZBL virial")
        _close(r["energy"].sum().item(), e_f[k], f"{name} slim {keep} frame {k} energy", rtol=1e-9)
        _close(r["forces"].numpy(), f_f[off[k]:off[k + 1]], f"{name} slim {keep} frame {k} forces",
               rtol=1e-9)


# --------------------------------------------------------------------------
# a training run

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("autograd", [False, True], ids=["analytical", "autograd"])
@pytest.mark.parametrize("name", ["PbHCsI_tw25", "PbHCsI_flex", "PbHCsI_pc_tw25", "CrCoNi_tw20"])
def test_training_run_predictions_equal_calculator(name, autograd, device, tmp_path):
    """train_nep from nep.in: the end-of-run predictions (*_train.out, made by
    the training model) equal the calculator on the nep_best.txt the run
    wrote. This is the train / inference consistency PR #29 broke; on a GPU
    the run takes the default compiled path."""
    system = CASES[name]["system"]
    nep_in = tmp_path / "nep.in"
    nep_in.write_text((ZBL_DIR / f"{name}.nep.in").read_text() + "epoch 1\nbatch 8\nlr 0\nstage2 0\n")
    if CASES[name]["flex"] is not None:
        (tmp_path / f"{name}.zbl.in").write_text(zbl_in_text(name))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_nep(str(nep_in), str(xyz(system)), output_dir=str(tmp_path / "out"), device=device,
                  precision="float64", use_autograd_forces=autograd, restart=False,
                  print_interval=100, checkpoint_interval=10**6, prediction_interval=10**6,
                  run_seed=0)
    frames = _frames(system)
    calc = NEPCalculator(str(tmp_path / "out" / "nep_best.txt"), dtype=torch.float64)
    e = np.loadtxt(tmp_path / "out" / "energy_train.out")[:, 0]
    f = np.loadtxt(tmp_path / "out" / "force_train.out")[:, :3]
    nat = np.diff(_offsets(frames))
    e_c = np.array([calc.compute(fr["species"], fr["positions"], fr["cell"])["energy"].sum().item()
                    for fr in frames]) / nat
    f_c = np.concatenate([calc.compute(fr["species"], fr["positions"], fr["cell"])["forces"].numpy()
                          for fr in frames])
    tol = {"rtol": 1e-9, "atol": 1e-8} if device == "cpu" else {"rtol": 1e-6, "atol": 1e-5}
    _close(e, e_c, f"{name} energy_train.out", **tol)
    _close(f, f_c, f"{name} force_train.out", **tol)


@pytest.mark.skipif(os.environ.get("TORCHNEP_TEST_DDP") != "1",
                    reason="multi-process test is local-only (set TORCHNEP_TEST_DDP=1)")
@pytest.mark.parametrize("name", ["PbHCsI_tw25", "PbHCsI_flex"])
def test_sharded_training_run_predictions_equal_calculator(name, tmp_path):
    """The same for train_nep_sharded on 2 ranks (its own model set-up and
    sharded prediction)."""
    from test_sharded_features import _run
    system = CASES[name]["system"]
    nep_in = tmp_path / "nep.in"
    nep_in.write_text((ZBL_DIR / f"{name}.nep.in").read_text() + "epoch 1\nbatch 8\nlr 0\nstage2 0\n")
    if CASES[name]["flex"] is not None:
        (tmp_path / f"{name}.zbl.in").write_text(zbl_in_text(name))
    _run(tmp_path, str(nep_in), str(xyz(system)), tmp_path / "out", run_seed=0)
    frames = _frames(system)
    calc = NEPCalculator(str(tmp_path / "out" / "nep_best.txt"), dtype=torch.float64)
    nat = np.diff(_offsets(frames))
    e_c = np.array([calc.compute(fr["species"], fr["positions"], fr["cell"])["energy"].sum().item()
                    for fr in frames]) / nat
    f_c = np.concatenate([calc.compute(fr["species"], fr["positions"], fr["cell"])["forces"].numpy()
                          for fr in frames])
    _close(np.loadtxt(tmp_path / "out" / "energy_train.out")[:, 0], e_c, f"{name} energy_train.out",
           rtol=1e-9)
    _close(np.loadtxt(tmp_path / "out" / "force_train.out")[:, :3], f_c, f"{name} force_train.out",
           rtol=1e-9)
