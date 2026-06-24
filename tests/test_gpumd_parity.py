# Copyright 2025 Yongchao Wu and the GPUMD development team
# This file is part of GPUMD (Torchnep project).
# GPUMD is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# GPUMD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.

"""End-to-end parity with GPUMD, plus gradient self-consistency.

Every fixture in ``_common.FIXTURES`` ships a ``data/<name>.gpumd.npz`` blob
baked from GPUMD's ``nep`` binary in prediction mode (see ``bake_fixtures.py``).

Three checks, each over (fixture * device * dtype):

  test_forward_vs_gpumd
      Per-atom E, F, V (6 comp.) and the scaled descriptor match the frozen
      GPUMD reference for every frame. The CrCoNi fixture is multi-frame:
      compressed/rattled frames push pairs to/below the ZBL inner cutoff,
      so the repulsive ZBL branch (forces up to ~120 eV/A) is covered here.
      GPUMD runs float32 and writes %g, so float32 torchnep is the tightest
      comparison; both dtypes agree to ~1e-5.

  test_analytical_vs_autograd
      The closed-form analytical force/virial path matches autograd-on-rij in
      the same context (formula re-ordering only). Run on the original and a
      strongly-compressed frame so the analytical ZBL gradient is exercised.

  test_train_vs_predict
      The training path (NEPModel.compute_properties_cached) matches the
      predict path (NEPCalculator.compute_batch) for identical weights. The
      two dispatch to different kernels under different autograd contexts, so
      the float64 bound is looser (~1e-6) — accumulation order, not a bug.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from torchnep.data import read_xyz
from torchnep.nep import NEPCalculator
from torchnep.model import NEPModel
from torchnep.predict import _build_batch, _preprocess_for_prediction
from _common import (DTYPE_MAP, NP_DTYPE_MAP, FIXTURES, devices, dtypes,
                     load_reference)


# Tolerance vs the GPUMD fixture. GPUMD computes in float32 and writes %g
# (6 sig figs), so the floor is relative ~1e-5; near-zero / cancelling
# components (e.g. forces from large opposing ZBL terms) are floored by atol.
# Both torchnep dtypes agree with GPUMD at this level.
RTOL, ATOL = 1e-5, 2e-4
# (A) analytical vs autograd — same context, formula re-ordering only, so the
# bound is relative (forces reach ~120 eV/A on the ZBL frames). (B) train vs
# predict — different autograd contexts dispatch to different kernels with
# subtly different accumulation order, hence the looser float64 bound.
TOL_ANA = {"float64": 1e-9, "float32": 1e-4}   # relative
TOL_TVP = {"float64": 5e-6, "float32": 5e-3}   # absolute


def _ids(seq, prefix):
    return [f"{prefix}={x}" for x in seq]


_FX = pytest.mark.parametrize(
    "fixture", FIXTURES, ids=_ids([f["name"] for f in FIXTURES], "fx"))
_DEV = pytest.mark.parametrize("device", devices(), ids=_ids(devices(), "dev"))
_DT = pytest.mark.parametrize("dtype_s", dtypes(), ids=_ids(dtypes(), "dt"))


@_FX
@_DEV
@_DT
def test_forward_vs_gpumd(fixture, device, dtype_s):
    """E/F/V/Descriptor agree with the baked GPUMD reference, all frames."""
    ref = load_reference(fixture["ref"])
    frames = read_xyz(str(fixture["xyz"]))

    torch_dtype = DTYPE_MAP[dtype_s]
    np_dtype = NP_DTYPE_MAP[dtype_s]
    calc = NEPCalculator(str(fixture["nep"]), dtype=torch_dtype, device=device)

    # Build pre-cached batches frame-by-frame to keep memory bounded.
    structures = _preprocess_for_prediction(frames, calc, np_dtype)
    natoms = np.asarray([s["natoms"] for s in structures], dtype=np.int64)
    N_total = int(natoms.sum())

    E_pa = np.empty(len(frames), dtype=np.float64)
    V_pa = np.empty((len(frames), 6), dtype=np.float64)
    F = np.empty((N_total, 3), dtype=np.float64)
    D = np.empty((N_total, calc.dim), dtype=np.float64)

    off = 0
    for i, s in enumerate(structures):
        batch = _build_batch([s], [0], calc, torch_dtype, torch.device(device))
        with torch.no_grad():
            r = calc.compute_batch(batch)
        n = s["natoms"]
        E_pa[i] = r["Etot"].item() / n
        # Virial: sum per-atom 9-vector, fold to GPUMD's 6 entries, divide by Na.
        v9 = r["virial"].sum(0).cpu().numpy()
        # GPUMD order (xx, yy, zz, xy, yz, zx) from the row-major 3*3 sum.
        V_pa[i] = np.array([v9[0], v9[4], v9[8],
                            v9[1], v9[5], v9[6]]) / n
        F[off:off + n] = r["forces"].cpu().numpy()
        D[off:off + n] = r["descriptor"].cpu().numpy()
        off += n

    checks = [("E", E_pa, ref["E_per_atom"]), ("F", F, ref["F"]),
              ("V", V_pa, ref["V_per_atom"]), ("D", D, ref["D_per_atom"])]
    for name, pred, rf in checks:
        if rf is None:
            continue
        d = float(np.abs(pred - rf).max())
        print(f"\n[{name} {fixture['name']:8s} {device:4s} {dtype_s:7s}] max|d|={d:.2e}")
        assert np.allclose(pred, rf, rtol=RTOL, atol=ATOL), (
            f"[{name} {fixture['name']} {device} {dtype_s}] max|d|={d:.2e} "
            f"exceeds rtol={RTOL}, atol={ATOL}")


# Original frame (index 0) and the rattled, strongly-compressed last frame
# (deepest into the ZBL repulsion) — covers both regimes for the gradient path.
_FRAME_IDX = [0, -1]


@_FX
@_DEV
@_DT
@pytest.mark.parametrize("frame_idx", _FRAME_IDX, ids=_ids(_FRAME_IDX, "frame"))
def test_analytical_vs_autograd(fixture, device, dtype_s, frame_idx):
    """Analytical force / virial path matches autograd on pair vectors."""
    torch_dtype = DTYPE_MAP[dtype_s]
    np_dtype = NP_DTYPE_MAP[dtype_s]
    calc = NEPCalculator(str(fixture["nep"]), dtype=torch_dtype, device=device)
    frame = read_xyz(str(fixture["xyz"]))[frame_idx]

    structures = _preprocess_for_prediction([frame], calc, np_dtype)
    batch = _build_batch(structures, [0], calc, torch_dtype, torch.device(device))
    with torch.no_grad():
        r_ana = calc.compute_batch(batch)
    f_ana = r_ana["forces"].cpu().double().numpy()
    v_ana = r_ana["virial"].cpu().double().numpy()

    r_grad = calc.compute(
        species=list(frame["species"]),
        positions=np.asarray(frame["positions"], dtype=np_dtype),
        cell=np.asarray(frame["cell"], dtype=np_dtype),
    )
    f_grad = r_grad["forces"].cpu().double().numpy()
    v_grad = r_grad["virial"].cpu().double().numpy()

    # Relative bound: ZBL forces reach ~120 eV/A, so scale by the magnitude.
    scaleF = np.abs(f_grad).max() + 1.0
    scaleV = np.abs(v_grad).max() + 1.0
    dF = float(np.abs(f_ana - f_grad).max()) / scaleF
    dV = float(np.abs(v_ana - v_grad).max()) / scaleV

    tol = TOL_ANA[dtype_s]
    msg = (f"[{fixture['name']:8s} {device:4s} {dtype_s:7s} frame={frame_idx}]  "
           f"relF={dF:.2e}  relV={dV:.2e}")
    print("\n" + msg)
    assert dF < tol, f"{msg}  (tol={tol})"
    assert dV < tol, f"{msg}  (tol={tol})"


def _model_from_calc(calc: NEPCalculator, device) -> NEPModel:
    """Mirror a calculator's trained weights into a fresh NEPModel so the
    training-path and predict-path forwards use identical parameters."""
    config = {
        "type_names":          list(calc.type_names),
        "num_types":           calc.num_types,
        "cutoff_radial":       calc.rc_radial,
        "cutoff_angular":      calc.rc_angular,
        "n_max_radial":        calc.n_max_radial,
        "n_max_angular":       calc.n_max_angular,
        "basis_size_radial":   calc.basis_size_radial,
        "basis_size_angular":  calc.basis_size_angular,
        "l_max":               [calc.l_max_3b, calc.has_q_222, calc.has_q_1111,
                                calc.has_q_112],
        "neuron":              calc.num_neurons,
    }
    if calc.has_zbl:
        config["zbl"] = calc.zbl_rc_outer
        if calc.zbl_typewise_factor is not None:
            config["typewise_cutoff_zbl_factor"] = calc.zbl_typewise_factor

    model = NEPModel(config).to(calc.dtype).to(device)
    for t in range(calc.num_types):
        model.fitting_nets[t].w0.data.copy_(calc.w0[t].T)
        model.fitting_nets[t].b0.data.copy_(calc.b0[t])
        model.fitting_nets[t].w1.data.copy_(calc.w1[t])
    model.b1.data.copy_(calc.b1)
    model.c_param_2.data.copy_(calc.c2)
    if calc.l_max_3b > 0:
        model.c_param_3.data.copy_(calc.c3)
    model.q_scaler.data.copy_(calc.q_scaler)
    return model


@_FX
@_DEV
@_DT
def test_train_vs_predict(fixture, device, dtype_s):
    """Training-path forward agrees with the predict-path forward."""
    torch_dtype = DTYPE_MAP[dtype_s]
    np_dtype = NP_DTYPE_MAP[dtype_s]
    calc = NEPCalculator(str(fixture["nep"]), dtype=torch_dtype, device=device)
    model = _model_from_calc(calc, device)
    frame = read_xyz(str(fixture["xyz"]))[0]

    structures = _preprocess_for_prediction([frame], calc, np_dtype)
    batch = _build_batch(structures, [0], calc, torch_dtype, torch.device(device))

    with torch.enable_grad():
        r_train = model.compute_properties_cached(
            batch, need_forces=True, need_virial=True, backend="loop")
    f_train = r_train["forces"].detach().cpu().double().numpy()
    v_train = r_train["virial"].detach().cpu().double().numpy()

    with torch.no_grad():
        r_pred = calc.compute_batch(batch)
    f_pred = r_pred["forces"].cpu().double().numpy()
    v_pred = r_pred["virial"].cpu().double().numpy()

    dF = float(np.abs(f_train - f_pred).max())
    dV = float(np.abs(v_train - v_pred).max())

    tol = TOL_TVP[dtype_s]
    msg = (f"[{fixture['name']:8s} {device:4s} {dtype_s:7s}]  "
           f"|dF|={dF:.2e}  |dV|={dV:.2e}")
    print("\n" + msg)
    assert dF < tol, f"{msg}  (F tol={tol})"
    assert dV < tol, f"{msg}  (V tol={tol})"
