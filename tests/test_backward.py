# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Yongchao Wu.
# Part of the torchnep project — https://github.com/mushroomfire/torchnep.

"""Backward / gradient self-consistency.

If the forward expression matches GPUMD (test_forward.py), PyTorch autograd
is by construction the correct gradient. So we only need to confirm the
closed-form analytical force / virial path agrees with autograd-on-rij,
and that the training-path forward matches the predict-path forward.

Two checks, per (fixture * device * dtype):

  A. predict-path analytical (NEPCalculator.compute_batch, with the
     analytical chain rule) vs autograd on pair vectors (NEPCalculator.compute,
     which calls torch.autograd.grad on rij).

  B. training-path (NEPModel.compute_properties_cached) vs predict-path
     (NEPCalculator.compute_batch). Same weights, different autograd
     contexts (grad-on vs no_grad) — verifies the two pipelines don't
     diverge.

Tolerances differ between A and B because (B) accumulates floats in a
slightly different order under autograd-on; ~1e-6 in float64 is expected.
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
from _common import DTYPE_MAP, NP_DTYPE_MAP, FIXTURES, devices, dtypes


# (A) analytical chain rule vs torch.autograd.grad — same context, formula
# re-ordering only -> tight bound. (B) train-vs-predict — different autograd
# contexts dispatch to different kernels with subtly different accumulation
# order, hence the looser float64 bound.
TOL_A = {
    "float64": {"F": 1e-10, "V": 1e-10},
    "float32": {"F": 5e-3,  "V": 5e-3},
}
TOL_B = {
    "float64": {"F": 5e-6,  "V": 5e-6},
    "float32": {"F": 5e-3,  "V": 5e-3},
}


def _ids(seq, prefix):
    return [f"{prefix}={x}" for x in seq]


# ---------------------------------------------------------------------------
# Mirror an NEPCalculator's trained weights into a fresh NEPModel so the
# training-path and predict-path forwards are guaranteed to use identical
# parameters.
# ---------------------------------------------------------------------------

def _model_from_calc(calc: NEPCalculator, device) -> NEPModel:
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


# ---------------------------------------------------------------------------
# A — autograd vs analytical (NEPCalculator only)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", FIXTURES, ids=_ids([f["name"] for f in FIXTURES], "fx"))
@pytest.mark.parametrize("device", devices(), ids=_ids(devices(), "dev"))
@pytest.mark.parametrize("dtype_s", dtypes(), ids=_ids(dtypes(), "dt"))
def test_analytical_vs_autograd(fixture, device, dtype_s):
    """Analytical force / virial path matches autograd on pair vectors."""
    torch_dtype = DTYPE_MAP[dtype_s]
    np_dtype = NP_DTYPE_MAP[dtype_s]
    calc = NEPCalculator(str(fixture["nep"]), dtype=torch_dtype, device=device)
    frame = read_xyz(str(fixture["xyz"]))[0]

    # Analytical via compute_batch
    structures = _preprocess_for_prediction([frame], calc, np_dtype)
    batch = _build_batch(structures, [0], calc, torch_dtype, torch.device(device))
    with torch.no_grad():
        r_ana = calc.compute_batch(batch)
    f_ana = r_ana["forces"].cpu().double().numpy()
    v_ana = r_ana["virial"].cpu().double().numpy()

    # Autograd via compute(species, positions, cell)
    r_grad = calc.compute(
        species=list(frame["species"]),
        positions=np.asarray(frame["positions"], dtype=np_dtype),
        cell=np.asarray(frame["cell"], dtype=np_dtype),
    )
    f_grad = r_grad["forces"].cpu().double().numpy()
    v_grad = r_grad["virial"].cpu().double().numpy()

    dF = float(np.abs(f_ana - f_grad).max())
    dV = float(np.abs(v_ana - v_grad).max())

    tol = TOL_A[dtype_s]
    msg = (f"[{fixture['name']:8s} {device:4s} {dtype_s:7s}]  "
           f"|dF|={dF:.2e}  |dV|={dV:.2e}")
    print("\n" + msg)
    assert dF < tol["F"], f"{msg}  (F tol={tol['F']})"
    assert dV < tol["V"], f"{msg}  (V tol={tol['V']})"


# ---------------------------------------------------------------------------
# B — train-path NEPModel vs predict-path NEPCalculator (same weights)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", FIXTURES, ids=_ids([f["name"] for f in FIXTURES], "fx"))
@pytest.mark.parametrize("device", devices(), ids=_ids(devices(), "dev"))
@pytest.mark.parametrize("dtype_s", dtypes(), ids=_ids(dtypes(), "dt"))
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

    tol = TOL_B[dtype_s]
    msg = (f"[{fixture['name']:8s} {device:4s} {dtype_s:7s}]  "
           f"|dF|={dF:.2e}  |dV|={dV:.2e}")
    print("\n" + msg)
    assert dF < tol["F"], f"{msg}  (F tol={tol['F']})"
    assert dV < tol["V"], f"{msg}  (V tol={tol['V']})"
