# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Yongchao Wu.
# Part of the torchnep project — https://github.com/mushroomfire/torchnep.

"""Forward pass: torchnep predictions vs frozen GPUMD reference fixtures.

Each fixture in ``_common.FIXTURES`` ships a ``data/<name>.gpumd.npz`` blob
that was baked from GPUMD's ``nep`` binary in prediction mode (see
``bake_fixtures.py``). This test loads the same nep.txt + xyz, runs
torchnep, and asserts element-wise agreement on:

  - E / atom            (column 0 of ``energy_train.out``)
  - F / atom            (columns 0..2 of ``force_train.out``)
  - V / atom (6 comp.)  (columns 0..5 of ``virial_train.out``)
  - scaled descriptor   (per-atom row of ``descriptor.out``, output_descriptor 2)

GPUMD writes ``%g`` (~6 sig figs) and runs float32 internally, so
tolerances are set accordingly:
  abs: 1e-4 on per-atom scalars, 1e-3 on forces / descriptor components.
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
from torchnep.predict import _build_batch, _preprocess_for_prediction
from _common import (DTYPE_MAP, NP_DTYPE_MAP, FIXTURES, devices, dtypes,
                     load_reference)


# Tolerances vs the GPUMD fixture (float32 + %g truncation set the floor)
TOL = {
    "float64": {"E": 1e-4, "F": 1e-3, "V": 1e-3, "D": 1e-3},
    "float32": {"E": 5e-4, "F": 2e-3, "V": 2e-3, "D": 2e-3},
}


def _ids(seq, prefix):
    return [f"{prefix}={x}" for x in seq]


@pytest.mark.parametrize("fixture", FIXTURES, ids=_ids([f["name"] for f in FIXTURES], "fx"))
@pytest.mark.parametrize("device", devices(), ids=_ids(devices(), "dev"))
@pytest.mark.parametrize("dtype_s", dtypes(), ids=_ids(dtypes(), "dt"))
def test_forward_vs_gpumd(fixture, device, dtype_s):
    """E/F/V/Descriptor agree with the baked GPUMD reference per fixture."""
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
        # Virial: sum per-atom 9-vector, fold to GPUMD's 6 entries, divide by Na
        v9 = r["virial"].sum(0).cpu().numpy()
        # GPUMD order (xx, yy, zz, xy, yz, zx) from the row-major 3×3 sum
        V_pa[i] = np.array([v9[0], v9[4], v9[8],
                            v9[1], v9[5], v9[6]]) / n
        F[off:off + n] = r["forces"].cpu().numpy()
        D[off:off + n] = r["descriptor"].cpu().numpy()
        off += n

    tol = TOL[dtype_s]
    dE = float(np.abs(E_pa - ref["E_per_atom"]).max())
    dF = float(np.abs(F - ref["F"]).max())
    dV = float(np.abs(V_pa - ref["V_per_atom"]).max())
    dD = (float(np.abs(D - ref["D_per_atom"]).max())
          if ref["D_per_atom"] is not None else 0.0)

    # Detailed message on failure
    msg = (f"[{fixture['name']:8s} {device:4s} {dtype_s:7s}]  "
           f"|ΔE/atom|={dE:.2e}  |ΔF|={dF:.2e}  "
           f"|ΔV/atom|={dV:.2e}  |ΔD|={dD:.2e}")
    print("\n" + msg)

    assert dE < tol["E"], f"{msg}  (E tol={tol['E']})"
    assert dF < tol["F"], f"{msg}  (F tol={tol['F']})"
    assert dV < tol["V"], f"{msg}  (V tol={tol['V']})"
    if ref["D_per_atom"] is not None:
        assert dD < tol["D"], f"{msg}  (D tol={tol['D']})"
