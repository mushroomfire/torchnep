"""Forward-pass validation against mdapy (NEP_CPU reference).

Tests every (device × dtype × model) combination available.

  float64 vs mdapy : strict (tol 1e-10). NEP_CPU itself uses double precision,
                    so float64 agreement should be at float-round-off.
  float32 vs mdapy : loose (tol 5e-4 for E/atom, 1e-3 for force/virial
                    component). The bound is an upper-estimate for float32
                    accumulation over ~10³ pair contributions; on these
                    small frames the measured error is typically 1e-5.

Run:
    python test_vs_mdapy.py                         # everything
    python test_vs_mdapy.py <nep.txt> <xyz>         # restrict to one case
    TEST_DEVICE=cuda python test_vs_mdapy.py        # restrict to one device
    TEST_DTYPE=float32 python test_vs_mdapy.py      # restrict to one dtype
"""
import os
os.environ.setdefault("TORCHNEP_PREPROC_WORKERS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import mdapy

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from torchnep.data import read_xyz
from torchnep.nep import NEPCalculator
from torchnep.predict import _preprocess_for_prediction, _build_batch


DTYPE_MAP = {"float32": torch.float32, "float64": torch.float64}
NP_DTYPE  = {"float32": np.float32,    "float64": np.float64}

# Tolerances vs the mdapy (double-precision) reference
TOL = {
    "float64": {"E": 1e-10, "F": 1e-10, "V": 1e-10},
    "float32": {"E": 5e-4,  "F": 1e-3,  "V": 1e-3},
}


def _run_frame(frame, calc_tn, calc_md, device, dtype_str):
    positions = np.asarray(frame["positions"], dtype=np.float64)
    species = list(frame["species"])
    cell = np.asarray(frame["cell"], dtype=np.float64)

    torch_dtype = DTYPE_MAP[dtype_str]
    np_dtype    = NP_DTYPE[dtype_str]
    structures = _preprocess_for_prediction([frame], calc_tn, np_dtype)
    batch = _build_batch(structures, [0], calc_tn, torch_dtype, device)
    with torch.no_grad():
        r = calc_tn.compute_batch(batch)
    ei_tn = r["Ei"].cpu().double().numpy()
    f_tn  = r["forces"].cpu().double().numpy()
    v_tn  = r["virial"].cpu().double().numpy()

    data = pl.DataFrame({
        "x": positions[:, 0], "y": positions[:, 1], "z": positions[:, 2],
        "element": species,
    })
    box = mdapy.Box(cell, boundary=[1, 1, 1])
    calc_md.calculate(data, box)
    ei_md = np.asarray(calc_md.results["energies"])
    f_md  = np.asarray(calc_md.results["forces"])
    v_md  = np.asarray(calc_md.results["virials"])

    return {
        "N": frame["natoms"],
        "E_abs": float(np.abs(ei_tn - ei_md).max()),
        "F_abs": float(np.abs(f_tn - f_md).max()),
        "V_abs": float(np.abs(v_tn - v_md).max()),
    }


def _devices():
    env = os.environ.get("TEST_DEVICE")
    if env:
        return [env]
    ds = ["cpu"]
    if torch.cuda.is_available():
        ds.append("cuda")
    return ds


def _dtypes():
    env = os.environ.get("TEST_DTYPE")
    if env:
        return [env]
    return ["float32", "float64"]


def _cases():
    if len(sys.argv) >= 3:
        return [(sys.argv[1], sys.argv[2])]
    return [
        (str(THIS_DIR / "nep_PdCuNiP.txt"), str(THIS_DIR / "PdCuNiP.xyz")),
        (str(THIS_DIR / "nep_CrCoNi.txt"),  str(THIS_DIR / "CrCoNi.xyz")),
    ]


def main():
    devices = _devices()
    dtypes  = _dtypes()
    cases   = _cases()

    print(f"devices: {devices}   dtypes: {dtypes}\n")
    all_pass = True
    for nep_txt, xyz_file in cases:
        print(f"=== {Path(nep_txt).name}  +  {Path(xyz_file).name} ===")
        calc_md = mdapy.NEP(nep_txt)
        frames = read_xyz(xyz_file)

        for dtype_str in dtypes:
            torch_dtype = DTYPE_MAP[dtype_str]
            tol = TOL[dtype_str]
            for device in devices:
                calc_tn = NEPCalculator(nep_txt, dtype=torch_dtype, device=device)
                max_E = max_F = max_V = 0.0
                for fr in frames:
                    r = _run_frame(fr, calc_tn, calc_md, device, dtype_str)
                    max_E = max(max_E, r["E_abs"])
                    max_F = max(max_F, r["F_abs"])
                    max_V = max(max_V, r["V_abs"])
                ok = (max_E < tol["E"] and max_F < tol["F"] and max_V < tol["V"])
                verdict = "PASS" if ok else "FAIL"
                print(f"  [{device:4s} {dtype_str:7s}]  "
                      f"|ΔE|≤{max_E:.2e}  |ΔF|≤{max_F:.2e}  "
                      f"|ΔV|≤{max_V:.2e}  → {verdict}")
                all_pass = all_pass and ok
        print()

    print("=" * 60)
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
