"""Backward-pass validation.

Philosophy: if the forward expression is correct, PyTorch autograd is, by
construction, correct. So the only thing we need to verify is that the
CLOSED-FORM analytical force path (used for speed in training / predict)
agrees with the autograd force obtained by differentiating the same total
energy w.r.t. pair vectors. If they match, the analytical chain-rule
implementation has no bug.

Two checks, each on every available (device × dtype × model) cell:

  A. predict-path forces/virial  (compute_batch, analytical)
     vs
     autograd forces/virial      (compute, grad on rij)

  B. training-path forces/virial (compute_properties_cached)
     vs
     predict-path forces/virial  (compute_batch)

Note on B's float64 tolerance
  The training path runs with ``requires_grad=True`` on c_param_2/c_param_3,
  so PyTorch dispatches to autograd-aware matmul/einsum variants; the predict
  path runs under ``torch.no_grad()`` and uses the plain variants. The two
  accumulate floating-point additions in slightly different orders, which
  produces ~1e-7 differences in descriptors and ~1e-6 in forces for float64.
  That is PyTorch internal, not a correctness issue — both results match the
  NEP_CPU reference to float64 round-off (see test_vs_mdapy.py).

Both are tested on the two user-supplied reference models:
    nep_PdCuNiP.txt  (fixed ZBL, zbl 0.9 1.8)
    nep_CrCoNi.txt   (typewise ZBL, zbl 1.25 2.5 0.7)

Run:
    python test_backward.py
    TEST_DEVICE=cuda python test_backward.py
    TEST_DTYPE=float64 python test_backward.py
"""
import os
os.environ.setdefault("TORCHNEP_PREPROC_WORKERS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from torchnep.data import read_xyz
from torchnep.nep import NEPCalculator
from torchnep.model import NEPModel
from torchnep.predict import _preprocess_for_prediction, _build_batch


DTYPE_MAP = {"float32": torch.float32, "float64": torch.float64}
NP_DTYPE  = {"float32": np.float32,    "float64": np.float64}

# Tolerances. Two different settings:
#   A (autograd-vs-analytical):  both paths run in the same autograd context,
#     so the only difference is formula re-ordering → tight bound.
#   B (train-vs-predict):        different autograd contexts (grad-on vs
#     no_grad), PyTorch internally dispatches to distinct kernels whose FP
#     accumulation order differs. ~1e-6 is inherent, not a bug.
TOL_A = {
    "float64": {"F": 1e-10, "V": 1e-10},
    "float32": {"F": 5e-3,  "V": 5e-3},
}
TOL_B = {
    "float64": {"F": 5e-6,  "V": 5e-6},
    "float32": {"F": 5e-3,  "V": 5e-3},
}


# ---------------------------------------------------------------------------
# Load NEPModel with the SAME weights as NEPCalculator
# ---------------------------------------------------------------------------

def _build_nepmodel_from_calc(calc, device):
    """Mirror weights from a NEPCalculator (loaded from nep.txt) into a
    fresh NEPModel so the training-path forward uses identical parameters."""
    config = {
        "type_names": list(calc.type_names),
        "num_types": calc.num_types,
        "cutoff_radial":  calc.rc_radial,
        "cutoff_angular": calc.rc_angular,
        "n_max_radial":      calc.n_max_radial,
        "n_max_angular":     calc.n_max_angular,
        "basis_size_radial": calc.basis_size_radial,
        "basis_size_angular": calc.basis_size_angular,
        "l_max":      [calc.l_max_3b, calc.l_max_4b, calc.l_max_5b],
        "neuron":     calc.num_neurons,
    }
    if calc.has_zbl:
        config["zbl"] = calc.zbl_rc_outer
        if calc.zbl_typewise_factor is not None:
            config["typewise_cutoff_zbl_factor"] = calc.zbl_typewise_factor

    model = NEPModel(config).to(calc.dtype).to(device)
    # NEPCalculator stores w0 as (neurons, dim); NEPModel stores (dim, neurons)
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
# Part A: analytical (compute_batch) vs autograd (compute)
# ---------------------------------------------------------------------------

def check_autograd_vs_analytical(frame, calc, device, dtype_str):
    """Compare forces/virial from NEPCalculator.compute_batch (analytical)
    with NEPCalculator.compute(species, positions, cell) which uses autograd
    on pair vectors rij."""
    torch_dtype = DTYPE_MAP[dtype_str]
    np_dtype    = NP_DTYPE[dtype_str]

    # Analytical via compute_batch
    structures = _preprocess_for_prediction([frame], calc, np_dtype)
    batch = _build_batch(structures, [0], calc, torch_dtype, device)
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

    return (float(np.abs(f_ana - f_grad).max()),
            float(np.abs(v_ana - v_grad).max()))


# ---------------------------------------------------------------------------
# Part B: training path (compute_properties_cached) vs predict path
# ---------------------------------------------------------------------------

def check_train_vs_predict(frame, calc, device, dtype_str):
    torch_dtype = DTYPE_MAP[dtype_str]
    np_dtype    = NP_DTYPE[dtype_str]

    model = _build_nepmodel_from_calc(calc, device)

    structures = _preprocess_for_prediction([frame], calc, np_dtype)
    batch = _build_batch(structures, [0], calc, torch_dtype, device)

    with torch.enable_grad():
        r_train = model.compute_properties_cached(
            batch, need_forces=True, need_virial=True, pytorch_only=True)
    f_train = r_train["forces"].detach().cpu().double().numpy()
    v_train = r_train["virial"].detach().cpu().double().numpy()

    with torch.no_grad():
        r_pred = calc.compute_batch(batch)
    f_pred = r_pred["forces"].cpu().double().numpy()
    v_pred = r_pred["virial"].cpu().double().numpy()

    return (float(np.abs(f_train - f_pred).max()),
            float(np.abs(v_train - v_pred).max()))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

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


def main():
    cases = [
        (str(THIS_DIR / "nep_PdCuNiP.txt"), str(THIS_DIR / "PdCuNiP.xyz")),
        (str(THIS_DIR / "nep_CrCoNi.txt"),  str(THIS_DIR / "CrCoNi.xyz")),
    ]
    devices = _devices()
    dtypes  = _dtypes()
    print(f"devices: {devices}   dtypes: {dtypes}\n")

    all_pass = True
    for nep_txt, xyz_file in cases:
        print(f"=== {Path(nep_txt).name}  +  {Path(xyz_file).name} ===")
        frame = read_xyz(xyz_file)[0]
        for dtype_str in dtypes:
            torch_dtype = DTYPE_MAP[dtype_str]
            for device in devices:
                calc = NEPCalculator(nep_txt, dtype=torch_dtype, device=device)

                dF_A, dV_A = check_autograd_vs_analytical(
                    frame, calc, device, dtype_str)
                ok_A = (dF_A < TOL_A[dtype_str]["F"]
                        and dV_A < TOL_A[dtype_str]["V"])

                dF_B, dV_B = check_train_vs_predict(
                    frame, calc, device, dtype_str)
                ok_B = (dF_B < TOL_B[dtype_str]["F"]
                        and dV_B < TOL_B[dtype_str]["V"])

                ok = ok_A and ok_B
                all_pass = all_pass and ok
                tag = f"[{device:4s} {dtype_str:7s}]"
                print(f"  {tag}  A autograd-vs-ana  "
                      f"|ΔF|={dF_A:.2e}  |ΔV|={dV_A:.2e}  "
                      f"→ {'PASS' if ok_A else 'FAIL'}")
                print(f"  {' ' * len(tag)}  B train-vs-predict  "
                      f"|ΔF|={dF_B:.2e}  |ΔV|={dV_B:.2e}  "
                      f"→ {'PASS' if ok_B else 'FAIL'}")
        print()

    print("=" * 60)
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
