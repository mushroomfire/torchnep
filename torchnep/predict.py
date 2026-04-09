"""
Full-dataset prediction and output for NEP models.

Computes energy, forces, and virial for all frames in a dataset
and saves results in GPUMD-compatible output format.
"""

import os
import numpy as np
from typing import Optional

from .nep import NEPCalculator
from .data import read_xyz


def predict_dataset(
    model_file: str,
    xyz_file: str,
    output_dir: str = ".",
    dtype: str = "float64",
    device: str = "cpu",
):
    """Run prediction on full dataset and save results.

    Each output file has two columns: predicted  target
    (target is taken from the xyz file labels if present, otherwise omitted).

    Outputs (GPUMD-compatible format):
        - energy_predict.out:  e_pred  e_target   (eV, per frame)
        - force_predict.out:   fx_pred fy_pred fz_pred  fx_ref fy_ref fz_ref
        - virial_predict.out:  xx yy zz xy yz zx  (pred then target, eV)

    Parameters
    ----------
    model_file : str
        Path to nep.txt model file.
    xyz_file : str
        Path to extended XYZ data file.
    output_dir : str
        Directory for output files.
    dtype : str
        Precision: 'float32' or 'float64'.
    device : str
        Compute device.
    """
    import torch

    dt = torch.float64 if dtype == "float64" else torch.float32
    calc = NEPCalculator(model_file, dtype=dt, device=device)
    frames = read_xyz(xyz_file)

    os.makedirs(output_dir, exist_ok=True)

    energy_file = open(os.path.join(output_dir, "energy_predict.out"), "w")
    force_file  = open(os.path.join(output_dir, "force_predict.out"),  "w")
    virial_file = open(os.path.join(output_dir, "virial_predict.out"), "w")

    # Write headers
    energy_file.write("# energy_pred(eV)  energy_target(eV)\n")
    force_file.write("# fx_pred  fy_pred  fz_pred  fx_target  fy_target  fz_target\n")
    virial_file.write("# xx_pred yy_pred zz_pred xy_pred yz_pred zx_pred  xx_tgt yy_tgt zz_tgt xy_tgt yz_tgt zx_tgt\n")

    try:
        for i, frame in enumerate(frames):
            result = calc.compute(
                frame["species"], frame["positions"], frame["cell"]
            )

            # --- Energy ---
            e_pred = result["energy"].sum().item()
            e_ref  = frame.get("energy", None)
            if e_ref is not None:
                energy_file.write(f"{e_pred:.10f}  {float(e_ref):.10f}\n")
            else:
                energy_file.write(f"{e_pred:.10f}\n")

            # --- Forces ---
            forces_pred = result["forces"].cpu().numpy()
            forces_ref  = frame.get("forces", None)
            for j, f in enumerate(forces_pred):
                if forces_ref is not None:
                    r = forces_ref[j]
                    force_file.write(
                        f"{f[0]:.10f} {f[1]:.10f} {f[2]:.10f}  "
                        f"{r[0]:.10f} {r[1]:.10f} {r[2]:.10f}\n"
                    )
                else:
                    force_file.write(f"{f[0]:.10f} {f[1]:.10f} {f[2]:.10f}\n")

            # --- Virial (6 unique: xx yy zz xy yz zx) ---
            v_pred = result["virial"].sum(dim=0).cpu().numpy()
            # index map: xx=0, yy=4, zz=8, xy=1, yz=5, zx=2  (9-component flat)
            vp = [v_pred[0], v_pred[4], v_pred[8],
                  v_pred[1], v_pred[5], v_pred[2]]
            v_ref = frame.get("virial", None)
            if v_ref is not None:
                vr = np.asarray(v_ref).flatten()
                # xyz file virial is typically stored as 9 components or 6
                if vr.size == 9:
                    vr6 = [vr[0], vr[4], vr[8], vr[1], vr[5], vr[2]]
                else:
                    vr6 = vr[:6].tolist()
                virial_file.write(
                    "  ".join(f"{x:.10f}" for x in vp) + "    " +
                    "  ".join(f"{x:.10f}" for x in vr6) + "\n"
                )
            else:
                virial_file.write("  ".join(f"{x:.10f}" for x in vp) + "\n")

            if (i + 1) % 100 == 0:
                print(f"Predicted {i + 1}/{len(frames)} frames")

    finally:
        energy_file.close()
        force_file.close()
        virial_file.close()

    print(f"Prediction complete. {len(frames)} frames processed.")
    print(f"Results saved to {output_dir}/")
    print("  energy_predict.out : pred  target")
    print("  force_predict.out  : fx fy fz (pred)  fx fy fz (target)")
    print("  virial_predict.out : xx yy zz xy yz zx (pred)  ... (target)")
