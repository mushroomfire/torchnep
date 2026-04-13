"""
Full-dataset prediction and output for NEP models.

Computes energy, forces, and virial for all frames in a dataset
and saves results in GPUMD-compatible output format.

Uses batched GPU inference: neighbor lists are built once on CPU,
Chebyshev/angular basis is cached on GPU, then structures are evaluated
in batches using compute_batch() — the same analytical-force path as training.
"""

import os
import torch
import numpy as np
from typing import Optional

from .nep import NEPCalculator
from .data import read_xyz, build_neighbor_list_np
from . import ops


def _preprocess_for_prediction(frames, calc, np_dtype):
    """Build neighbor lists for all frames. Returns list of structure dicts."""
    rc_rad = calc.rc_radial
    rc_ang = calc.rc_angular
    max_rc = max(rc_rad, rc_ang)

    structures = []
    for frame in frames:
        positions = np.array(frame["positions"], dtype=np_dtype)
        cell = np.array(frame["cell"], dtype=np_dtype)
        atom_types = np.array(
            [calc.type_names.index(s) for s in frame["species"]],
            dtype=np.int64)

        pair_i, pair_j, rij = build_neighbor_list_np(positions, cell, max_rc)
        dij = np.linalg.norm(rij, axis=1)

        s = {
            "natoms": frame["natoms"],
            "atom_types": atom_types,
            "pair_i_rad": pair_i[dij < rc_rad],
            "pair_j_rad": pair_j[dij < rc_rad],
            "rij_rad": rij[dij < rc_rad],
            "pair_i_ang": pair_i[dij < rc_ang],
            "pair_j_ang": pair_j[dij < rc_ang],
            "rij_ang": rij[dij < rc_ang],
            "energy": frame.get("energy"),
            "forces": frame.get("forces"),
            "virial": frame.get("virial"),
        }
        structures.append(s)
    return structures


def _build_batch(structures, indices, calc, dtype, device):
    """Collate a list of structure indices into a GPU batch with cached basis."""
    rc_rad = calc.rc_radial
    rc_ang = calc.rc_angular
    l_max_3b = calc.l_max_3b
    basis_r = calc.basis_size_radial
    basis_a = calc.basis_size_angular
    num_lm = calc.num_lm

    natoms_list = [structures[i]["natoms"] for i in indices]
    N_total = sum(natoms_list)
    B = len(indices)

    # Atom offsets for global indexing within the batch
    offsets = [0]
    for n in natoms_list:
        offsets.append(offsets[-1] + n)

    atom_types = torch.tensor(
        np.concatenate([structures[i]["atom_types"] for i in indices]),
        dtype=torch.long, device=device)

    struct_idx = torch.cat([
        torch.full((natoms_list[k],), k, dtype=torch.long, device=device)
        for k in range(B)
    ])

    # Radial pairs
    pi_r = torch.tensor(
        np.concatenate([structures[indices[k]]["pair_i_rad"] + offsets[k]
                        for k in range(B)]).astype(np.int64),
        dtype=torch.long, device=device)
    pj_r = torch.tensor(
        np.concatenate([structures[indices[k]]["pair_j_rad"] + offsets[k]
                        for k in range(B)]).astype(np.int64),
        dtype=torch.long, device=device)
    rij_r = torch.tensor(
        np.concatenate([structures[indices[k]]["rij_rad"] for k in range(B)]),
        dtype=dtype, device=device)

    # Angular pairs
    pi_a = torch.tensor(
        np.concatenate([structures[indices[k]]["pair_i_ang"] + offsets[k]
                        for k in range(B)]).astype(np.int64),
        dtype=torch.long, device=device)
    pj_a = torch.tensor(
        np.concatenate([structures[indices[k]]["pair_j_ang"] + offsets[k]
                        for k in range(B)]).astype(np.int64),
        dtype=torch.long, device=device)
    rij_a = torch.tensor(
        np.concatenate([structures[indices[k]]["rij_ang"] for k in range(B)]),
        dtype=dtype, device=device)

    # Cached basis functions
    dr = torch.norm(rij_r, dim=-1)
    fk_r, fkp_r = ops.chebyshev_basis_and_deriv(dr, rc_rad, basis_r)
    d12inv_r = 1.0 / dr.clamp(min=1e-10)

    if rij_a.shape[0] > 0:
        da = torch.norm(rij_a, dim=-1)
        fk_a, fkp_a = ops.chebyshev_basis_and_deriv(da, rc_ang, basis_a)
        d12inv_a = 1.0 / da.clamp(min=1e-10)
        blm = ops.angular_basis(
            rij_a[:, 0] * d12inv_a,
            rij_a[:, 1] * d12inv_a,
            rij_a[:, 2] * d12inv_a,
            l_max_3b)
    else:
        fk_a = torch.zeros(0, basis_a + 1, dtype=dtype, device=device)
        fkp_a = torch.zeros(0, basis_a + 1, dtype=dtype, device=device)
        d12inv_a = torch.zeros(0, dtype=dtype, device=device)
        blm = torch.zeros(0, num_lm, dtype=dtype, device=device)

    return {
        "N": N_total,
        "num_structures": B,
        "atom_types": atom_types,
        "struct_idx": struct_idx,
        "pair_i_rad": pi_r,
        "pair_j_rad": pj_r,
        "rij_rad": rij_r,
        "fk_rad": fk_r,
        "fkp_rad": fkp_r,
        "d12inv_rad": d12inv_r,
        "pair_i_ang": pi_a,
        "pair_j_ang": pj_a,
        "rij_ang": rij_a,
        "fk_ang": fk_a,
        "fkp_ang": fkp_a,
        "d12inv_ang": d12inv_a,
        "blm": blm,
    }


def predict_dataset(
    model_file: str,
    xyz_file: str,
    output_dir: str = ".",
    dtype: str = "float64",
    device: str = None,
    batch_size: int = 64,
):
    """Run batched prediction on a full dataset and save results.

    Builds all neighbor lists once on CPU, then evaluates structures in
    batches on GPU using analytical forces — the same code path as training.
    This is typically 10-50x faster than the per-frame calculator path.

    Each output file has two columns: predicted  target
    (target is taken from the xyz file labels if present, otherwise 'nan').

    Outputs (GPUMD-compatible format):
        - energy_predict.out:  e_pred  e_target   (eV/atom, per frame)
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
    batch_size : int
        Number of structures per GPU batch.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dt = torch.float64 if dtype == "float64" else torch.float32
    np_dtype = np.float64 if dtype == "float64" else np.float32

    calc = NEPCalculator(model_file, dtype=dt, device=device)
    frames = read_xyz(xyz_file)

    # Build all neighbor lists on CPU (once)
    structures = _preprocess_for_prediction(frames, calc, np_dtype)

    os.makedirs(output_dir, exist_ok=True)
    energy_file = open(os.path.join(output_dir, "energy_predict.out"), "w")
    force_file  = open(os.path.join(output_dir, "force_predict.out"),  "w")
    virial_file = open(os.path.join(output_dir, "virial_predict.out"), "w")

    energy_file.write("# energy_pred(eV/atom)  energy_target(eV/atom)\n")
    force_file.write("# fx_pred  fy_pred  fz_pred  fx_target  fy_target  fz_target\n")
    virial_file.write(
        "# xx_pred yy_pred zz_pred xy_pred yz_pred zx_pred  "
        "xx_tgt yy_tgt zz_tgt xy_tgt yz_tgt zx_tgt\n")

    n_frames = len(structures)
    atom_offset = 0  # track which atom row we're on across batches

    try:
        with torch.no_grad():
            for start in range(0, n_frames, batch_size):
                indices = list(range(start, min(start + batch_size, n_frames)))
                batch = _build_batch(structures, indices, calc, dt, device)
                result = calc.compute_batch(batch)

                Etot = result["Etot"].cpu()
                forces_all = result["forces"].cpu().numpy()
                virial_all = result["virial"].cpu().numpy()

                # Accumulate per-structure outputs
                f_ptr = 0
                for k, si in enumerate(indices):
                    natoms = structures[si]["natoms"]
                    e_pred = Etot[k].item() / natoms

                    # Energy
                    e_ref = structures[si]["energy"]
                    e_ref_str = (f"{float(e_ref) / natoms:.10f}"
                                 if e_ref is not None else "nan")
                    energy_file.write(f"{e_pred:.10f}  {e_ref_str}\n")

                    # Forces
                    forces_pred = forces_all[f_ptr:f_ptr + natoms]
                    forces_ref = structures[si]["forces"]
                    for j in range(natoms):
                        f = forces_pred[j]
                        if forces_ref is not None:
                            r = np.asarray(forces_ref)[j]
                            force_file.write(
                                f"{f[0]:.10f} {f[1]:.10f} {f[2]:.10f}  "
                                f"{r[0]:.10f} {r[1]:.10f} {r[2]:.10f}\n")
                        else:
                            force_file.write(
                                f"{f[0]:.10f} {f[1]:.10f} {f[2]:.10f}  "
                                f"nan nan nan\n")

                    # Virial — sum per-atom virial to get per-structure
                    v_sum = virial_all[f_ptr:f_ptr + natoms].sum(axis=0)
                    # index map: xx=0,yy=4,zz=8,xy=1,yz=5,zx=2 (9-component)
                    vp = [v_sum[0], v_sum[4], v_sum[8],
                          v_sum[1], v_sum[5], v_sum[2]]
                    v_ref = structures[si]["virial"]
                    if v_ref is not None:
                        vr = np.asarray(v_ref).flatten()
                        if vr.size == 9:
                            vr6 = [vr[0], vr[4], vr[8], vr[1], vr[5], vr[2]]
                        else:
                            vr6 = vr[:6].tolist()
                    else:
                        vr6 = [float("nan")] * 6
                    virial_file.write(
                        "  ".join(f"{x:.10f}" for x in vp) + "    " +
                        "  ".join(f"{x}" for x in vr6) + "\n")

                    f_ptr += natoms
    finally:
        energy_file.close()
        force_file.close()
        virial_file.close()

    print(f"  {n_frames} frames predicted → {output_dir}/"
          "{energy,force,virial}_predict.out")
