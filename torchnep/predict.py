"""
Full-dataset prediction for NEP models.

Pipeline:
  1. read_xyz
  2. preprocess_structures(...)  — multi-process neighbor lists (CPU)
  3. concatenate all structures into flat tensors uploaded to the GPU once
  4. batched compute_batch loop (basis recomputed per batch, no host transfer)
  5. vectorised numpy.savetxt for outputs (per-atom virial matches GPUMD)
"""

import os
import time
import torch
import numpy as np

from .nep import NEPCalculator
from .data import read_xyz, build_neighbor_list_np
from . import ops


# ---------------------------------------------------------------------------
# Single-frame helpers — kept so tests/test_backward.py and external callers
# that build a one-off batch by hand keep working. The fast prediction path
# (predict_dataset) does not use them.
# ---------------------------------------------------------------------------

def _preprocess_for_prediction(frames, calc, np_dtype):
    """Build neighbor lists for a list of frames. Returns structure dicts
    with the same schema the training pipeline produces."""
    rc_rad, rc_ang = calc.rc_radial, calc.rc_angular
    max_rc = max(rc_rad, rc_ang)

    structures = []
    for frame in frames:
        positions = np.asarray(frame["positions"], dtype=np_dtype)
        cell = np.asarray(frame["cell"], dtype=np_dtype)
        atom_types = np.array(
            [calc.type_names.index(s) for s in frame["species"]],
            dtype=np.int64)
        pair_i, pair_j, rij = build_neighbor_list_np(positions, cell, max_rc)
        dij = np.linalg.norm(rij, axis=1)
        structures.append({
            "natoms": frame["natoms"],
            "atom_types": atom_types,
            "pair_i_rad": pair_i[dij < rc_rad],
            "pair_j_rad": pair_j[dij < rc_rad],
            "rij_rad":    rij[dij < rc_rad],
            "pair_i_ang": pair_i[dij < rc_ang],
            "pair_j_ang": pair_j[dij < rc_ang],
            "rij_ang":    rij[dij < rc_ang],
            "energy": frame.get("energy"),
            "forces": frame.get("forces"),
            "virial": frame.get("virial"),
        })
    return structures


def _build_batch(structures, indices, calc, dtype, device):
    """Collate a list of structure indices into a GPU batch with cached basis,
    matching the dict shape NEPCalculator.compute_batch expects."""
    rc_rad, rc_ang = calc.rc_radial, calc.rc_angular
    basis_r, basis_a = calc.basis_size_radial, calc.basis_size_angular
    l_max_3b, num_lm = calc.l_max_3b, calc.num_lm

    natoms_list = [structures[i]["natoms"] for i in indices]
    N_total = sum(natoms_list)
    B = len(indices)
    offsets = [0]
    for n in natoms_list:
        offsets.append(offsets[-1] + n)

    atom_types = torch.tensor(
        np.concatenate([structures[i]["atom_types"] for i in indices]),
        dtype=torch.long, device=device)
    struct_idx = torch.cat([
        torch.full((natoms_list[k],), k, dtype=torch.long, device=device)
        for k in range(B)])

    def _cat_int(key):
        return torch.tensor(
            np.concatenate([structures[indices[k]][key] + offsets[k]
                            for k in range(B)]).astype(np.int64),
            dtype=torch.long, device=device)

    def _cat_rij(key):
        return torch.tensor(
            np.concatenate([structures[indices[k]][key] for k in range(B)]),
            dtype=dtype, device=device)

    pi_r, pj_r = _cat_int("pair_i_rad"), _cat_int("pair_j_rad")
    rij_r = _cat_rij("rij_rad")
    pi_a, pj_a = _cat_int("pair_i_ang"), _cat_int("pair_j_ang")
    rij_a = _cat_rij("rij_ang")

    dr = torch.norm(rij_r, dim=-1)
    fk_r, fkp_r = ops.chebyshev_basis_and_deriv(dr, rc_rad, basis_r)
    d12inv_r = 1.0 / dr.clamp(min=1e-10)

    if rij_a.shape[0] > 0:
        da = torch.norm(rij_a, dim=-1)
        fk_a, fkp_a = ops.chebyshev_basis_and_deriv(da, rc_ang, basis_a)
        d12inv_a = 1.0 / da.clamp(min=1e-10)
        blm = ops.angular_basis(rij_a[:, 0] * d12inv_a,
                                rij_a[:, 1] * d12inv_a,
                                rij_a[:, 2] * d12inv_a, l_max_3b)
    else:
        fk_a = torch.zeros(0, basis_a + 1, dtype=dtype, device=device)
        fkp_a = torch.zeros(0, basis_a + 1, dtype=dtype, device=device)
        d12inv_a = torch.zeros(0, dtype=dtype, device=device)
        blm = torch.zeros(0, num_lm, dtype=dtype, device=device)

    return {
        "N": N_total, "num_structures": B,
        "atom_types": atom_types, "struct_idx": struct_idx,
        "pair_i_rad": pi_r, "pair_j_rad": pj_r, "rij_rad": rij_r,
        "fk_rad": fk_r, "fkp_rad": fkp_r, "d12inv_rad": d12inv_r,
        "pair_i_ang": pi_a, "pair_j_ang": pj_a, "rij_ang": rij_a,
        "fk_ang": fk_a, "fkp_ang": fkp_a, "d12inv_ang": d12inv_a,
        "blm": blm,
    }


def _virial9_to_6(v9):
    """Re-order a length-9 virial (xx,xy,xz,yx,yy,yz,zx,zy,zz)
    into the GPUMD 6-vector (xx,yy,zz,xy,yz,zx)."""
    out = np.empty((v9.shape[0], 6), dtype=v9.dtype)
    out[:, 0] = v9[:, 0]   # xx
    out[:, 1] = v9[:, 4]   # yy
    out[:, 2] = v9[:, 8]   # zz
    out[:, 3] = v9[:, 1]   # xy
    out[:, 4] = v9[:, 5]   # yz
    out[:, 5] = v9[:, 2]   # zx
    return out


def predict_dataset(
    model_file: str,
    xyz_file: str,
    output_dir: str = ".",
    dtype: str = "float64",
    device: str = None,
    batch_size: int = 1000,
    verbose: bool = True,
    backend: str = "auto",
):
    """Run batched prediction on a full dataset and save GPUMD-format outputs.

    Outputs (per-atom for energy and virial; per-atom raw for forces):
      - energy_predict.out:  e_pred  e_target              (eV/atom, per frame)
      - force_predict.out:   fx fy fz  fx_t fy_t fz_t      (eV/A, per atom)
      - virial_predict.out:  xx yy zz xy yz zx (pred, ref) (eV/atom, per frame)

    The format mirrors GPUMD's *_train.out files, so the two can be diffed
    column by column.

    ``backend`` ∈ {"auto", "loop", "fast", "cuda"} — see
    ``torchnep.ops.resolve_backend``.
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

    def _log(msg):
        if verbose:
            print(msg)

    t_total = time.time()
    calc = NEPCalculator(model_file, dtype=dt, device=device)
    rc_rad, rc_ang = calc.rc_radial, calc.rc_angular
    basis_r, basis_a = calc.basis_size_radial, calc.basis_size_angular
    l_max_3b = calc.l_max_3b
    num_lm = calc.num_lm

    # Resolve backend now that num_types is known. "auto" probes the kernel
    # if ntypes < 8, falls back to "fast" if nvcc is unavailable.
    backend = ops.resolve_backend(backend, num_types=calc.num_types)
    _log(f"  backend: {backend}")

    # 1) Read xyz
    t0 = time.time()
    frames = read_xyz(xyz_file)
    n_struct = len(frames)
    _log(f"  read_xyz:    {time.time() - t0:5.1f}s   ({n_struct} frames)")

    # 2) Multi-process neighbor-list construction
    from .train import preprocess_structures  # local import: avoid cycle

    t0 = time.time()
    pp_config = {
        "cutoff_radial": rc_rad,
        "cutoff_angular": rc_ang,
        "type_names": calc.type_names,
    }
    structures = preprocess_structures(frames, pp_config, np_dtype)
    del frames
    _log(f"  neighbors:   {time.time() - t0:5.1f}s")

    # 3) Concatenate everything once and move raw data to the GPU.
    t0 = time.time()
    natoms_arr = np.asarray([s["natoms"] for s in structures], dtype=np.int64)
    nrad_arr = np.asarray([len(s["pair_i_rad"]) for s in structures],
                          dtype=np.int64)
    nang_arr = np.asarray([len(s["pair_i_ang"]) for s in structures],
                          dtype=np.int64)
    nat_cum = np.concatenate([[0], np.cumsum(natoms_arr)])
    nrad_cum = np.concatenate([[0], np.cumsum(nrad_arr)])
    nang_cum = np.concatenate([[0], np.cumsum(nang_arr)])
    N_atoms_total = int(nat_cum[-1])

    # Pair indices are global (atom positions in the concatenation), so a
    # per-batch slice only needs to subtract the batch's first-atom offset.
    all_at = np.concatenate([s["atom_types"] for s in structures])
    all_pi_r = np.concatenate(
        [s["pair_i_rad"] + nat_cum[i] for i, s in enumerate(structures)])
    all_pj_r = np.concatenate(
        [s["pair_j_rad"] + nat_cum[i] for i, s in enumerate(structures)])
    all_rij_r = np.concatenate([s["rij_rad"] for s in structures])
    all_pi_a = np.concatenate(
        [s["pair_i_ang"] + nat_cum[i] for i, s in enumerate(structures)])
    all_pj_a = np.concatenate(
        [s["pair_j_ang"] + nat_cum[i] for i, s in enumerate(structures)])
    all_rij_a = np.concatenate([s["rij_ang"] for s in structures])

    energy_ref = np.array(
        [s["energy"] if s.get("energy") is not None else np.nan
         for s in structures], dtype=np.float64)

    has_forces_global = any(s.get("forces") is not None for s in structures)
    forces_ref = None
    if has_forces_global:
        forces_ref = np.full((N_atoms_total, 3), np.nan, dtype=np.float64)
        for i, s in enumerate(structures):
            f = s.get("forces")
            if f is not None:
                forces_ref[nat_cum[i]:nat_cum[i + 1]] = \
                    np.asarray(f).reshape(-1, 3)

    has_virial_global = any(s.get("virial") is not None for s in structures)
    virial_ref = np.full((n_struct, 6), np.nan, dtype=np.float64)
    if has_virial_global:
        # Per-atom virial, to match GPUMD's *_train.out columns.
        for i, s in enumerate(structures):
            v = s.get("virial")
            if v is None:
                continue
            v = np.asarray(v).flatten()
            inv_n = 1.0 / float(s["natoms"])
            if v.size == 9:
                virial_ref[i] = [v[0] * inv_n, v[4] * inv_n, v[8] * inv_n,
                                 v[1] * inv_n, v[5] * inv_n, v[2] * inv_n]
            elif v.size >= 6:
                virial_ref[i] = v[:6] * inv_n

    del structures  # free CPU memory

    at_gpu    = torch.from_numpy(all_at).to(device=device)
    pi_r_gpu  = torch.from_numpy(all_pi_r).to(device=device)
    pj_r_gpu  = torch.from_numpy(all_pj_r).to(device=device)
    rij_r_gpu = torch.from_numpy(all_rij_r).to(device=device, dtype=dt)
    pi_a_gpu  = torch.from_numpy(all_pi_a).to(device=device)
    pj_a_gpu  = torch.from_numpy(all_pj_a).to(device=device)
    rij_a_gpu = torch.from_numpy(all_rij_a).to(device=device, dtype=dt)
    natoms_gpu = torch.from_numpy(natoms_arr).to(device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    _log(f"  upload:      {time.time() - t0:5.1f}s")
    del all_at, all_pi_r, all_pj_r, all_rij_r, all_pi_a, all_pj_a, all_rij_a

    # 4) Batched compute loop ------------------------------------------------
    e_pred_arr = np.empty(n_struct, dtype=np.float64)
    f_pred_arr = (np.empty((N_atoms_total, 3), dtype=np.float64)
                  if has_forces_global else None)
    v_pred_arr = np.empty((n_struct, 6), dtype=np.float64)

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n_struct, batch_size):
            end = min(start + batch_size, n_struct)
            B = end - start
            a_lo, a_hi = int(nat_cum[start]), int(nat_cum[end])
            r_lo, r_hi = int(nrad_cum[start]), int(nrad_cum[end])
            g_lo, g_hi = int(nang_cum[start]), int(nang_cum[end])
            N = a_hi - a_lo

            atom_types = at_gpu[a_lo:a_hi]
            pi_r = pi_r_gpu[r_lo:r_hi] - a_lo
            pj_r = pj_r_gpu[r_lo:r_hi] - a_lo
            rij_r = rij_r_gpu[r_lo:r_hi]
            pi_a = pi_a_gpu[g_lo:g_hi] - a_lo
            pj_a = pj_a_gpu[g_lo:g_hi] - a_lo
            rij_a = rij_a_gpu[g_lo:g_hi]

            struct_idx = torch.repeat_interleave(
                torch.arange(B, device=device, dtype=torch.long),
                natoms_gpu[start:end])

            dr = torch.norm(rij_r, dim=-1)
            fk_r, fkp_r = ops.chebyshev_basis_and_deriv(dr, rc_rad, basis_r)
            d12inv_r = 1.0 / dr.clamp(min=1e-10)

            if rij_a.shape[0] > 0:
                da = torch.norm(rij_a, dim=-1)
                fk_a, fkp_a = ops.chebyshev_basis_and_deriv(
                    da, rc_ang, basis_a)
                d12inv_a = 1.0 / da.clamp(min=1e-10)
                blm = ops.angular_basis(
                    rij_a[:, 0] * d12inv_a,
                    rij_a[:, 1] * d12inv_a,
                    rij_a[:, 2] * d12inv_a,
                    l_max_3b)
            else:
                fk_a = torch.zeros(0, basis_a + 1, dtype=dt, device=device)
                fkp_a = torch.zeros(0, basis_a + 1, dtype=dt, device=device)
                d12inv_a = torch.zeros(0, dtype=dt, device=device)
                blm = torch.zeros(0, num_lm, dtype=dt, device=device)

            batch = {
                "N": N, "num_structures": B,
                "atom_types": atom_types, "struct_idx": struct_idx,
                "pair_i_rad": pi_r, "pair_j_rad": pj_r, "rij_rad": rij_r,
                "fk_rad": fk_r, "fkp_rad": fkp_r, "d12inv_rad": d12inv_r,
                "pair_i_ang": pi_a, "pair_j_ang": pj_a, "rij_ang": rij_a,
                "fk_ang": fk_a, "fkp_ang": fkp_a, "d12inv_ang": d12inv_a,
                "blm": blm,
            }
            result = calc.compute_batch(batch, backend=backend)

            Etot = result["Etot"]
            v_per_frame = torch.zeros(B, 9, dtype=dt, device=device)
            v_per_frame.scatter_add_(
                0, struct_idx.unsqueeze(-1).expand(-1, 9), result["virial"])

            # Single H2D copy per batch for the per-frame outputs
            Etot_np = Etot.cpu().numpy()
            v_np = v_per_frame.cpu().numpy()
            nat_slice = natoms_arr[start:end].astype(np.float64)
            e_pred_arr[start:end] = Etot_np / nat_slice
            v_pred = _virial9_to_6(v_np) / nat_slice[:, None]
            v_pred_arr[start:end] = v_pred

            if f_pred_arr is not None:
                f_pred_arr[a_lo:a_hi] = result["forces"].cpu().numpy()

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    _log(f"  compute:     {time.time() - t0:5.1f}s")

    # 5) Vectorised text output --------------------------------------------
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)

    e_ref_pa = energy_ref / natoms_arr.astype(np.float64)
    np.savetxt(os.path.join(output_dir, "energy_predict.out"),
               np.column_stack([e_pred_arr, e_ref_pa]), fmt="%.10g")

    if forces_ref is None:
        forces_ref = np.full((N_atoms_total, 3), np.nan, dtype=np.float64)
    if f_pred_arr is None:
        f_pred_arr = np.zeros((N_atoms_total, 3), dtype=np.float64)
    np.savetxt(os.path.join(output_dir, "force_predict.out"),
               np.column_stack([f_pred_arr, forces_ref]), fmt="%.10g")

    np.savetxt(os.path.join(output_dir, "virial_predict.out"),
               np.column_stack([v_pred_arr, virial_ref]), fmt="%.10g")
    _log(f"  write:       {time.time() - t0:5.1f}s")

    _log(f"  TOTAL:       {time.time() - t_total:5.1f}s   "
         f"→ {output_dir}/(energy|force|virial)_predict.out")


# ---------------------------------------------------------------------------
# End-of-training prediction that reuses the in-memory model + GPUDataStore
# (no xyz re-read, no neighbor-list rebuild, no second GPU upload).
# ---------------------------------------------------------------------------

def predict_from_store(model, data_store, output_dir: str,
                       batch_size: int = 1000,
                       backend: str = "auto",
                       verbose: bool = True):
    """Run prediction using an already-loaded NEPModel + GPUDataStore.

    Designed for the end of training: reuses the preprocessed data_store so
    there is no xyz re-read / neighbor-list rebuild / GPU upload. The
    prediction dtype matches the training dtype (= data_store dtype).

    Writes GPUMD-format outputs in ``output_dir`` (same columns and format as
    ``predict_dataset``):
      energy_predict.out  — per-frame (pred, ref) in eV/atom
      force_predict.out   — per-atom (fx,fy,fz pred, ref) in eV/Å
      virial_predict.out  — per-frame (xx,yy,zz,xy,yz,zx pred, ref) in eV/atom
    """
    def _log(msg):
        if verbose:
            print(msg)

    dev   = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n_struct = data_store.n
    backend = ops.resolve_backend(backend, num_types=model.num_types)

    nat_arr = np.asarray(data_store.natoms, dtype=np.int64)
    nat_cum = np.concatenate([[0], np.cumsum(nat_arr)])
    N_atoms_total = int(nat_cum[-1])

    e_pred = np.empty(n_struct, dtype=np.float64)
    f_pred = np.full((N_atoms_total, 3), np.nan, dtype=np.float64)
    v_pred = np.empty((n_struct, 6), dtype=np.float64)

    was_training = model.training
    model.eval()
    t_compute = time.time()
    with torch.no_grad():
        for start in range(0, n_struct, batch_size):
            end = min(start + batch_size, n_struct)
            idx = list(range(start, end))
            B = end - start
            batch = data_store.collate(idx)
            r = model.compute_properties_cached(
                batch, need_forces=True, need_virial=True, backend=backend)

            nat_slice = nat_arr[start:end].astype(np.float64)
            e_pred[start:end] = r["Etot"].cpu().numpy() / nat_slice

            # Sum per-atom (N,9) virial into per-frame (B,9), then reorder.
            v_per = torch.zeros(B, 9, dtype=dtype, device=dev)
            v_per.scatter_add_(0,
                batch["struct_idx"].unsqueeze(-1).expand(-1, 9), r["virial"])
            v9 = v_per.cpu().numpy()
            v_pred[start:end] = _virial9_to_6(v9) / nat_slice[:, None]

            a_lo = int(nat_cum[start]); a_hi = int(nat_cum[end])
            f_pred[a_lo:a_hi] = r["forces"].cpu().numpy()
    if was_training:
        model.train()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    _log(f"  compute:  {time.time() - t_compute:5.1f}s")

    # Reference values (from data_store; only fill where the flag is set).
    energy_ref = np.array(
        [data_store.energy[i] if data_store.has_energy_flag[i] else np.nan
         for i in range(n_struct)], dtype=np.float64)
    e_ref_pa = energy_ref / nat_arr.astype(np.float64)

    forces_ref = np.full((N_atoms_total, 3), np.nan, dtype=np.float64)
    for i in range(n_struct):
        if data_store.has_forces_flag[i]:
            a_lo, a_hi = int(nat_cum[i]), int(nat_cum[i + 1])
            forces_ref[a_lo:a_hi] = data_store.forces[i].cpu().numpy()

    virial_ref = np.full((n_struct, 6), np.nan, dtype=np.float64)
    for i in range(n_struct):
        if data_store.has_virial_flag[i]:
            v9 = data_store.virial[i].cpu().numpy().flatten()  # length-9
            n = float(nat_arr[i])
            virial_ref[i] = [v9[0]/n, v9[4]/n, v9[8]/n,
                             v9[1]/n, v9[5]/n, v9[2]/n]

    os.makedirs(output_dir, exist_ok=True)
    t_write = time.time()
    np.savetxt(os.path.join(output_dir, "energy_predict.out"),
               np.column_stack([e_pred, e_ref_pa]), fmt="%.10g")
    np.savetxt(os.path.join(output_dir, "force_predict.out"),
               np.column_stack([f_pred, forces_ref]), fmt="%.10g")
    np.savetxt(os.path.join(output_dir, "virial_predict.out"),
               np.column_stack([v_pred, virial_ref]), fmt="%.10g")
    _log(f"  write:    {time.time() - t_write:5.1f}s   "
         f"→ {output_dir}/(energy|force|virial)_predict.out")
