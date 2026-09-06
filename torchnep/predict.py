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

"""
Full-dataset prediction for NEP models.

Pipeline (streamed, 1.0.3a2):
  1. index_xyz — byte offsets + atom counts, one I/O pass, no parsing
  2. per chunk of ~chunk_atoms atoms: read_xyz_at -> preprocess_structures
     (multi-process neighbor lists, CPU) -> flat host arrays
  3. batched compute_batch loop over the chunk (basis recomputed per batch,
     only the batch slice goes to the device; auto batch size + OOM retry)
  4. numpy.savetxt appends the chunk's rows (per-atom virial matches GPUMD)
Host memory is bounded by the chunk, device memory by the batch: any dataset
finishes on any machine, only the wall time differs.
"""

import os
import time
import torch
import numpy as np

from .nep import NEPCalculator
from . import ops


# GPUMD writes this sentinel into the reference column of virial_train.out /
# stress_train.out for structures that carry no reference virial (see GPUMD
# src/main_nep/structure.cu: ``structure.virial[m] = -1e6``). For stress it
# also skips the unit conversion when the reference is below -1e5 (fitness.cu,
# ``if (ref_value > -1e5)``). We mirror both so a missing virial reads as
# -1e6 (not NaN) and the files stay drop-in comparable for third-party tools.
_MISSING_VIRIAL = -1e6
_MISSING_VIRIAL_CUTOFF = -1e5


def _stress_from_virial(virial_pa, nat_col, vol_col, keep_missing):
    """Per-atom virial (eV/atom) -> stress (GPa), matching GPUMD's sign:
    ``stress = +virial_total / V`` (so stress shares virial's sign, unlike a
    physical Cauchy stress).

    ``virial_pa``   : (n, 6) per-atom virial.
    ``nat_col``     : (n, 1) atom counts.
    ``vol_col``     : (n, 1) cell volumes (<=0 treated as 1 to avoid div-by-0).
    ``keep_missing``: if True, rows holding the -1e6 missing-virial sentinel are
                      passed through unscaled (GPUMD's reference-column rule).
    """
    from .constants import EV_PER_A3_TO_GPa
    vol_safe = np.where(vol_col > 0, vol_col, 1.0)
    stress = virial_pa * (nat_col / vol_safe * EV_PER_A3_TO_GPa)
    if keep_missing:
        stress = np.where(virial_pa > _MISSING_VIRIAL_CUTOFF, stress, virial_pa)
    return stress


def _virial9_to_6(v9):
    r"""Re-order length-9 row-major virial (xx,xy,xz,yx,yy,yz,zx,zy,zz) into
    the GPUMD 6-vector (xx,yy,zz,xy,yz,zx).

    Picks single-triangular components to match GPUMD's per-atom virial
    convention (src/main_nep/nep.cu: s_virial_xy = -r[0]*f[1], s_virial_yz
    = -r[1]*f[2], s_virial_zx = -r[2]*f[0]). GPUMD does not average with
    the symmetric partner — it stores only one entry of each off-diagonal
    pair during force accumulation. Per-FRAME totals are still symmetric
    because the sum \Sigma -r_\alpha f_\beta over directed pairs equals
    \Sigma -r_\beta f_\alpha by Newton's 3rd law, so this choice is
    numerically equivalent to averaging at the per-frame output level but
    gives cheaper bit-identical match with GPUMD outputs."""
    out = np.empty((v9.shape[0], 6), dtype=v9.dtype)
    out[:, 0] = v9[:, 0]   # xx = -r_x f_x
    out[:, 1] = v9[:, 4]   # yy = -r_y f_y
    out[:, 2] = v9[:, 8]   # zz = -r_z f_z
    out[:, 3] = v9[:, 1]   # xy = -r_x f_y
    out[:, 4] = v9[:, 5]   # yz = -r_y f_z
    out[:, 5] = v9[:, 6]   # zx = -r_z f_x
    return out


class _Progress:
    """Frames-done progress line for predict_dataset: tqdm when installed,
    otherwise a dependency-free single-line bar. Silent when verbose=False."""

    def __init__(self, total, enabled):
        self.total, self.enabled, self.n, self.t0 = total, enabled, 0, time.time()
        self.bar = None
        if enabled:
            try:
                from tqdm import tqdm
                self.bar = tqdm(total=total, unit="frame", unit_scale=True,
                                desc="  predict", dynamic_ncols=True, leave=True)
            except Exception:
                self.bar = None
                self._draw()

    def update(self, k):
        if not self.enabled:
            return
        self.n += k
        if self.bar is not None:
            self.bar.update(k)
        else:
            self._draw()

    def _draw(self):
        frac = self.n / max(1, self.total); el = time.time() - self.t0
        eta = el / frac - el if frac > 0 else 0.0
        width = 30; done = int(width * frac)
        print(f"\r  predict [{'#' * done}{'-' * (width - done)}] {self.n}/{self.total} frames "
              f"{100 * frac:5.1f}%  {el:6.1f}s  ETA {eta:6.1f}s", end="", flush=True)

    def close(self):
        if not self.enabled:
            return
        if self.bar is not None:
            self.bar.close()
        else:
            print(flush=True)

    def write(self, msg):
        if not self.enabled:
            return
        if self.bar is not None:
            self.bar.write(msg)
        else:
            print("\r" + msg + " " * 20, flush=True); self._draw()


def _chunk_bounds(natoms, chunk_atoms, max_frames):
    """Split frames [0, n) into consecutive chunks holding ~``chunk_atoms`` atoms
    (at most ``max_frames`` frames, always >= 1 frame). Returns (lo, hi) pairs."""
    bounds, lo, acc = [], 0, 0
    for i, n in enumerate(natoms):
        acc += int(n)
        if acc >= chunk_atoms or (i + 1 - lo) >= max_frames:
            bounds.append((lo, i + 1)); lo, acc = i + 1, 0
    if lo < len(natoms):
        bounds.append((lo, len(natoms)))
    return bounds


def _chunk_arrays(structures, np_dtype):
    """Concatenate one chunk of preprocessed structures into flat host arrays."""
    natoms_arr = np.asarray([s["natoms"] for s in structures], dtype=np.int64)
    volumes_arr = np.asarray([s.get("volume", 0.0) for s in structures],
                             dtype=np.float64)
    nrad_arr = np.asarray([len(s["pair_i_rad"]) for s in structures],
                          dtype=np.int64)
    nang_arr = np.asarray([len(s["pair_i_ang"]) for s in structures],
                          dtype=np.int64)
    nat_cum = np.concatenate([[0], np.cumsum(natoms_arr)])
    c = dict(
        natoms=natoms_arr, volumes=volumes_arr, nrad=nrad_arr, nang=nang_arr,
        nat_cum=nat_cum,
        nrad_cum=np.concatenate([[0], np.cumsum(nrad_arr)]),
        nang_cum=np.concatenate([[0], np.cumsum(nang_arr)]),
        # Pair indices are global within the chunk (atom positions in the
        # concatenation); a batch slice only subtracts its first-atom offset.
        at=np.concatenate([s["atom_types"] for s in structures]),
        pi_r=np.concatenate([s["pair_i_rad"] + nat_cum[i] for i, s in enumerate(structures)]),
        pj_r=np.concatenate([s["pair_j_rad"] + nat_cum[i] for i, s in enumerate(structures)]),
        rij_r=np.concatenate([s["rij_rad"] for s in structures]),
        pi_a=np.concatenate([s["pair_i_ang"] + nat_cum[i] for i, s in enumerate(structures)]),
        pj_a=np.concatenate([s["pair_j_ang"] + nat_cum[i] for i, s in enumerate(structures)]),
        rij_a=np.concatenate([s["rij_ang"] for s in structures]),
    )
    n_struct, N_atoms = len(structures), int(nat_cum[-1])
    c["energy_ref"] = np.array(
        [s["energy"] if s.get("energy") is not None else np.nan
         for s in structures], dtype=np.float64)
    forces_ref = np.full((N_atoms, 3), np.nan, dtype=np.float64)
    for i, s in enumerate(structures):
        f = s.get("forces")
        if f is not None:
            forces_ref[nat_cum[i]:nat_cum[i + 1]] = np.asarray(f).reshape(-1, 3)
    c["forces_ref"] = forces_ref
    # Per-atom reference virial, GPUMD 6-vector (xx yy zz xy yz zx). The
    # off-diagonal pick matches _virial9_to_6 so both columns share one
    # definition; frames without a virial carry the -1e6 sentinel.
    virial_ref = np.full((n_struct, 6), _MISSING_VIRIAL, dtype=np.float64)
    for i, s in enumerate(structures):
        v = s.get("virial")
        if v is None:
            continue
        v = np.asarray(v).flatten()
        inv_n = 1.0 / float(s["natoms"])
        if v.size == 9:
            virial_ref[i] = [v[0] * inv_n, v[4] * inv_n, v[8] * inv_n,
                             v[1] * inv_n, v[5] * inv_n, v[6] * inv_n]
        elif v.size >= 6:
            virial_ref[i] = v[:6] * inv_n
    c["virial_ref"] = virial_ref
    return c


def _auto_batch_size(calc, dt, chunk, n_struct, log):
    """Bound the per-batch device footprint by a fraction of the free GPU
    memory, from the chunk's average per-frame pair counts and a conservative
    bytes-per-pair model of the batch tensors plus the kernels' intermediates
    (the safety factor absorbs backend differences). CPU/other devices: 1000."""
    try:
        free_b, _ = torch.cuda.mem_get_info()
    except Exception:
        return 1000
    esize = 8 if dt == torch.float64 else 4
    basis_r, basis_a = calc.basis_size_radial, calc.basis_size_angular
    num_lm, nmax_a = calc.num_lm, calc.n_max_angular
    avg_nrad = max(1.0, float(chunk["nrad"].mean()))
    avg_nang = max(1.0, float(chunk["nang"].mean()))
    avg_nat = max(1.0, float(chunk["natoms"].mean()))
    # dominant term: the analytical-force angular intermediates hold
    # ~(n_max_a+1) x num_lm floats per angular pair, a few times over.
    per_frame = (avg_nrad * (esize * (2 * (basis_r + 1) + 8) + 16)
                 + avg_nang * (esize * (3 * (nmax_a + 1) * num_lm
                                        + 2 * (basis_a + 1) + 2 * num_lm
                                        + 12) + 16)
                 + avg_nat * esize * 4 * calc.dim)
    bs = int(0.25 * free_b / (3.0 * per_frame))
    # Throughput is flat beyond a few thousand frames per batch (measured on
    # V100/MI250X), so cap the auto choice — larger batches only inflate the
    # memory peak.
    bs = max(16, min(bs, n_struct, 16384))
    log(f"  batch_size:  auto -> {bs} (free {free_b/2**30:.1f} GiB, "
        f"~{per_frame/1024:.0f} KiB/frame est.)")
    return bs


def _predict_chunk(calc, chunk, batch_size, device, dt, backend,
                   output_descriptor, log, progress=None):
    """Batched compute over one chunk. Returns (e_pred, f_pred, v_pred,
    d_pred, batch_size) — batch_size may have shrunk after an OOM retry."""
    n_struct = len(chunk["natoms"]); N_atoms = int(chunk["nat_cum"][-1])
    nat_cum, nrad_cum, nang_cum = chunk["nat_cum"], chunk["nrad_cum"], chunk["nang_cum"]
    natoms_arr = chunk["natoms"]
    rc_rad, rc_ang = calc.rc_radial, calc.rc_angular
    basis_r, basis_a = calc.basis_size_radial, calc.basis_size_angular
    # Host-resident staging: only each batch's slice is shipped to the device
    # inside the loop, so GPU memory scales with batch_size, not chunk size.
    at_cpu = torch.from_numpy(chunk["at"])
    pi_r_cpu, pj_r_cpu = torch.from_numpy(chunk["pi_r"]), torch.from_numpy(chunk["pj_r"])
    rij_r_cpu = torch.from_numpy(chunk["rij_r"])
    pi_a_cpu, pj_a_cpu = torch.from_numpy(chunk["pi_a"]), torch.from_numpy(chunk["pj_a"])
    rij_a_cpu = torch.from_numpy(chunk["rij_a"])
    natoms_gpu = torch.from_numpy(natoms_arr).to(device=device)

    e_pred = np.empty(n_struct, dtype=np.float64)
    f_pred = np.empty((N_atoms, 3), dtype=np.float64)
    v_pred = np.empty((n_struct, 6), dtype=np.float64)
    d_pred = None
    if output_descriptor == 1:
        d_pred = np.empty((n_struct, calc.dim), dtype=np.float64)
    elif output_descriptor == 2:
        d_pred = np.empty((N_atoms, calc.dim), dtype=np.float64)

    with torch.no_grad():
        start = 0
        while start < n_struct:
            end = min(start + batch_size, n_struct)
            B = end - start
            try:
                a_lo, a_hi = int(nat_cum[start]), int(nat_cum[end])
                r_lo, r_hi = int(nrad_cum[start]), int(nrad_cum[end])
                g_lo, g_hi = int(nang_cum[start]), int(nang_cum[end])
                N = a_hi - a_lo
                atom_types = at_cpu[a_lo:a_hi].to(device)
                pi_r = pi_r_cpu[r_lo:r_hi].to(device) - a_lo
                pj_r = pj_r_cpu[r_lo:r_hi].to(device) - a_lo
                rij_r = rij_r_cpu[r_lo:r_hi].to(device=device, dtype=dt)
                pi_a = pi_a_cpu[g_lo:g_hi].to(device) - a_lo
                pj_a = pj_a_cpu[g_lo:g_hi].to(device) - a_lo
                rij_a = rij_a_cpu[g_lo:g_hi].to(device=device, dtype=dt)
                struct_idx = torch.repeat_interleave(
                    torch.arange(B, device=device, dtype=torch.long),
                    natoms_gpu[start:end])

                dr = torch.norm(rij_r, dim=-1)
                fk_r, fkp_r = ops.chebyshev_basis_and_deriv(dr, rc_rad, basis_r)
                d12inv_r = 1.0 / dr.clamp(min=1e-10)
                if rij_a.shape[0] > 0:
                    da = torch.norm(rij_a, dim=-1)
                    fk_a, fkp_a = ops.chebyshev_basis_and_deriv(da, rc_ang, basis_a)
                    d12inv_a = 1.0 / da.clamp(min=1e-10)
                    blm = ops.angular_basis(rij_a[:, 0] * d12inv_a,
                                            rij_a[:, 1] * d12inv_a,
                                            rij_a[:, 2] * d12inv_a,
                                            calc.l_max_3b)
                else:
                    fk_a = torch.zeros(0, basis_a + 1, dtype=dt, device=device)
                    fkp_a = torch.zeros(0, basis_a + 1, dtype=dt, device=device)
                    d12inv_a = torch.zeros(0, dtype=dt, device=device)
                    blm = torch.zeros(0, calc.num_lm, dtype=dt, device=device)

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

                v_per_frame = torch.zeros(B, 9, dtype=dt, device=device)
                v_per_frame.scatter_add_(
                    0, struct_idx.unsqueeze(-1).expand(-1, 9), result["virial"])
                nat_slice = natoms_arr[start:end].astype(np.float64)
                e_pred[start:end] = result["Etot"].cpu().numpy() / nat_slice
                v_pred[start:end] = _virial9_to_6(v_per_frame.cpu().numpy()) / nat_slice[:, None]
                f_pred[a_lo:a_hi] = result["forces"].cpu().numpy()
                if output_descriptor:
                    # ``compute_batch`` already returns ``q * q_scaler``.
                    desc_np = result["descriptor"].cpu().numpy()
                    if output_descriptor == 2:
                        d_pred[a_lo:a_hi] = desc_np
                    else:
                        sums = np.zeros((B, calc.dim), dtype=np.float64)
                        np.add.at(sums, struct_idx.cpu().numpy(), desc_np)
                        d_pred[start:end] = sums / nat_slice[:, None]
            except torch.OutOfMemoryError:
                # Adaptive backoff: free the pool, halve the batch and redo
                # this batch — robust across models/hardware where any static
                # memory estimate can be off.
                if batch_size <= 16:
                    raise
                torch.cuda.empty_cache()
                batch_size = max(16, batch_size // 2)
                (progress.write if progress is not None else log)(
                    f"  OOM at batch of {B} frames -> retrying with batch_size={batch_size}")
                continue
            if progress is not None:
                progress.update(B)
            start = end
    return e_pred, f_pred, v_pred, d_pred, batch_size


def _predict_frames(calc, xyz_file, offsets, natoms, output_dir, dt, np_dtype,
                    device, backend, batch_size, chunk_atoms, energy_key,
                    output_descriptor, progress, log):
    """Streamed core shared by predict_dataset and predict_dataset_sharded:
    predict the frames at ``offsets`` (byte offsets from index_xyz, with their
    ``natoms``) chunk by chunk and append the rows to the ``*_train.out`` files
    in ``output_dir``. Returns (metric sums, timings, batch_size)."""
    from .train import preprocess_structures, make_preproc_pool
    from .data import read_xyz_at
    n_struct = len(offsets)
    bounds = _chunk_bounds(natoms, chunk_atoms, max_frames=50_000)
    pp_config = {"cutoff_radial": calc.rc_radial,
                 "cutoff_angular": calc.rc_angular,
                 "type_names": calc.type_names}
    os.makedirs(output_dir, exist_ok=True)
    fh = {name: open(os.path.join(output_dir, name), "w")
          for name in ("energy_train.out", "force_train.out",
                       "virial_train.out", "stress_train.out")}
    if output_descriptor:
        fh["descriptor.out"] = open(os.path.join(output_dir, "descriptor.out"), "w")
    acc = {"e": [0.0, 0.0, 0], "f": [0.0, 0.0, 0], "v": [0.0, 0.0, 0]}

    def _accum(key, d):
        d = d[np.isfinite(d)]
        if d.size:
            acc[key][0] += float(np.sum(d * d)); acc[key][1] += float(np.sum(np.abs(d)))
            acc[key][2] += int(d.size)

    tm = {"read": 0.0, "neighbors": 0.0, "compute": 0.0, "write": 0.0}
    # one neighbor-list worker pool for all chunks (fork; workers never touch CUDA)
    pool = make_preproc_pool() if n_struct >= 64 else None
    for lo, hi in bounds:
        t0 = time.time()
        frames = read_xyz_at(xyz_file, offsets[lo:hi], energy_key=energy_key)
        tm["read"] += time.time() - t0
        t0 = time.time()
        structures = preprocess_structures(frames, pp_config, np_dtype, pool=pool)
        del frames
        chunk = _chunk_arrays(structures, np_dtype)
        del structures
        tm["neighbors"] += time.time() - t0
        if batch_size is None:
            batch_size = _auto_batch_size(calc, dt, chunk, n_struct, progress.write) \
                if torch.device(device).type == "cuda" else 1000
        t0 = time.time()
        e_pred, f_pred, v_pred, d_pred, batch_size = _predict_chunk(
            calc, chunk, batch_size, device, dt, backend, output_descriptor, log, progress)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        tm["compute"] += time.time() - t0
        if "first_chunk" not in tm:
            tm["first_chunk"] = time.time() - t0     # includes kernel warm-up of this process
        t0 = time.time()
        nat = chunk["natoms"].astype(np.float64)
        e_ref_pa = chunk["energy_ref"] / nat
        np.savetxt(fh["energy_train.out"], np.column_stack([e_pred, e_ref_pa]), fmt="%.10g")
        np.savetxt(fh["force_train.out"], np.column_stack([f_pred, chunk["forces_ref"]]), fmt="%.10g")
        np.savetxt(fh["virial_train.out"], np.column_stack([v_pred, chunk["virial_ref"]]), fmt="%.10g")
        # Stress (GPa) = +virial_total / V * EV_PER_A3_TO_GPa, matching GPUMD's
        # convention so stress and virial carry the same sign (see
        # _stress_from_virial). Missing references keep the -1e6 sentinel unscaled.
        nat_col, vol_col = nat[:, None], chunk["volumes"][:, None]
        np.savetxt(fh["stress_train.out"], np.column_stack([
            _stress_from_virial(v_pred, nat_col, vol_col, keep_missing=False),
            _stress_from_virial(chunk["virial_ref"], nat_col, vol_col, keep_missing=True)]),
            fmt="%.10g")
        if d_pred is not None:
            np.savetxt(fh["descriptor.out"], d_pred, fmt="%.10g")
        _accum("e", e_pred - e_ref_pa)
        _accum("f", (f_pred - chunk["forces_ref"]).ravel())
        vmask = chunk["virial_ref"][:, 0] > _MISSING_VIRIAL / 2
        _accum("v", (v_pred[vmask] - chunk["virial_ref"][vmask]).ravel())
        tm["write"] += time.time() - t0
        del chunk, e_pred, f_pred, v_pred, d_pred
    for h in fh.values():
        h.close()
    if pool is not None:
        pool.close(); pool.join()
    return acc, tm, batch_size


def _metrics_table(acc, log):
    rows = []
    for key, label, per in (("e", "Energy (eV/atom)", "frames"),
                            ("f", "Force  (eV/A)", "atoms"),
                            ("v", "Virial (eV/atom)", "frames")):
        sq, ab, n = acc[key]
        if n:
            cov = n // 3 if key == "f" else (n // 6 if key == "v" else n)
            rows.append((label, np.sqrt(sq / n), ab / n, f"{cov} {per}"))
    if rows:
        log("  " + "-" * 58)
        log(f"  {'':18s} {'RMSE':>12s} {'MAE':>12s}")
        for label, rmse, mae, cov in rows:
            log(f"  {label:18s} {rmse:12.6f} {mae:12.6f}   ({cov})")
        log("  " + "-" * 58)


def _pick_device(device):
    if device is not None:
        return device
    # cuda probe also catches ROCm (PyTorch-HIP uses the cuda namespace).
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def predict_dataset(
    model_file: str,
    xyz_file: str,
    output_dir: str = ".",
    dtype: str = "float32",
    device: str = None,
    batch_size: int = None,
    verbose: bool = True,
    energy_key: str = "energy",
    output_descriptor: int = 0,
    chunk_atoms: int = None,
):
    """Run streamed, batched prediction on a full dataset and save GPUMD-format
    outputs.

    The file is first indexed (byte offsets + atom counts, one I/O pass, no
    parsing), then processed in consecutive **chunks** of roughly
    ``chunk_atoms`` atoms: each chunk is read, neighbor-listed (multi-process),
    predicted in device-sized batches and appended to the output files before
    the next chunk is touched. Host memory is therefore bounded by the chunk
    size and device memory by the batch size — any dataset finishes on any
    machine, only the wall time differs. Outputs are identical to the former
    whole-dataset path.

    Outputs (per-atom for energy and virial; per-atom raw for forces):
      - energy_train.out:  e_pred  e_target              (eV/atom, per frame)
      - force_train.out:   fx fy fz  fx_t fy_t fz_t      (eV/A, per atom)
      - virial_train.out:  xx yy zz xy yz zx (pred, ref) (eV/atom, per frame)
      - stress_train.out:  same layout in GPa (GPUMD sign convention)
      - descriptor.out (only when ``output_descriptor != 0``):
          mode 1 — per-frame averaged scaled descriptor, one row per frame
          mode 2 — per-atom scaled descriptor, one row per atom
        Matches GPUMD's ``output_descriptor`` / ``descriptor.out`` schema.

    The format mirrors GPUMD's *_train.out files, so the two can be diffed
    column by column. When reference labels are present, a summary of the
    energy/force/virial RMSE and MAE is printed at the end.

    Parameters
    ----------
    batch_size : int or None
        Frames per compute batch. ``None`` (default) sizes the batch
        automatically from the device's free memory and the first chunk's
        average pair counts (CUDA/ROCm; other devices fall back to 1000),
        with an OOM-halving retry.
    chunk_atoms : int or None
        Atoms per streamed chunk (host-memory bound; ~1 GB per 200k atoms of
        dense metal at cutoff 6/5). ``None`` -> ``TORCHNEP_PREDICT_CHUNK_ATOMS``
        env var, else 200000.
    output_descriptor : int
        0 — disabled (default).
        1 — write per-frame averaged ``q * q_scaler`` to descriptor.out.
        2 — write per-atom ``q * q_scaler`` to descriptor.out.

    The contraction backend is chosen automatically (see
    ``torchnep.ops.resolve_backend``).
    """
    from .train import _default_alloc_conf
    from .data import index_xyz
    _default_alloc_conf()
    device = _pick_device(device)
    dt = torch.float64 if dtype == "float64" else torch.float32
    np_dtype = np.float64 if dtype == "float64" else np.float32
    if chunk_atoms is None:
        chunk_atoms = int(os.environ.get("TORCHNEP_PREDICT_CHUNK_ATOMS", 200_000))
    chunk_atoms = max(1, int(chunk_atoms))

    def _log(msg):
        if verbose:
            print(msg, flush=True)

    t_total = time.time()
    calc = NEPCalculator(model_file, dtype=dt, device=device)
    backend = ops.resolve_backend("auto", num_types=calc.num_types,
                                  device_type=torch.device(device).type)
    # 1) Index the file (offsets + atom counts only; no parsing, O(n_frames) memory)
    t0 = time.time()
    offsets, natoms_all = index_xyz(xyz_file)
    n_struct = len(offsets)
    n_chunks = len(_chunk_bounds(natoms_all, chunk_atoms, max_frames=50_000))
    _log(f"  index_xyz:   {time.time() - t0:5.1f}s   ({n_struct} frames, "
         f"{int(natoms_all.sum())} atoms, {n_chunks} chunk(s) of "
         f"~{chunk_atoms} atoms, energy label: {energy_key})")
    # 2-5) streamed chunks -> rows appended to the output files
    progress = _Progress(n_struct, verbose)
    acc, tm, _ = _predict_frames(calc, xyz_file, offsets, natoms_all, output_dir, dt, np_dtype,
                                 device, backend, batch_size, chunk_atoms, energy_key,
                                 output_descriptor, progress, _log)
    progress.close()
    for k in ("read", "neighbors", "compute", "write"):
        _log(f"  {k + ':':12s} {tm[k]:5.1f}s" + (f"   (first chunk incl. warm-up {tm['first_chunk']:.1f}s)" if k == "compute" and "first_chunk" in tm else ""))
    _log(f"  TOTAL:       {time.time() - t_total:5.1f}s   "
         f"-> {output_dir}/(energy|force|virial|stress)_train.out")
    if verbose:
        _metrics_table(acc, _log)


def predict_dataset_sharded(
    model_file: str,
    xyz_file: str,
    output_dir: str = ".",
    dtype: str = "float32",
    batch_size: int = None,
    verbose: bool = True,
    energy_key: str = "energy",
    output_descriptor: int = 0,
    chunk_atoms: int = None,
):
    """Multi-rank (multi-GPU / multi-node) version of :func:`predict_dataset`.

    Launch one process per GPU with ``torchrun`` or ``srun`` (the launcher
    sets RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT). Rank 0
    indexes the file once and broadcasts the frame index; the frames are then
    split into WORLD_SIZE **contiguous** ranges balanced by atom count, every
    rank streams its own range exactly like ``predict_dataset`` (same chunking,
    auto batch size and OOM retry, bounded memory) into a per-rank part
    directory, and rank 0 finally concatenates the parts in rank order into
    the usual ``*_train.out`` files and prints the global E/F/V RMSE/MAE
    table. The output is identical to a single-process ``predict_dataset``.
    Rank 0 shows the progress of its own share; the other ranks are silent.
    With WORLD_SIZE == 1 (or no launcher) this is plain ``predict_dataset``.
    """
    import torch.distributed as dist
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return predict_dataset(model_file, xyz_file, output_dir=output_dir, dtype=dtype,
                               batch_size=batch_size, verbose=verbose, energy_key=energy_key,
                               output_descriptor=output_descriptor, chunk_atoms=chunk_atoms)
    from .train import _default_alloc_conf
    from .train_sharded import _register_pg_atexit
    from .data import index_xyz
    _default_alloc_conf()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    # ---- distributed init (same rules as train_nep_sharded) ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cuda_available = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if cuda_available else 0
    if cuda_available:
        gpu_id = local_rank % max(1, n_gpus)
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"
    if not dist.is_initialized():
        local_world_env = int(os.environ.get("LOCAL_WORLD_SIZE",
                              os.environ.get("SLURM_NTASKS_PER_NODE", world_size)))
        pg_backend = "nccl" if cuda_available and local_world_env <= n_gpus else "gloo"
        dist.init_process_group(backend=pg_backend,
                                device_id=torch.device(device) if cuda_available else None)
        _register_pg_atexit()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    is_main = rank == 0
    bcast_dev = torch.device(device) if dist.get_backend() == "nccl" else torch.device("cpu")

    def _log(msg):
        if verbose and is_main:
            print(msg, flush=True)

    dt = torch.float64 if dtype == "float64" else torch.float32
    np_dtype = np.float64 if dtype == "float64" else np.float32
    if chunk_atoms is None:
        chunk_atoms = int(os.environ.get("TORCHNEP_PREDICT_CHUNK_ATOMS", 200_000))
    chunk_atoms = max(1, int(chunk_atoms))
    t_total = time.time()

    # ---- rank 0 indexes, everyone gets (offsets, natoms) ----
    t0 = time.time()
    if is_main:
        offs_np, nat_np = index_xyz(xyz_file)
        n_all_t = torch.tensor([len(offs_np)], dtype=torch.long, device=bcast_dev)
    else:
        n_all_t = torch.zeros(1, dtype=torch.long, device=bcast_dev)
    dist.broadcast(n_all_t, 0)
    n_all = int(n_all_t.item())
    idx_t = torch.empty(2, n_all, dtype=torch.long, device=bcast_dev)
    if is_main:
        idx_t[0].copy_(torch.from_numpy(offs_np).to(bcast_dev))
        idx_t[1].copy_(torch.from_numpy(nat_np).to(bcast_dev))
    dist.broadcast(idx_t, 0)
    offsets, natoms_all = idx_t[0].cpu().numpy(), idx_t[1].cpu().numpy()
    del idx_t
    # contiguous ranges balanced by atoms (each rank reads one sequential byte range)
    cum = np.concatenate([[0], np.cumsum(natoms_all)])
    edges = np.searchsorted(cum, cum[-1] * np.arange(world_size + 1) / world_size)
    edges[0], edges[-1] = 0, n_all
    lo, hi = int(edges[rank]), int(edges[rank + 1])
    _log(f"  index_xyz:   {time.time() - t0:5.1f}s   ({n_all} frames, {int(cum[-1])} atoms; "
         f"{world_size} ranks, ~{int(cum[-1]) // world_size} atoms each, "
         f"chunks of ~{chunk_atoms} atoms, energy label: {energy_key})")

    calc = NEPCalculator(model_file, dtype=dt, device=device)
    backend = ops.resolve_backend("auto", num_types=calc.num_types,
                                  device_type=torch.device(device).type)
    t_setup = time.time() - t_total          # dist init + index broadcast + model load
    part_dir = os.path.join(output_dir, ".parts", f"rank{rank:04d}")
    progress = _Progress(hi - lo, verbose and is_main)
    acc, tm, _ = _predict_frames(calc, xyz_file, offsets[lo:hi], natoms_all[lo:hi], part_dir,
                                 dt, np_dtype, device, backend, batch_size, chunk_atoms,
                                 energy_key, output_descriptor, progress, _log)
    progress.close()
    t_pred = time.time() - t_total
    if verbose or os.environ.get("TORCHNEP_PREDICT_RANK_TIMING") == "1":
        print(f"  [rank {rank}/{world_size}] frames {lo}-{hi - 1} ({int(cum[hi] - cum[lo])} atoms): "
              f"setup {t_setup:.1f}s read {tm['read']:.1f}s neighbors {tm['neighbors']:.1f}s "
              f"compute {tm['compute']:.1f}s (first chunk {tm.get('first_chunk', 0.0):.1f}s) "
              f"write {tm['write']:.1f}s total {t_pred:.1f}s", flush=True)

    # ---- global metrics (sum of squares / abs / counts) ----
    vec = torch.tensor([v for k in ("e", "f", "v") for v in acc[k]], dtype=torch.float64,
                       device=bcast_dev)
    dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    g = vec.cpu().numpy()
    acc_g = {k: [float(g[3 * i]), float(g[3 * i + 1]), int(g[3 * i + 2])]
             for i, k in enumerate(("e", "f", "v"))}
    dist.barrier()
    # ---- rank 0 concatenates the parts in rank order ----
    if is_main:
        import shutil
        t0 = time.time()
        names = ["energy_train.out", "force_train.out", "virial_train.out", "stress_train.out"]
        if output_descriptor:
            names.append("descriptor.out")
        for name in names:
            with open(os.path.join(output_dir, name), "w") as dst:
                for r in range(world_size):
                    src = os.path.join(output_dir, ".parts", f"rank{r:04d}", name)
                    if os.path.exists(src):
                        with open(src) as s:
                            shutil.copyfileobj(s, dst, 1 << 22)
        shutil.rmtree(os.path.join(output_dir, ".parts"), ignore_errors=True)
        _log(f"  rank 0 share: read {tm['read']:.1f}s neighbors {tm['neighbors']:.1f}s "
             f"compute {tm['compute']:.1f}s write {tm['write']:.1f}s; "
             f"all ranks done in {t_pred:.1f}s; merge {time.time() - t0:.1f}s")
        _log(f"  TOTAL:       {time.time() - t_total:5.1f}s   "
             f"-> {output_dir}/(energy|force|virial|stress)_train.out")
        if verbose:
            _metrics_table(acc_g, _log)
    dist.barrier()


# ---------------------------------------------------------------------------
# End-of-training prediction that reuses the in-memory model + data store
# (no xyz re-read, no neighbor-list rebuild, no second GPU upload).
# ---------------------------------------------------------------------------

def predict_from_store(model, data_store, output_dir: str,
                       batch_size: int = 1000,
                                          verbose: bool = True,
                       suffix: str = "train"):
    """Run prediction using an already-loaded NEPModel + StreamDataStore.

    Designed for the end of training: reuses the preprocessed data_store so
    there is no xyz re-read / neighbor-list rebuild / GPU upload. The
    prediction dtype matches the training dtype (= data_store dtype).

    Writes GPUMD-format outputs in ``output_dir`` (same columns and format as
    ``predict_dataset``):
      energy_train.out  — per-frame (pred, ref) in eV/atom
      force_train.out   — per-atom (fx,fy,fz pred, ref) in eV/A
      virial_train.out  — per-frame (xx,yy,zz,xy,yz,zx pred, ref) in eV/atom

    ``suffix`` picks the file-name tail: "train" (default) for the training
    set, "test" for the validation set (GPUMD's *_test.out naming).
    """
    def _log(msg):
        if verbose:
            print(msg, flush=True)

    dev   = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n_struct = data_store.n
    backend = ops.resolve_backend("auto", num_types=model.num_types,
                                  device_type=dev.type)

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

    virial_ref = np.full((n_struct, 6), _MISSING_VIRIAL, dtype=np.float64)
    for i in range(n_struct):
        if data_store.has_virial_flag[i]:
            v9 = data_store.virial[i].cpu().numpy().flatten()  # length-9
            n = float(nat_arr[i])
            # Single-triangular pick to match GPUMD (see _virial9_to_6).
            virial_ref[i] = [
                v9[0] / n,   # xx
                v9[4] / n,   # yy
                v9[8] / n,   # zz
                v9[1] / n,   # xy
                v9[5] / n,   # yz
                v9[6] / n,   # zx
            ]

    os.makedirs(output_dir, exist_ok=True)
    t_write = time.time()
    np.savetxt(os.path.join(output_dir, f"energy_{suffix}.out"),
               np.column_stack([e_pred, e_ref_pa]), fmt="%.10g")
    np.savetxt(os.path.join(output_dir, f"force_{suffix}.out"),
               np.column_stack([f_pred, forces_ref]), fmt="%.10g")
    np.savetxt(os.path.join(output_dir, f"virial_{suffix}.out"),
               np.column_stack([v_pred, virial_ref]), fmt="%.10g")

    # Stress (GPa) = +virial_total / V * conversion (GPUMD sign; see
    # predict_dataset / _stress_from_virial). Missing refs keep -1e6 unscaled.
    vol_arr = data_store.volumes.detach().cpu().numpy().astype(np.float64)
    nat_col = nat_arr.astype(np.float64)[:, None]
    vol_col = vol_arr[:, None]
    stress_pred = _stress_from_virial(v_pred, nat_col, vol_col,
                                      keep_missing=False)
    stress_ref = _stress_from_virial(virial_ref, nat_col, vol_col,
                                     keep_missing=True)
    np.savetxt(os.path.join(output_dir, f"stress_{suffix}.out"),
               np.column_stack([stress_pred, stress_ref]), fmt="%.10g")

    _log(f"  write:    {time.time() - t_write:5.1f}s   "
         f"-> {output_dir}/(energy|force|virial|stress)_{suffix}.out")


# ---------------------------------------------------------------------------
# Sharded variant: each DDP rank predicts its own data_store shard, then all
# per-frame arrays are gathered to rank 0 and written out in input-xyz order.
# No xyz re-read, no temp nep.txt, no second neighbor-list build.
# ---------------------------------------------------------------------------

def _compute_local_predictions(model, data_store, batch_size):
    """Run prediction on one rank's local data_store -> numpy arrays.

    Returns a dict of per-frame / per-atom arrays (pred and ref) plus the
    natoms and volume metadata needed to merge + write on rank 0.
    """
    dev   = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n_struct = data_store.n
    backend = ops.resolve_backend("auto", num_types=model.num_types,
                                  device_type=dev.type)

    nat_arr = np.asarray(data_store.natoms, dtype=np.int64)
    nat_cum = np.concatenate([[0], np.cumsum(nat_arr)])
    N_atoms_local = int(nat_cum[-1])

    e_pred = np.zeros(n_struct, dtype=np.float64)
    f_pred = np.zeros((N_atoms_local, 3), dtype=np.float64)
    v_pred = np.zeros((n_struct, 6), dtype=np.float64)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, n_struct, batch_size):
            end = min(start + batch_size, n_struct)
            B = end - start
            batch = data_store.collate(list(range(start, end)))
            r = model.compute_properties_cached(
                batch, need_forces=True, need_virial=True, backend=backend)

            nat_slice = nat_arr[start:end].astype(np.float64)
            e_pred[start:end] = r["Etot"].cpu().numpy() / nat_slice

            v_per = torch.zeros(B, 9, dtype=dtype, device=dev)
            v_per.scatter_add_(
                0, batch["struct_idx"].unsqueeze(-1).expand(-1, 9), r["virial"])
            v_pred[start:end] = (_virial9_to_6(v_per.cpu().numpy())
                                 / nat_slice[:, None])

            a_lo, a_hi = int(nat_cum[start]), int(nat_cum[end])
            f_pred[a_lo:a_hi] = r["forces"].cpu().numpy()
    if was_training:
        model.train()
    if dev.type == "cuda":
        torch.cuda.synchronize()

    # Reference values (only fill where the flag is set)
    energy_ref = np.array(
        [data_store.energy[i] if data_store.has_energy_flag[i] else np.nan
         for i in range(n_struct)], dtype=np.float64)
    e_ref_pa = energy_ref / nat_arr.astype(np.float64)

    forces_ref = np.full((N_atoms_local, 3), np.nan, dtype=np.float64)
    for i in range(n_struct):
        if data_store.has_forces_flag[i]:
            a_lo, a_hi = int(nat_cum[i]), int(nat_cum[i + 1])
            forces_ref[a_lo:a_hi] = data_store.forces[i].cpu().numpy()

    virial_ref = np.full((n_struct, 6), _MISSING_VIRIAL, dtype=np.float64)
    for i in range(n_struct):
        if data_store.has_virial_flag[i]:
            v9 = data_store.virial[i].cpu().numpy().flatten()
            n = float(nat_arr[i])
            virial_ref[i] = [v9[0] / n, v9[4] / n, v9[8] / n,
                             v9[1] / n, v9[5] / n, v9[6] / n]

    vol_arr = data_store.volumes.detach().cpu().numpy().astype(np.float64)
    return {
        "natoms":    nat_arr,
        "volumes":   vol_arr,
        "e_pred":    e_pred,    "e_ref":    e_ref_pa,
        "f_pred":    f_pred,    "f_ref":    forces_ref,
        "v_pred":    v_pred,    "v_ref":    virial_ref,
    }


def _write_predictions(output_dir: str, n_total_frames: int,
                       natoms, volumes,
                       e_pred, e_ref, f_pred, f_ref, v_pred, v_ref,
                       suffix: str = "train"):
    """Write the four *_{suffix}.out files in frame-order. All arrays are
    already in global input-xyz order (frame 0 first)."""
    os.makedirs(output_dir, exist_ok=True)

    def _save(name, cols, block=1_000_000):
        """np.savetxt in row blocks: a single call formats the whole table
        into one Python string first (~2x the text size — 12 GiB for the
        force file of a 13M-frame set); blocks bound that at ~100 MB."""
        n = len(cols[0])
        with open(os.path.join(output_dir, name), "w") as fh:
            for st in range(0, n, block):
                np.savetxt(fh, np.column_stack([c[st:st + block] for c in cols]),
                           fmt="%.10g")

    _save(f"energy_{suffix}.out", [e_pred, e_ref])
    _save(f"force_{suffix}.out", [f_pred, f_ref])
    _save(f"virial_{suffix}.out", [v_pred, v_ref])

    # Stress (GPa), GPUMD sign (+virial/V); missing refs keep -1e6 unscaled.
    nat_col = natoms.astype(np.float64)[:, None]
    vol_col = volumes[:, None]
    stress_pred = _stress_from_virial(v_pred, nat_col, vol_col,
                                      keep_missing=False)
    stress_ref = _stress_from_virial(v_ref, nat_col, vol_col,
                                     keep_missing=True)
    _save(f"stress_{suffix}.out", [stress_pred, stress_ref])


def predict_from_store_sharded(model, data_store, local_global_idx,
                                n_total_frames: int, output_dir: str,
                                batch_size: int = 1000,
                                                            verbose: bool = True,
                                suffix: str = "train"):
    """DDP equivalent of ``predict_from_store``: each rank predicts its local
    data-store shard, arrays are gathered to rank 0, rank 0 writes the four
    ``*_{suffix}.out`` files in input-xyz order ("train" or "test").

    No xyz re-read, no neighbor-list rebuild, no temp nep.txt model file.

    Parameters
    ----------
    model : NEPModel  (DDP replica — parameters are in sync across ranks).
    data_store : StreamDataStore  (this rank's local shard).
    local_global_idx : list[int]  original xyz-frame index for each local
        frame (length == ``data_store.n``). Supplied by the random shard
        assignment in ``train_nep_sharded``.
    n_total_frames : int  total frames across all ranks (pre-drop).
    """
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_main = rank == 0

    def _log(msg):
        if verbose and is_main:
            print(msg, flush=True)

    t_compute = time.time()
    local = _compute_local_predictions(model, data_store, batch_size)
    local["global_idx"] = np.asarray(local_global_idx, dtype=np.int64)
    _log(f"  compute:  {time.time() - t_compute:5.1f}s")

    # Gather per-rank arrays onto rank 0, one rank at a time (point-to-point,
    # pickled bytes). all_gather_object would give EVERY rank the whole
    # dataset's predictions — ~10 GiB per rank on a 13M-frame set — which
    # was the largest host-memory item of a sharded run. Rank 0 folds each
    # part into the global arrays as it arrives and drops it, so its peak
    # is the global arrays plus one part; other ranks hold only their own.
    t_gather = time.time()
    comm_dev = (data_store.device if dist.is_initialized()
                and dist.get_backend() == "nccl" else torch.device("cpu"))

    def _parts(payload):
        """Yield the ranks' payloads on rank 0 (rank 0's own first); on the
        other ranks send and yield nothing."""
        import pickle
        if world_size == 1:
            yield payload
            return
        if is_main:
            yield payload
            for r in range(1, world_size):
                n = torch.zeros(1, dtype=torch.long, device=comm_dev)
                dist.recv(n, src=r)
                buf = torch.empty(int(n.item()), dtype=torch.uint8, device=comm_dev)
                dist.recv(buf, src=r)
                yield pickle.loads(buf.cpu().numpy().tobytes())
                del buf
        else:
            b = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            t = torch.frombuffer(bytearray(b), dtype=torch.uint8).to(comm_dev)
            dist.send(torch.tensor([t.numel()], dtype=torch.long, device=comm_dev), dst=0)
            dist.send(t, dst=0)
            del t, b

    # Round 1 (small): frame-level arrays + natoms, which size the
    # atom-level arrays. Padding duplicates (added in train_nep_sharded so
    # n_total divides evenly across ranks) write the same value twice into
    # the same slot — harmless. Every input frame appears in some rank, so
    # the output has no NaN rows.
    natoms_g = np.zeros(n_total_frames, dtype=np.int64)
    volumes_g = np.zeros(n_total_frames, dtype=np.float64)
    e_pred_g = np.full(n_total_frames, np.nan)
    e_ref_g  = np.full(n_total_frames, np.nan)
    v_pred_g = np.full((n_total_frames, 6), np.nan)
    v_ref_g  = np.full((n_total_frames, 6), _MISSING_VIRIAL)
    small = {k: local[k] for k in ("global_idx", "natoms", "volumes",
                                   "e_pred", "e_ref", "v_pred", "v_ref")}
    for part in _parts(small):
        gi = part["global_idx"]
        natoms_g[gi]  = part["natoms"]
        volumes_g[gi] = part["volumes"]
        e_pred_g[gi]  = part["e_pred"]
        e_ref_g[gi]   = part["e_ref"]
        v_pred_g[gi]  = part["v_pred"]
        v_ref_g[gi]   = part["v_ref"]
    del small

    # Global per-atom offsets are determined by input-xyz order so they match
    # what predict_dataset / predict_from_store would emit.
    if is_main:
        nat_cum_g = np.concatenate([[0], np.cumsum(natoms_g)])
        N_atoms_global = int(nat_cum_g[-1])
        f_pred_g = np.full((N_atoms_global, 3), np.nan)
        f_ref_g  = np.full((N_atoms_global, 3), np.nan)

    # Round 2 (large): per-atom forces, one rank at a time.
    big = {k: local[k] for k in ("global_idx", "natoms", "f_pred", "f_ref")}
    del local
    for part in _parts(big):
        gi = part["global_idx"]
        nat_part = np.asarray(part["natoms"], dtype=np.int64)
        part_cum = np.concatenate([[0], np.cumsum(nat_part)])
        # vectorised scatter: destination atom index for every local atom
        ga = np.repeat(nat_cum_g[gi], nat_part)
        la = np.repeat(part_cum[:-1], nat_part)
        dst = ga + (np.arange(int(part_cum[-1])) - la)
        f_pred_g[dst] = part["f_pred"]
        f_ref_g[dst]  = part["f_ref"]
        del part
    del big
    if is_main:
        _log(f"  gather:   {time.time() - t_gather:5.1f}s")
    if not is_main:
        return

    t_write = time.time()
    _write_predictions(output_dir, n_total_frames,
                       natoms_g, volumes_g,
                       e_pred_g, e_ref_g,
                       f_pred_g, f_ref_g,
                       v_pred_g, v_ref_g, suffix=suffix)
    _log(f"  write:    {time.time() - t_write:5.1f}s   "
         f"-> {output_dir}/(energy|force|virial|stress)_{suffix}.out")
