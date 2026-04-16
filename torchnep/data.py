"""
Data loading utilities for NEP training and prediction.

Supports extended XYZ format (as used by GPUMD) and nep.in parameter files.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def _parse_properties_schema(comment: str):
    """Parse the extended-XYZ ``Properties=...`` field.

    Returns a dict mapping field name → (type_code, token_offset, width).
    ``token_offset`` is the starting column in the FULL per-atom token list
    (so species and numeric fields share the same coordinate). This lets
    downstream parsing handle arbitrary field ordering, including the
    ``pos:R:3:species:S:1`` form produced by some exporters.
    """
    key = "Properties="
    idx = comment.find(key)
    if idx < 0:
        return None
    start = idx + len(key)
    end = start
    while end < len(comment) and comment[end] not in (" ", "\t"):
        end += 1
    spec = comment[start:end]
    toks = spec.split(":")
    if len(toks) % 3 != 0:
        return None
    schema = {}
    col = 0
    for i in range(0, len(toks), 3):
        name, tp, cnt = toks[i], toks[i + 1], int(toks[i + 2])
        schema[name] = (tp, col, cnt)
        col += cnt
    return schema


def _parse_frame_block(block):
    """Parse one frame from a list of text lines (picklable for mp.Pool).

    Reads standard extended XYZ with a ``Properties=...`` schema. Recognises
    ``pos``, ``force`` / ``forces``, and skips any other columns (e.g.
    ``Z:I:1`` atomic number). Energy / virial / lattice come from the
    comment-line key=value tags (see ``_parse_comment``).
    """
    natoms = int(block[0].strip())
    comment = block[1].strip()
    frame = _parse_comment(comment, natoms)

    schema = _parse_properties_schema(comment)
    if schema is None or "pos" not in schema:
        raise ValueError(
            "extended-xyz parser requires a Properties=... header with "
            "at least species and pos fields; got: " + comment[:120])

    # Find species column (only field with type 'S')
    species_key = next((k for k, v in schema.items() if v[0] == "S"), None)
    if species_key is None:
        raise ValueError("Properties schema missing species (type S) field")
    _, sp_col, _ = schema[species_key]

    atoms = block[2:2 + natoms]
    species = [None] * natoms
    numeric_rows = []
    for j, line in enumerate(atoms):
        toks = line.split()
        species[j] = toks[sp_col]
        # collect everything except the species token, preserving order
        numeric_rows.append([t for k, t in enumerate(toks) if k != sp_col])

    ncol_numeric = len(numeric_rows[0])
    flat = " ".join(" ".join(r) for r in numeric_rows)
    arr = np.fromstring(flat, sep=" ", dtype=np.float64)
    arr = arr.reshape(natoms, ncol_numeric)

    frame["natoms"] = natoms
    frame["species"] = species

    # Column offsets in schema are relative to the full token list including
    # species; convert to offsets within the numeric tail by subtracting 1 for
    # any field whose token position is beyond species.
    def _numeric_offset(field_col):
        return field_col if field_col < sp_col else field_col - 1

    _, pos_col, pos_w = schema["pos"]
    pos_off = _numeric_offset(pos_col)
    frame["positions"] = arr[:, pos_off:pos_off + pos_w].copy()

    force_key = "force" if "force" in schema else (
                "forces" if "forces" in schema else None)
    if force_key is not None:
        _, f_col, f_w = schema[force_key]
        f_off = _numeric_offset(f_col)
        frame["forces"] = arr[:, f_off:f_off + f_w].copy()
    return frame


def _split_frames(lines):
    """Split an XYZ text into per-frame line blocks."""
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        natoms = int(lines[i].strip())
        end = i + 2 + natoms
        blocks.append(lines[i:end])
        i = end
    return blocks


def read_xyz(filename: str) -> List[Dict]:
    """Read extended XYZ file (GPUMD format)."""
    with open(filename) as f:
        lines = f.readlines()

    blocks = _split_frames(lines)
    del lines
    return [_parse_frame_block(b) for b in blocks]


def _parse_comment(comment: str, natoms: int) -> Dict:
    """Parse extended XYZ comment line."""
    frame = {}

    # Extract Lattice
    lat_start = comment.find('Lattice="')
    if lat_start >= 0:
        lat_start += len('Lattice="')
        lat_end = comment.find('"', lat_start)
        lat_str = comment[lat_start:lat_end]
        lat_vals = [float(x) for x in lat_str.split()]
        frame["cell"] = np.array(lat_vals).reshape(3, 3)

    # Extract energy
    for prefix in ["energy=", "Energy="]:
        idx = comment.find(prefix)
        if idx >= 0:
            start = idx + len(prefix)
            end = start
            while end < len(comment) and comment[end] not in (" ", '"', "\t"):
                end += 1
            frame["energy"] = float(comment[start:end])
            break

    # Extract virial
    vir_start = comment.find('virial="')
    if vir_start >= 0:
        vir_start += len('virial="')
        vir_end = comment.find('"', vir_start)
        vir_str = comment[vir_start:vir_end]
        vir_vals = [float(x) for x in vir_str.split()]
        frame["virial"] = np.array(vir_vals)

    return frame


def parse_nep_in(filename: str) -> Dict:
    """Parse nep.in parameter file.

    Parameters
    ----------
    filename : str
        Path to nep.in file.

    Returns
    -------
    dict
        Dictionary of NEP parameters.
    """
    params = {}

    with open(filename) as f:
        for line in f:
            # Remove comments
            line = line.split("#")[0].strip()
            if not line:
                continue

            parts = line.split()
            key = parts[0].lower()

            if key == "type":
                params["num_types"] = int(parts[1])
                params["type_names"] = parts[2 : 2 + int(parts[1])]
            elif key == "version":
                params["version"] = int(parts[1])
            elif key == "zbl":
                params["zbl"] = float(parts[1])
            elif key == "use_typewise_cutoff_zbl":
                params["typewise_cutoff_zbl_factor"] = float(parts[1])
            elif key == "cutoff":
                params["cutoff_radial"] = float(parts[1])
                params["cutoff_angular"] = float(parts[2])
            elif key == "n_max":
                params["n_max_radial"] = int(parts[1])
                params["n_max_angular"] = int(parts[2])
            elif key == "basis_size":
                params["basis_size_radial"] = int(parts[1])
                params["basis_size_angular"] = int(parts[2])
            elif key == "l_max":
                params["l_max"] = [int(x) for x in parts[1:]]
            elif key == "neuron":
                params["neuron"] = int(parts[1])
            elif key == "lambda_1":
                params["lambda_1"] = float(parts[1])
            elif key == "lambda_e":
                params["lambda_e"] = float(parts[1])
            elif key == "lambda_f":
                params["lambda_f"] = float(parts[1])
            elif key == "lambda_v":
                params["lambda_v"] = float(parts[1])
            elif key == "lambda_2":
                params["lambda_2"] = float(parts[1])
            elif key == "batch":
                params["batch_size"] = int(parts[1])
            elif key == "save_potential":
                params["save_interval"] = int(parts[1])
                if len(parts) > 2:
                    params["save_start"] = int(parts[2])
                if len(parts) > 3:
                    params["save_count"] = int(parts[3])
            # --- torchnep training parameters ---
            elif key == "epoch":
                params["num_epochs"] = int(parts[1])
            elif key == "lr":
                params["lr"] = float(parts[1])
            elif key == "scheduler_patience":
                params["scheduler_patience"] = int(parts[1])
            elif key == "scheduler_factor":
                params["scheduler_factor"] = float(parts[1])
            elif key == "stop_lr":
                params["stop_lr"] = float(parts[1])
            elif key == "max_grad_norm":
                params["max_grad_norm"] = float(parts[1])
            elif key == "huber_delta":
                params["huber_delta"] = float(parts[1])
            elif key == "stage2":
                params["stage2"] = int(parts[1]) != 0
            elif key == "start_stage2":
                params["start_stage2"] = int(parts[1])
            elif key == "stage2_lr":
                params["stage2_lr"] = float(parts[1])
            elif key == "stage2_lambda_e":
                params["stage2_pref_e"] = float(parts[1])
            elif key == "stage2_lambda_f":
                params["stage2_pref_f"] = float(parts[1])
            elif key == "stage2_lambda_v":
                params["stage2_pref_v"] = float(parts[1])
            elif key == "use_swa":
                params["use_swa"] = int(parts[1]) != 0

    # Defaults — model
    params.setdefault("version", 4)
    params.setdefault("cutoff_radial", 6.0)
    params.setdefault("cutoff_angular", 6.0)
    params.setdefault("n_max_radial", 4)
    params.setdefault("n_max_angular", 4)
    params.setdefault("basis_size_radial", 12)
    params.setdefault("basis_size_angular", 12)
    params.setdefault("l_max", [4, 2, 0])
    params.setdefault("neuron", 40)
    params.setdefault("lambda_e", 1.0)
    params.setdefault("lambda_f", 1.0)
    params.setdefault("lambda_v", 0.0)
    params.setdefault("lambda_1", 0.0)
    params.setdefault("batch_size", 1000)

    return params


# ---------------------------------------------------------------------------
# Neighbor list construction (numpy, CPU — shared by training and prediction)
# ---------------------------------------------------------------------------

def build_neighbor_list_np(positions, cell, cutoff):
    """Build neighbor list using numpy (for preprocessing). Returns arrays."""
    N = positions.shape[0]
    inv_cell = np.linalg.inv(cell)

    n_rep = [int(np.ceil(cutoff / (1.0 / np.linalg.norm(inv_cell[i])))) for i in range(3)]

    a_r = np.arange(-n_rep[0], n_rep[0] + 1)
    b_r = np.arange(-n_rep[1], n_rep[1] + 1)
    c_r = np.arange(-n_rep[2], n_rep[2] + 1)
    shifts_frac = np.stack(np.meshgrid(a_r, b_r, c_r, indexing="ij"), axis=-1)
    shifts_frac = shifts_frac.reshape(-1, 3).astype(positions.dtype)
    shifts_cart = shifts_frac @ cell
    S = shifts_cart.shape[0]

    if N * N * S < 8_000_000:
        disp = (positions[None, :, None, :] + shifts_cart[None, None, :, :]
                - positions[:, None, None, :])
        dist = np.linalg.norm(disp, axis=-1)
        zero_shift = np.all(shifts_frac == 0, axis=1)
        self_mask = np.eye(N, dtype=bool)[:, :, None] & zero_shift[None, None, :]
        valid = (dist < cutoff) & (dist > 1e-10) & ~self_mask
        idx_i, idx_j, idx_s = np.where(valid)
        return idx_i.astype(np.int64), idx_j.astype(np.int64), disp[idx_i, idx_j, idx_s]

    zero_shift = np.all(shifts_frac == 0, axis=1)
    all_i, all_j, all_rij = [], [], []
    for si in range(S):
        shifted = positions + shifts_cart[si]
        disp = shifted[None, :, :] - positions[:, None, :]
        dist = np.linalg.norm(disp, axis=-1)
        valid = (dist < cutoff) & (dist > 1e-10)
        if zero_shift[si]:
            np.fill_diagonal(valid, False)
        ii, jj = np.where(valid)
        if len(ii) > 0:
            all_i.append(ii)
            all_j.append(jj)
            all_rij.append(disp[ii, jj])
    if not all_i:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64),
                np.zeros((0, 3), positions.dtype))
    return (np.concatenate(all_i).astype(np.int64),
            np.concatenate(all_j).astype(np.int64),
            np.concatenate(all_rij))
