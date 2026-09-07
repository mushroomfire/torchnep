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
Data loading utilities for NEP training and prediction.

Supports extended XYZ format (as used by GPUMD) and nep.in parameter files.
"""

import os
import numpy as np
from typing import Dict, List


def _parse_properties_schema(comment: str):
    """Parse the extended-XYZ ``Properties=...`` field.

    Returns a dict mapping field name -> (type_code, token_offset, width).
    ``token_offset`` is the starting column in the FULL per-atom token list
    (so species and numeric fields share the same coordinate). This lets
    downstream parsing handle arbitrary field ordering, including the
    ``pos:R:3:species:S:1`` form produced by some exporters.
    """
    # Case-insensitive match: ASE uses "Properties=" while some exporters
    # use "properties=". Look in a lower-cased copy but slice from original
    # so we preserve field names (species, pos, force, ...).
    key = "properties="
    idx = comment.lower().find(key)
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


def _parse_frame_block(block, energy_key="energy"):
    """Parse one frame from a list of text lines (picklable for mp.Pool).

    Reads extended XYZ with a ``Properties=...`` schema. Only
    ``species:S:1``, ``pos:R:3``, and ``force:R:3`` / ``forces:R:3`` are
    consumed — any other columns are silently ignored. Energy / virial /
    stress / lattice come from the comment-line key=value tags; see
    ``_parse_comment`` for strict-mode validation rules.
    """
    natoms = int(block[0].strip())
    comment = block[1].strip()
    frame = _parse_comment(comment, natoms, energy_key=energy_key)

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
        numeric_rows.append([t for k, t in enumerate(toks) if k != sp_col])

    ncol_numeric = len(numeric_rows[0])
    flat = " ".join(" ".join(r) for r in numeric_rows)
    arr = np.fromstring(flat, sep=" ", dtype=np.float64)
    arr = arr.reshape(natoms, ncol_numeric)

    frame["natoms"] = natoms
    frame["species"] = species

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


def index_xyz(filename: str):
    """Single streaming pass over an extended-XYZ file: frame index only.

    Returns ``(offsets, natoms)`` — two int64 numpy arrays with, for every
    frame, the byte offset of its ``natoms`` line and its atom count. No
    frame content is parsed and no line list is kept, so the pass runs at
    I/O speed with O(n_frames) memory — this is what makes multi-rank
    sharded loading of very large files (tens of GB) feasible: one rank
    indexes, the offsets are broadcast, and every rank then seek-reads only
    its own frames (see :func:`read_xyz_at`).
    """
    offsets, natoms = [], []
    with open(filename, "rb") as f:
        pos = 0
        line = f.readline()
        while line:
            stripped = line.strip()
            if not stripped:          # tolerate stray blank lines between frames
                pos = f.tell()
                line = f.readline()
                continue
            n = int(stripped)
            offsets.append(pos)
            natoms.append(n)
            for _ in range(n + 1):
                f.readline()
            pos = f.tell()
            line = f.readline()
    return (np.asarray(offsets, dtype=np.int64),
            np.asarray(natoms, dtype=np.int64))


def read_xyz_at(filename: str, offsets, energy_key: str = "energy"):
    """Parse only the frames starting at the given byte offsets.

    ``offsets`` come from :func:`index_xyz`; they may be in any order — the
    file is swept in ascending-offset order (sequential-friendly for
    parallel file systems) and the result list matches the ORDER OF
    ``offsets`` as passed in. Only the requested frames' bytes are read.
    """
    offsets = np.asarray(offsets, dtype=np.int64)
    order = np.argsort(offsets, kind="stable")
    out = [None] * len(offsets)
    with open(filename, "rb") as f:
        for k in order:
            f.seek(int(offsets[k]))
            first = f.readline()
            n = int(first)
            block = [first.decode()]
            block.extend(f.readline().decode() for _ in range(n + 1))
            out[int(k)] = _parse_frame_block(block, energy_key=energy_key)
    return out


def read_xyz(filename: str, energy_key: str = "energy") -> List[Dict]:
    """Read extended XYZ file (GPUMD format).

    Parameters
    ----------
    filename : str
        Path to the .xyz file.
    energy_key : str
        Name of the per-frame energy tag to read. Defaults to ``"energy"``;
        set to e.g. ``"atomization_energy"`` to use atomization energies
        instead of total energies.
    """
    with open(filename) as f:
        lines = f.readlines()

    blocks = _split_frames(lines)
    del lines
    return [_parse_frame_block(b, energy_key=energy_key) for b in blocks]


def _find_quoted(comment: str, key: str):
    """Return the content inside key="..." or None if absent.

    Matches ``key`` case-insensitively (e.g. ``Virial="..."`` and
    ``virial="..."`` both work — GPUMD / ASE / pymatgen differ on casing).
    """
    low = comment.lower()
    needle = key.lower() + '="'
    i = low.find(needle)
    if i < 0:
        return None
    i += len(needle)
    j = comment.find('"', i)
    if j < 0:
        return None
    return comment[i:j]


def _find_scalar(comment: str, key: str):
    """Return the string value after ``key=`` (unquoted) or None if absent.

    Matches ``key`` case-insensitively. Only returns a match when ``key=``
    is preceded by whitespace or start of string — prevents
    ``atomization_energy=`` from matching ``energy=``.
    """
    low = comment.lower()
    needle = key.lower() + "="
    start = 0
    while True:
        i = low.find(needle, start)
        if i < 0:
            return None
        if i == 0 or low[i - 1] in (" ", "\t"):
            i += len(needle)
            j = i
            while j < len(comment) and comment[j] not in (" ", "\t", '"'):
                j += 1
            return comment[i:j]
        start = i + 1


def _parse_comment(comment: str, natoms: int, energy_key: str = "energy") -> Dict:
    """Parse extended XYZ comment line with strict-mode validation.

    Rules:
      - ``Lattice="ax ay az bx by bz cx cy cz"`` is mandatory (9 floats).
        Every frame is treated as fully periodic; the ``pbc=`` tag is
        ignored (isolated clusters / molecules must be wrapped in a large
        vacuum box by the user before loading).
      - Energy tag name is configurable via ``energy_key`` (default "energy");
        if missing, the frame simply has no energy (handled downstream).
      - ``virial="..."`` and ``stress="..."`` are optional but, when present,
        must have exactly 9 components. If both are given, virial wins.
        stress (eV/A**3) is converted to virial (eV) as
        ``virial = -stress * |det(lattice)|`` — opposite sign convention.
    """
    frame = {}

    lat_str = _find_quoted(comment, "Lattice")
    if lat_str is None:
        raise ValueError(
            "extended-xyz frame is missing mandatory Lattice=\"...\" tag; "
            "comment: " + comment[:160])
    lat_vals = [float(x) for x in lat_str.split()]
    if len(lat_vals) != 9:
        raise ValueError(
            f"Lattice must have exactly 9 components, got {len(lat_vals)}; "
            "comment: " + comment[:160])
    frame["cell"] = np.array(lat_vals).reshape(3, 3)

    e_val = _find_scalar(comment, energy_key)
    if e_val is not None:
        frame["energy"] = float(e_val)

    vir_str = _find_quoted(comment, "virial")
    if vir_str is not None:
        vir_vals = [float(x) for x in vir_str.split()]
        if len(vir_vals) != 9:
            raise ValueError(
                f"virial must have exactly 9 components, got {len(vir_vals)}; "
                "comment: " + comment[:160])
        frame["virial"] = np.array(vir_vals)
    else:
        stress_str = _find_quoted(comment, "stress")
        if stress_str is not None:
            stress_vals = [float(x) for x in stress_str.split()]
            if len(stress_vals) != 9:
                raise ValueError(
                    f"stress must have exactly 9 components, got "
                    f"{len(stress_vals)}; comment: " + comment[:160])
            volume = abs(float(np.linalg.det(frame["cell"])))
            frame["virial"] = -np.array(stress_vals) * volume

    return frame


def zbl_pair_index(t1: int, t2: int, num_types: int) -> int:
    """Row of the (t1, t2) element pair in a GPUMD zbl.in table (pairs are
    listed 1-1, 1-2, ..., 1-n, 2-2, ..., n-n; order of t1/t2 irrelevant)."""
    if t1 > t2:
        t1, t2 = t2, t1
    return t1 * num_types - (t1 * (t1 - 1)) // 2 + (t2 - t1)


def read_zbl_in(filename: str, num_types: int) -> List[List[float]]:
    """Read a GPUMD ``zbl.in`` (flexible ZBL) file.

    One row per element pair, ``num_types * (num_types + 1) / 2`` rows in
    the order 1-1, 1-2, ..., 1-n, 2-2, ..., n-n; each row holds
    ``rc_inner rc_outer a1 a2 a3 a4 a5 a6 a7 a8`` with
    ``phi(x) = a1 exp(-a2 x) + a3 exp(-a4 x) + a5 exp(-a6 x) + a7 exp(-a8 x)``.
    Whitespace / line breaks are free (GPUMD reads the numbers as a stream).
    Returns the table as a list of 10-element rows.
    """
    n_pairs = num_types * (num_types + 1) // 2
    with open(filename) as f:
        text = f.read()
    vals = []
    for line in text.splitlines():           # '#' starts a comment (rest of line)
        for tok in line.split("#")[0].replace(",", " ").split():
            vals.append(float(tok))
    if len(vals) != 10 * n_pairs:
        raise ValueError(
            f"{filename}: expected {10 * n_pairs} numbers "
            f"({n_pairs} element pairs x 10: rc_inner rc_outer a1..a8) for "
            f"{num_types} types, found {len(vals)}")
    table = [vals[10 * k:10 * (k + 1)] for k in range(n_pairs)]
    for k, row in enumerate(table):
        if not (0.0 <= row[0] < row[1]):
            raise ValueError(f"{filename}: pair row {k + 1}: need "
                             f"0 <= rc_inner < rc_outer, got {row[0]} {row[1]}")
    return table


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
            line = line.split("#")[0].strip()
            if not line:
                continue

            parts = line.split()
            key = parts[0].lower()

            if key == "type":
                params["num_types"] = int(parts[1])
                params["type_names"] = parts[2 : 2 + int(parts[1])]
            elif key == "version":
                v = int(parts[1])
                # torchnep only implements NEP4.
                if v != 4:
                    raise ValueError(
                        f"nep.in version {v} is not supported — torchnep "
                        f"only implements NEP4 (set 'version 4').")
                params["version"] = v
            elif key == "zbl":
                # "zbl <cutoff>"  -> universal ZBL with that outer cutoff;
                # "zbl <file>"    -> flexible ZBL: per-element-pair cutoffs
                #                    and phi coefficients read from a GPUMD
                #                    zbl.in file (path relative to nep.in).
                try:
                    params["zbl"] = float(parts[1])
                except ValueError:
                    zbl_path = parts[1]
                    if not os.path.isabs(zbl_path):
                        zbl_path = os.path.join(
                            os.path.dirname(os.path.abspath(filename)), zbl_path)
                    params["zbl_file"] = zbl_path
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
            elif key == "lambda_e":
                params["lambda_e"] = float(parts[1])
            elif key == "lambda_f":
                params["lambda_f"] = float(parts[1])
            elif key == "lambda_v":
                params["lambda_v"] = float(parts[1])
            elif key == "weight_decay":
                params["weight_decay"] = float(parts[1])
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
            elif key == "early_stop":
                params["early_stop"] = int(parts[1])
            elif key == "scheduler_factor":
                params["scheduler_factor"] = float(parts[1])
            elif key == "stop_lr":
                params["stop_lr"] = float(parts[1])
            elif key == "lr_scheduler":
                # "plateau" (default, ReduceLROnPlateau) or "step" (StepLR)
                mode = parts[1].lower()
                if mode not in ("plateau", "step"):
                    raise ValueError(
                        f"lr_scheduler must be 'plateau' or 'step', got {parts[1]!r}")
                params["lr_scheduler"] = mode
            elif key == "max_grad_norm":
                params["max_grad_norm"] = float(parts[1])
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
            elif key == "stage2_scheduler_patience":
                params["stage2_scheduler_patience"] = int(parts[1])
            elif key == "stage2_scheduler_factor":
                params["stage2_scheduler_factor"] = float(parts[1])

    if "zbl_file" in params:
        if "num_types" not in params:
            raise ValueError("nep.in: 'zbl <file>' needs the 'type' line")
        params["zbl_flexible"] = read_zbl_in(params["zbl_file"],
                                             params["num_types"])
        # the largest outer cutoff stands in for the universal cutoff value
        params["zbl"] = max(row[1] for row in params["zbl_flexible"])

    # Snapshot the explicit (user-set) keys before applying defaults so the
    # trainer can report which values came from nep.in vs which fell back
    # to a default.
    explicit = set(params.keys())

    # Defaults — model architecture
    params.setdefault("version", 4)
    params.setdefault("cutoff_radial", 8.0)
    params.setdefault("cutoff_angular", 4.0)
    params.setdefault("n_max_radial", 6)
    params.setdefault("n_max_angular", 6)
    params.setdefault("basis_size_radial", 6)
    params.setdefault("basis_size_angular", 6)
    params.setdefault("l_max", [4, 1, 0])
    params.setdefault("neuron", 30)

    # Defaults — training hyperparameters (match train_nep / train_nep_sharded)
    params.setdefault("num_epochs", 600)
    params.setdefault("batch_size", 32)
    params.setdefault("lr", 0.01)
    params.setdefault("stop_lr", 1e-6)
    params.setdefault("scheduler_patience", 15)
    params.setdefault("early_stop", 0)
    params.setdefault("scheduler_factor", 0.7)
    params.setdefault("lr_scheduler", "plateau")
    params.setdefault("max_grad_norm", 10.0)
    params.setdefault("lambda_e", 0.01)
    params.setdefault("lambda_f", 1.0)
    params.setdefault("lambda_v", 0.01)
    params.setdefault("weight_decay", 1e-4)
    params.setdefault("stage2", False)

    # Defaults for optional stage-2 parameters (only used if stage2=1).
    params.setdefault("stage2_lr", 1e-3)
    params.setdefault("stage2_pref_e", 1.0)
    params.setdefault("stage2_pref_f", 0.05)
    params.setdefault("stage2_pref_v", 0.1)
    # start_stage2 defaults to 0.5 * num_epochs if not set — handled in trainer

    # Stash explicit-key set in the dict itself; consumers can read it (and
    # safely ignore it). Leading underscore so it can't collide with nep.in
    # tokens.
    params["_explicit"] = explicit
    return params


# ---------------------------------------------------------------------------
# Neighbor list construction (numpy, CPU — shared by training and prediction)
# ---------------------------------------------------------------------------

def wrap_positions(positions, cell):
    """Wrap ``positions`` into the primary cell (fractional coords in [0, 1)).

    Under full PBC the physics is translation-invariant, so every neighbor
    builder works on wrapped coordinates; the wrapped array is also what the
    compact / on-the-fly data stores keep, so their displacement vectors
    are computed from exactly the coordinates the numpy builder used.
    """
    inv_cell = np.linalg.inv(cell)
    frac = positions @ inv_cell
    frac -= np.floor(frac)
    return frac @ cell, inv_cell


def image_repeats(inv_cell, cutoff):
    """Periodic image repeats per lattice direction needed to cover ``cutoff``.

    The perpendicular distance between planes spanned by (b,c), (a,c), (a,b)
    is ``1/|inv_cell[:, i]|`` (columns of inv_cell are the reciprocal
    vectors). Using the cell ROWS instead silently undercounts image replicas
    for heavily skewed triclinic cells and drops real neighbors — bug fixed
    2025.
    """
    return [int(np.ceil(cutoff * np.linalg.norm(inv_cell[:, i])))
            for i in range(3)]


def build_neighbor_list_np_ex(positions, cell, cutoff):
    """Numpy neighbor list that also reports the periodic image of each pair.

    Returns ``(idx_i, idx_j, rij, shift_frac, positions_wrapped)``:
    ``shift_frac`` is the (P, 3) integer lattice translation of the neighbor
    image (``rij = pos_w[j] + shift_frac @ cell - pos_w[i]``), so a pair list
    can be stored without its displacement vectors (7 bytes per pair instead
    of 28) and ``rij`` recomputed on the device from the wrapped positions.
    :func:`build_neighbor_list_np` is this function minus the extras and is
    numerically identical to it (same arithmetic, same pair order).

    Cell is stored with lattice vectors as ROWS. Input positions may lie
    outside the primary cell; they are wrapped first (see
    :func:`wrap_positions`) so the image estimate covers every neighbor.
    """
    N = positions.shape[0]
    positions, inv_cell = wrap_positions(positions, cell)
    n_rep = image_repeats(inv_cell, cutoff)

    a_r = np.arange(-n_rep[0], n_rep[0] + 1)
    b_r = np.arange(-n_rep[1], n_rep[1] + 1)
    c_r = np.arange(-n_rep[2], n_rep[2] + 1)
    shifts_int = np.stack(np.meshgrid(a_r, b_r, c_r, indexing="ij"),
                          axis=-1).reshape(-1, 3)
    shifts_frac = shifts_int.astype(positions.dtype)
    shifts_cart = shifts_frac @ cell
    S = shifts_cart.shape[0]
    zero_shift = np.all(shifts_int == 0, axis=1)

    if N * N * S < 8_000_000:
        disp = (positions[None, :, None, :] + shifts_cart[None, None, :, :]
                - positions[:, None, None, :])
        dist = np.linalg.norm(disp, axis=-1)
        self_mask = np.eye(N, dtype=bool)[:, :, None] & zero_shift[None, None, :]
        valid = (dist < cutoff) & (dist > 1e-10) & ~self_mask
        idx_i, idx_j, idx_s = np.where(valid)
        return (idx_i.astype(np.int64), idx_j.astype(np.int64),
                disp[idx_i, idx_j, idx_s], shifts_int[idx_s], positions)

    all_i, all_j, all_s, all_rij = [], [], [], []
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
            all_s.append(np.full(len(ii), si, dtype=np.int64))
            all_rij.append(disp[ii, jj])
    if not all_i:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64),
                np.zeros((0, 3), positions.dtype),
                np.zeros((0, 3), np.int64), positions)
    return (np.concatenate(all_i).astype(np.int64),
            np.concatenate(all_j).astype(np.int64),
            np.concatenate(all_rij),
            shifts_int[np.concatenate(all_s)], positions)


def build_neighbor_list_np(positions, cell, cutoff):
    """Build neighbor list using numpy (for preprocessing). Returns
    ``(idx_i, idx_j, rij)`` — see :func:`build_neighbor_list_np_ex`."""
    idx_i, idx_j, rij, _, _ = build_neighbor_list_np_ex(positions, cell, cutoff)
    return idx_i, idx_j, rij


def valid_split_indices(n_frames: int, valid_ratio: float, run_seed: int):
    """Train/validation split indices — the exact split ``train_nep`` makes.

    ``train_nep(valid_ratio=r, run_seed=s)`` holds out
    ``max(1, round(r * n))`` frames drawn from a dedicated torch generator
    seeded with ``run_seed``. This helper is that draw, factored out so the
    trainers and :func:`export_valid_split` can never disagree.

    Returns ``(train_idx, valid_idx)`` — both sorted in input-file order.
    """
    import torch
    if run_seed is None:
        raise ValueError("run_seed is required: the split is drawn from it "
                         "(train_nep uses the same seed to reproduce it)")
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError(f"valid_ratio must be in (0, 1), got {valid_ratio}")
    g = torch.Generator()
    g.manual_seed(run_seed)
    perm = torch.randperm(n_frames, generator=g).tolist()
    n_val = max(1, int(round(valid_ratio * n_frames)))
    if n_val >= n_frames:
        raise ValueError(f"valid_ratio={valid_ratio} leaves no training "
                         f"frames ({n_frames} total)")
    val_set = set(perm[:n_val])
    train_idx = [i for i in range(n_frames) if i not in val_set]
    return train_idx, sorted(val_set)


def export_valid_split(data_file: str, valid_ratio: float, run_seed: int,
                       output_dir: str = "split", strategy: str = "stratified",
                       min_stratum: int = 20):
    """Write GPUMD-ready ``train.xyz`` / ``test.xyz`` with train_nep's split.

    Reproduces exactly the validation split that
    ``train_nep(data_file, valid_ratio=r, run_seed=s, valid_strategy=...)``
    uses internally, so the exported pair can train the SAME data partition
    in GPUMD (or any other code) and loss curves stay comparable. Frames
    are copied verbatim (raw text, untouched fields and precision), in
    input-file order.

    ``strategy``: "random" (default) or "stratified" — see
    :func:`stratified_split_indices`.

    Returns ``(train_path, test_path, n_train, n_valid)``.
    """
    import os
    with open(data_file) as f:
        blocks = _split_frames(f.readlines())
    if strategy == "stratified":
        metas = []
        for b in blocks:
            na = int(b[0].split()[0])
            metas.append((na, {line.split()[0] for line in b[2:2 + na]}))
        train_idx, val_idx, _ = stratified_split_indices(
            metas, valid_ratio, run_seed, min_stratum=min_stratum)
    elif strategy == "random":
        train_idx, val_idx = valid_split_indices(len(blocks), valid_ratio,
                                                 run_seed)
    else:
        raise ValueError(f"unknown split strategy: {strategy!r}")
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.xyz")
    test_path = os.path.join(output_dir, "test.xyz")
    src = os.path.abspath(data_file)
    for path, idxs in ((train_path, train_idx), (test_path, val_idx)):
        if os.path.abspath(path) == src:
            raise ValueError(f"output would overwrite the input: {src}")
        with open(path, "w") as out:
            for k in idxs:
                out.writelines(blocks[k])
    return train_path, test_path, len(train_idx), len(val_idx)


def _size_class(natoms: int) -> int:
    """Size class for stratified splitting: 0 = tiny cells (<=4 atoms,
    dimers/trimers — the pair-specific short-range information), 1 = small
    (5-15), 2 = bulk (>=16)."""
    if natoms <= 4:
        return 0
    if natoms <= 15:
        return 1
    return 2


def stratified_split_indices(metas, valid_ratio: float, run_seed: int,
                             min_stratum: int = 20,
                             tiny_to_train: bool = True):
    """Coverage-aware train/validation split.

    Frames are grouped into strata keyed by (element combination, size
    class — see :func:`_size_class`). Within each stratum ``valid_ratio``
    of the frames is held out for validation; strata with fewer than
    ``min_stratum`` frames go ENTIRELY to training. Rationale: with many
    element types a random split inevitably drops some rare stratum — e.g.
    the only few Mo-Pd dimer curves — fully into validation, so the model
    never sees that pair's short-range physics and can only fail on it.
    Stratifying guarantees every represented (composition, size) group is
    learned, and rare groups are never wasted on validation. The held-out
    fraction is therefore slightly below ``valid_ratio`` (rare strata
    contribute nothing); the validation set measures within-stratum
    generalization only.

    ``tiny_to_train`` (default True): tiny cells (size class 0, <= 4
    atoms — dimer/trimer short-range scans) go ENTIRELY to training
    regardless of stratum size. Their per-pair curves are sparse in
    configuration space even when the stratum is populous, and their job
    is to teach the short-range physics — holding some out both starves
    the model and produces the dominant validation-error tail.

    ``metas``: sequence of (natoms, iterable_of_species) per frame, in file
    order. Deterministic for a given ``run_seed``; the trainers and
    :func:`export_valid_split` share this implementation.

    Returns ``(train_idx, valid_idx, stats)`` — index lists sorted in input
    order plus a stats dict (n_strata, n_rare_strata, n_rare_frames,
    n_tiny_frames).
    """
    import torch
    if run_seed is None:
        raise ValueError("run_seed is required: the split is drawn from it")
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError(f"valid_ratio must be in (0, 1), got {valid_ratio}")
    strata = {}
    for i, (na, sp) in enumerate(metas):
        key = ("-".join(sorted(set(sp))), _size_class(na))
        strata.setdefault(key, []).append(i)

    g = torch.Generator()
    g.manual_seed(run_seed)
    val, n_rare, n_rare_frames, n_tiny = [], 0, 0, 0
    for key in sorted(strata):
        idxs = strata[key]
        if tiny_to_train and key[1] == 0:
            n_tiny += len(idxs)
            continue
        if len(idxs) < min_stratum:
            n_rare += 1
            n_rare_frames += len(idxs)
            continue
        perm = torch.randperm(len(idxs), generator=g).tolist()
        n_val = min(max(1, int(round(valid_ratio * len(idxs)))),
                    len(idxs) - 1)
        val.extend(idxs[p] for p in perm[:n_val])
    # Fallback: when the eligible pool is too small for a meaningful
    # holdout (e.g. a dataset made ENTIRELY of tiny cells, which
    # tiny_to_train sends to training), a starved validation set would be
    # useless-to-empty — fall back to a plain random split (same seed) and
    # report it via stats["fallback"] so callers can log it.
    if len(val) < max(1, int(round(0.5 * valid_ratio * len(metas)))):
        train_idx, val_idx = valid_split_indices(len(metas), valid_ratio,
                                                 run_seed)
        stats = {"n_strata": len(strata), "n_rare_strata": n_rare,
                 "n_rare_frames": n_rare_frames, "n_tiny_frames": n_tiny,
                 "fallback": "random"}
        return train_idx, val_idx, stats
    val_set = set(val)
    train_idx = [i for i in range(len(metas)) if i not in val_set]
    stats = {"n_strata": len(strata), "n_rare_strata": n_rare,
             "n_rare_frames": n_rare_frames, "n_tiny_frames": n_tiny}
    return train_idx, sorted(val_set), stats
