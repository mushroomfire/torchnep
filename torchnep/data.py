"""
Data loading utilities for NEP training and prediction.

Supports extended XYZ format (as used by GPUMD) and nep.in parameter files.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def read_xyz(filename: str) -> List[Dict]:
    """Read extended XYZ file (GPUMD format).

    Each frame contains:
        - natoms: number of atoms
        - cell: (3, 3) lattice vectors
        - species: list of element symbols
        - positions: (N, 3) atomic positions
        - energy: total energy (if available)
        - forces: (N, 3) forces (if available)
        - virial: (6,) or (9,) virial tensor (if available)

    Parameters
    ----------
    filename : str
        Path to XYZ file.

    Returns
    -------
    list of dict
        List of frame dictionaries.
    """
    frames = []
    with open(filename) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        # Number of atoms
        natoms = int(lines[i].strip())
        i += 1

        # Comment line with properties
        comment = lines[i].strip()
        i += 1

        # Parse comment line
        frame = _parse_comment(comment, natoms)

        # Read atom data
        species = []
        positions = []
        forces = []
        has_forces = False

        for j in range(natoms):
            parts = lines[i].split()
            i += 1
            species.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(parts) >= 7:
                forces.append([float(parts[4]), float(parts[5]), float(parts[6])])
                has_forces = True

        frame["natoms"] = natoms
        frame["species"] = species
        frame["positions"] = np.array(positions)
        if has_forces:
            frame["forces"] = np.array(forces)

        frames.append(frame)

    return frames


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
