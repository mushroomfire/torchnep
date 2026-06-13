# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Yongchao Wu.
# Part of the torchnep project — https://github.com/mushroomfire/torchnep.

"""PyTorch cell-list neighbor search for MD-sized structures.

The training / prediction path uses :func:`torchnep.data.build_neighbor_list_np`,
an O(N**2 * n_images) brute-force builder. That is perfectly fine for the small
periodic cells in a fit set (tens to a few hundred atoms), but it materialises a
dense (N, N, n_images) displacement tensor and so blows up — in both memory and
time — once an ASE-driven MD run reaches thousands of atoms.

:func:`build_neighbor_list` here is the MD-oriented replacement. It runs a
linked-cell (a.k.a. cell-list) search entirely in PyTorch so it stays on the
model's device (CPU / CUDA / MPS), avoiding the numpy round-trip, and is O(N)
in the common condensed-matter regime. When the cell is too small for the
linked-cell decomposition (fewer than 3 bins along some lattice direction — i.e.
a tiny training-style box) it transparently falls back to the numpy builder,
which is both correct for that regime and cheap there.

The pair *set* produced is identical (to round-off) to the numpy builder; pair
ordering may differ, which is irrelevant because every downstream consumer
scatter-sums over pairs.
"""

import numpy as np
import torch

from .data import build_neighbor_list_np


def _search_dtype(device: torch.device) -> torch.dtype:
    """Float precision for the geometry of the search.

    float64 everywhere it is supported (CPU, CUDA) so the pair geometry is
    bit-identical to the numpy reference; MPS has no float64, so fall back to
    float32 there (models on MPS are float32 anyway).
    """
    return torch.float32 if device.type == "mps" else torch.float64


def build_neighbor_list(positions, cell, cutoff, device="cpu",
                        dtype=torch.float64, max_pairs_chunk=2_000_000):
    """Build a directed neighbor list (each physical pair appears as i->j and j->i).

    Parameters
    ----------
    positions : (N, 3) array-like        atomic positions (A); may lie outside the cell.
    cell : (3, 3) array-like              lattice vectors as ROWS.
    cutoff : float                        neighbor cutoff (A).
    device : str or torch.device          device for the search and the output tensors.
    dtype : torch.dtype                   dtype of the returned ``rij`` (matches the model).
    max_pairs_chunk : int                 candidate-tensor size budget per chunk (tunes peak memory).

    Returns
    -------
    pair_i, pair_j : (P,) int64 tensors   central / neighbor atom indices, on ``device``.
    rij : (P, 3) ``dtype`` tensor         displacement vectors r_j(image) - r_i, on ``device``.
    """
    device = torch.device(device)
    sdtype = _search_dtype(device)
    pos = torch.as_tensor(np.asarray(positions), dtype=sdtype, device=device)
    box = torch.as_tensor(np.asarray(cell), dtype=sdtype, device=device)

    out = _cell_list(pos, box, float(cutoff), device, sdtype,
                     max_pairs_chunk=max_pairs_chunk)
    if out is None:
        # Small / degenerate cell — defer to the numpy brute-force builder.
        pi, pj, rij = build_neighbor_list_np(
            np.asarray(positions), np.asarray(cell), float(cutoff))
        return (torch.from_numpy(pi).to(device),
                torch.from_numpy(pj).to(device),
                torch.from_numpy(np.ascontiguousarray(rij)).to(device=device, dtype=dtype))

    pi, pj, rij = out
    return pi, pj, rij.to(dtype)


def _cell_list(pos, cell, cutoff, device, sdtype, max_pairs_chunk):
    """Linked-cell neighbor search. Returns (pi, pj, rij) torch tensors or None.

    None signals "not applicable" (a lattice direction holds < 3 bins, so the
    +/-1 bin stencil would alias onto itself) and the caller should fall back.
    """
    N = pos.shape[0]
    if N == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z.clone(), torch.zeros(0, 3, dtype=sdtype, device=device)

    inv = torch.linalg.inv(cell)
    # Perpendicular width between lattice planes along reciprocal direction i is
    # 1/|inv[:, i]| (columns of inv are the reciprocal vectors). Same quantity the
    # numpy builder uses to size its image replication.
    widths = 1.0 / torch.linalg.norm(inv, dim=0)              # (3,)
    n_cells = torch.floor(widths / cutoff).to(torch.long)      # (3,)

    # The +/-1 stencil only covers the cutoff when each direction has >= 3 bins
    # (bin width >= cutoff and the bin and its two periodic neighbours are
    # distinct). Tiny boxes go to the numpy fallback.
    if bool((n_cells < 3).any()):
        return None

    ncx, ncy, ncz = (int(n_cells[0]), int(n_cells[1]), int(n_cells[2]))
    total = ncx * ncy * ncz

    # Wrap into the primary cell. Physics is translation-invariant under full
    # PBC, so this never changes the pair set, only keeps bin indices in range.
    frac = pos @ inv
    frac = frac - torch.floor(frac)
    pos_w = frac @ cell

    # Per-atom bin index (clamp guards frac rounding to exactly 1.0).
    bin_xyz = torch.floor(frac * n_cells).to(torch.long)
    bin_xyz = torch.minimum(bin_xyz, n_cells - 1).clamp_(min=0)
    cell_id = (bin_xyz[:, 0] * ncy + bin_xyz[:, 1]) * ncz + bin_xyz[:, 2]   # (N,)

    # Dense bin table: (total, max_per) of atom indices, -1 padded. Built with a
    # single sort so the intra-bin slot of each atom is arange - bin_start.
    counts = torch.bincount(cell_id, minlength=total)
    max_per = int(counts.max())
    order = torch.argsort(cell_id)
    offsets = torch.zeros(total + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(counts, 0)
    intra = torch.arange(N, device=device) - offsets[cell_id[order]]
    table = torch.full((total * max_per,), -1, dtype=torch.long, device=device)
    table[cell_id[order] * max_per + intra] = order
    table = table.view(total, max_per)

    # 27-bin stencil offsets.
    rng = torch.tensor([-1, 0, 1], dtype=torch.long, device=device)
    offs = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), dim=-1).reshape(-1, 3)
    S = offs.shape[0]                                                       # 27

    # Chunk over centre atoms so peak memory stays ~ max_pairs_chunk elements.
    chunk = max(1, max_pairs_chunk // (S * max(max_per, 1)))
    pis, pjs, rijs = [], [], []
    atom_ids = torch.arange(N, device=device)

    for start in range(0, N, chunk):
        cidx = atom_ids[start:start + chunk]                               # (C,)
        C = cidx.shape[0]
        nb = bin_xyz[cidx].unsqueeze(1) + offs.unsqueeze(0)                 # (C, S, 3)
        # How many whole cells each stencil bin crosses -> the periodic image
        # translation applied to the candidate atom.
        img = torch.div(nb, n_cells, rounding_mode="floor")                # (C, S, 3)
        nb_w = nb - img * n_cells                                          # in [0, n_cells)
        nb_flat = (nb_w[..., 0] * ncy + nb_w[..., 1]) * ncz + nb_w[..., 2]  # (C, S)

        cand = table[nb_flat]                                              # (C, S, max_per)
        shift = (img.to(sdtype) @ cell)                                    # (C, S, 3)

        valid = cand >= 0
        q = torch.where(valid, cand, torch.zeros_like(cand))               # safe gather idx
        qpos = pos_w[q]                                                    # (C, S, max_per, 3)
        rij = (qpos + shift.unsqueeze(2)
               - pos_w[cidx].unsqueeze(1).unsqueeze(1))                    # (C, S, max_per, 3)
        d2 = (rij * rij).sum(-1)                                           # (C, S, max_per)

        keep = valid & (d2 < cutoff * cutoff) & (d2 > 1e-20)
        if not bool(keep.any()):
            continue
        ci = cidx.unsqueeze(1).unsqueeze(2).expand_as(cand)                # (C, S, max_per)
        pis.append(ci[keep])
        pjs.append(q[keep])
        rijs.append(rij[keep])

    if not pis:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z.clone(), torch.zeros(0, 3, dtype=sdtype, device=device)
    return (torch.cat(pis), torch.cat(pjs), torch.cat(rijs))
