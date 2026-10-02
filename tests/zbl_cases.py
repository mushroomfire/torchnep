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
"""The ZBL test matrix: element systems, ZBL settings and structures.

Shared by ``make_zbl_fixtures.py`` (writes tests/data/zbl/) and
``test_zbl_matrix.py``. Every case is a nep.in (plus a zbl.in for the
flexible ones); its nep.txt holds random weights written by torchnep and its
``.nepcpu.npz`` the energies / forces / virials NEP_CPU computes from that
nep.txt in double precision.

Systems (the type order is not sorted by Z on purpose):
  PbHCsI  Pb H Cs I: with ``use_typewise_cutoff_zbl 0.7`` and ``zbl 2.5`` the
          pairs Cs-Cs (3.90 A), Cs-I (3.18 A) and Cs-Pb (3.16 A) are capped at
          2.5 A, Pb-Pb, Pb-I, I-I and every H pair are not
  Cs      a single type (T = 1 tables)
  CrCoNi  ``zbl 2`` with typewise 0.7: only Cr-Cr (2.05 A) is capped

Structures per system: isolated dimers of every element pair at distances
on both sides of every cutoff of every case (rc_inner, rc_outer and the
uncapped typewise value), randomly oriented; dense random cells; a strongly
sheared cell; a cell thinner than the ZBL cutoff (an atom repels its own
periodic image).
"""
import itertools
from pathlib import Path

import numpy as np

from zbl_oracle import COVALENT_RADIUS, SYMBOLS, ZBL_UNIVERSAL, ZBLSpec

ZBL_DIR = Path(__file__).resolve().parent / "data" / "zbl"

ARCH = "n_max 4 4\nbasis_size 4 4\nl_max 4 2 0\nneuron 8\n"

SYSTEMS = {
    "PbHCsI": ["Pb", "H", "Cs", "I"],
    "Cs": ["Cs"],
    "CrCoNi": ["Cr", "Co", "Ni"],
}

_PCUT = "cutoff 6 4 5 3.5 5.5 3 4.5 3.5\n"       # per species, angular min 3 A


def _flex_rows(num_types, seed, rc_max):
    """A zbl.in table: per-pair cutoffs (some with rc_inner 0) and screening
    coefficients scaled by 0.8 .. 1.2, rounded to 6 decimals (GPUMD reads
    zbl.in as float)."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(num_types * (num_types + 1) // 2):
        rc_o = rng.uniform(1.6, rc_max)
        rc_i = 0.0 if k % 3 == 0 else rng.uniform(0.3, 0.8) * rc_o
        coef = [c * rng.uniform(0.8, 1.2) for c in ZBL_UNIVERSAL]
        rows.append([round(v, 6) for v in [rc_i, rc_o] + coef])
    return rows


def _case(system, zbl, cutoff="cutoff 6 4\n", flex=None):
    return {"system": system, "types": SYSTEMS[system], "cutoff": cutoff,
            "zbl": zbl, "flex": flex}


CASES = {
    # universal
    "PbHCsI_uni25":    _case("PbHCsI", "zbl 2.5\n"),
    "PbHCsI_uni14":    _case("PbHCsI", "zbl 1.4\n"),
    # typewise: cap active for some pairs / for none / for almost all
    "PbHCsI_tw25":     _case("PbHCsI", "zbl 2.5\nuse_typewise_cutoff_zbl 0.7\n"),
    "PbHCsI_tw30f05":  _case("PbHCsI", "zbl 3\nuse_typewise_cutoff_zbl 0.5\n"),
    "PbHCsI_tw12f10":  _case("PbHCsI", "zbl 1.2\nuse_typewise_cutoff_zbl 1.0\n"),
    # flexible (zbl.in); with a typewise line too, which GPUMD ignores
    "PbHCsI_flex":     _case("PbHCsI", "zbl PbHCsI_flex.zbl.in\n",
                             flex=_flex_rows(4, 1, 2.8)),
    "PbHCsI_flextw":   _case("PbHCsI", "zbl PbHCsI_flextw.zbl.in\nuse_typewise_cutoff_zbl 0.7\n",
                             flex=_flex_rows(4, 1, 2.8)),
    # per-species cutoffs with each ZBL mode
    "PbHCsI_pc_uni30": _case("PbHCsI", "zbl 3\n", cutoff=_PCUT),
    "PbHCsI_pc_tw25":  _case("PbHCsI", "zbl 2.5\nuse_typewise_cutoff_zbl 0.7\n", cutoff=_PCUT),
    "PbHCsI_pc_flex":  _case("PbHCsI", "zbl PbHCsI_pc_flex.zbl.in\n", cutoff=_PCUT,
                             flex=_flex_rows(4, 2, 3.0)),
    # one type
    "Cs_uni20":        _case("Cs", "zbl 2\n"),
    "Cs_tw25":         _case("Cs", "zbl 2.5\nuse_typewise_cutoff_zbl 0.7\n"),
    "Cs_flex":         _case("Cs", "zbl Cs_flex.zbl.in\n", flex=_flex_rows(1, 3, 2.8)),
    # transition metals, cap active for Cr-Cr only
    "CrCoNi_tw20":     _case("CrCoNi", "zbl 2\nuse_typewise_cutoff_zbl 0.7\n"),
}


def nep_in_text(name):
    c = CASES[name]
    return f"type {len(c['types'])} {' '.join(c['types'])}\n" + c["cutoff"] + ARCH + c["zbl"]


def zbl_in_text(name):
    rows = CASES[name]["flex"]
    if rows is None:
        return None
    return "\n".join(" ".join(f"{v:.6f}" for v in r) for r in rows) + "\n"


def spec(name):
    return ZBLSpec(nep_in_text(name), zbl_in_text(name))


def nep_txt(name):
    return ZBL_DIR / f"nep_{name}.txt"


def nepcpu_ref(name):
    return ZBL_DIR / f"{name}.nepcpu.npz"


def xyz(system):
    return ZBL_DIR / f"{system}.xyz"


# --------------------------------------------------------------------------
# structures

def _radius(sym):
    return COVALENT_RADIUS[SYMBOLS.index(sym)]


def dimer_distances(system, a, b):
    """Distances for the element pair (a, b): a coarse grid; for every case of
    ``system``, +-4 mA around its cutoffs of this pair, 1/4, 1/2, 3/4 of the
    way through its switching window and two points below rc_inner; +-4 mA
    and the middle of the window between the cap and the uncapped typewise
    cutoff f * (R_a + R_b) (where PR #29 was)."""
    types = SYSTEMS[system]
    ta, tb = types.index(a), types.index(b)
    d = set(np.round(np.arange(0.6, 3.61, 0.2), 4))
    for name, c in CASES.items():
        if c["system"] != system:
            continue
        s = spec(name)
        rc_i, rc_o, _ = s.pair(ta, tb)
        marks = [rc_i, rc_o] if rc_i > 0 else [rc_o]
        d.update(rc_i + (rc_o - rc_i) * w for w in (0.25, 0.5, 0.75))
        if rc_i > 0:
            d.update([0.6 * rc_i, 0.85 * rc_i])
        if s.factor is not None and s.table is None:
            uncapped = s.factor * (_radius(a) + _radius(b))
            marks.append(uncapped)
            if uncapped > rc_o:
                d.add(0.5 * (rc_o + uncapped))
        for m in marks:
            d.update([m - 0.004, m + 0.004])
    return sorted({round(float(x), 6) for x in d if 0.25 <= x <= 3.7})


def _random_direction(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def dimer_frames(system, rng, spacing=7.0, grid=(3, 3, 3)):
    """All dimers of all element pairs, one per grid site (sites `spacing`
    apart, box >= 3 x the radial cutoff so the tiled calculator tiles)."""
    types = SYSTEMS[system]
    dimers = [(a, b, d) for a, b in itertools.combinations_with_replacement(types, 2)
              for d in dimer_distances(system, a, b)]
    sites = [np.array(s) * spacing for s in itertools.product(*[range(n) for n in grid])]
    cell = np.diag([spacing * n for n in grid]).astype(float)
    frames = []
    for k in range(0, len(dimers), len(sites)):
        species, pos = [], []
        for (a, b, d), site in zip(dimers[k:k + len(sites)], sites):
            c = site + rng.uniform(-0.5, 0.5, 3)
            u = _random_direction(rng)
            species += [a, b]
            pos += [c - 0.5 * d * u, c + 0.5 * d * u]
        frames.append({"species": species, "positions": np.array(pos), "cell": cell})
    return frames


def _min_image_distance(cell, p, q):
    inv = np.linalg.inv(cell)
    best = np.inf
    for s in itertools.product((-1, 0, 1), repeat=3):
        dvec = q - p
        frac = dvec @ inv
        frac -= np.round(frac)
        best = min(best, np.linalg.norm((frac + s) @ cell))
    return best


def random_frame(types, rng, cell, natoms, dmin=0.7):
    species, pos = [], []
    while len(pos) < natoms:
        x = rng.uniform(0, 1, 3) @ cell
        if all(_min_image_distance(cell, x, y) >= dmin for y in pos):
            pos.append(x)
            species.append(types[rng.integers(len(types))])
    return {"species": species, "positions": np.array(pos), "cell": cell}


def bulk_frames(system, rng):
    types = SYSTEMS[system]
    frames = []
    for _ in range(3):                                      # dense random cells
        cell = np.diag(rng.uniform(5.0, 6.0, 3)) + rng.uniform(-0.6, 0.6, (3, 3))
        frames.append(random_frame(types, rng, cell, 16))
    sheared = np.array([[6.5, 0.0, 0.0], [3.9, 5.6, 0.0], [-2.7, 2.2, 5.9]])
    frames.append(random_frame(types, rng, sheared, 18))
    thin = np.array([[2.4, 0.0, 0.0], [0.3, 6.5, 0.0], [0.2, -0.4, 6.8]])  # own image at 2.4 A
    frames.append(random_frame(types, rng, thin, 6, dmin=0.9))
    return frames


def system_frames(system, seed=0):
    rng = np.random.default_rng(seed)
    return dimer_frames(system, rng) + bulk_frames(system, rng)
