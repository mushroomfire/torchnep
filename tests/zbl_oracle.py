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
"""Independent reference for the ZBL term: a line-by-line numpy port of
GPUMD's ``find_force_ZBL`` (src/main_nep/nep.cu) and ``find_f_and_fp_zbl``
(src/utilities/nep_utilities.cuh), in float64.

It does not import torchnep. Its input is the TEXT of nep.in (``type``,
``zbl``, ``use_typewise_cutoff_zbl``) and of zbl.in, read with GPUMD's
rules, so it cannot inherit an interpretation torchnep makes internally
(the PR #29 bug was such an interpretation: the cap of the typewise pair
cutoff came from an internal attribute instead of the ``zbl`` value). The
constants are copied from the GPUMD source; ``test_zbl_matrix`` checks
torchnep's copies against them.

GPUMD's rules, per element pair (t1, t2):
  universal  ``zbl X``                     rc_inner = X / 2, rc_outer = X
  typewise   ``zbl X`` + ``use_typewise_cutoff_zbl f``
                                           rc_inner = 0,
                                           rc_outer = min(f * (R_1 + R_2), X)
  flexible   ``zbl <file>`` (GPUMD: zbl.in) row of the pair (t1 <= t2,
             index t1*T - t1*(t1-1)/2 + t2-t1): rc_inner rc_outer a1 b1 .. a4 b4;
             a typewise factor is ignored
and E_pair = K Z_1 Z_2 / d * phi(d * a_inv) * fc(d), split half / half
over the two atoms, with fc = 1 below rc_inner, 0.5 cos(pi (d - rc_inner) /
(rc_outer - rc_inner)) + 0.5 up to rc_outer, 0 beyond.
"""
import itertools

import numpy as np

# GPUMD nep_utilities.cuh / NEP_CPU nep_utilities.h
K_C_SP = 14.399645
ZBL_UNIVERSAL = (0.18175, 3.1998, 0.50986, 0.94229, 0.28022, 0.4029, 0.02817, 0.20162)
COVALENT_RADIUS = (
    0.426667, 0.613333, 1.6, 1.25333, 1.02667, 1.0, 0.946667, 0.84, 0.853333,
    0.893333, 1.86667, 1.66667, 1.50667, 1.38667, 1.46667, 1.36, 1.32, 1.28,
    2.34667, 2.05333, 1.77333, 1.62667, 1.61333, 1.46667, 1.42667, 1.38667, 1.33333,
    1.32, 1.34667, 1.45333, 1.49333, 1.45333, 1.53333, 1.46667, 1.52, 1.56,
    2.52, 2.22667, 1.96, 1.85333, 1.76, 1.65333, 1.53333, 1.50667, 1.50667,
    1.44, 1.53333, 1.64, 1.70667, 1.68, 1.68, 1.64, 1.76, 1.74667,
    2.78667, 2.34667, 2.16, 1.96, 2.10667, 2.09333, 2.08, 2.06667, 2.01333,
    2.02667, 2.01333, 2.0, 1.98667, 1.98667, 1.97333, 2.04, 1.94667, 1.82667,
    1.74667, 1.64, 1.57333, 1.54667, 1.48, 1.49333, 1.50667, 1.76, 1.73333,
    1.73333, 1.81333, 1.74667, 1.84, 1.89333, 2.68, 2.41333, 2.22667, 2.10667,
    2.02667, 2.04, 2.05333, 2.06667)
SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu").split()


class ZBLSpec:
    """The ZBL settings of a nep.in, read the way GPUMD reads them."""

    def __init__(self, nep_in_text, zbl_in_text=None):
        self.type_names, self.zbl, self.factor, self.table = None, None, None, None
        for line in nep_in_text.splitlines():
            tok = line.split("#")[0].split()
            if not tok:
                continue
            if tok[0] == "type":
                self.type_names = tok[2:2 + int(tok[1])]
            elif tok[0] == "zbl":
                try:
                    self.zbl = float(tok[1])
                except ValueError:                 # zbl <file>: flexible
                    self.table = np.array(zbl_in_text.split(), dtype=float).reshape(-1, 10)
            elif tok[0] == "use_typewise_cutoff_zbl":
                self.factor = float(tok[1]) if len(tok) > 1 else 0.7
        self.Z = [SYMBOLS.index(s) + 1 for s in self.type_names]

    def pair(self, t1, t2):
        """(rc_inner, rc_outer, (a1, b1, ..., a4, b4)) of the element pair."""
        T = len(self.type_names)
        if self.table is not None:
            a, b = min(t1, t2), max(t1, t2)
            row = self.table[a * T - (a * (a - 1)) // 2 + (b - a)]
            return row[0], row[1], tuple(row[2:])
        if self.factor is not None:
            z1, z2 = self.Z[t1], self.Z[t2]
            rc = min((COVALENT_RADIUS[z1 - 1] + COVALENT_RADIUS[z2 - 1]) * self.factor, self.zbl)
            return 0.0, rc, ZBL_UNIVERSAL
        return self.zbl / 2.0, self.zbl, ZBL_UNIVERSAL

    def max_rc(self):
        T = len(self.type_names)
        return max(self.pair(a, b)[1] for a in range(T) for b in range(T))


def f_and_fp(zizj, a_inv, rc_inner, rc_outer, d, coef):
    """GPUMD find_f_and_fp_zbl: pair energy f and its derivative fp = df/dd."""
    x = d * a_inv
    f = fp = 0.0
    for k in range(4):
        tmp = coef[2 * k] * np.exp(-coef[2 * k + 1] * x)
        f += tmp
        fp -= coef[2 * k + 1] * tmp
    f *= zizj
    fp *= zizj * a_inv
    fp = fp / d - f / d ** 2
    f /= d
    if d < rc_inner:
        fc, fcp = 1.0, 0.0
    elif d < rc_outer:
        w = np.pi / (rc_outer - rc_inner)
        fc = np.cos(w * (d - rc_inner)) * 0.5 + 0.5
        fcp = -np.sin(w * (d - rc_inner)) * w * 0.5
    else:
        fc, fcp = 0.0, 0.0
    return f * fc, fp * fc + f * fcp


def _pairs(positions, cell, rc):
    """Every directed pair (i, j, r_ij = x_j + shift - x_i) with |r_ij| < rc,
    over all periodic images (brute force, any cell size)."""
    cell = np.asarray(cell, float).reshape(3, 3)
    vol = abs(np.linalg.det(cell))
    heights = [vol / np.linalg.norm(np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])) for k in range(3)]
    reps = [int(np.ceil(rc / h)) for h in heights]
    pos = np.asarray(positions, float)
    out = []
    for s in itertools.product(*[range(-r, r + 1) for r in reps]):
        shift = np.array(s, float) @ cell
        rij = pos[None, :, :] + shift - pos[:, None, :]
        d = np.linalg.norm(rij, axis=-1)
        for i, j in zip(*np.nonzero(d < rc)):
            if i == j and not any(s):
                continue
            out.append((i, j, rij[i, j]))
    return out


def zbl_reference(spec, species, positions, cell):
    """Per-atom energy (N,), forces (N, 3) and the total virial (3, 3) of the
    ZBL term, GPUMD's accumulation: for each directed pair (n1, n2),
    pe[n1] += f / 2, f12 = r12 * fp / d / 2, F[n1] += f12, F[n2] -= f12,
    virial -= r12 (x) f12."""
    types = [spec.type_names.index(s) for s in species]
    N = len(types)
    pe, F, W = np.zeros(N), np.zeros((N, 3)), np.zeros((3, 3))
    for n1, n2, r12 in _pairs(positions, cell, spec.max_rc() + 1e-9):
        t1, t2 = types[n1], types[n2]
        rc_inner, rc_outer, coef = spec.pair(t1, t2)
        d = float(np.linalg.norm(r12))
        z1, z2 = spec.Z[t1], spec.Z[t2]
        a_inv = (z1 ** 0.23 + z2 ** 0.23) * 2.134563
        f, fp = f_and_fp(K_C_SP * z1 * z2, a_inv, rc_inner, rc_outer, d, coef)
        f12 = r12 * fp / d * 0.5
        pe[n1] += 0.5 * f
        F[n1] += f12
        F[n2] -= f12
        W -= np.outer(r12, f12)
    return pe, F, W
