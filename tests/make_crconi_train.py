# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project (GPL-3.0-or-later, see LICENSE).
"""Build data/CrCoNi_train.xyz, the labelled training fixture used by the
trainer / data-store tests: 24 frames derived from the first CrCoNi frame
(random rattle sigma 0.08 A, isotropic cell scale 0.97-1.03) and labelled
with energy / forces / virial from data/nep_CrCoNi.txt, so the file is
self-consistent and fully owned by the project.  Run once:

    python make_crconi_train.py
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from torchnep.data import read_xyz
from torchnep.nep import NEPCalculator
from _common import DATA_DIR

base = read_xyz(str(DATA_DIR / "CrCoNi.xyz"))[0]
calc = NEPCalculator(str(DATA_DIR / "nep_CrCoNi.txt"), dtype=torch.float64)
rng = np.random.default_rng(7)
lines = []
for k in range(24):
    s = float(rng.uniform(0.97, 1.03))
    cell = base["cell"] * s
    pos = base["positions"] * s + rng.normal(0.0, 0.08, base["positions"].shape)
    r = calc.compute(base["species"], pos, cell)
    e = float(r["energy"].sum())
    f = r["forces"].numpy()
    v = r["virial"].numpy()            # (N, 9) per-atom -> frame total
    vt = v.sum(0) if v.ndim == 2 else v
    lat = " ".join(f"{x:.8f}" for x in cell.reshape(-1))
    vir = " ".join(f"{x:.8f}" for x in vt.reshape(-1))
    lines.append(str(base["natoms"]))
    lines.append(f'Lattice="{lat}" Properties=species:S:1:pos:R:3:force:R:3 '
                 f'energy={e:.8f} virial="{vir}" pbc="T T T"')
    for sp, p, ff in zip(base["species"], pos, f):
        lines.append(f"{sp} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {ff[0]:.8f} {ff[1]:.8f} {ff[2]:.8f}")
(DATA_DIR / "CrCoNi_train.xyz").write_text("\n".join(lines) + "\n")
print("wrote", DATA_DIR / "CrCoNi_train.xyz", "24 frames x", base["natoms"], "atoms")
