# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project (GPL-3.0-or-later, see LICENSE).
"""Derive the two extra ZBL fixtures from the CrCoNi model (run once; the
GPUMD references are then baked with bake_fixtures.py):

  data/nep_CrCoNi_unizbl.txt   same weights, universal ZBL (header
                               "zbl 1.25 2.5", no typewise factor)
  data/nep_CrCoNi_flexzbl.txt  same weights, FLEXIBLE ZBL: a zbl.in with
                               pair-specific cutoffs / screening coefficients
                               (data/CrCoNi_flexzbl.zbl.in), written by
                               torchnep itself ("zbl 0 0" + table at the end)
Both use data/CrCoNi.xyz (original + compressed frames).

    python make_flexzbl_fixture.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from torchnep.data import read_xyz
from torchnep.model import NEPModel
from torchnep.train import preprocess_structures, compute_max_neighbors
from torchnep.constants import ZBL_PARA
from _common import DATA_DIR, parse_nep_header

src = DATA_DIR / "nep_CrCoNi.txt"
hdr = parse_nep_header(src)
T = hdr["num_types"]

# 1) universal ZBL: same file, typewise factor dropped from the zbl line
lines = src.read_text().splitlines()
assert lines[1].startswith("zbl ")
lines[1] = "zbl 1.25 2.5"
(DATA_DIR / "nep_CrCoNi_unizbl.txt").write_text("\n".join(lines) + "\n")

# 2) flexible ZBL: per-pair table, rc_inner 0.7-1.0 A, rc_outer 1.6-2.4 A,
#    coefficients = universal * (0.8-1.2)
rng = np.random.default_rng(2025)
rows = []
for k in range(T * (T + 1) // 2):
    rc_o = float(rng.uniform(1.6, 2.4)); rc_i = float(rng.uniform(0.7, 1.0))
    rows.append([rc_i, rc_o] + [float(c * rng.uniform(0.8, 1.2)) for c in ZBL_PARA])
zbl_in = DATA_DIR / "CrCoNi_flexzbl.zbl.in"
zbl_in.write_text("\n".join(" ".join(f"{v:.10g}" for v in r) for r in rows) + "\n")

cfg = dict(num_types=T, type_names=hdr["type_names"], cutoff_radial=hdr["rc_radial"],
           cutoff_angular=hdr["rc_angular"], n_max_radial=hdr["n_max_radial"],
           n_max_angular=hdr["n_max_angular"], basis_size_radial=hdr["basis_size_radial"],
           basis_size_angular=hdr["basis_size_angular"], l_max=hdr["l_max"], neuron=hdr["neuron"],
           zbl=hdr["zbl_outer"], zbl_flexible=rows)
model = NEPModel(cfg).double()
model.load_weights_from_nep_txt(str(src))
frames = read_xyz(str(DATA_DIR / "CrCoNi.xyz"))
structs = preprocess_structures(frames, cfg, np.float64)
nn_r, nn_a = compute_max_neighbors(structs)
model.save_nep_txt(str(DATA_DIR / "nep_CrCoNi_flexzbl.txt"), nn_r, nn_a)
print("wrote", DATA_DIR / "nep_CrCoNi_unizbl.txt", DATA_DIR / "nep_CrCoNi_flexzbl.txt", zbl_in)
for k, s in enumerate(structs):
    print("frame", k, "natoms", s["natoms"], "min NN", round(float(np.linalg.norm(s["rij_rad"], axis=1).min()), 3))
