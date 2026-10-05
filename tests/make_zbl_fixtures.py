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
"""Write the ZBL matrix fixtures in tests/data/zbl/ (see zbl_cases.py).

    python tests/make_zbl_fixtures.py inputs
        structures (<system>.xyz), <case>.nep.in (+ .zbl.in) and nep_<case>.txt
        (random weights, q_scaler from the structures, written by torchnep)
    NEP_CPU=/path/to/NEP_CPU python tests/make_zbl_fixtures.py nepcpu
        compiles a small driver (DRIVER_CPP) against NEP_CPU ($CXX, default c++) and
        writes <case>.nepcpu.npz: per-atom energy, forces, per-frame virial
        (3 x 3), double precision
    python tests/make_zbl_fixtures.py gpumd-prepare <case> <dir>
    python tests/make_zbl_fixtures.py gpumd-collect <case> <dir>
        inputs for GPUMD's nep (prediction mode) on a GPU machine, and
        <case>.gpumd.npz from its *_train.out files

Rerun ``inputs`` only when zbl_cases.py changes; then rerun ``nepcpu``.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
sys.path.insert(0, str(THIS_DIR))

from torchnep.data import parse_nep_in, read_xyz
from torchnep.model import NEPModel
from torchnep.train import (StreamDataStore, compute_max_neighbors, compute_q_scaler,
                            preprocess_structures)
from _common import write_gpumd_xyz
from zbl_cases import (CASES, SYSTEMS, ZBL_DIR, nep_in_text, nep_txt, nepcpu_ref,
                       spec, system_frames, xyz, zbl_in_text)


def write_inputs():
    ZBL_DIR.mkdir(parents=True, exist_ok=True)
    for system in SYSTEMS:
        frames = system_frames(system)
        for f in frames:
            f["natoms"] = len(f["species"])
        write_gpumd_xyz(frames, xyz(system))
        print(f"{xyz(system).name}: {len(frames)} frames, "
              f"{sum(f['natoms'] for f in frames)} atoms")
    for name, case in CASES.items():
        nep_in = ZBL_DIR / f"{name}.nep.in"
        nep_in.write_text(nep_in_text(name))
        if case["flex"] is not None:
            (ZBL_DIR / f"{name}.zbl.in").write_text(zbl_in_text(name))
        cfg = parse_nep_in(str(nep_in))
        torch.manual_seed(zlib.crc32(name.encode()))
        model = NEPModel(cfg).double()
        frames = read_xyz(str(xyz(case["system"])))
        structs = preprocess_structures(frames, cfg, np.float64)
        nn_r, nn_a = compute_max_neighbors(structs)        # before the store takes the pairs
        store = StreamDataStore(structs, torch.device("cpu"), torch.float64, config=cfg)
        q_min, q_max = compute_q_scaler(model, store)
        model.q_scaler.copy_(1.0 / torch.clamp(q_max - q_min, min=1e-10))
        model.save_nep_txt(str(nep_txt(name)), nn_r, nn_a)
        print(f"{nep_txt(name).name}: max rc {spec(name).max_rc():.4f} A")


# The C++ driver compiled against NEP_CPU (kept here as a string: the
# repository holds no C++ source of its own).
DRIVER_CPP = r"""// NEP_CPU reference driver: E / F / per-atom virial for every frame of a plain
// text file, in double precision.
//   input : nframes, then per frame: N / "ax ay az bx by bz cx cy cz" / N x "type x y z"
//   output: per frame N lines "e fx fy fz vxx vxy vxz vyx vyy vyz vzx vzy vzz"
#include "nep.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::cerr << "usage: driver nep.txt frames.txt out.txt\n";
    return 1;
  }
  NEP nep(argv[1]);
  std::ifstream in(argv[2]);
  FILE* out = fopen(argv[3], "w");
  int nframes;
  in >> nframes;
  for (int f = 0; f < nframes; ++f) {
    int N;
    in >> N;
    double a[9];
    for (int k = 0; k < 9; ++k) in >> a[k];
    // NEP_CPU: box = ax, bx, cx, ay, by, cy, az, bz, cz (lattice vectors as columns)
    std::vector<double> box = {a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]};
    std::vector<int> type(N);
    std::vector<double> pos(3 * N), pe(N), force(3 * N), virial(9 * N);
    for (int n = 0; n < N; ++n) in >> type[n] >> pos[n] >> pos[n + N] >> pos[n + 2 * N];
    nep.compute(type, box, pos, pe, force, virial);
    for (int n = 0; n < N; ++n) {
      fprintf(out, "%.17g %.17g %.17g %.17g", pe[n], force[n], force[n + N], force[n + 2 * N]);
      for (int k = 0; k < 9; ++k) fprintf(out, " %.17g", virial[n + k * N]);
      fprintf(out, "\n");
    }
  }
  fclose(out);
  return 0;
}
"""


def _driver():
    src = Path(os.environ["NEP_CPU"]).expanduser() / "src"
    build = Path(tempfile.mkdtemp(prefix="nepcpu_"))
    (build / "driver.cpp").write_text(DRIVER_CPP)
    exe = build / "nepcpu"
    cxx = os.environ.get("CXX", "c++")
    subprocess.run([cxx, "-O2", "-std=c++17", f"-I{src}", str(build / "driver.cpp"),
                    str(src / "nep.cpp"), str(src / "neighbor_nep.cpp"),
                    str(src / "ewald_nep.cpp"), "-o", str(exe)], check=True)
    return exe


def nepcpu_compute(exe, nep_file, frames, type_names):
    """Run the NEP_CPU driver; per-atom energy (N,), forces (N, 3), the
    per-frame total virial (F, 3, 3) and the per-atom virial (N, 9)."""
    with tempfile.TemporaryDirectory() as d:
        fin, fout = Path(d) / "frames.txt", Path(d) / "out.txt"
        lines = [str(len(frames))]
        for fr in frames:
            lines.append(str(len(fr["species"])))
            lines.append(" ".join(f"{v:.17g}" for v in np.asarray(fr["cell"], float).reshape(9)))
            for s, p in zip(fr["species"], np.asarray(fr["positions"], float)):
                lines.append(f"{type_names.index(s)} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g}")
        fin.write_text("\n".join(lines) + "\n")
        subprocess.run([str(exe), str(nep_file), str(fin), str(fout)], check=True,
                       capture_output=True)
        out = np.loadtxt(fout).reshape(-1, 13)
    vir, off = [], 0
    for fr in frames:
        n = len(fr["species"])
        vir.append(out[off:off + n, 4:].sum(0).reshape(3, 3))
        off += n
    return out[:, 0], out[:, 1:4], np.array(vir), out[:, 4:]


def _zbl_only_nep_txt(name, dst):
    """The case's nep.txt with the network output switched off (w1 = 0,
    b1 = 0): NEP_CPU then returns the ZBL term alone."""
    model = NEPModel(parse_nep_in(str(ZBL_DIR / f"{name}.nep.in"))).double()
    model.load_weights_from_nep_txt(str(nep_txt(name)))
    with torch.no_grad():
        for net in model.fitting_nets:
            net.w1.zero_()
        model.b1.zero_()
    cut = next(ln for ln in nep_txt(name).read_text().splitlines() if ln.startswith("cutoff"))
    nn_r, nn_a = (int(x) for x in cut.split()[-2:])
    model.save_nep_txt(str(dst), nn_r, nn_a)


def bake_nepcpu():
    exe = _driver()
    for name, case in CASES.items():
        frames = read_xyz(str(xyz(case["system"])))
        e, f, v, va = nepcpu_compute(exe, nep_txt(name), frames, case["types"])
        with tempfile.TemporaryDirectory() as d:
            _zbl_only_nep_txt(name, Path(d) / "nep.txt")
            ez, fz, vz, _ = nepcpu_compute(exe, Path(d) / "nep.txt", frames, case["types"])
        np.savez_compressed(nepcpu_ref(name), e_atom=e, forces=f, virial=v, virial_atom=va,
                 e_atom_zbl=ez, forces_zbl=fz, virial_zbl=vz)
        print(f"{nepcpu_ref(name).name}: |F| max {np.abs(f).max():.1f} eV/A, "
              f"ZBL |e| max {np.abs(ez).max():.1f} eV")


def gpumd_prepare(name, workdir):
    """nep.in / nep.txt / train.xyz (+ zbl.in) for GPUMD's nep in prediction
    mode. GPUMD takes the ZBL settings from nep.in, and the flexible table from
    zbl.in in the working directory with a numeric `zbl` line."""
    case = CASES[name]
    workdir.mkdir(parents=True, exist_ok=True)
    text = nep_in_text(name).replace("l_max 4 2 0", "l_max 4 2 0 0 0")
    if case["flex"] is not None:
        text = "\n".join(f"zbl {spec(name).max_rc():.6f}" if ln.startswith("zbl ") else ln
                         for ln in text.splitlines()) + "\n"
        (workdir / "zbl.in").write_text(zbl_in_text(name))
    (workdir / "nep.in").write_text(text + "prediction 1\nbatch 1\n")
    shutil.copy(nep_txt(name), workdir / "nep.txt")
    shutil.copy(xyz(case["system"]), workdir / "train.xyz")


def gpumd_collect(name, workdir):
    e = np.loadtxt(workdir / "energy_train.out").reshape(-1, 2)
    f = np.loadtxt(workdir / "force_train.out").reshape(-1, 6)
    v = np.loadtxt(workdir / "virial_train.out").reshape(-1, 12)
    np.savez(ZBL_DIR / f"{name}.gpumd.npz", E_per_atom=e[:, 0], F=f[:, :3], V_per_atom=v[:, :6])


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "inputs":
        write_inputs()
    elif cmd == "nepcpu":
        bake_nepcpu()
    elif cmd in ("gpumd-prepare", "gpumd-collect") and len(sys.argv) == 4:
        (gpumd_prepare if cmd == "gpumd-prepare" else gpumd_collect)(sys.argv[2], Path(sys.argv[3]))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
