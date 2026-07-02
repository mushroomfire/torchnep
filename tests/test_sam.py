# Copyright 2025 Yongchao Wu and the GPUMD development team
# This file is part of GPUMD (Torchnep project).
# GPUMD is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# GPUMD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.

"""Tests for SAM (sharpness-aware minimization, ``sam_rho``)."""
import numpy as np

from torchnep import train_nep
from _common import DATA_DIR

PBTE = DATA_DIR.parent.parent / "example" / "PbTe" / "train.xyz"
NEP_IN = ("type 2 Te Pb\ncutoff 6 4\nn_max 4 4\n"
          "basis_size 6 6\nl_max 4 2 1\nneuron 30\n")


def _write_run_files(tmp_path, n_frames=20, epochs=3):
    nepin = tmp_path / "nep.in"
    nepin.write_text(NEP_IN + f"epoch {epochs}\nbatch 8\n")
    xyz = tmp_path / "train.xyz"
    raw = PBTE.read_text().splitlines()
    out, i, k = [], 0, 0
    while i < len(raw) and k < n_frames:
        na = int(raw[i].strip())
        out += raw[i:i + na + 2]
        i += na + 2; k += 1
    xyz.write_text("\n".join(out) + "\n")
    return str(nepin), str(xyz)


def _train(nepin, xyz, out, **kw):
    kw.setdefault("device", "cpu")
    kw.setdefault("precision", "float64")
    kw.setdefault("print_interval", 100)
    kw.setdefault("restart", False)
    kw.setdefault("checkpoint_interval", 10000)
    kw.setdefault("prediction_interval", 10000)
    train_nep(config_file=nepin, data_file=xyz, output_dir=str(out), **kw)


def _last_loss(out):
    rows = [ln.split() for ln in (out / "loss.out").read_text().splitlines()
            if not ln.startswith("#")]
    return float(rows[-1][1])


def test_sam_runs_and_is_reproducible(tmp_path):
    """sam_rho > 0 trains to a finite loss and is seed-reproducible."""
    nepin, xyz = _write_run_files(tmp_path)
    _train(nepin, xyz, tmp_path / "a", run_seed=7, sam_rho=0.05)
    _train(nepin, xyz, tmp_path / "b", run_seed=7, sam_rho=0.05)
    assert np.isfinite(_last_loss(tmp_path / "a"))
    assert ((tmp_path / "a" / "nep_final.txt").read_text()
            == (tmp_path / "b" / "nep_final.txt").read_text())


def test_sam_changes_training(tmp_path):
    """Same seed, sam on vs off -> different weights (the perturbed
    gradient actually reaches the optimizer)."""
    nepin, xyz = _write_run_files(tmp_path)
    _train(nepin, xyz, tmp_path / "off", run_seed=3, sam_rho=0.0)
    _train(nepin, xyz, tmp_path / "on", run_seed=3, sam_rho=0.05)
    assert ((tmp_path / "off" / "nep_final.txt").read_text()
            != (tmp_path / "on" / "nep_final.txt").read_text())
    # SAM still converges on this easy set: loss within 3x of plain Adam
    assert _last_loss(tmp_path / "on") < 3 * _last_loss(tmp_path / "off")
