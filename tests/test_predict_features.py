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
"""predict_dataset options beyond E/F/V (CPU): descriptor.out against GPUMD,
the OOM-halving retry and small chunks leave the results unchanged, and the
progress line works without tqdm."""
import sys

import numpy as np
import torch

from torchnep import predict_dataset
from torchnep.nep import NEPCalculator
from _common import DATA_DIR

MODEL = str(DATA_DIR / "nep_CrCoNi.txt")
OUTS = ("energy_train.out", "force_train.out", "virial_train.out", "stress_train.out")


def _predict(out, xyz=DATA_DIR / "CrCoNi.xyz", **kw):
    kw.setdefault("verbose", False)
    predict_dataset(MODEL, str(xyz), output_dir=str(out), dtype="float64", device="cpu", **kw)
    return {name: np.loadtxt(out / name, ndmin=2) for name in OUTS}


def test_descriptor_out_matches_gpumd(tmp_path):
    """Mode 2 writes GPUMD's per-atom descriptor.out (baked reference, 5 frames,
    540 atoms); mode 1 is its per-frame average."""
    ref = np.load(DATA_DIR / "CrCoNi.gpumd.npz")["D_per_atom"]
    _predict(tmp_path / "atom", output_descriptor=2)
    per_atom = np.loadtxt(tmp_path / "atom" / "descriptor.out")
    np.testing.assert_allclose(per_atom, ref, rtol=1e-5, atol=2e-4)
    _predict(tmp_path / "frame", output_descriptor=1)
    per_frame = np.loadtxt(tmp_path / "frame" / "descriptor.out")
    natoms = np.loadtxt(tmp_path / "frame" / "force_train.out").shape[0] // len(per_frame)
    assert per_frame.shape == (5, ref.shape[1]) and natoms * 5 == len(ref)
    np.testing.assert_allclose(per_frame, per_atom.reshape(5, natoms, -1).mean(1), rtol=1e-9, atol=1e-12)


def test_oom_retry_and_small_chunks_change_nothing(tmp_path, monkeypatch):
    """A batch that runs out of device memory is redone at half the size, and
    a file streamed in many small chunks is predicted frame by frame: the
    output files must equal the one-batch, one-chunk run."""
    xyz = DATA_DIR / "CrCoNi_train.xyz"                       # 24 frames
    ref = _predict(tmp_path / "ref", xyz, batch_size=24)
    real, sizes = NEPCalculator.compute_batch, []

    def flaky(self, batch, **kw):
        sizes.append(batch["num_structures"])
        if len(sizes) == 1:
            raise torch.OutOfMemoryError("simulated")
        return real(self, batch, **kw)

    monkeypatch.setattr(NEPCalculator, "compute_batch", flaky)
    got = _predict(tmp_path / "oom", xyz, batch_size=24)
    assert sizes == [24, 16, 8]
    monkeypatch.setattr(NEPCalculator, "compute_batch", real)
    chunked = _predict(tmp_path / "chunks", xyz, batch_size=24, chunk_atoms=300)
    for name in OUTS:
        np.testing.assert_allclose(got[name], ref[name], rtol=1e-12, atol=0, err_msg=name)
        np.testing.assert_allclose(chunked[name], ref[name], rtol=1e-12, atol=0, err_msg=name)


def test_progress_line_without_tqdm(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "tqdm", None)            # import tqdm -> ImportError
    _predict(tmp_path, verbose=True)
    out = capsys.readouterr().out
    assert "0/5 frames   0.0%" in out and "5/5 frames 100.0%" in out
