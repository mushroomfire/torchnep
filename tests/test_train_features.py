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
"""Restart, fine-tuning and data features of train_nep (single process, CPU).

The fine-tuning and restart options documented in the user guide
(finetune_from, slim_types, recompute_q_scaler, resume_from) and the random
validation split. Each test checks an invariant of the feature: frozen weights
keep their forces, redoing stage 2 from its checkpoint reproduces the run,
parallel preprocessing equals serial.
"""
import numpy as np
import pytest
import torch

from torchnep.data import read_xyz, parse_nep_in
from torchnep.nep import NEPCalculator
from torchnep.train import preprocess_structures
from test_run_seed_and_valid import NEP_IN, XYZ, _train, _write_run_files


def _forces(nep_txt, frame):
    return NEPCalculator(str(nep_txt)).compute(frame["species"], frame["positions"], frame["cell"])["forces"]


def test_finetune_keeps_frozen_weights_nep_txt_and_checkpoint(tmp_path):
    """finetune_from a nep.txt (with slim_types dropping an absent element) and
    from a checkpoint.pt: lr 0 freezes the weights, so the forces — independent
    of the re-solved energy offset b1 — must equal the parent's.
    recompute_q_scaler then rescales the inputs on purpose: the new scaler is
    1 / (max - min) of the descriptors the loaded (trained) model produces on
    the new data — not the parent's scaler, which came from its initial
    descriptor coefficients."""
    _, xyz = _write_run_files(tmp_path, n_frames=16)
    frame = read_xyz(xyz)[0]
    nep4 = NEP_IN.replace("type 3 Cr Co Ni", "type 4 Cr Co Ni Fe")
    (tmp_path / "parent.in").write_text(nep4 + "epoch 2\nbatch 8\n")
    (tmp_path / "frozen.in").write_text(nep4 + "epoch 1\nbatch 8\nlr 0\n")
    parent = tmp_path / "parent"
    _train(str(tmp_path / "parent.in"), xyz, parent, run_seed=0, checkpoint_interval=1)

    slim = tmp_path / "slim"
    _train(str(tmp_path / "frozen.in"), xyz, slim, run_seed=0, slim_types=True,
           finetune_from=str(parent / "nep_final.txt"))
    log = (slim / "output.log").read_text()
    assert "slim_types: ['Cr', 'Co', 'Ni', 'Fe'] -> ['Cr', 'Co', 'Ni']" in log and "[4 -> 3 types]" in log
    assert (slim / "nep_final.txt").read_text().split()[1] == "3"
    torch.testing.assert_close(_forces(slim / "nep_final.txt", frame),
                               _forces(parent / "nep_final.txt", frame), rtol=1e-10, atol=1e-10)

    ckpt = tmp_path / "from_ckpt"
    _train(str(tmp_path / "frozen.in"), xyz, ckpt, run_seed=0, finetune_from=str(parent / "checkpoint.pt"))
    assert "q_scaler: kept from finetune source" in (ckpt / "output.log").read_text()
    torch.testing.assert_close(_forces(ckpt / "nep_final.txt", frame),
                               _forces(parent / "nep_final.txt", frame), rtol=1e-10, atol=1e-10)

    rescaled = tmp_path / "rescaled"
    _train(str(tmp_path / "frozen.in"), xyz, rescaled, run_seed=0, recompute_q_scaler=True,
           finetune_from=str(parent / "nep_final.txt"))
    assert "q_scaler: RECOMPUTED" in (rescaled / "output.log").read_text()
    calc = NEPCalculator(str(parent / "nep_final.txt"))
    q = np.concatenate([np.asarray(calc.get_descriptor(f["species"], f["positions"], f["cell"]))
                        / calc.q_scaler.numpy() for f in read_xyz(xyz)])
    expected = 1.0 / np.maximum(q.max(0) - q.min(0), 1e-10)
    np.testing.assert_allclose(NEPCalculator(str(rescaled / "nep_final.txt")).q_scaler.numpy(), expected, rtol=1e-5)


def test_slim_types_with_all_types_present_changes_nothing(tmp_path):
    nepin, xyz = _write_run_files(tmp_path, n_frames=12, epochs=1)
    out = tmp_path / "out"
    _train(nepin, xyz, out, run_seed=0, slim_types=True)
    assert "slim_types: all types present in data, nothing to remove" in (out / "output.log").read_text()
    assert (out / "nep_final.txt").read_text().split()[1] == "3"


def test_redo_stage2_from_its_checkpoint_reproduces_the_run(tmp_path):
    """The end-of-stage-1 checkpoint exists to redo stage 2 (resume_from).
    With unchanged settings the redone stage 2 must equal the original one."""
    _, xyz = _write_run_files(tmp_path, n_frames=16)
    (tmp_path / "nep.in").write_text(NEP_IN + "epoch 4\nbatch 8\nstage2 1\n")
    first = tmp_path / "first"
    _train(str(tmp_path / "nep.in"), xyz, first, run_seed=0, valid_ratio=0.25)
    assert (first / "checkpoint_stage1.pt").exists()
    redo = tmp_path / "redo"
    _train(str(tmp_path / "nep.in"), xyz, redo, run_seed=0, valid_ratio=0.25,
           resume_from=str(first / "checkpoint_stage1.pt"))
    assert "Resumed from" in (redo / "output.log").read_text()
    a = np.loadtxt(first / "loss.out", ndmin=2)
    b = np.loadtxt(redo / "loss.out", ndmin=2)
    stage2 = a[a[:, 0] >= b[0, 0]]                 # the epochs the redo covers
    np.testing.assert_allclose(b, stage2, rtol=1e-12, atol=0)


def test_resume_reports_changed_validation_and_loss_weights(tmp_path):
    """A restart whose validation split or loss weights differ from the
    checkpoint's warns and resets the best losses instead of silently mixing."""
    nepin, xyz = _write_run_files(tmp_path, n_frames=16, epochs=2)
    out = tmp_path / "out"
    _train(nepin, xyz, out, run_seed=0, valid_ratio=0.25, checkpoint_interval=1)
    (tmp_path / "nep3.in").write_text(NEP_IN + "epoch 3\nbatch 8\nlambda_e 2\n")
    _train(str(tmp_path / "nep3.in"), xyz, out, run_seed=0, valid_ratio=0.5, restart=True)
    log = (out / "output.log").read_text()
    assert "validation settings changed since the checkpoint" in log
    assert "Loss weights changed since checkpoint was saved" in log


def test_random_validation_split_and_unknown_strategy(tmp_path):
    nepin, xyz = _write_run_files(tmp_path, n_frames=16, epochs=1)
    out = tmp_path / "out"
    _train(nepin, xyz, out, run_seed=0, valid_ratio=0.25, valid_strategy="random", prediction_interval=1)
    assert "valid_ratio=0.25: held out 4 frames" in (out / "output.log").read_text()
    assert len(np.loadtxt(out / "energy_test.out", ndmin=2)) == 4
    with pytest.raises(ValueError, match="valid_strategy"):
        _train(nepin, xyz, tmp_path / "bad", run_seed=0, valid_ratio=0.25, valid_strategy="nearest")


def test_parallel_preprocessing_equals_serial(tmp_path):
    """Large sets are preprocessed in worker processes (>= 64 frames): the
    neighbor lists must be identical to the serial ones."""
    frames = read_xyz(str(XYZ)) * 3                        # 72 frames
    (tmp_path / "nep.in").write_text(NEP_IN)
    cfg = parse_nep_in(str(tmp_path / "nep.in"))
    serial = preprocess_structures(frames, cfg, dtype=np.float64, n_workers=1)
    parallel = preprocess_structures(frames, cfg, dtype=np.float64, n_workers=2)
    assert len(parallel) == len(serial) == 72
    for s, p in zip(serial, parallel):
        assert s.keys() == p.keys()
        for k in s:
            if isinstance(s[k], np.ndarray):
                np.testing.assert_array_equal(s[k], p[k])
