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
"""Features of the sharded trainer, 2 ranks on CPU/gloo (opt-in: TORCHNEP_TEST_DDP=1).

train_nep_sharded has its own implementation of data loading, validation,
resume, fine-tuning and early stopping. Large runs depend on exactly these
paths — a multi-GB training file is streamed, a separate validation file is
sharded, long trainings resume from checkpoints — so every test checks what
the feature must do, not only that it runs.
"""
import json
import os
import subprocess

import numpy as np
import pytest
import torch

from _common import DATA_DIR, torchrun_cmd

pytestmark = pytest.mark.skipif(os.environ.get("TORCHNEP_TEST_DDP") != "1",
                                reason="multi-process test is local-only (set TORCHNEP_TEST_DDP=1)")

XYZ = DATA_DIR / "CrCoNi_train.xyz"          # 24 frames
NEP_IN = "cutoff 6 4\nn_max 4 4\nbasis_size 8 8\nl_max 4 2 0\nneuron 8\nbatch 4\n"
TYPES3 = "type 3 Cr Co Ni\n"

_RUNNER = """
import json, sys
from torchnep.train_sharded import train_nep_sharded
train_nep_sharded(sys.argv[1], sys.argv[2], output_dir=sys.argv[3], precision="float64",
                  print_interval=1, **json.loads(sys.argv[4]))
"""


def _frames(path, first, n):
    """Write frames [first, first + n) of the CrCoNi set to ``path``."""
    raw = XYZ.read_text().splitlines()
    out, i, k = [], 0, 0
    while i < len(raw):
        na = int(raw[i].strip())
        if first <= k < first + n:
            out += raw[i:i + na + 2]
        i += na + 2
        k += 1
    path.write_text("\n".join(out) + "\n")
    return str(path)


def _nep_in(path, head, tail):
    path.write_text(head + NEP_IN + tail)
    return str(path)


def _run(tmp_path, nep_in, xyz, out, env=None, **kw):
    """One 2-rank run; returns output.log. ``kw`` goes to train_nep_sharded."""
    cmd = torchrun_cmd(2)
    if not cmd:
        pytest.skip("torchrun not on PATH")
    runner = tmp_path / "runner.py"
    runner.write_text(_RUNNER)
    root = str(DATA_DIR.parent.parent)
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="",
             PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    e.update(env or {})
    kw.setdefault("checkpoint_interval", 10**6)
    kw.setdefault("prediction_interval", 10**6)
    r = subprocess.run(cmd + [str(runner), nep_in, xyz, str(out), json.dumps(kw)],
                       capture_output=True, text=True, env=e, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return (out / "output.log").read_text()


def _loss(out):
    return np.loadtxt(out / "loss.out", ndmin=2)


def test_streamed_loading_matches_in_memory(tmp_path):
    """A file above TORCHNEP_STREAM_THRESHOLD is indexed by rank 0 and seek-read
    per rank (the path a multi-GB training set takes). With the same seed it
    must train exactly like the in-memory path. A separate validation file is
    sharded across the ranks in both cases, and interim predictions run."""
    xyz = _frames(tmp_path / "train.xyz", 0, 16)
    valid = _frames(tmp_path / "valid.xyz", 16, 8)
    nepin = _nep_in(tmp_path / "nep.in", TYPES3, "epoch 3\n")
    kw = dict(restart=False, run_seed=3, valid_file=valid, prediction_interval=1)
    log_s = _run(tmp_path, nepin, xyz, tmp_path / "streamed", env={"TORCHNEP_STREAM_THRESHOLD": "0"}, **kw)
    log_m = _run(tmp_path, nepin, xyz, tmp_path / "memory", **kw)
    assert "streamed shard loading" in log_s and "streamed shard loading" not in log_m
    for log in (log_s, log_m):
        assert "validation file" in log and "sharded across 2 ranks" in log
    np.testing.assert_allclose(_loss(tmp_path / "streamed"), _loss(tmp_path / "memory"), rtol=1e-12, atol=0)
    assert len(np.loadtxt(tmp_path / "streamed" / "energy_test.out", ndmin=2)) == 8


def test_resume_continues_exactly(tmp_path):
    """Long trainings run in segments: a run stopped after epoch 2 and resumed
    with restart=True must end exactly like one that ran 4 epochs straight
    (model, optimizer, scheduler and shard order all restored). With a
    validation set, as in production runs: without one, the end-of-run
    true-loss evaluation re-solves b1 (a documented side effect), so extending
    a finished run is exact only up to that one offset update."""
    xyz = _frames(tmp_path / "train.xyz", 0, 16)
    valid = _frames(tmp_path / "valid.xyz", 16, 8)
    kw = dict(run_seed=7, checkpoint_interval=1, valid_file=valid)
    straight = tmp_path / "straight"
    _run(tmp_path, _nep_in(tmp_path / "nep4.in", TYPES3, "stage2 0\nepoch 4\n"), xyz, straight,
         restart=False, **kw)
    split = tmp_path / "split"
    _run(tmp_path, _nep_in(tmp_path / "nep2.in", TYPES3, "stage2 0\nepoch 2\n"), xyz, split,
         restart=False, **kw)
    log = _run(tmp_path, str(tmp_path / "nep4.in"), xyz, split, restart=True, **kw)
    assert "Resumed from" in log
    np.testing.assert_allclose(_loss(split), _loss(straight), rtol=1e-12, atol=0)
    assert (split / "nep_final.txt").read_text() == (straight / "nep_final.txt").read_text()


def test_finetune_with_slim_types_and_random_split(tmp_path):
    """Fine-tuning a 4-type model on data without Fe, with slim_types: the Fe
    network is dropped and the other three keep their weights (lr 0 freezes
    them, so the forces — independent of the re-solved energy offset b1 — must
    equal the parent's). Validation drawn with valid_strategy="random"."""
    from torchnep.nep import NEPCalculator
    from torchnep.data import read_xyz
    xyz = _frames(tmp_path / "train.xyz", 0, 16)
    types4 = "type 4 Cr Co Ni Fe\n"
    parent = tmp_path / "parent"
    _run(tmp_path, _nep_in(tmp_path / "nep.in", types4, "epoch 2\n"), xyz, parent, restart=False, run_seed=1)
    child = tmp_path / "child"
    log = _run(tmp_path, _nep_in(tmp_path / "ft.in", types4, "epoch 1\nlr 0\n"), xyz, child, restart=False,
               run_seed=1, slim_types=True, finetune_from=str(parent / "nep_final.txt"),
               valid_ratio=0.25, valid_strategy="random")
    assert "slim_types: ['Cr', 'Co', 'Ni', 'Fe'] -> ['Cr', 'Co', 'Ni']" in log
    assert "[4 -> 3 types]" in log
    assert "valid_ratio=0.25: held out 4 frames" in log
    assert (child / "nep_final.txt").read_text().split()[1] == "3"
    f = read_xyz(xyz)[0]
    got = [NEPCalculator(str(p / "nep_final.txt")).compute(f["species"], f["positions"], f["cell"])["forces"]
           for p in (parent, child)]
    torch.testing.assert_close(got[1], got[0], rtol=1e-10, atol=1e-10)


def test_early_stop_jumps_to_stage2_then_stops(tmp_path):
    """With frozen weights (lr 0 in both stages) the loss plateaus: stage 1 hands
    over to stage 2 early, stage 2 then stops early, long before `epoch`."""
    xyz = _frames(tmp_path / "train.xyz", 0, 16)
    log = _run(tmp_path, _nep_in(tmp_path / "nep.in", TYPES3,
                                 "epoch 40\nlr 0\nstage2_lr 0\nearly_stop 2\nstage2 1\n"),
               xyz, tmp_path / "out", restart=False, run_seed=2, valid_ratio=0.25)
    assert "Early stop (stage 1)" in log and "Early stop:" in log
    assert len(_loss(tmp_path / "out")) < 40


def test_autograd_forces_match_analytical(tmp_path):
    """Autograd and analytical forces are two routes to the same numbers: the
    losses of the two runs agree to float64 round-off. Also covers GPUMD's
    parameter init (every rank gets rank 0's) and an SWA window that is
    never reached."""
    xyz = _frames(tmp_path / "train.xyz", 0, 16)
    nepin = _nep_in(tmp_path / "nep.in", TYPES3, "epoch 2\nstage2 1\n")      # SWA lives in stage 2
    kw = dict(restart=False, run_seed=4, use_gpumd_qscaler=True, use_swa=True, swa_start=100)
    _run(tmp_path, nepin, xyz, tmp_path / "an", use_autograd_forces=False, **kw)
    log = _run(tmp_path, nepin, xyz, tmp_path / "ag", use_autograd_forces=True, **kw)
    assert "SWA window never reached" in log and "use_gpumd_qscaler" in log
    np.testing.assert_allclose(_loss(tmp_path / "ag"), _loss(tmp_path / "an"), rtol=1e-8)
