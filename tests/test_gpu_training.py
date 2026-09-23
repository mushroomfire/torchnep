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
"""Training on the GPU with torch.compile — the path production runs take
(use_compile is on by default on CUDA/ROCm). Needs a GPU, so it runs on our
own machines, not in CI.

A compiled run must train like the eager one: the same loss curve through
stage 1 and stage 2 and the same final model, up to Inductor's reordered
floating-point sums. Covered for analytical and autograd forces, per-species
cutoffs with typewise ZBL, a zbl.in table, float32 (the production
precision) and float64, and data-parallel training on two GPUs.
"""
import json
import os
import subprocess

import numpy as np
import pytest
import torch

from torchnep import train_nep
from torchnep.data import read_xyz
from torchnep.nep import NEPCalculator
from _common import DATA_DIR, torchrun_cmd

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA/ROCm GPU")

XYZ = DATA_DIR / "CrCoNi_train.xyz"                       # 24 labelled frames
BASE = "type 3 Cr Co Ni\nn_max 6 4\nbasis_size 8 8\nl_max 4 2 0\nneuron 30\nbatch 8\nepoch 4\nstage2 1\n"
SETUPS = {
    "analytical": ("cutoff 6 4\n", {}),
    "autograd": ("cutoff 6 4\n", {"use_autograd_forces": True}),
    "species_cutoff_zbl": ("cutoff 6 4 5 3.5 4.5 3\nzbl 2\nuse_typewise_cutoff_zbl 0.7\n", {}),
    "zbl_in": ("cutoff 6 4\nzbl zbl.in\n", {}),
}
# compiled vs eager after 4 epochs (2 of them stage 2): the reordered sums of
# Inductor kernels (~1 ulp per op) grow through the optimizer steps
TOL = {"float64": 1e-8, "float32": 1e-3}


def _compile_available():
    from torchnep.train import _compile_check
    return _compile_check(torch.device("cuda"))[0]


def _write(tmp_path, setup):
    (tmp_path / "zbl.in").write_text((DATA_DIR / "CrCoNi_flexzbl.zbl.in").read_text())
    p = tmp_path / "nep.in"
    p.write_text(BASE + SETUPS[setup][0])
    return str(p)


def _forces(model_file, frames):
    calc = NEPCalculator(str(model_file), dtype=torch.float64)
    return np.concatenate([np.asarray(calc.compute(f["species"], f["positions"], f["cell"])["forces"])
                           for f in frames])


def _compare(eager, compiled, precision):
    a, b = np.loadtxt(eager / "loss.out"), np.loadtxt(compiled / "loss.out")
    rel = np.abs(a[:, 1:] - b[:, 1:]) / np.abs(a[:, 1:])
    frames = read_xyz(str(XYZ))[:4]
    fa, fb = _forces(eager / "nep_final.txt", frames), _forces(compiled / "nep_final.txt", frames)
    frel = np.abs(fa - fb).max() / np.abs(fa).max()
    print(f"\n[{precision}] loss.out max rel diff {rel.max():.2e}, forces {frel:.2e}")
    assert a.shape == b.shape
    assert rel.max() < TOL[precision] and frel < TOL[precision]


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("setup", list(SETUPS))
def test_compiled_training_matches_eager(tmp_path, setup, precision):
    if not _compile_available():
        pytest.skip("torch.compile unavailable here (no Triton or no C/C++ compiler)")
    nep_in = _write(tmp_path, setup)
    kw = dict(device="cuda", precision=precision, restart=False, run_seed=0, valid_ratio=0.25,
              print_interval=1, checkpoint_interval=10**6, prediction_interval=10**6, **SETUPS[setup][1])
    train_nep(nep_in, str(XYZ), output_dir=str(tmp_path / "eager"), use_compile=False, **kw)
    train_nep(nep_in, str(XYZ), output_dir=str(tmp_path / "compiled"), use_compile=True, **kw)
    assert "torch.compile: enabled" in (tmp_path / "compiled" / "output.log").read_text()
    assert "torch.compile: enabled" not in (tmp_path / "eager" / "output.log").read_text()
    _compare(tmp_path / "eager", tmp_path / "compiled", precision)


_RUNNER = """
import json, sys
from torchnep.train_sharded import train_nep_sharded
train_nep_sharded(sys.argv[1], sys.argv[2], output_dir=sys.argv[3], print_interval=1,
                  checkpoint_interval=10**6, prediction_interval=10**6, **json.loads(sys.argv[4]))
"""


@pytest.mark.skipif(os.environ.get("TORCHNEP_TEST_DDP") != "1" or torch.cuda.device_count() < 2,
                    reason="needs 2 GPUs and TORCHNEP_TEST_DDP=1")
@pytest.mark.parametrize("setup", ["analytical", "species_cutoff_zbl"])
def test_compiled_ddp_training_matches_eager(tmp_path, setup):
    """train_nep_sharded on 2 GPUs (NCCL/RCCL), compiled vs eager, float32."""
    if not _compile_available():
        pytest.skip("torch.compile unavailable here (no Triton or no C/C++ compiler)")
    cmd = torchrun_cmd(2)
    if not cmd:
        pytest.skip("torchrun not on PATH")
    nep_in = _write(tmp_path, setup)
    (tmp_path / "runner.py").write_text(_RUNNER)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1",
               PYTHONPATH=str(DATA_DIR.parent.parent) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    for name, compile_on in (("eager", False), ("compiled", True)):
        kw = dict(restart=False, run_seed=0, valid_ratio=0.25, precision="float32", use_compile=compile_on)
        r = subprocess.run(cmd + [str(tmp_path / "runner.py"), nep_in, str(XYZ), str(tmp_path / name),
                                  json.dumps(kw)], capture_output=True, text=True, env=env, timeout=900)
        assert r.returncode == 0, r.stderr[-3000:]
    assert "torch.compile: enabled" in (tmp_path / "compiled" / "output.log").read_text()
    _compare(tmp_path / "eager", tmp_path / "compiled", "float32")
