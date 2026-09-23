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
"""CUDA machines without a C compiler (torchnep/_runtime.py).

The first four tests fake the probe and run anywhere; the last one reproduces
the real situation — empty Triton cache, no compiler — and needs a GPU.
"""
import os
import shutil
import subprocess
import sys
import types

import pytest
import torch

import torchnep._runtime as rt
from torchnep.train import _compile_check

NO_COMPILER = "Failed to find C compiler. Please specify via CC environment variable"


@pytest.fixture
def fresh(monkeypatch):
    """A clean per-process probe state for every test."""
    monkeypatch.setattr(rt, "_state", {"checked": False, "ok": True, "reason": ""})


def test_off_cuda_is_a_no_op(fresh):
    assert rt.ensure_triton_runtime(torch.device("cpu"))
    assert rt.ensure_triton_runtime(torch.device("mps"))
    assert not rt._state["checked"]                     # never probed Triton


def test_failure_switches_triton_ops_off_and_warns_once(fresh, monkeypatch):
    def failing_probe():
        rt._state.update(checked=True, ok=False, reason=NO_COMPILER)
        return False, NO_COMPILER
    monkeypatch.setattr(rt, "triton_runtime_ok", failing_probe)
    calls = []
    fake = types.ModuleType("torch._native.registry")
    fake.deregister_op_overrides = lambda **kw: calls.append(kw)
    monkeypatch.setitem(sys.modules, "torch._native.registry", fake)

    lines = []
    assert rt.ensure_triton_runtime(torch.device("cuda"), log=lines.append) is False
    assert calls == [{"disable_dsl_names": "triton"}]
    assert len(lines) == 1 and "Failed to find C compiler" in lines[0] and "CC and CXX" in lines[0]
    assert rt.ensure_triton_runtime(torch.device("cuda"), log=lines.append) is False
    assert len(lines) == 1 and len(calls) == 1          # once per process


def test_compile_check_refuses_cuda_when_triton_cannot_run(fresh, monkeypatch):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: object() if name == "triton" else real(name, *a))
    monkeypatch.setattr(rt, "triton_runtime_ok", lambda: (False, NO_COMPILER))
    ok, msg = _compile_check(torch.device("cuda"))
    assert not ok and "Triton cannot launch kernels" in msg


def test_compile_check_needs_a_cxx_compiler(fresh, monkeypatch):
    monkeypatch.setattr(rt, "cxx_compiler_found", lambda: False)
    ok, msg = _compile_check(torch.device("cpu"))
    assert not ok and "C++" in msg


_REPRO = """
import warnings, torch
warnings.simplefilter("always")
from torchnep._runtime import ensure_triton_runtime
with warnings.catch_warnings(record=True) as w:
    ok = ensure_triton_runtime(torch.device("cuda"))
a = torch.randn(64, 500, 1, device="cuda", requires_grad=True)   # outer-product bmm: the shape
b = torch.randn(64, 1, 80, device="cuda", requires_grad=True)    # torch._native gives to Triton
out = torch.bmm(a, b); out.sum().backward(); torch.cuda.synchronize()
err = float((out.detach() - a.detach() * b.detach()).abs().max())
print("RESULT", ok, err, sum("C compiler" in str(x.message) for x in w))
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_eager_cuda_runs_without_a_compiler(tmp_path):
    """No compiler and an empty Triton cache: eager work must still run, with one warning."""
    env = dict(os.environ, TRITON_CACHE_DIR=str(tmp_path / "triton"),
               TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "inductor"),
               PATH=os.path.dirname(sys.executable) + ":/usr/bin:/bin")
    env.pop("CC", None); env.pop("CXX", None)
    if any(shutil.which(c, path=env["PATH"]) for c in ("cc", "gcc", "clang")):
        pytest.skip("a C compiler is still on the stripped PATH; nothing to test here")
    r = subprocess.run([sys.executable, "-c", _REPRO], env=env, capture_output=True, text=True, timeout=600)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT")]
    assert line, r.stdout + r.stderr
    _, ok, err, n_warn = line[0].split()
    assert ok == "False" and float(err) == 0.0 and int(n_warn) == 1, line[0]
