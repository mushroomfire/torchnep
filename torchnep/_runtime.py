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

"""Runtime checks for CUDA machines without a C compiler.

Triton compiles a small C module, its CUDA launcher, the first time it runs for
a given Triton version, Python version and CPU architecture, and caches it in
``~/.triton/cache`` (or ``TRITON_CACHE_DIR``). With no C compiler and no cached
launcher, every Triton kernel launch fails with "Failed to find C compiler".

Since PyTorch 2.12 this is not only a ``torch.compile`` matter: ``torch._native``
routes some built-in CUDA ops through Triton kernels in eager mode too (the
outer-product ``bmm`` that backward passes produce, for one), so a plain eager
run can die mid-training on such a machine — and only on the first run, which
makes it look random. :func:`ensure_triton_runtime` probes once, up front, and
hands those ops back to the regular CUDA kernels when Triton cannot run.
"""
import os
import shutil
import warnings

_state = {"checked": False, "ok": True, "reason": ""}


def _compiler_found(env_var, names):
    """A compiler named by ``env_var`` (CC / CXX, possibly with flags), else one of ``names`` on PATH."""
    cmd = os.environ.get(env_var, "").split()
    if cmd:
        return shutil.which(cmd[0]) is not None
    return any(shutil.which(n) for n in names)


def cxx_compiler_found():
    """Whether Inductor can build C++ (it needs one for the CPU parts of a graph, also on CUDA)."""
    return _compiler_found("CXX", ("g++", "clang++", "c++"))


def triton_runtime_ok():
    """``(ok, reason)``: can Triton launch CUDA kernels in this process?

    Initialising Triton's CUDA driver builds (or loads from the cache) the
    launcher module, so this either succeeds for good or reproduces the exact
    error the first kernel launch would raise. Checked once per process.
    """
    if not _state["checked"]:
        _state["checked"] = True
        try:
            from triton.runtime.driver import driver
        except Exception:               # no Triton: nothing in torch routes through it
            return True, ""
        try:
            driver.active.get_current_device()
        except Exception as e:
            _state["ok"] = False
            _state["reason"] = (str(e).splitlines() or [type(e).__name__])[0]
    return _state["ok"], _state["reason"]


def ensure_triton_runtime(dev, log=None):
    """Keep CUDA runs working where Triton cannot build its launcher.

    On failure the Triton-backed built-in ops of ``torch._native`` are switched
    back to the regular CUDA kernels (same results; PyTorch < 2.12 has none) and
    one message says why and how to fix it for good. Returns ``False`` when
    Triton cannot run, so callers keep ``torch.compile`` off. Silent and ``True``
    on non-CUDA devices and whenever Triton works.
    """
    if getattr(dev, "type", str(dev)) != "cuda":
        return True
    first = not _state["checked"]
    ok, reason = triton_runtime_ok()
    if ok:
        return True
    if first:
        try:
            from torch._native.registry import deregister_op_overrides
            deregister_op_overrides(disable_dsl_names="triton")
            fallback = ("PyTorch's Triton-backed built-in ops are switched back to the regular CUDA kernels "
                        "for this run")
        except Exception:
            fallback = "if a built-in op still fails, set TORCH_DISABLE_NATIVE_JIT=1 before starting Python"
        msg = (f"Triton cannot launch CUDA kernels here ({reason}). It builds a small C launcher on first "
               f"use, cached in ~/.triton/cache afterwards: load a compiler (e.g. `module load gcc`) or set "
               f"CC and CXX. Meanwhile {fallback}, and torch.compile stays off.")
        if log is not None:
            log(f"  WARNING: {msg}")
        else:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return False
