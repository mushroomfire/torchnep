# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Yongchao Wu.
# Part of the torchnep project — https://github.com/mushroomfire/torchnep.

"""pytest conftest — sets sys.path so ``torchnep`` is importable when running
``pytest`` from the repo root or from ``tests/`` itself, with no need for
``pip install -e .``.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent

for p in (ROOT, TESTS):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# Keep OpenMP / numpy thread fan-out predictable in CI.
os.environ.setdefault("TORCHNEP_PREPROC_WORKERS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
