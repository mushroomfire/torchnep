# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Yongchao Wu.
# Part of the torchnep project — https://github.com/mushroomfire/torchnep.

"""Regression tests for legacy nep.in / nep.txt parsing.

Pins down acceptance of the historical 3-field ``l_max <L> <l_max_4b>
<l_max_5b>`` form (and the intermediate 4-field bridge), as well as the
new 5-field ``l_max <L> <has_q_222> <has_q_1111> <has_q_112> <has_q_1122>``.
Anything past index 0 is treated as a boolean flag, so the old
``l_max 4 2 1`` semantics (4-body and 5-body invariants enabled) are
preserved.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torchnep.model import NEPModel
from torchnep.nep import NEPCalculator


def _minimal_config(l_max_list):
    return {
        "num_types": 1, "type_names": ["X"],
        "cutoff_radial": 6.0, "cutoff_angular": 5.0,
        "n_max_radial": 4, "n_max_angular": 4,
        "basis_size_radial": 8, "basis_size_angular": 8,
        "l_max": l_max_list, "neuron": 10,
    }


@pytest.mark.parametrize("l_max,expected", [
    # 3-field legacy form (used by every nep.in / nep.txt before the
    # mixed-body extension; '2' and '1' were "max L used in 4b/5b" but
    # had boolean semantics everywhere they entered the code).
    ([4, 2, 1], (4, 1, 1, 0, 0)),
    ([4, 2, 0], (4, 1, 0, 0, 0)),
    ([4, 0, 0], (4, 0, 0, 0, 0)),
    # 4-field bridge form (q_112 added, q_1122 still off)
    ([4, 1, 1, 1],    (4, 1, 1, 1, 0)),
    ([4, 1, 0, 1],    (4, 1, 0, 1, 0)),
    # 5-field new form
    ([4, 1, 1, 1, 1], (4, 1, 1, 1, 1)),
    ([4, 1, 0, 0, 1], (4, 1, 0, 0, 1)),
    # 1-field (3-body only; future-compat — every trailing flag defaults to 0)
    ([4],             (4, 0, 0, 0, 0)),
])
def test_l_max_field_normalisation(l_max, expected):
    """Trailing fields normalise to 0/1 regardless of input width."""
    m = NEPModel(_minimal_config(l_max))
    got = (m.l_max_3b, m.has_q_222, m.has_q_1111, m.has_q_112, m.has_q_1122)
    assert got == expected


def test_legacy_3field_nep_txt_loads():
    """Existing nep_CrCoNi.txt with ``l_max 4 2 1`` still loads correctly.

    Catches regressions where parsing of older nep.txt files breaks after
    a refactor of the header parser.
    """
    nep_txt = Path(__file__).resolve().parent / "data" / "nep_CrCoNi.txt"
    calc = NEPCalculator(str(nep_txt))
    assert calc.l_max_3b == 4
    assert calc.has_q_222 == 1
    assert calc.has_q_1111 == 1
    assert calc.has_q_112 == 0
    assert calc.has_q_1122 == 0
    # Sanity on derived dim:
    # radial (n_max_r+1=9) + 3-body (9*4=36) + q_222 (9) + q_1111 (9) = 63
    assert calc.dim == 63
