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

"""Cutoff rules enforced on nep.in and on every NEPModel: angular >= 3 A
and <= radial per species, radial <= 100 A, ZBL outer in [1, 3] A and <=
the smallest angular cutoff (ZBL runs on the angular neighbor list) — for
the universal `zbl` value and every zbl.in row — and typewise factor >= 0.5 (required)."""
import re

import pytest

from torchnep.data import parse_nep_in, read_zbl_in, validate_cutoffs
from torchnep.model import NEPModel

BASE = "type 3 Cr Co Ni\nn_max 4 4\nbasis_size 6 6\nl_max 4 2 1\nneuron 30\n"
UNI_ROW = "0.18175 3.1998 0.50986 0.94229 0.28022 0.4029 0.02817 0.20162"


def _parse(tmp_path, lines):
    p = tmp_path / "nep.in"
    p.write_text(BASE + lines)
    return parse_nep_in(str(p))


def _zbl_in(tmp_path, rc_outer, name="zbl.in"):
    (tmp_path / name).write_text("\n".join([f"0.5 {rc_outer} {UNI_ROW}"] * 6) + "\n")
    return name


@pytest.mark.parametrize("lines", [
    "cutoff 6 4\nzbl 2.5\n",
    "cutoff 6 4 5 3.5 4.5 3\nzbl 3\n",        # ZBL equal to the smallest angular cutoff: allowed
    "cutoff 6 4\nzbl 3\nuse_typewise_cutoff_zbl 0.7\n",
    "cutoff 3 3\nzbl 1\n",
    "cutoff 6 4\n",                            # no ZBL at all
])
def test_valid_configs(tmp_path, lines):
    NEPModel(_parse(tmp_path, lines))


@pytest.mark.parametrize("lines, msg", [
    ("cutoff 6 2.5\n", "angular cutoff 2.5 A is below the minimum"),
    ("cutoff 6 4 5 3.5 4.5 2.9\n", "angular cutoff (Ni) 2.9 A is below"),
    ("cutoff 4 5\n", "radial cutoff 4.0 A is smaller than the angular"),
    ("cutoff 6 4 5 3.5 3 3.5\n", "radial cutoff (Ni) 3.0 A is smaller"),
    ("cutoff 120 4\n", "radial cutoff 120.0 A exceeds 100.0 A"),
    ("cutoff 6 4\nzbl 0.8\n", "ZBL outer cutoff 0.8 A outside"),
    ("cutoff 6 4\nzbl 3.5\n", "ZBL outer cutoff 3.5 A outside"),
    ("cutoff 6 4\nzbl 2.5\nuse_typewise_cutoff_zbl 0.4\n", "use_typewise_cutoff_zbl factor 0.4 is below 0.5"),
    ("cutoff 6 4\nzbl 2.5\nuse_typewise_cutoff_zbl\n", "use_typewise_cutoff_zbl needs the factor"),
])
def test_invalid_configs(tmp_path, lines, msg):
    with pytest.raises(ValueError, match=re.escape(msg)):
        _parse(tmp_path, lines)


def test_zbl_above_smallest_angular(monkeypatch):
    """Implied by angular >= 3 and ZBL <= 3, but checked explicitly as a
    guard: widen the ZBL range to reach it."""
    from torchnep import data
    monkeypatch.setattr(data, "ZBL_RC_OUTER_RANGE", (1.0, 5.0))
    cfg = {"cutoff_radial": 6.0, "cutoff_angular": 3.0, "type_names": ["Cr", "Ni"],
           "cutoff_radial_per_type": [6.0, 6.0], "cutoff_angular_per_type": [3.0, 4.0], "zbl": 3.5}
    with pytest.raises(ValueError, match="exceeds the smallest angular"):
        validate_cutoffs(cfg)
    cfg["zbl"] = 3.0
    validate_cutoffs(cfg)                                   # equal is fine
    cfg["zbl_flexible"] = [[0.5, 3.2] + [0.1] * 8] * 3      # a zbl.in row above it
    with pytest.raises(ValueError, match="exceeds the smallest angular"):
        validate_cutoffs(cfg)


def test_zbl_in_rows(tmp_path):
    ok = _zbl_in(tmp_path, 2.8, "ok.in")
    assert len(read_zbl_in(str(tmp_path / ok), 3)) == 6
    bad = _zbl_in(tmp_path, 3.2, "bad.in")
    with pytest.raises(ValueError, match="rc_outer 3.2 outside"):
        read_zbl_in(str(tmp_path / bad), 3)
    _zbl_in(tmp_path, 2.8)
    NEPModel(_parse(tmp_path, "cutoff 6 4 5 3.5 4.5 3\nzbl zbl.in\n"))   # 2.8 <= 3: fine


def test_model_config_is_checked():
    cfg = {"num_types": 1, "type_names": ["Cu"], "cutoff_radial": 5.0, "cutoff_angular": 2.0,
           "n_max_radial": 4, "n_max_angular": 4, "basis_size_radial": 6, "basis_size_angular": 6,
           "l_max": [4, 2, 0], "neuron": 10}
    with pytest.raises(ValueError, match="below the minimum"):
        NEPModel(cfg)
