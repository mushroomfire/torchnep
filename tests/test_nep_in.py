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
"""nep.in parsing: every keyword lands where the trainer reads it, anything
that is not a TorchNEP keyword (a typo, a GPUMD-only option) or has an invalid
value is rejected up front, and the documented keywords are exactly the
supported ones."""
import re
from pathlib import Path

import pytest

from torchnep.data import NEP_IN_KEYWORDS, parse_nep_in

DOCS = Path(__file__).resolve().parent.parent / "docs" / "guide" / "nep-in.md"


def _parse(tmp_path, text):
    p = tmp_path / "nep.in"
    p.write_text(text)
    return parse_nep_in(str(p))


def test_every_training_key(tmp_path):
    c = _parse(tmp_path, """type 2 Cr Ni
version 4
cutoff 6 5
n_max 6 4
basis_size 10 8
l_max 4 2 1
neuron 50
lambda_e 1.5
lambda_f 2
lambda_v 0.2
weight_decay 0.001
batch 128
epoch 700
lr 0.02
scheduler_patience 7
early_stop 30
scheduler_factor 0.6
stop_lr 1e-5
lr_scheduler step
max_grad_norm 5
stage2 1
start_stage2 400
stage2_lr 0.002
stage2_lambda_e 3
stage2_lambda_f 0.1
stage2_lambda_v 0.3
stage2_scheduler_patience 11
stage2_scheduler_factor 0.8
""")
    expect = dict(
        num_types=2, type_names=["Cr", "Ni"], version=4, cutoff_radial=6.0, cutoff_angular=5.0,
        n_max_radial=6, n_max_angular=4, basis_size_radial=10, basis_size_angular=8, neuron=50,
        lambda_e=1.5, lambda_f=2.0, lambda_v=0.2, weight_decay=1e-3, batch_size=128, num_epochs=700, lr=0.02,
        scheduler_patience=7, early_stop=30, scheduler_factor=0.6, stop_lr=1e-5,
        lr_scheduler="step", max_grad_norm=5.0, stage2=True, start_stage2=400, stage2_lr=0.002,
        stage2_pref_e=3.0, stage2_pref_f=0.1, stage2_pref_v=0.3,
        stage2_scheduler_patience=11, stage2_scheduler_factor=0.8)
    for key, value in expect.items():
        assert c[key] == value, (key, c[key], value)


def test_unknown_keywords_are_rejected(tmp_path):
    """A GPUMD nep.in (here the UNEP16 one) carries the options of GPUMD's
    evolutionary optimiser: they mean nothing to TorchNEP and must be removed,
    and a misspelt keyword must not silently fall back to the default. The
    error names the keyword and its line."""
    gpumd = """type       16 Ag Al Au Cr Cu Mg Mo Ni Pb Pd Pt Ta Ti V W Zr
version    4
cutoff     6 5
lambda_1   0
population 60
"""
    with pytest.raises(ValueError, match=r"line 4: unknown keyword 'lambda_1'"):
        _parse(tmp_path, gpumd)
    with pytest.raises(ValueError, match=r"line 2: unknown keyword 'stage2_lambda'"):
        _parse(tmp_path, "type 1 Cr\nstage2_lambda 0.1\n")


def test_documented_keywords_are_the_supported_ones():
    """The tables of the nep.in reference list exactly the keywords the parser takes."""
    documented = set(re.findall(r"^\| `([a-z_0-9]+)`", DOCS.read_text(), flags=re.M))
    assert documented == set(NEP_IN_KEYWORDS)


@pytest.mark.parametrize("line, message", [
    ("version 3", "only implements NEP4"),
    ("lr_scheduler cosine", "lr_scheduler must be"),
    ("use_typewise_cutoff_zbl", "needs the factor"),
])
def test_invalid_values_are_rejected(tmp_path, line, message):
    with pytest.raises(ValueError, match=message):
        _parse(tmp_path, "type 1 Cr\n" + line + "\n")
