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
"""Typewise ZBL: the per-pair outer cutoff is min(factor * (R_i + R_j), zbl),
as in GPUMD (main_nep/nep.cu) and NEP_CPU, with the `zbl` value of nep.in as
the cap. The CrCoNi fixtures never reach the cap (0.7 * 2 * 1.47 = 2.05 A
< 2.5 A); heavy elements do: Cs has R = 2.787 A, so Cs-I gives 3.18 A and
Cs-Pb 3.16 A against zbl 2.5. Without the cap a model is trained with a
repulsion that GPUMD never applies when it runs the written nep.txt."""
import pytest
import torch

from torchnep import ops
from torchnep.constants import COVALENT_RADIUS
from torchnep.data import parse_nep_in
from torchnep.model import NEPModel

NEP_IN = ("type 3 Cs Pb I\ncutoff 6 4\nn_max 4 4\nbasis_size 4 4\nl_max 4 0\n"
          "neuron 20\nzbl 2.5\nuse_typewise_cutoff_zbl 0.7\n")
Z = {"Cs": 55, "Pb": 82, "I": 53}


def _model(tmp_path):
    f = tmp_path / "nep.in"
    f.write_text(NEP_IN)
    return NEPModel(parse_nep_in(str(f))).to(torch.float64)


def _pair_cutoff(a, b):
    return min(0.7 * (COVALENT_RADIUS[Z[a] - 1] + COVALENT_RADIUS[Z[b] - 1]), 2.5)


def test_pair_cutoff_table_capped_at_zbl(tmp_path):
    """The (T, T) table of the compiled path and the global fallback both
    stop at the zbl value; pairs below it keep their typewise cutoff."""
    model = _model(tmp_path)
    assert model.zbl_rc_outer == pytest.approx(2.5)
    table = model.zbl_rc_outer_pair
    assert float(table.max()) <= 2.5 + 1e-6
    assert table[0, 2].item() == pytest.approx(_pair_cutoff("Cs", "I"))   # capped: 2.5
    assert table[1, 2].item() == pytest.approx(_pair_cutoff("Pb", "I"), abs=1e-6)  # 2.4453, below the cap


def _zbl_dimer(model, types, r):
    """ZBL energy of a dimer on the eager path, called as NEPModel.compute does."""
    dtype, device = torch.float64, "cpu"
    rij = torch.tensor([[r, 0.0, 0.0], [-r, 0.0, 0.0]], dtype=dtype)
    return ops.compute_zbl(
        torch.tensor(types), torch.tensor([0, 1]), torch.tensor([1, 0]), rij, 2,
        model.atomic_numbers.tolist(), model.zbl_rc_inner, model.zbl_rc_outer,
        model.zbl_typewise_factor, model.zbl_rc_inner_per_type,
        model.zbl_rc_outer_per_type, dtype, device, **model._flexible_zbl_kwargs())


def test_no_repulsion_beyond_cap(tmp_path):
    """Cs-I at 2.9 A: inside the uncapped typewise cutoff (3.18 A), outside
    zbl 2.5 -> no ZBL energy; at 2.4 A (inside) the repulsion is there."""
    model = _model(tmp_path)
    assert float(_zbl_dimer(model, [0, 2], 2.9).abs().sum()) == 0.0
    assert float(_zbl_dimer(model, [0, 2], 2.4).sum()) > 0.0
