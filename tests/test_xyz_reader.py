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
"""Extended-XYZ reading: the stress label becomes a virial with GPUMD's sign
and volume, the indexed (streamed) reader returns the same frames as the
plain one, and malformed frames are rejected with a message instead of
being read wrongly."""
import numpy as np
import pytest

from torchnep.data import index_xyz, read_xyz, read_xyz_at

STRESS = [0.01, 0.002, -0.003, 0.002, -0.02, 0.004, -0.003, 0.004, 0.015]     # eV/A^3


def _frame(tags, lattice="4 0 0 0 5 0 0 1 6"):
    header = f'Lattice="{lattice}" energy=-7.5 {tags} Properties=species:S:1:pos:R:3:force:R:3'
    return f"2\n{header}\nCr 0 0 0 0.1 0 0\nNi 1.2 1.3 1.4 -0.1 0 0\n"


def test_stress_is_converted_to_virial(tmp_path):
    """virial = -stress * volume (GPUMD convention); an explicit virial wins."""
    p = tmp_path / "s.xyz"
    s = " ".join(map(str, STRESS))
    p.write_text(_frame(f'stress="{s}"') + _frame(f'virial="{s}" stress="{s}"'))
    by_stress, by_virial = read_xyz(str(p))
    volume = abs(np.linalg.det(np.array([[4, 0, 0], [0, 5, 0], [0, 1, 6]], float)))
    np.testing.assert_allclose(by_stress["virial"], -np.array(STRESS) * volume, rtol=1e-12)
    np.testing.assert_allclose(by_virial["virial"], STRESS, rtol=1e-12)


def test_indexed_reader_matches_plain_reader(tmp_path):
    """Streamed training/prediction read frames by byte offset; stray blank
    lines between frames must not shift the offsets."""
    p = tmp_path / "b.xyz"
    p.write_text(_frame("") + "\n\n" + _frame("", "5 0 0 0 5 0 0 0 5") + "\n" + _frame("", "6 0 0 0 6 0 0 0 6"))
    offsets, natoms = index_xyz(str(p))
    assert list(natoms) == [2, 2, 2]
    plain, indexed = read_xyz(str(p)), read_xyz_at(str(p), offsets[[2, 0]])
    assert len(plain) == 3
    for a, b in zip(indexed, [plain[2], plain[0]]):
        np.testing.assert_array_equal(a["cell"], b["cell"])
        np.testing.assert_array_equal(a["positions"], b["positions"])
    from torchnep.plot import frame_meta
    assert list(frame_meta(p)[0]) == [2, 2, 2]


@pytest.mark.parametrize("header, message", [
    ('energy=-1 Properties=species:S:1:pos:R:3', "missing mandatory Lattice"),
    ('Lattice="4 0 0 0 4 0 0 0" Properties=species:S:1:pos:R:3', "exactly 9 components"),
    ('Lattice="4 0 0 0 4 0 0 0 4" virial="1 2 3" Properties=species:S:1:pos:R:3', "virial must have"),
    ('Lattice="4 0 0 0 4 0 0 0 4" stress="1 2 3" Properties=species:S:1:pos:R:3', "stress must have"),
])
def test_malformed_frames_are_rejected(tmp_path, header, message):
    p = tmp_path / "bad.xyz"
    p.write_text(f"1\n{header}\nCr 0 0 0\n")
    with pytest.raises(ValueError, match=message):
        read_xyz(str(p))
