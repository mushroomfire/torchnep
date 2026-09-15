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

"""Smoke test of torchnep.plot: every figure builds from synthetic output
files and is written to disk (the figures' looks are checked by eye, not
here). Skipped without matplotlib."""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from torchnep.plot import NEPPlotter, read_loss, read_outputs, shift_energy, frame_meta


def _write_run(d, n_frames=40, natoms=8, test=True):
    rng = np.random.default_rng(0)
    ep = np.arange(1, 31)
    cols = [ep, 1e-2 / ep, 0.3 / ep, 0.4 / ep, 0.2 / ep, 2.0 / ep]
    if test:
        cols += [0.31 / ep, 0.41 / ep, 0.21 / ep, 2.1 / ep]
    np.savetxt(d / "loss.out", np.column_stack(cols), header="epoch loss ...")
    for split in ("train", "test") if test else ("train",):
        e_ref = rng.uniform(-8, -4, n_frames)
        np.savetxt(d / f"energy_{split}.out", np.column_stack([e_ref + rng.normal(0, 0.02, n_frames), e_ref]))
        f_ref = rng.normal(0, 2, (n_frames * natoms, 3))
        np.savetxt(d / f"force_{split}.out", np.column_stack([f_ref + rng.normal(0, 0.1, f_ref.shape), f_ref]))
        v_ref = rng.normal(0, 1, (n_frames, 6))
        v_ref[::5] = -1e6                      # frames without a virial label
        np.savetxt(d / f"virial_{split}.out", np.column_stack([v_ref + 0.05, v_ref]))
        np.savetxt(d / f"stress_{split}.out", np.column_stack([v_ref * 10 + 0.5, v_ref * 10]))
    lines = []
    for k in range(n_frames):
        lines.append(f"{natoms}\nLattice=\"10 0 0 0 10 0 0 0 10\" config_type={'bulk' if k % 2 else 'defect'} "
                     "Properties=species:S:1:pos:R:3")
        for i in range(natoms):
            lines.append(f"{'Cr' if i % 3 else 'Ni'} {i} 0 0")
    (d / "data.xyz").write_text("\n".join(lines) + "\n")


def test_readers_and_shift(tmp_path):
    _write_run(tmp_path)
    loss = read_loss(tmp_path)
    assert "e_test" in loss and len(loss["epoch"]) == 30
    d = read_outputs(tmp_path, "train")
    assert set(d) == {"E", "F", "V", "S"} and d["V"]["mask"].sum() == 32
    assert d["F"]["pred"].shape == (320, 3)
    natoms, species, ct = frame_meta(tmp_path / "data.xyz")
    assert natoms.sum() == 320 and set(ct) == {"bulk", "defect"}
    e = shift_energy(d, "mean")
    assert abs(np.mean(e - d["E"]["ref"])) < 1e-12
    e = shift_energy(d, "element", (natoms, species, ct))
    assert abs(np.mean(e - d["E"]["ref"])) < 1e-6
    with pytest.raises(ValueError):
        shift_energy(d, "element")


def test_all_figures(tmp_path):
    _write_run(tmp_path)
    p = NEPPlotter(font="Arial", dpi=60)
    fig = p.loss(tmp_path / "loss.out", out=tmp_path / "loss.png")
    assert (tmp_path / "loss.png").exists() and len(fig.axes) == 4
    for kind in ("scatter", "density"):
        p.parity(tmp_path, split="test", kind=kind, out=tmp_path / f"parity_{kind}.png",
                 quantities=("E", "F", "V", "S"))
    p.parity(tmp_path, shift_energy="element", xyz=tmp_path / "data.xyz", out=tmp_path / "shift.png")
    p.parity_train_test(tmp_path, out=tmp_path / "tt.png")
    p.errors(tmp_path, xyz=tmp_path / "data.xyz", out=tmp_path / "errors.png")
    p.prediction(tmp_path, xyz=tmp_path / "data.xyz", out=tmp_path / "pred.png", shift_energy="mean")
    p.dashboard(tmp_path, out=tmp_path / "dash.png")
    for name in ("parity_scatter", "parity_density", "shift", "tt", "errors", "pred", "dash"):
        assert (tmp_path / f"{name}.png").stat().st_size > 1000


def test_train_only_and_missing_font(tmp_path):
    _write_run(tmp_path, test=False)
    with pytest.warns(UserWarning):
        p = NEPPlotter(font="NoSuchFontXYZ", dpi=60)
    p.dashboard(tmp_path, out=tmp_path / "dash.png")
    p.errors(tmp_path, out=tmp_path / "errors.png")
    assert (tmp_path / "dash.png").exists()
