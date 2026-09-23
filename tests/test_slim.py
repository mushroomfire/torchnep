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
"""Slimming a model to a subset of its elements, for every ZBL variant.

A slimmed model must give exactly the parent's energies, forces and virial on
structures made of the kept elements — the per-species cutoffs, the typewise
ZBL cutoffs and the rows of a flexible zbl.in table all follow the kept types
(and their new order). Checked on compressed structures, where ZBL acts, for
both routes: slim_model + save_nep_txt (running a large model on a small
system), and fine-tuning with slim_types=True from a nep.txt.
"""
import numpy as np
import pytest
import torch

from torchnep import train_nep
from torchnep.data import parse_nep_in, read_xyz
from torchnep.model import NEPModel, slim_config, slim_model
from torchnep.nep import NEPCalculator
from _common import DATA_DIR

ARCH = "type 3 Cr Co Ni\nn_max 8 8\nbasis_size 12 12\nl_max 4 2 1\nneuron 80\n"
MODELS = {                      # fixture -> the nep.in lines that describe its cutoffs and ZBL
    "typewise_zbl": ("nep_CrCoNi.txt", "cutoff 6 4\nzbl 2.5\nuse_typewise_cutoff_zbl 0.7\n"),
    "universal_zbl": ("nep_CrCoNi_unizbl.txt", "cutoff 6 4\nzbl 2.5\n"),
    "zbl_in": ("nep_CrCoNi_flexzbl.txt", "cutoff 6 4\nzbl zbl.in\n"),
    "per_species_cutoff": ("nep_CrCoNi_multicut.txt",
                           "cutoff 6 4 5 3.5 4.5 3\nzbl 2.5\nuse_typewise_cutoff_zbl 0.7\n"),
}


def _nep_in(tmp_path, key, extra=""):
    # a zbl.in in nep.in replaces the table stored in nep.txt; write it from
    # that table at full precision (the fixture's zbl.in has one digit less,
    # which moves ZBL energies by ~1e-8) so only the slimming is compared
    table = (DATA_DIR / "nep_CrCoNi_flexzbl.txt").read_text().split()[-60:]
    (tmp_path / "zbl.in").write_text("\n".join(" ".join(table[i:i + 10]) for i in range(0, 60, 10)) + "\n")
    p = tmp_path / f"{key}.in"
    p.write_text(ARCH + MODELS[key][1] + extra)
    return p


def _without_co(frames):
    out = []
    for f in frames:
        keep = [i for i, s in enumerate(f["species"]) if s != "Co"]
        g = dict(f, species=[f["species"][i] for i in keep], positions=np.asarray(f["positions"])[keep],
                 natoms=len(keep))
        if f.get("forces") is not None:
            g["forces"] = np.asarray(f["forces"])[keep]
        out.append(g)
    return out


def _write_xyz(path, frames):
    """Labelled frames (the energy label only sets the offset b1 here)."""
    lines = []
    for f in frames:
        lines += [str(f["natoms"]),
                  'Lattice="%s" energy=%.10f Properties=species:S:1:pos:R:3:force:R:3'
                  % (" ".join(f"{x:.10f}" for x in np.ravel(f["cell"])), f["energy"])]
        lines += [f"{s} " + " ".join(f"{x:.10f}" for x in (*p, *fo))
                  for s, p, fo in zip(f["species"], f["positions"], f["forces"])]
    path.write_text("\n".join(lines) + "\n")


def _efv(model_file, frames):
    calc = NEPCalculator(str(model_file), dtype=torch.float64)
    res = [calc.compute(f["species"], f["positions"], f["cell"]) for f in frames]
    return [np.concatenate([np.asarray(r[k]).ravel() for r in res]) for k in ("energy", "forces", "virial")]


def _compressed():
    """Co-free CrCoNi structures incl. the rattled, strongly compressed last
    one; asserts that some pair lies inside the ZBL range (< 2.5 A)."""
    frames = _without_co(read_xyz(str(DATA_DIR / "CrCoNi.xyz")))
    f = frames[-1]
    d = np.linalg.norm(f["positions"][:, None] - f["positions"][None], axis=-1)
    assert (d[np.triu_indices(len(d), 1)] < 2.5).any()
    return frames


@pytest.mark.parametrize("key", list(MODELS))
def test_slim_model_is_exact_on_the_kept_elements(tmp_path, key):
    """slim_model to ["Ni", "Cr"] (reversed order, so every per-type table is
    re-indexed), saved as nep.txt and read back by the calculator."""
    cfg = parse_nep_in(str(_nep_in(tmp_path, key)))
    model = NEPModel(cfg).to(torch.float64)
    model.load_weights_from_nep_txt(str(DATA_DIR / MODELS[key][0]))
    slim = slim_model(model, ["Ni", "Cr"])
    slim.save_nep_txt(str(tmp_path / "slim.txt"), 200, 100)
    assert (tmp_path / "slim.txt").read_text().split()[:4] == ["nep4_zbl", "2", "Ni", "Cr"]
    frames = _compressed()
    for got, ref in zip(_efv(tmp_path / "slim.txt", frames), _efv(DATA_DIR / MODELS[key][0], frames)):
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)
    # the trainer's route: a model built from the slimmed nep.in config takes the slimmed weights
    NEPModel(slim_config(cfg, ["Ni", "Cr"])).to(torch.float64).load_state_dict(slim.state_dict())


@pytest.mark.parametrize("key", list(MODELS))
def test_finetune_with_slim_types_keeps_the_model(tmp_path, key):
    """Fine-tuning a Cr/Co/Ni model on Cr/Ni data with slim_types=True: the
    Co type is dropped from the model, its cutoffs and its ZBL table rows; lr 0
    freezes the weights, so energies up to the re-solved offset, forces and
    virials equal the parent's."""
    xyz = tmp_path / "noco.xyz"
    _write_xyz(xyz, _without_co(read_xyz(str(DATA_DIR / "CrCoNi_train.xyz"))[:16]))
    nepin = _nep_in(tmp_path, key, "epoch 1\nbatch 8\nlr 0\nstage2 0\n")
    out = tmp_path / "out"
    train_nep(str(nepin), str(xyz), output_dir=str(out), device="cpu", precision="float64",
              restart=False, run_seed=0, slim_types=True, finetune_from=str(DATA_DIR / MODELS[key][0]),
              print_interval=100, checkpoint_interval=10**6, prediction_interval=10**6)
    assert "[3 -> 2 types]" in (out / "output.log").read_text()
    frames = _compressed()
    (e, f, v), (e0, f0, v0) = _efv(out / "nep_final.txt", frames), _efv(DATA_DIR / MODELS[key][0], frames)
    np.testing.assert_allclose(f, f0, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(v, v0, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.diff(e), np.diff(e0), rtol=1e-10, atol=1e-10)   # per-atom energies up to b1
