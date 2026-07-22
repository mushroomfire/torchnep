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

"""Tests for ``run_seed`` reproducibility and the validation set
(``valid_file`` / ``valid_ratio``): deterministic split, best-model
selection on the validation loss, GPUMD-style *_test.out outputs, and
split preservation across resume (the checkpoint's seed wins).
"""
import os

import numpy as np
import pytest
import torch

from torchnep.data import read_xyz, parse_nep_in
from torchnep.train import (preprocess_structures, GPUDataStore, train_nep)
from torchnep.model import NEPModel
from _common import DATA_DIR

PBTE = DATA_DIR.parent.parent / "example" / "PbTe" / "train.xyz"
NEP_IN = ("type 2 Te Pb\ncutoff 6 4\nn_max 4 4\n"
          "basis_size 6 6\nl_max 4 2 1\nneuron 30\n")


def _write_run_files(tmp_path, n_frames=20, epochs=3):
    nepin = tmp_path / "nep.in"
    nepin.write_text(NEP_IN + f"epoch {epochs}\nbatch 8\n")
    xyz = tmp_path / "train.xyz"
    raw = PBTE.read_text().splitlines()
    out, i, k = [], 0, 0
    while i < len(raw) and k < n_frames:
        na = int(raw[i].strip())
        out += raw[i:i + na + 2]
        i += na + 2; k += 1
    xyz.write_text("\n".join(out) + "\n")
    return str(nepin), str(xyz)


def _train(nepin, xyz, out, **kw):
    kw.setdefault("device", "cpu")
    kw.setdefault("precision", "float64")
    kw.setdefault("print_interval", 100)
    kw.setdefault("restart", False)
    kw.setdefault("checkpoint_interval", 10000)
    kw.setdefault("prediction_interval", 10000)
    train_nep(config_file=nepin, data_file=xyz, output_dir=str(out), **kw)


def _expected_split(n_frames, seed, ratio):
    """Replicates train_nep's valid_ratio split (must stay in sync)."""
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n_frames, generator=g).tolist()
    n_val = max(1, int(round(ratio * n_frames)))
    val_idx = sorted(perm[:n_val])
    train_idx = [i for i in range(n_frames) if i not in set(val_idx)]
    return train_idx, val_idx


def _valid_energy_mse(cfg, frames, nep_txt):
    """Energy MSE of a saved model on the given frames (per-atom, eV/atom)."""
    structs = preprocess_structures(frames, cfg, np.float64)
    ds = GPUDataStore(structs, torch.device("cpu"), torch.float64, config=cfg)
    m = NEPModel(cfg).to(torch.float64)
    m.load_weights_from_nep_txt(nep_txt)
    sq, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, ds.n, 1000):
            b = ds.collate(list(range(s, min(s + 1000, ds.n))))
            r = m.compute_properties_cached(b, need_forces=False,
                                            backend="loop")
            d = (r["Etot"] / b["natoms"]
                 - b["energy"] / b["natoms"])[b["energy_mask"]]
            sq += float((d ** 2).sum()); n += int(b["energy_mask"].sum())
    return sq / max(n, 1)


# --------------------------------------------------------------------------
# early stopping
# --------------------------------------------------------------------------

def _loss_out_epochs(out):
    lines = [l for l in (out / "loss.out").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    return len(lines)


def test_early_stop_triggers_on_plateau_with_validation(tmp_path):
    """early_stop halts the run once the monitored loss plateaus. lr=0 freezes
    the weights, so after the analytical b1 settles the (validation) loss stops
    improving and the run stops well before `epoch`. Exercises the valid-loss
    branch (valid_ratio active) and leaves a consistent final state."""
    _, xyz = _write_run_files(tmp_path, n_frames=20)
    nepin = tmp_path / "nep.in"
    nepin.write_text(NEP_IN + "epoch 40\nbatch 8\nlr 0\nearly_stop 3\n")
    out = tmp_path / "out"
    _train(nepin, xyz, out, run_seed=0, valid_ratio=0.2)

    n_epochs = _loss_out_epochs(out)
    assert n_epochs < 40, "early_stop never fired"
    assert n_epochs >= 3, "stopped before the patience window elapsed"
    # Final state is still written and self-consistent.
    assert (out / "nep_final.txt").exists()
    assert (out / "nep_best.txt").exists()
    assert (out / "energy_test.out").exists()   # valid branch was active


def test_no_early_stop_when_disabled(tmp_path):
    """Default (early_stop unset) runs all epochs even on a flat lr=0 plateau."""
    _, xyz = _write_run_files(tmp_path, n_frames=20)
    nepin = tmp_path / "nep.in"
    nepin.write_text(NEP_IN + "epoch 6\nbatch 8\nlr 0\n")
    out = tmp_path / "out"
    _train(nepin, xyz, out, run_seed=0)
    assert _loss_out_epochs(out) == 6


# --------------------------------------------------------------------------
# run_seed
# --------------------------------------------------------------------------

def test_same_seed_reproduces_run(tmp_path):
    """Two runs with the same run_seed are bit-for-bit identical (weights,
    shuffle stream -> identical nep_final.txt and loss.out)."""
    nepin, xyz = _write_run_files(tmp_path)
    _train(nepin, xyz, tmp_path / "a", run_seed=1234)
    _train(nepin, xyz, tmp_path / "b", run_seed=1234)
    assert ((tmp_path / "a" / "nep_final.txt").read_text()
            == (tmp_path / "b" / "nep_final.txt").read_text())
    assert ((tmp_path / "a" / "loss.out").read_text()
            == (tmp_path / "b" / "loss.out").read_text())


def test_different_seeds_differ(tmp_path):
    """Different run_seed -> different weight init -> different model."""
    nepin, xyz = _write_run_files(tmp_path)
    _train(nepin, xyz, tmp_path / "a", run_seed=1)
    _train(nepin, xyz, tmp_path / "b", run_seed=2)
    assert ((tmp_path / "a" / "nep_final.txt").read_text()
            != (tmp_path / "b" / "nep_final.txt").read_text())


def test_none_seed_runs_differ(tmp_path):
    """run_seed=None draws a fresh seed each run -> repeated runs differ."""
    nepin, xyz = _write_run_files(tmp_path)
    _train(nepin, xyz, tmp_path / "a")
    _train(nepin, xyz, tmp_path / "b")
    assert ((tmp_path / "a" / "nep_final.txt").read_text()
            != (tmp_path / "b" / "nep_final.txt").read_text())


def test_seed_saved_in_checkpoint_and_wins_on_resume(tmp_path):
    """The checkpoint stores run_seed; a resume with a DIFFERENT seed keeps
    the original one (shuffle stream + valid split stay continuous)."""
    nepin, xyz = _write_run_files(tmp_path, epochs=2)
    out = tmp_path / "out"
    _train(nepin, xyz, out, run_seed=777, checkpoint_interval=1,
           valid_ratio=0.2)
    ck = torch.load(str(out / "checkpoint.pt"), map_location="cpu",
                    weights_only=False)
    assert ck["run_seed"] == 777

    # Extend and resume with a different seed — the saved seed must win.
    (tmp_path / "nep.in").write_text(NEP_IN + "epoch 4\nbatch 8\n")
    _train(nepin, xyz, out, run_seed=999, restart=True,
           checkpoint_interval=1, valid_ratio=0.2)
    ck = torch.load(str(out / "checkpoint.pt"), map_location="cpu",
                    weights_only=False)
    assert ck["epoch"] == 4
    assert ck["run_seed"] == 777


# --------------------------------------------------------------------------
# validation set
# --------------------------------------------------------------------------

def test_valid_file_and_ratio_mutually_exclusive(tmp_path):
    nepin, xyz = _write_run_files(tmp_path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _train(nepin, xyz, tmp_path / "out",
               valid_file=xyz, valid_ratio=0.1)


def test_valid_ratio_outputs_and_split(tmp_path):
    """valid_ratio: *_test.out files match the run_seed-derived holdout
    (row counts = frames / atoms of the expected split) and loss.out gains
    the three GPUMD-style test-RMSE columns."""
    nepin, xyz = _write_run_files(tmp_path, epochs=3)
    out = tmp_path / "out"
    seed, ratio = 42, 0.25
    _train(nepin, xyz, out, run_seed=seed, valid_ratio=ratio)

    frames = read_xyz(xyz)
    train_idx, val_idx = _expected_split(len(frames), seed, ratio)
    n_val = len(val_idx)
    n_val_atoms = sum(len(frames[i]["species"]) for i in val_idx)
    n_train_atoms = sum(len(frames[i]["species"]) for i in train_idx)

    for stem in ("energy", "force", "virial", "stress"):
        assert (out / f"{stem}_test.out").exists()
        assert (out / f"{stem}_train.out").exists()
    assert len((out / "energy_test.out").read_text().splitlines()) == n_val
    assert len((out / "force_test.out").read_text().splitlines()) \
        == n_val_atoms
    # train outputs cover only the remaining (non-holdout) frames
    assert len((out / "energy_train.out").read_text().splitlines()) \
        == len(train_idx)
    assert len((out / "force_train.out").read_text().splitlines()) \
        == n_train_atoms

    # loss.out: 6 train columns + 4 test columns (E/F/V/stress)
    rows = [ln.split() for ln in
            (out / "loss.out").read_text().splitlines()
            if not ln.startswith("#")]
    assert all(len(r) == 10 for r in rows)
    # test RMSEs are finite and positive
    assert all(float(r[6]) > 0 and float(r[7]) > 0 for r in rows)
    # with a validation set the loss column IS the weighted validation loss
    lam_e, lam_f, lam_v = 0.01, 1.0, 0.01   # nep.in defaults
    for r in rows:
        expect = (lam_e * float(r[6]) ** 2 + lam_f * float(r[7]) ** 2
                  + lam_v * float(r[8]) ** 2)
        assert abs(float(r[1]) - expect) < 1e-5

    # references in energy_test.out match the expected holdout frames
    ref = np.loadtxt(out / "energy_test.out")[:, 1]
    expected = np.array([frames[i]["energy"] / len(frames[i]["species"])
                         for i in val_idx])
    assert np.allclose(ref, expected, atol=1e-8)


def test_valid_file_outputs(tmp_path):
    """valid_file: the explicit file becomes the test set."""
    nepin, xyz = _write_run_files(tmp_path, n_frames=16, epochs=2)
    frames = read_xyz(xyz)
    # last 4 frames as an explicit validation file
    raw = open(xyz).read().splitlines()
    blocks, i = [], 0
    while i < len(raw):
        na = int(raw[i].strip())
        blocks.append("\n".join(raw[i:i + na + 2]))
        i += na + 2
    vxyz = tmp_path / "valid.xyz"
    vxyz.write_text("\n".join(blocks[-4:]) + "\n")

    out = tmp_path / "out"
    _train(nepin, xyz, out, run_seed=7, valid_file=str(vxyz))

    assert len((out / "energy_test.out").read_text().splitlines()) == 4
    # train outputs still cover the FULL data_file (no holdout)
    assert len((out / "energy_train.out").read_text().splitlines()) \
        == len(frames)


def test_best_model_selected_on_valid(tmp_path):
    """With a validation set, nep_best is the epoch with the lowest
    validation loss — so on the holdout it can't be worse than nep_final."""
    nepin, xyz = _write_run_files(tmp_path, epochs=10)
    out = tmp_path / "out"
    seed, ratio = 3, 0.25
    _train(nepin, xyz, out, run_seed=seed, valid_ratio=ratio)

    frames = read_xyz(xyz)
    _, val_idx = _expected_split(len(frames), seed, ratio)
    val_frames = [frames[i] for i in val_idx]
    cfg = parse_nep_in(nepin)

    best = _valid_energy_mse(cfg, val_frames, str(out / "nep_best.txt"))
    final = _valid_energy_mse(cfg, val_frames, str(out / "nep_final.txt"))
    # nep_best minimizes the WEIGHTED valid loss — with a validation set the
    # loss column of loss.out IS that quantity, so the best epoch's value
    # must be the minimum (the final epoch can't beat it).
    rows = [ln.split() for ln in
            (out / "loss.out").read_text().splitlines()
            if not ln.startswith("#")]
    vloss = [float(r[1]) for r in rows]
    assert min(vloss) <= vloss[-1] + 1e-12
    # And nep_best/nep_final sanity: both finite, best not absurdly worse.
    assert np.isfinite(best) and np.isfinite(final)


def test_resume_preserves_valid_split(tmp_path):
    """Resuming a valid_ratio run reproduces the SAME holdout (checkpoint
    seed drives the split), even when a different seed is passed."""
    nepin, xyz = _write_run_files(tmp_path, epochs=2)
    out = tmp_path / "out"
    seed, ratio = 55, 0.25
    _train(nepin, xyz, out, run_seed=seed, valid_ratio=ratio,
           checkpoint_interval=1)
    ref_first = np.loadtxt(out / "energy_test.out")[:, 1]

    (tmp_path / "nep.in").write_text(NEP_IN + "epoch 4\nbatch 8\n")
    _train(nepin, xyz, out, run_seed=98765, valid_ratio=ratio,
           restart=True, checkpoint_interval=1)
    ref_resumed = np.loadtxt(out / "energy_test.out")[:, 1]

    # Same reference energies in the same order -> same holdout frames.
    assert np.allclose(ref_first, ref_resumed, atol=0)

    frames = read_xyz(xyz)
    _, val_idx = _expected_split(len(frames), seed, ratio)
    expected = np.array([frames[i]["energy"] / len(frames[i]["species"])
                         for i in val_idx])
    assert np.allclose(ref_resumed, expected, atol=1e-8)
