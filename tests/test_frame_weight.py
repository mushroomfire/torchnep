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
"""Per-frame weights (``weight=w`` on the comment line of the xyz).

The objective counts every squared error of a frame (its energy, the forces
of its atoms, its virial) w times, 0 < w <= 100, averaged over the labels:

    L = λ_e Σ w e² / N_e + λ_f Σ_atoms w |Δf|²/3 / N_f + λ_v Σ w |Δv|²/6 / N_v

The energy offset b1 minimises it (Σ w·r / Σ w), the reported RMSEs stay
unweighted. The tests check that the weights are read on every path and
that the logged loss IS this objective, recomputed independently from the
saved model with the calculator — for train_nep and the sharded trainer,
training and validation data.
"""
import os

import numpy as np
import pytest
import torch

from torchnep import train_nep
from torchnep.data import index_xyz, parse_nep_in, read_xyz, read_xyz_at
from torchnep.nep import NEPCalculator
from torchnep.train import StreamDataStore, preprocess_structures
from _common import DATA_DIR

XYZ = DATA_DIR / "CrCoNi_train.xyz"                                # 24 labelled frames
NEP_IN = ("type 3 Cr Co Ni\ncutoff 6 4\nn_max 4 4\nbasis_size 6 6\nl_max 4 2 0\nneuron 20\n"
          "lambda_e 1\nlambda_f 1\nlambda_v 0.1\nstage2 0\n")
WEIGHTS = [0.3, 0.5, 1, 2, 3, 10, 0.05, 1.5, 7, 0.25, 1, 4, 100, 0.1, 2.5, 1]
VIRIAL_6 = [0, 4, 8, 1, 5, 6]
LOSS_RTOL = 2e-6          # loss.out prints the loss with 7 significant digits
RMSE_ATOL = 1e-6          # ... and the RMSEs with 6 decimals


def _write(path, first, weights):
    """Frames first.. of CrCoNi_train.xyz with ``weight=`` tags (None: no tag)."""
    raw, out, i, k = XYZ.read_text().splitlines(), [], 0, 0
    while i < len(raw) and k < first + len(weights):
        n = int(raw[i])
        if k >= first:
            j = k - first
            head, atoms = raw[i + 1], raw[i + 2:i + 2 + n]
            if weights[j] is not None:
                head += f" weight={weights[j]}"
            out += [raw[i], head] + atoms
        i += n + 2
        k += 1
    path.write_text("\n".join(out) + "\n")
    return str(path)


def _objective(model_file, xyz, weights, lam=(1.0, 1.0, 0.1)):
    """(objective, rmse_e, rmse_f, rmse_v, weighted mean energy residual,
    unweighted mean energy residual) of a saved model, from the calculator."""
    calc = NEPCalculator(str(model_file), dtype=torch.float64)
    w = np.asarray(weights, float)
    e, f, v = [], [], []
    for fr in read_xyz(xyz):
        r = calc.compute(fr["species"], fr["positions"], fr["cell"])
        n = fr["natoms"]
        e.append(float(r["energy"].sum()) / n - fr["energy"] / n)
        f.append((np.asarray(r["forces"]) - fr["forces"]) ** 2)
        v.append(np.mean(((np.asarray(r["virial"]).sum(0) - fr["virial"])[VIRIAL_6] / n) ** 2))
    e, v = np.array(e), np.array(v)
    fw = np.concatenate([np.full(len(x), wi) for x, wi in zip(f, w)])
    f_err = np.concatenate([x.mean(1) for x in f])
    loss = (lam[0] * np.mean(w * e ** 2) + lam[1] * np.mean(fw * f_err)
            + lam[2] * np.mean(w * v))
    return (loss, np.sqrt(np.mean(e ** 2)), np.sqrt(f_err.mean()), np.sqrt(v.mean()),
            np.sum(w * e) / w.sum(), e.mean())


def _train(nep_in, xyz, out, **kw):
    kw.setdefault("run_seed", 0)
    kw.setdefault("device", "cpu")
    train_nep(nep_in, xyz, output_dir=str(out), precision="float64", restart=False,
              print_interval=100, checkpoint_interval=10**6, prediction_interval=10**6, **kw)
    return np.loadtxt(out / "loss.out", ndmin=2)


def test_weights_are_read_on_every_path(tmp_path):
    """weight= reaches the frames, the streamed reader and the batches the
    trainer sees; default 1, quoted values, energy_weight= is not weight=."""
    xyz = _write(tmp_path / "w.xyz", 0, WEIGHTS[:6] + [None, "\"2.5\""])
    text = open(xyz).read().replace("weight=0.5", "energy_weight=9 weight=0.5")
    open(xyz, "w").write(text)
    expect = WEIGHTS[:6] + [1.0, 2.5]
    frames = read_xyz(xyz)
    assert [f.get("weight", 1.0) for f in frames] == expect
    offsets, _ = index_xyz(xyz)
    assert [f.get("weight", 1.0) for f in read_xyz_at(xyz, offsets[::-1])] == expect[::-1]
    (tmp_path / "nep.in").write_text(NEP_IN)
    cfg = parse_nep_in(str(tmp_path / "nep.in"))
    store = StreamDataStore(preprocess_structures(frames, cfg, dtype=np.float64), torch.device("cpu"),
                            torch.float64, config=cfg)
    assert store.collate([7, 0, 5])["weight"].tolist() == [2.5, 0.3, 10.0]


@pytest.mark.parametrize("bad", ["0", "0.0", "-1", "100.5", "abc", "nan"])
def test_invalid_weights_are_rejected(tmp_path, bad):
    """A weight must be a number in (0, 100]; 0 is not a way to drop a frame
    (delete it instead). Rejected by both readers and before training."""
    xyz = _write(tmp_path / "w.xyz", 0, [1, bad])
    with pytest.raises(ValueError, match="weight must be"):
        read_xyz(xyz)
    offsets, _ = index_xyz(xyz)
    with pytest.raises(ValueError, match="weight must be"):
        read_xyz_at(xyz, offsets)
    (tmp_path / "nep.in").write_text(NEP_IN + "epoch 1\n")
    with pytest.raises(ValueError, match="weight must be"):
        _train(str(tmp_path / "nep.in"), xyz, tmp_path / "out")


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_logged_loss_is_the_weighted_objective(tmp_path, device):
    """lr 0: the model is fixed, so the logged loss must equal the weighted
    objective of the saved model (second epoch: b1 already at its optimum)
    and the logged RMSEs the unweighted errors. b1 minimises the weighted
    energy term: the weighted mean residual is 0, the plain one is not.
    On a GPU the run takes the default compiled path."""
    xyz = _write(tmp_path / "w.xyz", 0, WEIGHTS)
    (tmp_path / "nep.in").write_text(NEP_IN + "epoch 2\nbatch 5\nlr 0\n")
    loss = _train(str(tmp_path / "nep.in"), xyz, tmp_path / "out", device=device)
    obj, rmse_e, rmse_f, rmse_v, mean_w, mean = _objective(tmp_path / "out" / "nep_final.txt", xyz, WEIGHTS)
    np.testing.assert_allclose(loss[1, 1], obj, rtol=LOSS_RTOL)
    np.testing.assert_allclose(loss[1, 2:5], [rmse_e, rmse_f, rmse_v], rtol=0, atol=RMSE_ATOL)
    assert abs(mean_w) < 1e-9 and abs(mean) > 1e-3
    # the same data without weights: a different objective, the same errors
    plain = _train(str(tmp_path / "nep.in"), _write(tmp_path / "p.xyz", 0, [None] * 16), tmp_path / "plain",
                   device=device)
    assert abs(plain[1, 1] / loss[1, 1] - 1) > 1e-2


def test_validation_loss_is_weighted(tmp_path):
    """With a validation file the logged loss is the weighted objective on
    it (b1 stays train-fitted), the test columns the unweighted errors."""
    xyz = _write(tmp_path / "train.xyz", 0, [None] * 16)
    vw = [3, 0.4, 1, 0.5, 8, 1, 2, 0.2]
    valid = _write(tmp_path / "valid.xyz", 16, vw)
    (tmp_path / "nep.in").write_text(NEP_IN + "epoch 1\nbatch 8\nlr 0\n")
    loss = _train(str(tmp_path / "nep.in"), xyz, tmp_path / "out", valid_file=valid)
    obj, rmse_e, rmse_f, rmse_v, *_ = _objective(tmp_path / "out" / "nep_final.txt", valid, vw)
    np.testing.assert_allclose(loss[0, 1], obj, rtol=LOSS_RTOL)
    np.testing.assert_allclose(loss[0, 6:9], [rmse_e, rmse_f, rmse_v], rtol=0, atol=RMSE_ATOL)


def test_weight_one_equals_no_weight(tmp_path):
    """weight=1 on every frame is the unweighted training, bit for bit."""
    (tmp_path / "nep.in").write_text(NEP_IN + "epoch 2\nbatch 4\n")
    outs = []
    for name, w in (("none", [None] * 16), ("ones", [1] * 16)):
        _train(str(tmp_path / "nep.in"), _write(tmp_path / f"{name}.xyz", 0, w), tmp_path / name, valid_ratio=0.25)
        outs.append(tmp_path / name)
    for f in ("loss.out", "nep_final.txt", "nep_best.txt"):
        assert (outs[0] / f).read_text() == (outs[1] / f).read_text(), f


# --- the sharded trainer (2 ranks, TORCHNEP_TEST_DDP=1) --------------------

@pytest.mark.skipif(os.environ.get("TORCHNEP_TEST_DDP") != "1",
                    reason="multi-process test is local-only (set TORCHNEP_TEST_DDP=1)")
@pytest.mark.parametrize("streamed", [False, True], ids=["in_memory", "streamed"])
def test_sharded_weighted_objective(tmp_path, streamed):
    """train_nep_sharded logs the same weighted objective, on the training
    data and on a validation file, both with weights."""
    from test_sharded_features import _run
    env = {"TORCHNEP_STREAM_THRESHOLD": "0"} if streamed else None
    xyz = _write(tmp_path / "w.xyz", 0, WEIGHTS)
    (tmp_path / "nep.in").write_text(NEP_IN + "epoch 2\nbatch 3\nlr 0\n")
    log = _run(tmp_path, str(tmp_path / "nep.in"), xyz, tmp_path / "out", env=env, restart=False, run_seed=0)
    assert ("streamed shard loading" in log) == streamed
    loss = np.loadtxt(tmp_path / "out" / "loss.out", ndmin=2)
    obj, rmse_e, rmse_f, rmse_v, mean_w, _ = _objective(tmp_path / "out" / "nep_final.txt", xyz, WEIGHTS)
    np.testing.assert_allclose(loss[1, 1], obj, rtol=LOSS_RTOL)
    np.testing.assert_allclose(loss[1, 2:5], [rmse_e, rmse_f, rmse_v], rtol=0, atol=RMSE_ATOL)
    assert abs(mean_w) < 1e-9

    vw = [3, 0.4, 1, 0.5, 8, 1, 2, 0.2]
    valid = _write(tmp_path / "valid.xyz", 16, vw)
    (tmp_path / "v.in").write_text(NEP_IN + "epoch 1\nbatch 3\nlr 0\n")
    _run(tmp_path, str(tmp_path / "v.in"), xyz, tmp_path / "vout", env=env, restart=False, run_seed=0,
         valid_file=valid)
    vloss = np.loadtxt(tmp_path / "vout" / "loss.out", ndmin=2)
    np.testing.assert_allclose(vloss[0, 1], _objective(tmp_path / "vout" / "nep_final.txt", valid, vw)[0],
                               rtol=LOSS_RTOL)
