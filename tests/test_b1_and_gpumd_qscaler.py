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

"""Tests for the analytical b1 energy shift and the ``use_gpumd_qscaler``
training option (branch ``qscaler-optimizer-test``).

Two features under test:
  * ``recompute_b1_shift`` sets the global energy offset b1 to its analytical
    optimum (mean per-atom residual -> 0), and train_nep keeps b1 out of the
    gradient optimizer.
  * ``use_gpumd_qscaler=True`` trains with GPUMD's init: descriptor coeffs
    uniform(-1,1) and a q_scaler computed with all coefficients = 1.0.
"""
import numpy as np
import torch

from torchnep.data import read_xyz, parse_nep_in
from torchnep.train import (preprocess_structures, GPUDataStore,
                            compute_q_scaler, recompute_b1_shift, train_nep)
from torchnep.model import NEPModel, gpumd_init_parameters
from _common import DATA_DIR

# PbTe example carries per-frame energy + forces (the CrCoNi fixture has no
# energy labels, so it can't exercise the energy-offset b1 logic).
PBTE = DATA_DIR.parent.parent / "example" / "PbTe" / "train.xyz"
NEP_IN = ("type 2 Te Pb\ncutoff 6 4\nn_max 4 4\n"
          "basis_size 6 6\nl_max 4 2 1\nneuron 30\n")


def _store(cfg, n=20, dtype=torch.float64):
    frames = read_xyz(str(PBTE))[:n]
    structs = preprocess_structures(frames, cfg, np.float64)
    return GPUDataStore(structs, torch.device("cpu"), dtype, config=cfg)


def _cfg_from_nepin(tmp_path=None):
    import tempfile, os
    d = tmp_path or tempfile.mkdtemp()
    p = os.path.join(str(d), "nep.in")
    with open(p, "w") as f:
        f.write(NEP_IN)
    return parse_nep_in(p)


@torch.no_grad()
def _mean_residual(model, ds):
    tot, n = 0.0, 0
    for s in range(0, ds.n, 1000):
        b = ds.collate(list(range(s, min(s + 1000, ds.n))))
        r = model.compute_properties_cached(b, need_forces=False, backend="loop")
        d = (r["Etot"] / b["natoms"] - b["energy"] / b["natoms"])[b["energy_mask"]]
        tot += float(d.sum()); n += int(b["energy_mask"].sum())
    return tot / max(n, 1)


def test_recompute_b1_zeros_residual():
    """After recompute_b1_shift the mean per-atom energy residual is ~0,
    regardless of the starting b1."""
    torch.manual_seed(0)
    cfg = _cfg_from_nepin()
    ds = _store(cfg)
    m = NEPModel(cfg).to(torch.float64)
    qmin, qmax = compute_q_scaler(m, ds)
    m.set_q_scaler(qmin, qmax)

    m.b1.data.fill_(123.456)                 # deliberately wrong offset
    assert abs(_mean_residual(m, ds)) > 1.0
    recompute_b1_shift(m, ds, 1000, "loop")
    assert abs(_mean_residual(m, ds)) < 1e-9


def test_recompute_b1_is_additive_and_idempotent():
    """A second call leaves b1 unchanged (residual already ~0)."""
    torch.manual_seed(1)
    cfg = _cfg_from_nepin()
    ds = _store(cfg)
    m = NEPModel(cfg).to(torch.float64)
    qmin, qmax = compute_q_scaler(m, ds)
    m.set_q_scaler(qmin, qmax)

    b1a = recompute_b1_shift(m, ds, 1000, "loop")
    b1b = recompute_b1_shift(m, ds, 1000, "loop")
    assert abs(b1a - b1b) < 1e-9


def test_gpumd_init_qscaler_matches_c1_and_differs_from_self():
    """compute_q_scaler(gpumd_init=True) == forcing all coeffs to 1.0, and
    that differs from the self-consistent q_scaler."""
    torch.manual_seed(2)
    cfg = _cfg_from_nepin()
    ds = _store(cfg)
    m = NEPModel(cfg).to(torch.float64)

    g_min, g_max = compute_q_scaler(m, ds, gpumd_init=True)
    s_min, s_max = compute_q_scaler(m, ds, gpumd_init=False)

    # manual c=1 reference
    c2, c3 = m.c_param_2.data.clone(), m.c_param_3.data.clone()
    m.c_param_2.data.fill_(1.0)
    if m.c_param_3 is not None:
        m.c_param_3.data.fill_(1.0)
    r_min, r_max = compute_q_scaler(m, ds, gpumd_init=False)
    m.c_param_2.data.copy_(c2)
    if m.c_param_3 is not None:
        m.c_param_3.data.copy_(c3)

    assert torch.allclose(g_min, r_min) and torch.allclose(g_max, r_max)
    assert not torch.allclose(g_max - g_min, s_max - s_min)


def test_gpumd_init_parameters_reinits_nn_and_coeffs():
    """gpumd_init_parameters draws every parameter — descriptor coeffs AND the
    NN weights (w0/b0/w1) — from uniform(-1, 1), a much larger amplitude than
    torch's default small-variance NN init."""
    torch.manual_seed(3)
    cfg = _cfg_from_nepin()
    m = NEPModel(cfg).to(torch.float64)

    # torch's default init is small; b0 starts at exactly zero.
    assert m.fitting_nets[0].w0.abs().max().item() < 0.5
    assert torch.count_nonzero(m.fitting_nets[0].b0) == 0

    gpumd_init_parameters(m)

    for net in m.fitting_nets:
        for name, p in (("w0", net.w0), ("b0", net.b0), ("w1", net.w1)):
            assert p.min().item() >= -1.0 and p.max().item() <= 1.0, name
        # uniform(-1,1) reaches near the edges — far wider than the small init.
        assert net.w0.abs().max().item() > 0.5
        assert net.b0.abs().max().item() > 0.0     # no longer all-zero
    assert m.c_param_2.min().item() >= -1.0 and m.c_param_2.max().item() <= 1.0


def _weight_rms(cfg, nep_txt):
    m = NEPModel(cfg).to(torch.float64)
    m.load_weights_from_nep_txt(nep_txt)
    sq, n = 0.0, 0
    for net in m.fitting_nets:
        for p in (net.w0, net.b0, net.w1):
            sq += float((p.detach() ** 2).sum()); n += p.numel()
    return (sq / n) ** 0.5


def test_l2_regularization_shrinks_weights(tmp_path):
    """With the same seed/init/data order, a positive lambda_2 (GPUMD-form L2)
    drives the NN weights to a smaller RMS than an unregularized run — the only
    difference between the two runs is the reg gradient, which points to 0."""
    _, xyz = _write_run_files(tmp_path)
    base = tmp_path / "nep0.in"
    base.write_text(NEP_IN + "epoch 15\nbatch 8\nlambda_2 0\n")
    reg = tmp_path / "nepR.in"
    reg.write_text(NEP_IN + "epoch 15\nbatch 8\nlambda_2 0.2\n")

    kw = dict(data_file=xyz, device="cpu", precision="float64",
              print_interval=100, restart=False, checkpoint_interval=10000,
              prediction_interval=10000, run_seed=7)
    train_nep(config_file=str(base), output_dir=str(tmp_path / "o0"), **kw)
    train_nep(config_file=str(reg), output_dir=str(tmp_path / "oR"), **kw)

    cfg = parse_nep_in(str(base))
    rms0 = _weight_rms(cfg, str(tmp_path / "o0" / "nep_final.txt"))
    rmsR = _weight_rms(cfg, str(tmp_path / "oR" / "nep_final.txt"))
    assert rmsR < rms0, f"L2 did not shrink weights: reg={rmsR} vs none={rms0}"


def _write_run_files(tmp_path, n_frames=20):
    nepin = tmp_path / "nep.in"
    nepin.write_text(NEP_IN + "epoch 3\nbatch 8\n")
    xyz = tmp_path / "train.xyz"
    # Slice PbTe down to n_frames by re-reading the raw text blocks.
    raw = PBTE.read_text().splitlines()
    out, i, k = [], 0, 0
    while i < len(raw) and k < n_frames:
        na = int(raw[i].strip())
        out += raw[i:i + na + 2]
        i += na + 2; k += 1
    xyz.write_text("\n".join(out) + "\n")
    return str(nepin), str(xyz)


def test_train_nep_b1_not_in_optimizer_and_analytic(tmp_path):
    """End-to-end: train_nep runs, b1 is excluded from the optimizer and ends
    at the analytical optimum (mean residual ~0)."""
    nepin, xyz = _write_run_files(tmp_path)
    out = tmp_path / "out"
    train_nep(config_file=nepin, data_file=xyz, output_dir=str(out),
              device="cpu", precision="float64", print_interval=1, restart=False,
              checkpoint_interval=1000, prediction_interval=1000)

    # Reload the trained model and check the residual is ~0 on the train set.
    cfg = parse_nep_in(nepin)
    ds = _store(cfg)
    m = NEPModel(cfg).to(torch.float64)
    m.load_weights_from_nep_txt(str(out / "nep_final.txt"))
    assert abs(_mean_residual(m, ds)) < 1e-6


def test_nep_best_not_worse_than_final(tmp_path):
    """nep_best's energy fit must not be worse than nep_final's.

    Both are saved with their own exact analytical b1 and the final epoch is
    always a best candidate, so nep_best can never end up worse than nep_final.
    """
    _, xyz = _write_run_files(tmp_path)
    (tmp_path / "nep.in").write_text(NEP_IN + "epoch 12\nbatch 8\n")
    out = tmp_path / "out"
    train_nep(config_file=str(tmp_path / "nep.in"), data_file=xyz,
              output_dir=str(out), device="cpu", precision="float64",
              print_interval=100, restart=False, checkpoint_interval=10000,
              prediction_interval=10000)

    cfg = parse_nep_in(str(tmp_path / "nep.in"))
    ds = _store(cfg)

    def energy_mse(path):
        m = NEPModel(cfg).to(torch.float64)
        m.load_weights_from_nep_txt(path)
        sq, n = 0.0, 0
        with torch.no_grad():
            for s in range(0, ds.n, 1000):
                b = ds.collate(list(range(s, min(s + 1000, ds.n))))
                r = m.compute_properties_cached(b, need_forces=False, backend="loop")
                d = (r["Etot"] / b["natoms"] - b["energy"] / b["natoms"])[b["energy_mask"]]
                sq += float((d ** 2).sum()); n += int(b["energy_mask"].sum())
        return sq / max(n, 1)

    assert energy_mse(str(out / "nep_best.txt")) <= energy_mse(str(out / "nep_final.txt")) + 1e-9


def test_train_nep_use_gpumd_qscaler(tmp_path):
    """use_gpumd_qscaler=True -> the saved q_scaler equals the c=1 value, and
    differs from the default self-consistent run."""
    nepin, xyz = _write_run_files(tmp_path)

    out_g = tmp_path / "out_gpumd"
    train_nep(config_file=nepin, data_file=xyz, output_dir=str(out_g),
              device="cpu", precision="float64", print_interval=1, restart=False,
              checkpoint_interval=1000, prediction_interval=1000,
              use_gpumd_qscaler=True)

    out_s = tmp_path / "out_self"
    train_nep(config_file=nepin, data_file=xyz, output_dir=str(out_s),
              device="cpu", precision="float64", print_interval=1, restart=False,
              checkpoint_interval=1000, prediction_interval=1000,
              use_gpumd_qscaler=False)

    cfg = parse_nep_in(nepin)
    ds = _store(cfg)
    mg = NEPModel(cfg).to(torch.float64); mg.load_weights_from_nep_txt(str(out_g / "nep_final.txt"))
    ms = NEPModel(cfg).to(torch.float64); ms.load_weights_from_nep_txt(str(out_s / "nep_final.txt"))

    # GPUMD q_scaler is c=1; recompute the c=1 reference and compare.
    ref = NEPModel(cfg).to(torch.float64)
    ref.c_param_2.data.fill_(1.0)
    if ref.c_param_3 is not None:
        ref.c_param_3.data.fill_(1.0)
    r_min, r_max = compute_q_scaler(ref, ds, gpumd_init=False)
    qs_c1 = 1.0 / torch.clamp(r_max - r_min, min=1e-10)

    assert torch.allclose(mg.q_scaler, qs_c1, rtol=1e-5, atol=1e-8)
    assert not torch.allclose(mg.q_scaler, ms.q_scaler)
