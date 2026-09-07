# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project (GPL-3.0-or-later, see train.py).

"""weight_decay (AdamW with bias-exempt groups), the gathered NN, and the
compiled ZBL path."""
import numpy as np
import torch

from torchnep.data import read_xyz, parse_nep_in
from torchnep.train import preprocess_structures, StreamDataStore, train_nep
from _common import DATA_DIR

PBTE = DATA_DIR / "PbTe.xyz"
NEP_IN = ("type 2 Te Pb\ncutoff 6 4\nn_max 4 4\n"
          "basis_size 6 6\nl_max 4 2 1\nneuron 20\n")


def _store(tmp_path, n=10):
    p = tmp_path / "nep.in"
    p.write_text(NEP_IN)
    cfg = parse_nep_in(str(p))
    structs = preprocess_structures(read_xyz(str(PBTE))[:n], cfg, np.float64)
    return StreamDataStore(structs, torch.device("cpu"), torch.float64,
                           config=cfg), cfg


def _run(tmp_path, out, extra, seed=5):
    nepin = tmp_path / f"nep_{out}.in"
    nepin.write_text(NEP_IN + "epoch 3\nbatch 4\n" + extra)
    frames = read_xyz(str(PBTE))[:12]
    xyz = tmp_path / "train.xyz"
    raw = PBTE.read_text().splitlines()
    keep, i, k = [], 0, 0
    while i < len(raw) and k < 12:
        na = int(raw[i].strip()); keep += raw[i:i + na + 2]
        i += na + 2; k += 1
    xyz.write_text("\n".join(keep) + "\n")
    train_nep(config_file=str(nepin), data_file=str(xyz),
              output_dir=str(tmp_path / out), device="cpu",
              precision="float64", print_interval=100, restart=False,
              checkpoint_interval=1000, prediction_interval=1000,
              run_seed=seed)
    return (tmp_path / out / "loss.out").read_text()


def test_weight_decay_adamw_active(tmp_path):
    a = _run(tmp_path, "w1", "weight_decay 1e-2\n")
    b = _run(tmp_path, "w2", "weight_decay 1e-2\n")
    c = _run(tmp_path, "w0", "")
    assert a == b
    assert a != c


def test_gathered_nn_equivalence(tmp_path):
    """The gathered-weight NN (one bmm) must reproduce the autograd
    reference path (mask-based per-type nets) exactly: energies, forces,
    AND gradients w.r.t. every parameter — float64, tight tolerance.
    Also: the one-hot (grad) and plain-gather (no_grad) weight paths must
    agree bit-for-bit."""
    from torchnep.model import NEPModel
    store, cfg = _store(tmp_path, n=8)
    batch = store.collate([0, 2, 4, 6])

    torch.manual_seed(1)
    m = NEPModel(cfg).to(torch.float64)
    m.train()

    # reference: autograd path (unchanged mask-based per-type nets)
    ref = m.compute_properties(
        batch["rij_rad"].requires_grad_(True), batch["rij_ang"].requires_grad_(True),
        batch["pair_i_rad"], batch["pair_j_rad"],
        batch["pair_i_ang"], batch["pair_j_ang"],
        batch["atom_types"], batch["N"],
        batch["struct_idx"], batch["num_structures"],
        need_forces=True, need_virial=True, backend="loop")
    loss_ref = (ref["Ei"] ** 2).sum() + (ref["forces"] ** 2).sum()
    g_ref = torch.autograd.grad(loss_ref, list(m.parameters()),
                                allow_unused=True)

    got = m.compute_properties_cached(batch, need_forces=True,
                                      need_virial=True, backend="loop")
    torch.testing.assert_close(got["Ei"], ref["Ei"], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(got["forces"], ref["forces"],
                               rtol=1e-8, atol=1e-10)
    loss_got = (got["Ei"] ** 2).sum() + (got["forces"] ** 2).sum()
    g_got = torch.autograd.grad(loss_got, list(m.parameters()),
                                allow_unused=True)
    for p, a, b in zip(m.parameters(), g_got, g_ref):
        if a is None or b is None:
            assert (a is None) == (b is None)
            continue
        torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-9)

    # one-hot (grad path) vs plain gather (no_grad path): bit identical
    ei_grad, _, _ = m._cached_core(batch, need_forces=False)
    with torch.no_grad():
        ei_nograd, _, _ = m._cached_core(batch, need_forces=False)
    assert torch.equal(ei_grad.detach(), ei_nograd)


def test_zbl_static_equivalence(tmp_path):
    """The compiled branch-free ZBL (table gather + analytic pair gradient,
    ops.compute_zbl_pair inside _cached_core) must reproduce the eager
    reference (ops.compute_zbl + autograd through rij) — energies, forces,
    virial — in float64, for both plain and typewise cutoffs, with and
    without torch.no_grad()."""
    from torchnep.model import NEPModel
    store, cfg = _store(tmp_path, n=8)
    batch = store.collate([0, 2, 4, 6])

    for extra in ({"zbl": 3.5},
                  {"zbl": 3.5, "typewise_cutoff_zbl_factor": 1.2}):
        c = dict(cfg)
        c.update(extra)
        torch.manual_seed(3)
        m = NEPModel(c).to(torch.float64)
        m.train()

        ref = m.compute_properties(
            batch["rij_rad"], batch["rij_ang"],
            batch["pair_i_rad"], batch["pair_j_rad"],
            batch["pair_i_ang"], batch["pair_j_ang"],
            batch["atom_types"], batch["N"],
            batch["struct_idx"], batch["num_structures"],
            need_forces=True, need_virial=True, backend="loop")

        got = m.compute_properties_cached(batch, need_forces=True,
                                          need_virial=True, backend="loop")
        torch.testing.assert_close(got["Ei"], ref["Ei"],
                                   rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(got["forces"], ref["forces"],
                                   rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(got["virial"], ref["virial"],
                                   rtol=1e-8, atol=1e-10)

        # ZBL actually contributes on this geometry (identical seed ->
        # identical NN weights; the difference is pure ZBL)
        torch.manual_seed(3)
        m0 = NEPModel(dict(cfg)).to(torch.float64)
        got0 = m0.compute_properties_cached(batch, need_forces=False,
                                            backend="loop")
        assert (got["Ei"] - got0["Ei"]).abs().max() > 1e-6

        # prediction path: no_grad must still carry full ZBL forces
        # (the old eager block needed an enable_grad escape for this)
        with torch.no_grad():
            gp = m.compute_properties_cached(batch, need_forces=True,
                                             need_virial=True, backend="loop")
        torch.testing.assert_close(gp["forces"], ref["forces"],
                                   rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(gp["virial"], ref["virial"],
                                   rtol=1e-8, atol=1e-10)


def test_optimizer_all_params_decay(tmp_path):
    """Every trainable parameter (including b0 biases) sits in ONE decay
    group — the bias-exempt variant was tried and rejected (it fit train
    identically but lost 30-40% on independent-test energies)."""
    from torchnep.model import NEPModel
    from torchnep.train import _make_optimizer
    _, cfg = _store(tmp_path, n=4)
    torch.manual_seed(0)
    m = NEPModel(cfg).to(torch.float64)
    m.b1.requires_grad_(False)
    named = [(n, p) for n, p in m.named_parameters() if n != "b1"]
    opt = _make_optimizer(named, 1e-3, 1e-4)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["weight_decay"] == 1e-4
    assert len(opt.param_groups[0]["params"]) == len(named)


def test_stratified_fallback_all_tiny():
    """A dataset made entirely of tiny cells cannot be stratified
    (tiny_to_train sends everything to training) — the split must fall
    back to random instead of returning an empty validation set."""
    from torchnep.data import stratified_split_indices
    metas = [(2, {"Te", "Pb"}) for _ in range(200)]
    tr, va, st = stratified_split_indices(metas, 0.1, 0)
    assert st.get("fallback") == "random"
    assert len(va) == 20
    assert len(tr) == 180
    assert not set(tr) & set(va)


def test_swa_start_window(tmp_path):
    """swa_start gates the averaging window; a window past the end of the
    run must produce no nep_average.txt (with the run otherwise fine)."""
    nepin = tmp_path / "nep_swa.in"
    nepin.write_text(NEP_IN + "epoch 4\nbatch 4\nstage2 1\nstart_stage2 2\n")
    frames = read_xyz(str(PBTE))[:8]
    raw = PBTE.read_text().splitlines()
    keep, i, k = [], 0, 0
    while i < len(raw) and k < 8:
        na = int(raw[i].strip()); keep += raw[i:i + na + 2]
        i += na + 2; k += 1
    xyz = tmp_path / "train.xyz"
    xyz.write_text("\n".join(keep) + "\n")
    train_nep(config_file=str(nepin), data_file=str(xyz),
              output_dir=str(tmp_path / "o"), device="cpu",
              precision="float64", print_interval=100, restart=False,
              checkpoint_interval=1000, prediction_interval=1000,
              run_seed=5, use_swa=True, swa_start=99)
    assert (tmp_path / "o" / "nep_final.txt").exists()
    assert not (tmp_path / "o" / "nep_average.txt").exists()

    # window inside the run -> average IS written
    nepin2 = tmp_path / "nep_swa2.in"
    nepin2.write_text(NEP_IN + "epoch 4\nbatch 4\nstage2 1\nstart_stage2 2\n")
    train_nep(config_file=str(nepin2), data_file=str(xyz),
              output_dir=str(tmp_path / "o2"), device="cpu",
              precision="float64", print_interval=100, restart=False,
              checkpoint_interval=1000, prediction_interval=1000,
              run_seed=5, use_swa=True, swa_start=3)
    assert (tmp_path / "o2" / "nep_average.txt").exists()


def test_default_alloc_conf(monkeypatch):
    """expandable_segments is defaulted only when the user set nothing,
    on CUDA, before the context exists."""
    from torchnep.train import _default_alloc_conf
    import torchnep.train as T

    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    _default_alloc_conf()
    import os
    assert os.environ["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"

    # user setting wins
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "max_split_size_mb:64")
    _default_alloc_conf()
    assert os.environ["PYTORCH_ALLOC_CONF"] == "max_split_size_mb:64"


def test_nn_formulations_equivalent(tmp_path):
    """The bmm (CUDA/CPU) and multiply+reduce (ROCm) NN formulations must
    agree — energies, forces, parameter gradients — in float64."""
    from torchnep.model import NEPModel
    from torchnep import ops as _ops
    store, cfg = _store(tmp_path, n=6)
    batch = store.collate([0, 2, 4])
    torch.manual_seed(2)
    m = NEPModel(cfg).to(torch.float64)
    m.train()

    results = {}
    for flag in (False, True):
        old = _ops.NN_MULSUM
        _ops.NN_MULSUM = flag
        try:
            r = m.compute_properties_cached(batch, need_forces=True,
                                            need_virial=True, backend="loop")
            loss = (r["Ei"] ** 2).sum() + (r["forces"] ** 2).sum()
            g = torch.autograd.grad(loss, list(m.parameters()),
                                    allow_unused=True)
            results[flag] = (r["Ei"].detach(), r["forces"].detach(), g)
        finally:
            _ops.NN_MULSUM = old

    torch.testing.assert_close(results[True][0], results[False][0],
                               rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(results[True][1], results[False][1],
                               rtol=1e-9, atol=1e-11)
    for a, b in zip(results[True][2], results[False][2]):
        if a is None or b is None:
            assert (a is None) == (b is None)
            continue
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-10)
