# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project (GPL-3.0-or-later, see train.py).

"""pos_noise (training-time coordinate jitter) and weight_decay (AdamW).

The noise contract: per-atom Gaussian displacements drawn from a DEDICATED
generator (global RNG untouched), applied to the pair vectors of both the
radial and angular lists consistently (rij' = rij + d_j - d_i), labels
untouched, basis computed from the noisy geometry, fully reproducible.
"""
import numpy as np
import torch

from torchnep.data import read_xyz, parse_nep_in
from torchnep.train import preprocess_structures, StreamDataStore, train_nep
from _common import DATA_DIR

PBTE = DATA_DIR.parent.parent / "example" / "PbTe" / "train.xyz"
NEP_IN = ("type 2 Te Pb\ncutoff 6 4\nn_max 4 4\n"
          "basis_size 6 6\nl_max 4 2 1\nneuron 20\n")


def _store(tmp_path, n=10):
    p = tmp_path / "nep.in"
    p.write_text(NEP_IN)
    cfg = parse_nep_in(str(p))
    structs = preprocess_structures(read_xyz(str(PBTE))[:n], cfg, np.float64)
    return StreamDataStore(structs, torch.device("cpu"), torch.float64,
                           config=cfg), cfg


def test_pos_noise_collate_contract(tmp_path):
    store, cfg = _store(tmp_path)
    idx = [0, 2, 5]
    clean = store.collate(idx)

    g = torch.Generator(); g.manual_seed(77)
    noisy = store.collate(idx, noise_gen=g, noise_sigma=0.01)
    g2 = torch.Generator(); g2.manual_seed(77)
    noisy2 = store.collate(idx, noise_gen=g2, noise_sigma=0.01)

    # deterministic given the generator seed
    for k in ("rij_rad", "rij_ang", "fk_rad", "blm"):
        assert torch.equal(noisy[k], noisy2[k]), k

    # exact displacement algebra: rij' = rij + d_j - d_i, same d for both
    # pair lists
    g3 = torch.Generator(); g3.manual_seed(77)
    d = torch.randn(clean["N"], 3, generator=g3,
                    dtype=torch.float64) * 0.01
    exp_r = (clean["rij_rad"] + d[clean["pair_j_rad"]]
             - d[clean["pair_i_rad"]])
    exp_a = (clean["rij_ang"] + d[clean["pair_j_ang"]]
             - d[clean["pair_i_ang"]])
    assert torch.equal(noisy["rij_rad"], exp_r)
    assert torch.equal(noisy["rij_ang"], exp_a)

    # labels untouched; basis follows the NOISY geometry
    assert torch.equal(noisy["forces"], clean["forces"])
    assert torch.equal(noisy["energy"], clean["energy"])
    assert torch.allclose(1.0 / noisy["d12inv_rad"],
                          torch.norm(exp_r, dim=-1))
    assert not torch.equal(noisy["fk_rad"], clean["fk_rad"])

    # the "_clean" view riding along the noisy batch is bit-identical to
    # a plain clean collate — it is what the logged train metrics and the
    # analytic b1 update are computed from
    cv = noisy["_clean"]
    for k in ("rij_rad", "rij_ang", "fk_rad", "fkp_rad", "d12inv_rad",
              "fk_ang", "fkp_ang", "d12inv_ang", "blm"):
        assert torch.equal(cv[k], clean[k]), k
    assert "_clean" not in clean

    # sigma 0 (default) is bit-identical to clean
    again = store.collate(idx)
    assert torch.equal(again["rij_rad"], clean["rij_rad"])


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


def test_pos_noise_training_reproducible_and_active(tmp_path):
    a = _run(tmp_path, "n1", "pos_noise 0.02\n")
    b = _run(tmp_path, "n2", "pos_noise 0.02\n")
    c = _run(tmp_path, "n0", "")
    assert a == b          # same seed + same sigma -> identical run
    assert a != c          # noise actually changes training


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


def test_pos_noise_with_autograd_forces(tmp_path):
    """pos_noise + use_autograd_forces: the clean-metrics pass must not
    require autograd (it runs under no_grad) — regression for the crash
    'element 0 of tensors does not require grad' when the eager autograd
    path was used for clean train metrics."""
    nepin = tmp_path / "nep_ag.in"
    nepin.write_text(NEP_IN + "epoch 2\nbatch 4\npos_noise 0.01\n")
    frames = read_xyz(str(PBTE))[:8]
    raw = PBTE.read_text().splitlines()
    keep, i, k = [], 0, 0
    while i < len(raw) and k < 8:
        na = int(raw[i].strip()); keep += raw[i:i + na + 2]
        i += na + 2; k += 1
    xyz = tmp_path / "train.xyz"
    xyz.write_text("\n".join(keep) + "\n")
    train_nep(config_file=str(nepin), data_file=str(xyz),
              output_dir=str(tmp_path / "ag"), device="cpu",
              precision="float64", print_interval=100, restart=False,
              checkpoint_interval=1000, prediction_interval=1000,
              run_seed=5, use_autograd_forces=True)
    assert (tmp_path / "ag" / "loss.out").exists()
