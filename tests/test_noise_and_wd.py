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
