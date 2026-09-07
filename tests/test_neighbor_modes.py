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

"""The three ``StreamDataStore`` neighbor layouts must produce the same
batches.

``cached`` is the reference (full pair lists from the numpy builder).
``compact`` keeps int32 pairs + int8 image shifts and rebuilds ``rij`` on the
device; ``on_the_fly`` keeps only positions + cells and runs the neighbor
search on the device. Both must reproduce the cached pair sets exactly and
the displacement vectors to float round-off, and a short training run must
go through in every mode.
"""
import numpy as np
import pytest
import torch

from torchnep.data import read_xyz, parse_nep_in, build_neighbor_list_np
from torchnep.data import build_neighbor_list_np_ex
from torchnep.train import (preprocess_structures, StreamDataStore,
                            iter_collated, estimate_store_bytes,
                            choose_neighbor_mode, NEIGHBOR_MODES)
from _common import DATA_DIR, devices

PBTE = DATA_DIR / "PbTe.xyz"
NEP_IN = ("type 2 Te Pb\ncutoff 6 4\nn_max 4 4\n"
          "basis_size 6 6\nl_max 4 2 1\nneuron 30\n")


def _config(tmp_path):
    p = tmp_path / "nep.in"
    p.write_text(NEP_IN)
    return parse_nep_in(str(p))


def _order(batch, suffix):
    """Permutation sorting the pairs by (i, j, rij) — makes the pair order
    of a batch irrelevant when comparing layouts."""
    i = batch["pair_i_" + suffix].cpu().numpy()
    j = batch["pair_j_" + suffix].cpu().numpy()
    r = batch["rij_" + suffix].cpu().numpy().astype(np.float64)
    return torch.from_numpy(np.lexsort((np.round(r[:, 2], 4), np.round(r[:, 1], 4),
                                        np.round(r[:, 0], 4), j, i)))


def _sorted_pairs(batch, suffix):
    key = _order(batch, suffix).numpy()
    i = batch["pair_i_" + suffix].cpu().numpy()
    j = batch["pair_j_" + suffix].cpu().numpy()
    r = batch["rij_" + suffix].cpu().numpy().astype(np.float64)
    return i[key], j[key], r[key]


def test_np_ex_matches_np():
    """The extended numpy builder is the plain one plus shifts."""
    frames = read_xyz(str(PBTE))[:5]
    for f in frames:
        pos = f["positions"].astype(np.float32); cell = f["cell"].astype(np.float32)
        i0, j0, r0 = build_neighbor_list_np(pos, cell, 6.0)
        i1, j1, r1, sh, pw = build_neighbor_list_np_ex(pos, cell, 6.0)
        assert np.array_equal(i0, i1) and np.array_equal(j0, j1)
        assert np.array_equal(r0, r1)
        # rij reconstructed from wrapped positions + shifts
        rec = (pw[j1] + sh.astype(np.float32) @ cell) - pw[i1]
        assert np.allclose(rec, r1, atol=1e-5)


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("mode", ["compact", "on_the_fly"])
def test_modes_match_cached(tmp_path, device, mode):
    dev = torch.device(device)
    dtype = torch.float32
    cfg = _config(tmp_path)
    frames = read_xyz(str(PBTE))[:24]
    ref_structs = preprocess_structures(frames, cfg, np.float32, mode="cached")
    structs = preprocess_structures(frames, cfg, np.float32, mode=mode)
    ref = StreamDataStore(ref_structs, dev, dtype, config=cfg)
    store = StreamDataStore(structs, dev, dtype, config=cfg, neighbor_mode=mode)
    assert store.memory_bytes() < ref.memory_bytes()

    rng = np.random.default_rng(5)
    for idx in ([0], list(range(8)), rng.permutation(24)[:10].tolist(),
                list(range(24))):
        b = store.collate(idx)
        r = ref.collate(idx)
        assert b["N"] == r["N"]
        for key in ("atom_types", "struct_idx", "energy", "forces",
                    "force_mask", "virial", "natoms"):
            assert torch.equal(b[key], r[key]), key
        for suf in ("rad", "ang"):
            bi, bj, br = _sorted_pairs(b, suf)
            ri, rj, rr = _sorted_pairs(r, suf)
            assert len(bi) == len(ri), (mode, suf, len(bi), len(ri))
            assert np.array_equal(bi, ri) and np.array_equal(bj, rj), (mode, suf)
            assert np.allclose(br, rr, atol=2e-5), (mode, suf)
        # per-pair basis is elementwise: compare it on the matched pair order
        for suf, keys in (("rad", ("fk_rad", "d12inv_rad")),
                          ("ang", ("fk_ang", "d12inv_ang", "blm"))):
            bi, bj, br = _sorted_pairs(b, suf); ri, rj, rr = _sorted_pairs(r, suf)
            kb = np.lexsort((np.round(br[:, 2], 4), np.round(br[:, 1], 4),
                             np.round(br[:, 0], 4), bj, bi))
            i_b = b["pair_i_" + suf].cpu().numpy(); i_r = r["pair_i_" + suf].cpu().numpy()
            ob = _order(b, suf); orr = _order(r, suf)
            for key in keys:
                assert torch.allclose(b[key][ob], r[key][orr], atol=1e-5,
                                      rtol=1e-5), (mode, key)

    # prefetching iterator gives the same batches
    idx_lists = [list(range(k, k + 6)) for k in range(0, 24, 6)]
    for direct, pref in zip((store.collate(i) for i in idx_lists),
                            iter_collated(store, idx_lists)):
        assert torch.equal(direct["pair_i_rad"], pref["pair_i_rad"])
        assert torch.allclose(direct["rij_rad"], pref["rij_rad"])


@pytest.mark.parametrize("device", devices())
def test_on_the_fly_scan_max_neighbors(tmp_path, device):
    dev = torch.device(device)
    cfg = _config(tmp_path)
    frames = read_xyz(str(PBTE))[:24]
    cached = preprocess_structures(frames, cfg, np.float32, mode="cached")
    from torchnep.train import compute_max_neighbors
    exp = compute_max_neighbors(cached)
    store = StreamDataStore(preprocess_structures(frames, cfg, np.float32,
                                                  mode="on_the_fly"),
                            dev, torch.float32, config=cfg,
                            neighbor_mode="on_the_fly")
    assert store.scan_max_neighbors(batch_size=7) == exp


def test_estimate_and_auto_choice(tmp_path):
    cfg = _config(tmp_path)
    frames = read_xyz(str(PBTE))[:24]
    est = estimate_store_bytes(frames, cfg)
    assert est["cached"] > est["compact"] > est["on_the_fly"] > 0
    for m in NEIGHBOR_MODES:
        assert choose_neighbor_mode(frames, cfg, m)[0] == m
    with pytest.raises(ValueError):
        choose_neighbor_mode(frames, cfg, "bogus")
    mode, _, budget = choose_neighbor_mode(frames, cfg, "auto")
    assert mode in NEIGHBOR_MODES
    # a tiny budget forces the memory-saving layouts
    mode2, _, _ = choose_neighbor_mode(frames, cfg, "auto", fraction=1e-12)
    assert mode2 == "on_the_fly"


@pytest.mark.parametrize("mode", NEIGHBOR_MODES)
def test_train_nep_runs_in_every_mode(tmp_path, mode):
    """End-to-end: a 2-epoch single-device run in each layout finishes and
    the loss trajectories agree (they see identical batches)."""
    from torchnep import train_nep
    raw = PBTE.read_text().splitlines()
    out, i, k = [], 0, 0
    while i < len(raw) and k < 12:
        na = int(raw[i].strip()); out += raw[i:i + na + 2]; i += na + 2; k += 1
    xyz = tmp_path / "train.xyz"; xyz.write_text("\n".join(out) + "\n")
    nepin = tmp_path / "nep.in"
    nepin.write_text(NEP_IN + "epoch 2\nbatch 4\n")
    od = tmp_path / f"out_{mode}"
    train_nep(str(nepin), str(xyz), output_dir=str(od), device="cpu",
              run_seed=1, restart=False, neighbor_mode=mode,
              print_interval=1, checkpoint_interval=100,
              prediction_interval=100)
    assert (od / "nep_final.txt").exists()
    loss = np.loadtxt(od / "loss.out", ndmin=2, comments="#")
    assert loss.shape[0] == 2 and np.all(np.isfinite(loss))


def test_train_losses_agree_across_modes(tmp_path):
    from torchnep import train_nep
    raw = PBTE.read_text().splitlines()
    out, i, k = [], 0, 0
    while i < len(raw) and k < 12:
        na = int(raw[i].strip()); out += raw[i:i + na + 2]; i += na + 2; k += 1
    xyz = tmp_path / "train.xyz"; xyz.write_text("\n".join(out) + "\n")
    nepin = tmp_path / "nep.in"
    nepin.write_text(NEP_IN + "epoch 3\nbatch 4\n")
    finals = {}
    for mode in NEIGHBOR_MODES:
        od = tmp_path / f"out_{mode}"
        train_nep(str(nepin), str(xyz), output_dir=str(od), device="cpu",
                  run_seed=1, restart=False, neighbor_mode=mode,
                  print_interval=1, checkpoint_interval=100,
                  prediction_interval=100)
        finals[mode] = np.loadtxt(od / "loss.out", ndmin=2, comments="#")[:, 1:4]
    for mode in ("compact", "on_the_fly"):
        assert np.allclose(finals[mode], finals["cached"], rtol=1e-3, atol=1e-5), mode
