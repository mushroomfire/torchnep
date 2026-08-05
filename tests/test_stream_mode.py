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

"""Tests for ``StreamDataStore`` — the (only) training data store.

The store keeps everything in host memory and assembles device batches on
demand. ``collate`` must reproduce, bit-for-bit, a reference batch built
independently from the raw per-frame structures (concatenation + index
offsets + the Chebyshev/angular basis evaluated directly with the ops
functions on the batch's rij).
"""
import numpy as np
import pytest
import torch

from torchnep import ops
from torchnep.data import read_xyz, parse_nep_in
from torchnep.train import (preprocess_structures, StreamDataStore,
                            iter_collated)
from _common import DATA_DIR, devices

PBTE = DATA_DIR.parent.parent / "example" / "PbTe" / "train.xyz"
NEP_IN = ("type 2 Te Pb\ncutoff 6 4\nn_max 4 4\n"
          "basis_size 6 6\nl_max 4 2 1\nneuron 30\n")


def _config(tmp_path):
    p = tmp_path / "nep.in"
    p.write_text(NEP_IN)
    return parse_nep_in(str(p))


def _reference_batch(structs, indices, cfg, dev, dtype):
    """Independently assemble the batch straight from the structures."""
    sel = [structs[i] for i in indices]
    offsets = np.concatenate([[0], np.cumsum([s["natoms"] for s in sel])])

    def cat(key):
        return torch.from_numpy(np.concatenate([s[key] for s in sel])).to(dev)

    ref = {
        "N": int(offsets[-1]), "num_structures": len(sel),
        "atom_types": cat("atom_types"),
        "rij_rad": cat("rij_rad").to(dtype),
        "rij_ang": cat("rij_ang").to(dtype),
    }
    for key, cnt in (("pair_i_rad", "rij_rad"), ("pair_j_rad", "rij_rad"),
                     ("pair_i_ang", "rij_ang"), ("pair_j_ang", "rij_ang")):
        parts = [torch.from_numpy(s[key]) + int(offsets[k])
                 for k, s in enumerate(sel)]
        ref[key] = torch.cat(parts).to(dev)
    ref["struct_idx"] = torch.cat([
        torch.full((s["natoms"],), k, dtype=torch.long)
        for k, s in enumerate(sel)]).to(dev)

    # Basis straight from the ops functions on the batch rij.
    l3 = cfg["l_max"][0]
    dr = torch.norm(ref["rij_rad"], dim=-1)
    fk_r, fkp_r = ops.chebyshev_basis_and_deriv(
        dr, cfg["cutoff_radial"], cfg["basis_size_radial"])
    da = torch.norm(ref["rij_ang"], dim=-1)
    fk_a, fkp_a = ops.chebyshev_basis_and_deriv(
        da, cfg["cutoff_angular"], cfg["basis_size_angular"])
    dinv_a = 1.0 / da
    ref.update(fk_rad=fk_r, fkp_rad=fkp_r, d12inv_rad=1.0 / dr,
               fk_ang=fk_a, fkp_ang=fkp_a, d12inv_ang=dinv_a,
               blm=ops.angular_basis(
                   ref["rij_ang"][:, 0] * dinv_a,
                   ref["rij_ang"][:, 1] * dinv_a,
                   ref["rij_ang"][:, 2] * dinv_a, l3))
    return ref


@pytest.mark.parametrize("device", devices())
def test_collate_matches_reference(tmp_path, device):
    """collate output must equal the independently assembled reference
    bit-for-bit (indices, rij, and the on-the-fly basis)."""
    dev = torch.device(device)
    dtype = torch.float32
    cfg = _config(tmp_path)
    frames = read_xyz(str(PBTE))[:20]
    structs = preprocess_structures(frames, cfg, np.float32)
    store = StreamDataStore(structs, dev, dtype, config=cfg)

    rng = np.random.default_rng(3)
    for idx in ([0], list(range(8)), rng.permutation(20)[:8].tolist(),
                list(range(20))):
        batch = store.collate(idx)
        ref = _reference_batch(structs, idx, cfg, dev, dtype)
        assert batch["N"] == ref["N"]
        assert batch["num_structures"] == ref["num_structures"]
        for key in ("atom_types", "struct_idx", "pair_i_rad", "pair_j_rad",
                    "rij_rad", "pair_i_ang", "pair_j_ang", "rij_ang",
                    "fk_rad", "fkp_rad", "d12inv_rad", "fk_ang", "fkp_ang",
                    "d12inv_ang", "blm"):
            assert torch.equal(batch[key], ref[key]), key


@pytest.mark.parametrize("device", devices())
def test_store_metadata_and_masks(tmp_path, device):
    """Metadata (counts / flags / per-frame views) and batch masks must
    reflect the structures, including missing energy/forces channels."""
    dev = torch.device(device)
    cfg = _config(tmp_path)
    frames = read_xyz(str(PBTE))[:12]
    frames[3].pop("energy", None)
    frames[5].pop("forces", None)
    structs = preprocess_structures(frames, cfg, np.float64)
    store = StreamDataStore(structs, dev, torch.float64, config=cfg)

    assert store.n == 12
    assert store.natoms == [s["natoms"] for s in structs]
    assert store.has_energy_flag == ["energy" in s for s in structs]
    assert store.has_forces_flag == ["forces" in s for s in structs]
    assert store.n_energy == 11 and store.n_forces == 11
    for i in (0, 3, 5, 11):
        exp = structs[i].get("forces")
        if exp is not None:
            assert torch.equal(store.forces[i].cpu().double(),
                               torch.from_numpy(np.asarray(exp)).double())

    batch = store.collate(list(range(12)))
    e_mask = batch["energy_mask"].cpu().tolist()
    assert e_mask == store.has_energy_flag
    f_mask = batch["force_mask"].cpu()
    off = np.concatenate([[0], np.cumsum(store.natoms)])
    for i in range(12):
        seg = f_mask[int(off[i]):int(off[i + 1])]
        assert bool(seg.all()) == store.has_forces_flag[i]
        assert bool(seg.any()) == store.has_forces_flag[i]


def test_iter_collated_prefetch_matches_direct(tmp_path):
    """Prefetched iteration yields exactly the same batches as direct
    collate calls (same order, same tensors)."""
    cfg = _config(tmp_path)
    frames = read_xyz(str(PBTE))[:16]
    structs = preprocess_structures(frames, cfg, np.float64)
    store = StreamDataStore(structs, torch.device("cpu"), torch.float64,
                            config=cfg)
    idx_lists = [[0, 3, 5], [1, 2], list(range(16)), [15]]
    direct = [store.collate(i) for i in idx_lists]
    for got, want in zip(iter_collated(store, idx_lists), direct):
        for key in ("atom_types", "rij_rad", "fk_ang", "forces", "energy"):
            assert torch.equal(got[key], want[key]), key


_SHARDED_RUNNER = """
import sys
from torchnep.train_sharded import train_nep_sharded
train_nep_sharded(sys.argv[1], sys.argv[2], output_dir=sys.argv[3],
                  precision="float64", print_interval=100,
                  checkpoint_interval=10000, prediction_interval=10000,
                  restart=False, run_seed=99)
"""


def test_sharded_run_reproducible(tmp_path):
    """2-rank DDP (CPU/gloo): two identical runs with the same seed are
    byte-identical — DDP smoke coverage for the streamed store.

    Opt-in (local only): multi-process rendezvous can hang on constrained
    CI runners, so this is skipped unless TORCHNEP_TEST_DDP=1 is set.
    Run locally with:  TORCHNEP_TEST_DDP=1 pytest tests/test_stream_mode.py
    """
    import os
    import shutil
    import subprocess
    if os.environ.get("TORCHNEP_TEST_DDP") != "1":
        pytest.skip("DDP test is local-only (set TORCHNEP_TEST_DDP=1)")
    torchrun = shutil.which("torchrun")
    if torchrun is None:
        pytest.skip("torchrun not on PATH")

    nepin = tmp_path / "nep.in"
    nepin.write_text(NEP_IN + "epoch 3\nbatch 4\n")
    raw = PBTE.read_text().splitlines()
    out, i, k = [], 0, 0
    while i < len(raw) and k < 16:
        na = int(raw[i].strip())
        out += raw[i:i + na + 2]
        i += na + 2; k += 1
    xyz = tmp_path / "train.xyz"
    xyz.write_text("\n".join(out) + "\n")
    runner = tmp_path / "runner.py"
    runner.write_text(_SHARDED_RUNNER)

    root = str(DATA_DIR.parent.parent)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    for out_dir in ("out_a", "out_b"):
        r = subprocess.run(
            [torchrun, "--standalone", "--nproc_per_node=2", str(runner),
             str(nepin), str(xyz), str(tmp_path / out_dir)],
            capture_output=True, text=True, env=env, timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]

    a, b = tmp_path / "out_a", tmp_path / "out_b"
    assert (a / "loss.out").read_text() == (b / "loss.out").read_text()
    assert (a / "nep_final.txt").read_text() == (b / "nep_final.txt").read_text()
