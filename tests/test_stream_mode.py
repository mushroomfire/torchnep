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

"""Tests for ``stream_mode`` (host-resident dataset, batch streaming).

``StreamDataStore`` must be a drop-in replacement for ``GPUDataStore``:
same collate values (bit-exact, including the on-the-fly basis), same
metadata interface, and a training run in stream mode must reproduce the
default-mode run exactly (CPU float64, where the backward is deterministic).
"""
import numpy as np
import pytest
import torch

from torchnep.data import read_xyz, parse_nep_in
from torchnep.train import (preprocess_structures, GPUDataStore,
                            StreamDataStore, train_nep)
from _common import DATA_DIR, devices

PBTE = DATA_DIR.parent.parent / "example" / "PbTe" / "train.xyz"
NEP_IN = ("type 2 Te Pb\ncutoff 6 4\nn_max 4 4\n"
          "basis_size 6 6\nl_max 4 2 1\nneuron 30\n")

BATCH_KEYS = [
    "atom_types", "struct_idx", "pair_i_rad", "pair_j_rad", "rij_rad",
    "pair_i_ang", "pair_j_ang", "rij_ang", "energy", "natoms", "volumes",
    "energy_mask", "forces", "force_mask", "virial", "virial_mask",
    "fk_rad", "fkp_rad", "d12inv_rad", "fk_ang", "fkp_ang", "d12inv_ang",
    "blm",
]


def _config(tmp_path, extra=""):
    p = tmp_path / "nep.in"
    p.write_text(NEP_IN + extra)
    return parse_nep_in(str(p))


@pytest.mark.parametrize("device", devices())
def test_collate_bit_exact(tmp_path, device):
    """Stream collate must reproduce GPUDataStore's batches bit-for-bit,
    including the recomputed Chebyshev/angular basis."""
    dev = torch.device(device)
    dtype = torch.float32
    cfg = _config(tmp_path)
    frames = read_xyz(str(PBTE))[:20]
    structs = preprocess_structures(frames, cfg, np.float32)

    gpu_store = GPUDataStore(structs, dev, dtype, config=cfg)
    str_store = StreamDataStore(structs, dev, dtype, config=cfg)

    rng = np.random.default_rng(3)
    for idx in ([0], list(range(8)), rng.permutation(20)[:8].tolist(),
                list(range(20))):
        bg = gpu_store.collate(idx)
        bs = str_store.collate(idx)
        assert bg["N"] == bs["N"]
        assert bg["num_structures"] == bs["num_structures"]
        for key in BATCH_KEYS:
            tg, ts = bg[key], bs[key]
            assert tg.shape == ts.shape, f"{key}: shape mismatch"
            assert tg.dtype == ts.dtype, f"{key}: dtype mismatch"
            assert tg.device.type == ts.device.type, f"{key}: device mismatch"
            assert torch.equal(tg, ts), f"{key}: values differ"


@pytest.mark.parametrize("device", devices())
def test_store_metadata_parity(tmp_path, device):
    """Every metadata attribute consumers read must match GPUDataStore."""
    dev = torch.device(device)
    cfg = _config(tmp_path)
    frames = read_xyz(str(PBTE))[:12]
    # Exercise the missing-channel paths too (PbTe frames carry no virial,
    # so the virial-missing path is exercised by every frame).
    frames[3].pop("energy", None)
    frames[5].pop("forces", None)
    frames[7].pop("virial", None)
    structs = preprocess_structures(frames, cfg, np.float64)

    a = GPUDataStore(structs, dev, torch.float64, config=cfg)
    b = StreamDataStore(structs, dev, torch.float64, config=cfg)

    assert a.n == b.n
    assert a.natoms == b.natoms
    assert a.energy == b.energy
    assert a.has_energy_flag == b.has_energy_flag
    assert a.has_forces_flag == b.has_forces_flag
    assert a.has_virial_flag == b.has_virial_flag
    assert (a.n_energy, a.n_forces, a.n_virial) == \
           (b.n_energy, b.n_forces, b.n_virial)
    assert (a.has_forces, a.has_virial) == (b.has_forces, b.has_virial)
    assert a.has_cached_basis and b.has_cached_basis
    assert torch.equal(a.volumes, b.volumes)
    for i in range(a.n):
        assert torch.equal(a.forces[i].cpu().double(),
                           b.forces[i].cpu().double())
        assert torch.equal(a.virial[i].cpu().double(),
                           b.virial[i].cpu().double())
    # Masks in a mixed-coverage batch
    ba = a.collate(list(range(12)))
    bb = b.collate(list(range(12)))
    for key in ("energy_mask", "force_mask", "virial_mask"):
        assert torch.equal(ba[key], bb[key])


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


def test_train_stream_reproduces_default(tmp_path):
    """Same seed, stream_mode on vs off: identical loss.out, identical
    nep_final.txt and identical end-of-training predictions (CPU float64 is
    deterministic, and the streamed batches are bit-identical)."""
    nepin, xyz = _write_run_files(tmp_path, n_frames=16, epochs=3)

    out_a = tmp_path / "out_default"
    out_b = tmp_path / "out_stream"
    _train(nepin, xyz, out_a, run_seed=77, prediction_interval=2)
    _train(nepin, xyz, out_b, run_seed=77, prediction_interval=2,
           stream_mode=True)

    assert (out_a / "loss.out").read_text() == (out_b / "loss.out").read_text()
    assert (out_a / "nep_final.txt").read_text() == \
           (out_b / "nep_final.txt").read_text()
    assert (out_a / "nep_best.txt").read_text() == \
           (out_b / "nep_best.txt").read_text()
    for f in ("energy_train.out", "force_train.out", "virial_train.out",
              "stress_train.out"):
        assert (out_a / f).read_text() == (out_b / f).read_text(), f


def test_train_stream_with_validation(tmp_path):
    """stream_mode combined with valid_ratio: the validation store streams
    too, and the run matches the default-mode run exactly."""
    nepin, xyz = _write_run_files(tmp_path, n_frames=20, epochs=3)

    out_a = tmp_path / "out_default"
    out_b = tmp_path / "out_stream"
    _train(nepin, xyz, out_a, run_seed=5, valid_ratio=0.25)
    _train(nepin, xyz, out_b, run_seed=5, valid_ratio=0.25, stream_mode=True)

    assert (out_a / "loss.out").read_text() == (out_b / "loss.out").read_text()
    assert (out_a / "nep_best.txt").read_text() == \
           (out_b / "nep_best.txt").read_text()
    for f in ("energy_test.out", "force_test.out", "virial_test.out"):
        assert (out_a / f).read_text() == (out_b / f).read_text(), f
