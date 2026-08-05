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

"""Tests for CompiledAutogradForce (make_fx-materialized autograd forces).

CUDA-only (the Inductor/Triton pipeline is the point) — auto-skipped on
CPU-only hosts, so CI never runs the (slow) compile.
"""
import numpy as np
import pytest
import torch

from torchnep.data import read_xyz, parse_nep_in
from torchnep.train import GPUDataStore, preprocess_structures
from torchnep.model import NEPModel, gpumd_init_parameters
from _common import DATA_DIR

PBTE = DATA_DIR.parent.parent / "example" / "PbTe" / "train.xyz"
NEP_IN = ("type 2 Te Pb\ncutoff 6 4\nn_max 4 4\n"
          "basis_size 6 6\nl_max 4 2 1\nneuron 30\n")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="CompiledAutogradForce needs CUDA")


def _setup(tmp_path, n_frames=24):
    from torchnep.compiled_autograd import CompiledAutogradForce
    dev = torch.device("cuda")
    dtype = torch.float32
    p = tmp_path / "nep.in"
    p.write_text(NEP_IN)
    cfg = parse_nep_in(str(p))
    frames = read_xyz(str(PBTE))[:n_frames]
    structs = preprocess_structures(frames, cfg, np.float32)
    store = GPUDataStore(structs, dev, dtype, config=cfg)
    torch.manual_seed(11)
    model = NEPModel(cfg).to(dtype).to(dev)
    gpumd_init_parameters(model)
    model.set_q_scaler(torch.zeros(model.dim, device=dev),
                       torch.ones(model.dim, device=dev))
    model.train()
    return model, store, CompiledAutogradForce(model)


def _run(fn, batch):
    return fn(batch["rij_rad"], batch["rij_ang"],
              batch["pair_i_rad"], batch["pair_j_rad"],
              batch["pair_i_ang"], batch["pair_j_ang"],
              batch["atom_types"], batch["N"],
              batch["struct_idx"], batch["num_structures"],
              need_forces=True, need_virial=True, backend="bmm")


def _loss(r, batch):
    e = ((r["Etot"] / batch["natoms"]
          - batch["energy"] / batch["natoms"]) ** 2).mean()
    f = ((r["forces"] - batch["forces"]) ** 2).mean()
    v = (r["virial"] ** 2).mean()
    return 0.01 * e + f + 0.01 * v


def _rel(a, b):
    return (a - b).abs().max().item() / max(a.abs().max().item(), 1e-30)


def test_compiled_matches_eager_outputs_and_grads(tmp_path):
    """One dynamic graph, multiple batch shapes: outputs AND parameter
    gradients (second-order path through the force loss) must match the
    eager autograd reference to float32 kernel-fusion noise."""
    model, store, caf = _setup(tmp_path)
    rng = np.random.default_rng(1)

    for idx in (list(range(8)), rng.permutation(24)[:8].tolist(),
                rng.permutation(24)[:20].tolist()):
        batch = store.collate(idx)

        model.zero_grad(set_to_none=True)
        r_e = _run(model.compute_properties, batch)
        _loss(r_e, batch).backward()
        g_e = {n: p.grad.detach().clone()
               for n, p in model.named_parameters() if p.grad is not None}

        model.zero_grad(set_to_none=True)
        r_c = _run(caf.compute_properties, batch)
        _loss(r_c, batch).backward()
        g_c = {n: p.grad.detach().clone()
               for n, p in model.named_parameters() if p.grad is not None}

        for key in ("Ei", "Etot", "forces", "virial"):
            assert _rel(r_e[key].detach(), r_c[key].detach()) < 1e-4, key
        assert set(g_e) == set(g_c)
        for n in g_e:
            denom = max(g_e[n].abs().max().item(), 1e-12)
            assert (g_e[n] - g_c[n]).abs().max().item() / denom < 1e-3, n


def test_compiled_energy_only_falls_back(tmp_path):
    """need_forces=False routes to the eager model (graph is force-shaped)."""
    model, store, caf = _setup(tmp_path, n_frames=8)
    batch = store.collate(list(range(8)))
    r = caf.compute_properties(
        batch["rij_rad"], batch["rij_ang"],
        batch["pair_i_rad"], batch["pair_j_rad"],
        batch["pair_i_ang"], batch["pair_j_ang"],
        batch["atom_types"], batch["N"],
        batch["struct_idx"], batch["num_structures"],
        need_forces=False, backend="bmm")
    r_ref = model.compute_properties(
        batch["rij_rad"], batch["rij_ang"],
        batch["pair_i_rad"], batch["pair_j_rad"],
        batch["pair_i_ang"], batch["pair_j_ang"],
        batch["atom_types"], batch["N"],
        batch["struct_idx"], batch["num_structures"],
        need_forces=False, backend="bmm")
    # Not torch.equal: CUDA scatter_add atomics make even two identical
    # eager calls differ in the last ULP.
    torch.testing.assert_close(r["Etot"], r_ref["Etot"],
                               rtol=1e-6, atol=1e-4)
    assert "forces" not in r
