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

"""Extrapolation grade: parameter gradients, MaxVol, build / grade / select."""

import numpy as np
import pytest
import torch

from _common import DATA_DIR
from torchnep.data import read_xyz
from torchnep.extrapolation import (
    ActiveSet, _DescriptorStream, _Element, _load_calculators, b_vectors, build_active_set,
    compute_gamma, select_structures)
from torchnep.model import NEPModel
from torchnep.train import compute_max_neighbors, preprocess_structures

TRAIN = str(DATA_DIR / "CrCoNi_train.xyz")


def _small_model(path, seed=0, types=("Cr", "Co", "Ni")):
    """A small random NEP4 model (K = 6 x (21 + 2) = 138) for CrCoNi."""
    torch.manual_seed(seed)
    cfg = {"type_names": list(types), "num_types": len(types),
           "cutoff_radial": 5.0, "cutoff_angular": 4.0,
           "n_max_radial": 4, "n_max_angular": 3,
           "basis_size_radial": 6, "basis_size_angular": 6,
           "l_max": [4, 0, 0, 0], "neuron": 6}
    m = NEPModel(cfg).double()
    frames = read_xyz(TRAIN)
    nn_r, nn_a = compute_max_neighbors(preprocess_structures(frames, cfg, np.float64))
    m.save_nep_txt(str(path), nn_r + 8, nn_a + 8)
    return str(path)


def _write_xyz(path, frames):
    with open(path, "w") as f:
        for fr in frames:
            cell = " ".join(f"{v:.10f}" for v in np.asarray(fr["cell"]).reshape(-1))
            f.write(f"{fr['natoms']}\nLattice=\"{cell}\" Properties=species:S:1:pos:R:3 "
                    f"pbc=\"T T T\"\n")
            for s, p in zip(fr["species"], fr["positions"]):
                f.write(f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")


def test_b_vectors_match_autograd():
    torch.manual_seed(0)
    n, D, H = 7, 5, 4
    q = torch.randn(n, D, dtype=torch.float64)
    w0 = torch.randn(H, D, dtype=torch.float64, requires_grad=True)
    b0 = torch.randn(H, dtype=torch.float64, requires_grad=True)
    w1 = torch.randn(H, dtype=torch.float64, requires_grad=True)
    B = b_vectors(q, w0.detach(), b0.detach(), w1.detach())
    for i in range(n):
        E = torch.tanh(q[i] @ w0.T - b0) @ w1
        gw0, gb0, gw1 = torch.autograd.grad(E, [w0, b0, w1])
        ref = torch.cat([gw0, gb0.unsqueeze(1), gw1.unsqueeze(1)], 1).reshape(-1)
        assert torch.allclose(B[i], ref, atol=1e-12)


def _identity_element(r, dtype=torch.float64):
    e = _Element("X", None, None, None)
    e.V = torch.eye(r, dtype=dtype)
    e.scale = torch.ones(r, dtype=dtype)
    return e


def test_maxvol_gives_dominant_rows():
    torch.manual_seed(1)
    n, r, tol = 400, 12, 1.01
    x = torch.randn(n, r, dtype=torch.float64) * torch.logspace(0, -2, r, dtype=torch.float64)
    q = torch.randn(n, 3, dtype=torch.float64)
    src = torch.stack([torch.arange(n), torch.zeros(n, dtype=torch.long)], 1)
    e = _identity_element(r)
    e.init_from_rows(x[:4 * r].clone(), q[:4 * r], src[:4 * r])
    vol0 = abs(float(torch.linalg.det(e.X)))
    xs = x.clone()
    e.maxvol(xs, q.clone(), src.clone(), tol)
    C = x @ torch.linalg.inv(e.X)
    assert float(C.abs().max()) <= tol + 1e-9          # no single swap gains more than tol
    assert abs(float(torch.linalg.det(e.X))) >= vol0
    # the active rows are rows of x, recorded with their origin
    for k in range(r):
        assert torch.allclose(e.X[k], x[int(e.src[k, 0])], atol=1e-12)
    assert torch.allclose(e.Ainv, torch.linalg.inv(e.X), atol=1e-9)


def test_build_grade_reload(tmp_path):
    model = _small_model(tmp_path / "nep.txt")
    out = tmp_path / "as.pt"
    aset = build_active_set(model, TRAIN, str(out), device="cpu", precision="float64",
                            verbose=False)
    for e in aset.elements:
        assert 0 < e.r <= aset.K
    g = compute_gamma(model, str(out), TRAIN, per_atom=True, device="cpu", precision="float64",
                      verbose=False)
    assert g["gamma"].shape == (24,) and g["gamma_atoms"].shape == (24 * 108,)
    assert g["gamma"].max() <= aset.tol + 1e-6          # converged MaxVol on its training set
    assert np.all(g["gamma_res"] <= 1 + 1e-9)
    per_frame = g["gamma_atoms"].reshape(24, 108)
    assert np.allclose(per_frame.max(1), g["gamma"])
    assert np.array_equal(g["grade"], np.maximum(g["gamma"], g["gamma_res"]))
    grade_atoms = np.maximum(per_frame, g["gamma_res_atoms"].reshape(24, 108))
    assert np.all(grade_atoms[np.arange(24), g["atom"]] == g["grade"])
    # float32 grades the same structures almost identically
    g32 = compute_gamma(model, str(out), TRAIN, device="cpu", verbose=False)
    assert np.allclose(g32["gamma"], g["gamma"], rtol=1e-3)
    # an active set only loads with its own model
    other = _small_model(tmp_path / "other.txt", seed=1)
    with pytest.raises(ValueError):
        ActiveSet.load(str(out), other, device="cpu")


def test_select_skips_near_duplicates(tmp_path):
    model = _small_model(tmp_path / "nep.txt")
    aset = build_active_set(model, TRAIN, str(tmp_path / "as.pt"), device="cpu",
                            precision="float64", verbose=False)
    rng = np.random.default_rng(0)
    frames = read_xyz(TRAIN)[:6]
    cand = []
    for fr in frames:                           # strongly rattled frame + an exact copy
        fr = dict(fr)
        fr["positions"] = fr["positions"] + rng.normal(0, 0.25, fr["positions"].shape)
        cand += [fr, fr]
    path = tmp_path / "cand.xyz"
    _write_xyz(path, cand)
    res = select_structures(model, str(tmp_path / "as.pt"), str(path),
                            output_xyz=str(tmp_path / "chosen.xyz"),
                            output_active_set=str(tmp_path / "as2.pt"),
                            device="cpu", verbose=False)
    chosen = res["index"]
    assert len(chosen) > 0
    assert len(set(i // 2 for i in chosen)) == len(chosen)        # never both copies
    assert np.all(res["grade_at_choice"] > aset.tol)
    assert len(read_xyz(str(tmp_path / "chosen.xyz"))) == len(chosen)
    # against the extended active set nothing chosen extrapolates any more
    g2 = compute_gamma(model, str(tmp_path / "as2.pt"), str(tmp_path / "chosen.xyz"),
                       device="cpu", precision="float64", verbose=False)
    assert g2["gamma"].max() <= aset.tol + 1e-6
    # a budget stops the choice early
    res1 = select_structures(model, str(tmp_path / "as.pt"), str(path), max_frames=1,
                             device="cpu", verbose=False)
    assert len(res1["index"]) == 1 and res1["index"][0] == chosen[0]


@pytest.mark.parametrize("model", ["nep_CrCoNi.txt", "nep_CrCoNi_multicut.txt"])
def test_streamed_descriptors_match_the_calculator(model):
    """The batched, streamed descriptors equal NEPCalculator's, frame by frame
    (uniform and per-species cutoffs)."""
    calc, desc = _load_calculators(str(DATA_DIR / model), torch.device("cpu"), "float64")
    xyz = str(DATA_DIR / "CrCoNi.xyz")
    frames = read_xyz(xyz)
    stream = _DescriptorStream(desc, xyz, chunk_atoms=250)       # several chunks
    seen = 0
    for ids, nat, types, q in stream.chunks():
        a = 0
        for i, n in zip(ids, nat):
            f = frames[i]
            ref = calc.get_descriptor(f["species"], f["positions"], f["cell"])
            assert np.allclose(q[a:a + n].numpy(), ref, rtol=1e-10, atol=1e-12)
            assert [calc.type_names[t] for t in types[a:a + n].tolist()] == list(f["species"])
            a += n
            seen += 1
    stream.close()
    assert seen == len(frames)


def test_parallel_parsing_and_device_batches_match_the_calculator(tmp_path):
    """Parser workers, prefetch and several device batches per chunk give the
    calculator's descriptors, also for cells much smaller than the cutoff
    (many periodic images) and triclinic ones."""
    rng = np.random.default_rng(0)
    base = read_xyz(str(DATA_DIR / "CrCoNi.xyz"))
    frames = []
    for k in range(70):
        f = dict(base[k % len(base)])
        if k % 7 == 3:        # one-atom triclinic fcc primitive cell
            a = 3.52
            f = {"natoms": 1, "species": [("Cr", "Co", "Ni")[k % 3]],
                 "positions": np.zeros((1, 3)),
                 "cell": 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0.]])}
        elif k % 7 == 5:      # four-atom cubic cell, atoms outside the cell
            a = 3.55
            f = {"natoms": 4, "species": ["Ni", "Co", "Ni", "Cr"],
                 "positions": a * (np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
                                   + rng.integers(-2, 3, (4, 3))),
                 "cell": a * np.eye(3)}
        f["positions"] = np.asarray(f["positions"]) + rng.normal(0, 0.05, (f["natoms"], 3))
        frames.append(f)
    xyz = str(tmp_path / "mixed.xyz")
    _write_xyz(xyz, frames)
    frames = read_xyz(xyz)
    for model in ("nep_CrCoNi.txt", "nep_CrCoNi_multicut.txt"):
        calc, desc = _load_calculators(str(DATA_DIR / model), torch.device("cpu"), "float64")
        stream = _DescriptorStream(desc, xyz, chunk_atoms=600, workers=2)
        assert stream.pool is not None
        stream.max_pairs = 20_000                  # several device batches per chunk
        seen = 0
        for ids, nat, types, q in stream.chunks():
            a = 0
            for i, n in zip(ids, nat):
                f = frames[i]
                ref = calc.get_descriptor(f["species"], f["positions"], f["cell"])
                assert np.allclose(q[a:a + n].numpy(), ref, rtol=1e-10, atol=1e-12)
                a += n
                seen += 1
        stream.close()
        assert seen == len(frames)


def test_whitened_gamma_equals_textbook_maxvol():
    """In a full-rank subspace the grade from the whitened, projected rows is
    the plain MaxVol grade max |b A^-1| of the raw rows."""
    torch.manual_seed(3)
    n, H, D = 600, 3, 4                            # K = 3 x (4 + 2) = 18
    w0 = torch.randn(H, D, dtype=torch.float64)
    b0 = torch.randn(H, dtype=torch.float64) * 0.3
    w1 = torch.randn(H, dtype=torch.float64)
    q = torch.randn(n, D, dtype=torch.float64)
    B = b_vectors(q, w0, b0, w1)
    K = B.shape[1]
    lam, V = torch.linalg.eigh(B.T @ B)
    assert float(lam[0] / lam[-1]) > 1e-10          # well conditioned: full rank
    e = _Element("X", w0, b0, w1)
    e.V, e.scale = V.flip(1).contiguous(), 1.0 / torch.sqrt(lam.flip(0) / n)
    src = torch.stack([torch.arange(n), torch.zeros(n, dtype=torch.long)], 1)
    x, res = e.project(q)
    assert float(res.max()) < 1e-6 * float(torch.linalg.vector_norm(B, dim=1).max())
    e.init_from_rows(x[:4 * K].clone(), q[:4 * K], src[:4 * K])
    e.maxvol(x.clone(), q.clone(), src.clone(), 1.01)
    A_raw = b_vectors(e.q, w0, b0, w1)             # the active rows, unprojected
    q_new = torch.randn(50, D, dtype=torch.float64) * 1.5
    x_new, _ = e.project(q_new)
    g_lib = e.coefficients(x_new).abs().amax(1)
    g_ref = (b_vectors(q_new, w0, b0, w1) @ torch.linalg.inv(A_raw)).abs().amax(1)
    assert torch.allclose(g_lib, g_ref, rtol=1e-8)
    g_train = (B @ torch.linalg.inv(A_raw)).abs().amax(1)
    assert float(g_train.max()) <= 1.01 + 1e-9     # every row inside the active set


def test_element_without_training_data_is_infinite(tmp_path):
    model = _small_model(tmp_path / "nep.txt", types=("Cr", "Co", "Ni", "Fe"))
    aset = build_active_set(model, TRAIN, str(tmp_path / "as.pt"), device="cpu", verbose=False)
    assert aset.elements[3].r == 0                 # no Fe in the training set
    fr = dict(read_xyz(TRAIN)[0])
    fr["species"] = ["Fe"] + list(fr["species"][1:])
    _write_xyz(tmp_path / "fe.xyz", [fr])
    g = compute_gamma(model, str(tmp_path / "as.pt"), str(tmp_path / "fe.xyz"), per_atom=True,
                      device="cpu", verbose=False)
    assert np.isinf(g["gamma_atoms"][0]) and np.isfinite(g["gamma_atoms"][1:]).all()
    assert np.isinf(g["gamma"][0]) and g["atom"][0] == 0
    # a frame with an element never trained is always a candidate and chosen
    res = select_structures(model, str(tmp_path / "as.pt"), str(tmp_path / "fe.xyz"),
                            device="cpu", verbose=False)
    assert list(res["index"]) == [0] and np.isinf(res["grade_at_choice"][0])
    # GPUMD cannot grade Fe: its block is left out, with a warning
    with pytest.warns(UserWarning, match="Fe"):
        written = ActiveSet.load(str(tmp_path / "as.pt"), model).save_gpumd(
            str(tmp_path / "as.asi"))
    assert written == ["Cr", "Co", "Ni"]


_SHARDED_RUNNER = """
import sys
import numpy as np
from torchnep.extrapolation import (build_active_set_sharded, compute_gamma_sharded,
                                    select_structures_sharded)
model, train, cand, out = sys.argv[1:5]
build_active_set_sharded(model, train, out + "/as_ddp.pt", precision="float64", verbose=False)
g = compute_gamma_sharded(model, out + "/as_ddp.pt", cand, per_atom=True, precision="float64",
                          verbose=False)
res = select_structures_sharded(model, out + "/as_ddp.pt", cand, output_xyz=out + "/chosen_ddp.xyz",
                                precision="float64", verbose=False)
if g is not None:
    np.savez(out + "/g_ddp.npz", **g)
    np.save(out + "/index_ddp.npy", res["index"])
"""


def test_sharded_matches_single_process(tmp_path):
    """2-rank CPU/gloo run of the three sharded functions: the active set keeps
    every training atom at gamma <= tol, and grading and selection with it
    match the single-process functions (grades to round-off, choice exactly).

    Opt-in (local only), like the other DDP tests: TORCHNEP_TEST_DDP=1."""
    import os
    import subprocess
    if os.environ.get("TORCHNEP_TEST_DDP") != "1":
        pytest.skip("DDP test is local-only (set TORCHNEP_TEST_DDP=1)")
    from _common import torchrun_cmd
    cmd = torchrun_cmd(2)
    if not cmd:
        pytest.skip("torchrun not on PATH")
    model = _small_model(tmp_path / "nep.txt")
    rng = np.random.default_rng(1)
    cand = []
    for fr in read_xyz(TRAIN)[:8]:
        fr = dict(fr)
        fr["positions"] = fr["positions"] + rng.normal(0, 0.2, fr["positions"].shape)
        cand += [fr, fr]
    _write_xyz(tmp_path / "cand.xyz", cand)
    runner = tmp_path / "runner.py"
    runner.write_text(_SHARDED_RUNNER)
    root = str(DATA_DIR.parent.parent)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run(cmd + [str(runner), model, TRAIN, str(tmp_path / "cand.xyz"), str(tmp_path)],
                       capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]

    as_ddp = str(tmp_path / "as_ddp.pt")
    g_train = compute_gamma(model, as_ddp, TRAIN, device="cpu", precision="float64",
                            verbose=False)
    assert g_train["gamma"].max() <= 1.01 + 1e-6
    ref = compute_gamma(model, as_ddp, str(tmp_path / "cand.xyz"), per_atom=True,
                        device="cpu", precision="float64", verbose=False)
    ddp = np.load(tmp_path / "g_ddp.npz")
    # the ranks batch the descriptors differently: equal up to summation order
    for k in ("gamma", "gamma_res", "gamma_atoms", "gamma_res_atoms"):
        rel = np.abs(ddp[k] - ref[k]) / np.abs(ref[k])
        # gamma_res comes from |b|^2 - |b V|^2: the cancellation costs digits
        assert rel.max() <= (1e-10 if k in ("gamma", "gamma_atoms") else 1e-6), k
    assert np.array_equal(ddp["atom"], ref["atom"])
    sel = select_structures(model, as_ddp, str(tmp_path / "cand.xyz"), device="cpu",
                            precision="float64", verbose=False)
    assert np.array_equal(np.load(tmp_path / "index_ddp.npy"), sel["index"])
    assert len(read_xyz(str(tmp_path / "chosen_ddp.xyz"))) == len(sel["index"])


def test_float32_grades_track_float64(tmp_path):
    model = _small_model(tmp_path / "nep.txt")
    build_active_set(model, TRAIN, str(tmp_path / "as.pt"), device="cpu", precision="float64",
                     verbose=False)
    rng = np.random.default_rng(2)
    frames = []
    for fr in read_xyz(TRAIN)[:6]:
        fr = dict(fr)
        fr["positions"] = fr["positions"] + rng.normal(0, 0.15, fr["positions"].shape)
        frames.append(fr)
    _write_xyz(tmp_path / "cand.xyz", frames)
    kw = dict(per_atom=True, device="cpu", verbose=False)
    g64 = compute_gamma(model, str(tmp_path / "as.pt"), str(tmp_path / "cand.xyz"),
                        precision="float64", **kw)
    g32 = compute_gamma(model, str(tmp_path / "as.pt"), str(tmp_path / "cand.xyz"),
                        precision="float32", **kw)
    for k in ("gamma_atoms", "gamma_res_atoms"):
        rel = np.abs(g32[k] - g64[k]) / np.abs(g64[k])
        assert rel.max() < 1e-3, (k, rel.max())


def test_float32_build_keeps_the_training_set_inside(tmp_path):
    """The float32 build (descriptors and screening in float32) converges like
    the float64 one: every training atom within tol, to float32 round-off."""
    model = _small_model(tmp_path / "nep.txt")
    a32 = build_active_set(model, TRAIN, str(tmp_path / "a32.pt"), device="cpu", verbose=False)
    a64 = build_active_set(model, TRAIN, str(tmp_path / "a64.pt"), device="cpu",
                           precision="float64", verbose=False)
    assert a32.precision == "float32" and a32.meta["converged"] and a64.meta["converged"]
    for e32, e64 in zip(a32.elements, a64.elements):
        assert e32.r == e64.r
    for prec in ("float32", "float64"):
        g = compute_gamma(model, str(tmp_path / "a32.pt"), TRAIN, device="cpu", precision=prec,
                          verbose=False)
        assert g["gamma"].max() <= a32.tol * (1 + 1e-4), (prec, g["gamma"].max())


def test_gpumd_export_reproduces_gamma(tmp_path):
    """GPUMD grades an atom with the ASI matrix M read as a column-major
    K x K array (gemv, no transpose): max |M_colmajor @ b|. The exported
    file gives gamma of every atom."""
    model = _small_model(tmp_path / "nep.txt")
    aset = build_active_set(model, TRAIN, str(tmp_path / "as.pt"), device="cpu",
                            precision="float64", verbose=False)
    assert aset.save_gpumd(str(tmp_path / "as.asi")) == ["Cr", "Co", "Ni"]
    K = aset.K
    tokens = open(tmp_path / "as.asi").read().split()
    blocks, i = {}, 0
    while i < len(tokens):                         # GPUMD's reader: symbol, shape, values
        name, n1, n2 = tokens[i], int(tokens[i + 1]), int(tokens[i + 2])
        assert (n1, n2) == (K, K)
        blocks[name] = np.asarray(tokens[i + 3:i + 3 + n1 * n2], dtype=np.float64)
        i += 3 + n1 * n2
    rng = np.random.default_rng(4)
    frames = []
    for fr in read_xyz(TRAIN)[:3]:
        fr = dict(fr)
        fr["positions"] = fr["positions"] + rng.normal(0, 0.2, fr["positions"].shape)
        frames.append(fr)
    _write_xyz(tmp_path / "md.xyz", frames)
    g = compute_gamma(model, aset, str(tmp_path / "md.xyz"), per_atom=True, device="cpu",
                      precision="float64", verbose=False)
    _, desc = _load_calculators(model, torch.device("cpu"), "float64")
    stream = _DescriptorStream(desc, str(tmp_path / "md.xyz"))
    k = 0
    for ids, nat, types, q in stream.chunks():
        for t, qa in zip(types.tolist(), q):
            e = aset.elements[t]
            b = b_vectors(qa[None], e.w0, e.b0, e.w1)[0].numpy()
            A = blocks[e.name].reshape(K, K, order="F")        # column-major, lda = K
            assert np.isclose(np.abs(A @ b).max(), g["gamma_atoms"][k], rtol=1e-9)
            k += 1
    stream.close()
    assert k == len(g["gamma_atoms"]) and g["gamma"].max() > aset.tol   # some atoms extrapolate


def test_select_takes_frames_outside_the_subspace(tmp_path):
    """A frame whose atoms lie within the active set (gamma <= tol) but have
    more weight outside the subspace than any training atom (gamma_res > 1)
    is chosen; its exact copy is not, since the directions it added cover it."""
    model = _small_model(tmp_path / "nep.txt")
    build_active_set(model, TRAIN, str(tmp_path / "as.pt"), rcond=1e-2, device="cpu",
                     precision="float64", verbose=False)
    rng = np.random.default_rng(0)
    frames = []
    for fr in read_xyz(TRAIN)[:8]:
        fr = dict(fr)
        fr["positions"] = fr["positions"] + rng.normal(0, 0.03, fr["positions"].shape)
        frames.append(fr)
    _write_xyz(tmp_path / "pool.xyz", [frames[4], frames[4]])
    kw = dict(device="cpu", precision="float64", verbose=False)
    g = compute_gamma(model, str(tmp_path / "as.pt"), str(tmp_path / "pool.xyz"), **kw)
    assert g["gamma"][0] <= 1.01 < g["gamma_res"][0], (g["gamma"], g["gamma_res"])
    res = select_structures(model, str(tmp_path / "as.pt"), str(tmp_path / "pool.xyz"), **kw)
    assert list(res["index"]) == [0]
    assert np.isclose(res["grade_at_choice"][0], g["gamma_res"][0])
