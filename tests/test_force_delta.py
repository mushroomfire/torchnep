"""Regression tests for GPUMD-compatible ``force_delta`` weighting."""

import math

import pytest
import torch

from torchnep.data import parse_nep_in
from torchnep import train as train_module
from torchnep.train import train_nep


def _config(tmp_path, line=""):
    path = tmp_path / "nep.in"
    path.write_text(f"type 1 Cu\n{line}")
    return parse_nep_in(str(path))


def test_force_delta_defaults_to_disabled(tmp_path):
    assert _config(tmp_path)["force_delta"] == 0.0


def test_force_delta_parses_nonnegative_value(tmp_path):
    config = _config(tmp_path, "force_delta 2\n")
    assert config["force_delta"] == 2.0
    assert "force_delta" in config["_explicit"]


@pytest.mark.parametrize("line", [
    "force_delta\n", "force_delta 1 2\n", "force_delta -0.1\n",
    "force_delta nan\n", "force_delta inf\n", "force_delta -inf\n",
])
def test_force_delta_rejects_invalid_values(tmp_path, line):
    with pytest.raises(ValueError, match="force_delta"):
        _config(tmp_path, line)


def test_force_delta_uses_reference_norm_and_original_denominator():
    reference = torch.tensor([[0., 0., 0.], [2., 0., 0.],
                              [2., 2., 2.], [50., 0., 0.]])
    prediction = reference + torch.tensor([1., 2., 3.])
    mask = torch.tensor([True, True, True, False])

    weighted, ordinary = train_module._force_error_sums(
        prediction, reference, mask, force_delta=2.0)

    expected = 14.0 * (1.0 + 0.5 + 2.0 / (2.0 + math.sqrt(12.0)))
    assert weighted.item() == pytest.approx(expected)
    assert ordinary.item() == pytest.approx(42.0)
    assert (weighted / (3 * mask.sum())).item() == pytest.approx(expected / 9)


def test_force_delta_zero_is_exact_old_path():
    reference = torch.tensor([[10., 0., 0.], [1., 2., 3.]])
    prediction = reference + 1
    weighted, ordinary = train_module._force_error_sums(
        prediction, reference, torch.tensor([True, False]), 0.0)
    assert weighted is ordinary
    assert ordinary.item() == 3.0


@pytest.mark.parametrize("delta", [1e-40, 1e100])
def test_force_delta_is_stable_for_extreme_finite_values(delta):
    weighted, _ = train_module._force_error_sums(
        torch.ones((1, 3)), torch.zeros((1, 3)),
        torch.tensor([True]), delta)
    assert torch.isfinite(weighted)
    assert weighted.item() == pytest.approx(3.0)


def test_force_delta_scales_gradient_per_reference_atom():
    reference = torch.tensor([[0., 0., 0.], [2., 0., 0.]])
    prediction = torch.tensor([[1., 0., 0.], [3., 0., 0.]],
                              requires_grad=True)
    weighted, _ = train_module._force_error_sums(
        prediction, reference, torch.tensor([True, True]), 2.0)
    (weighted / 6).backward()
    torch.testing.assert_close(
        prediction.grad,
        torch.tensor([[1 / 3, 0., 0.], [1 / 6, 0., 0.]]))


class _OneBatchStore:
    n = 1
    has_forces = True
    has_virial = False

    def collate(self, _indices):
        return {"energy_mask": torch.tensor([False]),
                "force_mask": torch.tensor([True, True]),
                "forces": torch.tensor([[0., 0., 0.], [2., 0., 0.]])}


class _FrozenModel:
    training = False

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


def test_validation_objective_weighted_but_rmse_physical():
    def compute(_batch, **_kwargs):
        return {"forces": torch.tensor([[1., 0., 0.], [3., 0., 0.]])}

    loss, _, rmse_f, _, _ = train_module._evaluate_valid_loss(
        _OneBatchStore(), 1, _FrozenModel(), None, compute,
        False, "loop", 0., 1., 0., torch.float32, torch.device("cpu"), 2.)
    assert loss == pytest.approx(0.25)
    assert rmse_f == pytest.approx(math.sqrt(1 / 3))


def _run_one_epoch(tmp_path, name, force_delta, *, batch=1):
    xyz = tmp_path / "train.xyz"
    xyz.write_text(
        "2\n"
        'Lattice="12 0 0 0 12 0 0 0 12" energy=0 '
        'Properties=species:S:1:pos:R:3:force:R:3 pbc="T T T"\n'
        "Cu 1 1 1 0 0 0\nCu 3 1 1 2 0 0\n"
        "3\n"
        'Lattice="12 0 0 0 12 0 0 0 12" energy=0 '
        'Properties=species:S:1:pos:R:3:force:R:3 pbc="T T T"\n'
        "Cu 1 1 1 1 0 0\nCu 3 1 1 10 0 0\nCu 5 1 1 50 0 0\n")
    nepin = tmp_path / f"{name}.in"
    nepin.write_text(
        "type 1 Cu\ncutoff 4 4\nn_max 1 1\nbasis_size 1 1\n"
        "l_max 1 0 0\nneuron 2\nepoch 1\nlr 0\n"
        f"batch {batch}\nlambda_e 0\nlambda_f 1\nlambda_v 0\n"
        f"weight_decay 0\nforce_delta {force_delta}\n")
    out = tmp_path / name
    train_nep(str(nepin), str(xyz), output_dir=str(out), device="cpu",
              precision="float64", restart=False, run_seed=9,
              print_interval=100, checkpoint_interval=100,
              prediction_interval=100)
    row = [line for line in (out / "loss.out").read_text().splitlines()
           if line and not line.startswith("#")][0].split()
    return float(row[1]), float(row[3]), out


def test_training_uses_weighted_loss_but_unweighted_rmse(tmp_path):
    plain_loss, plain_rmse, plain_out = _run_one_epoch(tmp_path, "plain", 0)
    biased_loss, biased_rmse, biased_out = _run_one_epoch(tmp_path, "biased", 2)
    assert biased_loss < plain_loss
    assert biased_rmse == pytest.approx(plain_rmse)

    plain = torch.load(plain_out / "checkpoint.pt", map_location="cpu",
                       weights_only=False)
    biased = torch.load(biased_out / "checkpoint.pt", map_location="cpu",
                        weights_only=False)
    assert "force_delta" not in plain["loss_weights"]
    assert biased["loss_weights"]["force_delta"] == 2.0


def test_sharded_training_matches_single_device(tmp_path):
    """Opt-in CPU DDP check for force-weighted global normalisation."""
    import os
    from pathlib import Path
    import shutil
    import subprocess

    if os.environ.get("TORCHNEP_TEST_DDP") != "1":
        pytest.skip("set TORCHNEP_TEST_DDP=1 to run the two-rank test")
    torchrun = shutil.which("torchrun")
    if torchrun is None:
        pytest.skip("torchrun not found")

    single_loss, single_rmse, _ = _run_one_epoch(
        tmp_path, "single", 2, batch=2)
    runner = tmp_path / "run_sharded.py"
    runner.write_text(
        "import sys\n"
        "from torchnep.train_sharded import train_nep_sharded\n"
        "train_nep_sharded(sys.argv[1], sys.argv[2], output_dir=sys.argv[3], "
        "precision='float64', print_interval=100, checkpoint_interval=100, "
        "prediction_interval=100, restart=False, run_seed=9)\n")
    out = tmp_path / "sharded"
    root = str(Path(train_module.__file__).resolve().parent.parent)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    result = subprocess.run(
        [torchrun, "--standalone", "--nproc_per_node=2", str(runner),
         str(tmp_path / "single.in"), str(tmp_path / "train.xyz"), str(out)],
        capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, result.stderr[-4000:]
    row = [line for line in (out / "loss.out").read_text().splitlines()
           if line and not line.startswith("#")][0].split()
    assert float(row[1]) == pytest.approx(single_loss)
    assert float(row[3]) == pytest.approx(single_rmse)
