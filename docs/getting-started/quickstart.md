# Quick start

From a training set to a model running in GPUMD, in four steps.

## 1. Prepare the training data

Put the reference structures in one extended-XYZ file, `train.xyz`, with the lattice, energies, forces and (optionally) virials of each frame. The format is described in [Training data](training-data.md).

## 2. Write `nep.in`

`nep.in` holds the model architecture and the training hyperparameters. A starting point for a binary alloy:

```text title="nep.in"
type       2 Cu Zr
version    4
zbl        2.0
cutoff     6 5
n_max      6 6
basis_size 8 8
l_max      4 2 0
neuron     50

epoch      600
batch      32
lr         0.01
stage2     1
```

Every keyword and its default is listed in the [nep.in reference](../guide/nep-in.md).

## 3. Train

```python title="run_train.py"
from torchnep import train_nep

train_nep("nep.in", "train.xyz", output_dir="output", valid_ratio=0.1)
```

```bash
python run_train.py
```

`valid_ratio=0.1` holds out 10 % of the frames as a validation set: the best model and the learning-rate schedule then follow the validation loss. Training runs in two stages — forces first, energies second (`stage2 1`). More in [Training](../guide/training.md).

When the run finishes, `output/` contains:

- `nep_best.txt` — the best model, in GPUMD format;
- `loss.out` — the training curves;
- `energy_train.out`, `force_train.out`, … — predictions of the best model on the training set, and `*_test.out` on the validation set.

See [Output files](../reference/output-files.md) for every file.

## 4. Check and use the model

Plot the run (needs `pip install torchnep[plot]`):

```python
from torchnep.plot import NEPPlotter

NEPPlotter().dashboard("output", out="dashboard.png")
```

Run molecular dynamics in GPUMD with the model:

```text title="run.in"
potential nep_best.txt
```

or evaluate it from Python with ASE:

```python
from ase.io import read
from torchnep.ase_calculator import NEP

atoms = read("POSCAR")
atoms.calc = NEP("output/nep_best.txt", device="cuda")
print(atoms.get_potential_energy())
```

Predict a whole test set and get its error table with [`predict_dataset`](../guide/prediction.md#full-dataset-prediction).
