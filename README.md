# torchnep

PyTorch implementation of the NEP4 (Neuroevolution Potential) for molecular dynamics.

## Features

- **Prediction**: Load trained `nep.txt` models and compute energy, forces, virial, descriptors
- **Training**: Train NEP4 models from `nep.in` + `train.xyz` with GPU acceleration
- **CUDA kernels**: JIT-compiled custom CUDA kernels for descriptor forward+backward (363x speedup over pure PyTorch loops)
- **ZBL**: Universal ZBL potential with typewise cutoff support
- **Multi-device**: Auto-detects CUDA > MPS (Apple Silicon) > CPU
- **Precision**: Configurable float32 (training) / float64 (prediction)
- **Verified**: Machine-precision match with GPUMD/NEP_CPU reference (errors < 1e-14)

## Installation

```bash
pip install -e .
```

Requirements: `torch >= 2.0`, `numpy`.

Optional: `ninja` (for faster CUDA JIT compilation):

```bash
pip install ninja
```

## Quick Start

### Prediction from trained model

```python
from torchnep import NEPCalculator

calc = NEPCalculator("nep.txt")
result = calc.compute(species, positions, cell)
# result["energy"]  — per-atom energy (N,)
# result["forces"]  — forces (N, 3)
# result["virial"]  — per-atom virial (N, 9)
```

### Training

```python
from torchnep import train_nep

train_nep(
    config_file="nep.in",
    data_file="train.xyz",
    output_dir="output",
    # device auto-detected: CUDA → MPS → CPU
    precision="float32",
    num_epochs=200,
    batch_size=64,
    lr=1e-2,
)
```

Output files:

- `nep.txt` — best model (GPUMD-compatible)
- `nep_final.txt` — final epoch model
- `loss.out` — training loss per epoch
- `energy_predict.out` — predicted vs reference energies
- `force_predict.out` — predicted vs reference forces
- `virial_predict.out` — predicted vs reference virials
- `checkpoint.pt` — training state for restart

### Full-dataset prediction

```python
from torchnep import predict_dataset

predict_dataset("nep.txt", "test.xyz", output_dir="results")
# Writes: energy_predict.out, force_predict.out, virial_predict.out
```

## Input Parameters (nep.in)

| Parameter                  | Default   | Description                              |
| -------------------------- | --------- | ---------------------------------------- |
| `type`                     | required  | `N_types name1 name2 ...`                |
| `version`                  | 4         | NEP version (only 4 supported)           |
| `zbl`                      | none      | ZBL outer cutoff (enables ZBL potential)  |
| `use_typewise_cutoff_zbl`  | none      | Factor for type-dependent ZBL cutoffs     |
| `cutoff`                   | `6.0 6.0` | Radial and angular cutoffs (Angstrom)    |
| `n_max`                    | `4 4`    | Radial and angular expansion orders       |
| `basis_size`               | `12 12`  | Number of Chebyshev basis functions       |
| `l_max`                    | `4 2 0`  | Max L for 3-body, 4-body, 5-body         |
| `neuron`                   | 40        | Hidden layer size                        |
| `lambda_e`                 | 1.0       | Energy loss weight                       |
| `lambda_f`                 | 1.0       | Force loss weight                        |
| `lambda_v`                 | 0.1       | Virial loss weight                       |
| `lambda_1`                 | 0         | L1 regularization                        |
| `lambda_2`                 | 0         | L2 regularization (weight decay)         |
| `batch`                    | 1000      | Batch size                               |

## Project Structure

```text
torchnep/
  data.py        — XYZ parser, nep.in parser
  ops.py         — core operations (neighbor list, basis functions, descriptors)
  model.py       — NEPModel (trainable nn.Module)
  train.py       — training pipeline with GPU data pre-loading
  nep.py         — NEPCalculator (prediction from nep.txt)
  predict.py     — full-dataset prediction with output files
  cuda_ops.py    — torch.autograd.Function CUDA wrappers
  constants.py   — physical constants, element data, C3B/C4B/C5B coefficients
  csrc/          — CUDA kernels (JIT-compiled at runtime)
    nep_cached.cu    — type-contraction kernels for training
    nep_kernels.cu   — fused radial+angular descriptor forward+backward
    nep_ops.cu       — force/virial accumulation kernel
    nep_descriptor.cu — fused radial descriptor + force kernel
```

## Performance

Training benchmark (1024 atoms, 66k radial pairs, 18k angular pairs):

| Backend                        | Time/step | Throughput       |
| ------------------------------ | --------- | ---------------- |
| Pure PyTorch (type-pair loops) | 9690 ms   | 3 structures/s   |
| CUDA kernels                   | 26.6 ms   | 1126 structures/s |
| **Speedup**                    | **363x**  |                  |

Prediction accuracy: < 1e-14 vs NEP_CPU reference.

## CUDA Kernel Architecture

The training path uses custom CUDA kernels for the type-pair contraction operations:

- **Scatter contraction** (radial): Forward computes `q[i,n] = sum_p sum_k c[t1,t2,n,k] * basis[p,k]` via atomicAdd, and pre-accumulates `dfeat_c[i,t2,k]` for backward. Backward uses per-atom outer products (N atoms instead of P pairs).
- **Type contraction** (angular): Forward computes `gn[p,n] = sum_k c[t1,t2,n,k] * basis[p,k]`. Backward accumulates `grad_c` via per-pair atomicAdd.
- Falls back to pure PyTorch on CPU/MPS automatically.

## Architecture

### Descriptor computation

- Radial (2-body): Chebyshev polynomials with cosine cutoff, contracted with learnable `c_param`
- Angular (3-body): Solid harmonics basis accumulated per atom, contracted to q via C3B coefficients
- 4-body: Cubic invariants of L=2 angular moments
- 5-body: Quartic invariants of L=1 angular moments

### Neural network

- Per-type single-hidden-layer network: `tanh(q @ W - b) @ w1 - b1`
- GPUMD convention: bias subtracted (not added)

### Force computation

- Analytical forces via precomputed Chebyshev derivatives + angular derivatives
- No `create_graph=True` needed — stable for high-order descriptors
