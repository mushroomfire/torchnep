# torchnep

PyTorch implementation of the NEP4 (Neuroevolution Potential) for molecular dynamics.

## Features

- **Prediction**: Load trained `nep.txt` models and compute energy, forces, virial, descriptors
- **Training**: Train NEP4 models from `nep.in` + `train.xyz` with GPU acceleration
- **ZBL**: Universal ZBL potential with typewise cutoff support
- **Precision**: Configurable float32 (training) / float64 (prediction)
- **CUDA**: JIT-compiled CUDA kernels for force/virial accumulation; pure PyTorch fallback for CPU/Mac
- **Verified**: Machine-precision match with GPUMD/NEP_CPU reference (errors < 1e-14)

## Installation

```bash
pip install -e .
```

Requirements: `torch >= 2.0`, `numpy`. Optional: `ninja` (for CUDA JIT compilation).

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
    device="cuda",        # or "cpu"
    precision="float32",
    num_epochs=200,
    batch_size=64,
    lr=1e-2,
)
```

### Full-dataset prediction

```python
from torchnep import predict_dataset

predict_dataset("nep.txt", "test.xyz", output_dir="results")
# Writes: energy_predict.out, force_predict.out, virial_predict.out
```

## Input Parameters (nep.in)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `type` | required | `N_types name1 name2 ...` |
| `version` | 4 | NEP version (only 4 supported) |
| `zbl` | none | ZBL outer cutoff (enables ZBL potential) |
| `use_typewise_cutoff_zbl` | none | Factor for type-dependent ZBL cutoffs |
| `cutoff` | `6.0 6.0` | Radial and angular cutoffs (Angstrom) |
| `n_max` | `4 4` | Radial and angular expansion orders |
| `basis_size` | `12 12` | Number of Chebyshev basis functions |
| `l_max` | `4 2 0` | Max L for 3-body, 4-body, 5-body |
| `neuron` | 40 | Hidden layer size |
| `lambda_e` | 1.0 | Energy loss weight |
| `lambda_f` | 1.0 | Force loss weight |
| `lambda_v` | 0.1 | Virial loss weight |
| `lambda_1` | 0 | L1 regularization |
| `lambda_2` | 0 | L2 regularization (weight decay) |

## Project Structure

```
torchnep/
  constants.py   — physical constants, element data, C3B/C4B/C5B
  ops.py         — core operations (PyTorch + optional CUDA)
  nep.py         — NEPCalculator (prediction from nep.txt)
  model.py       — NEPModel (trainable nn.Module)
  train.py       — training pipeline with GPU data pre-loading
  data.py        — XYZ parser, nep.in parser
  predict.py     — full-dataset prediction
  cuda_ops.py    — two-level torch.autograd.Function CUDA wrappers
  csrc/          — CUDA kernels (JIT-compiled)
    nep_kernels.cu     — fused radial+angular descriptor forward+backward
    nep_ops.cu         — force/virial accumulation kernel
    nep_descriptor.cu  — fused radial descriptor + force kernel
```

## Performance

Tested on 3030 structures (Cr-Co-Ni, 2-256 atoms/structure), batch_size=64, single GPU:

| Operation | Time |
|-----------|------|
| Forward pass (descriptors + NN) | 14 ms/batch |
| Force autograd | 17 ms/batch |
| Full training step (PyTorch only) | ~285 ms/batch |
| Full training step (CUDA kernels) | ~163 ms/batch (1.75x faster) |
| Prediction accuracy | < 1e-14 vs NEP_CPU |

### CUDA kernel acceleration

The CUDA kernels implement a two-level `torch.autograd.Function` following MatPL's pattern:

- **Radial forward+backward**: fused Chebyshev basis + c_param contraction + scatter
- **Angular forward+backward**: fused spherical harmonics accumulation + GPUMD-style `accumulate_f12`
- **Second-order backward**: enables `create_graph=True` for force training
- Falls back to pure PyTorch on CPU/Mac automatically

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
- Autograd-based: `torch.autograd.grad(E, rij, create_graph=True)` for training
- Analytical: precomputed Chebyshev derivatives + angular derivatives (experimental)

### CUDA acceleration
- Force/virial accumulation: fused CUDA kernel (replaces 6 scatter_add ops)
- Radial descriptor: fused Chebyshev + contraction kernel
- Falls back to pure PyTorch on CPU/Mac automatically

## Roadmap

- [ ] Fused CUDA `torch.autograd.Function` for descriptor+force+backward (5-10x speedup potential)
- [ ] PyTorch DDP multi-GPU training
- [ ] Flexible ZBL (learnable screening parameters)
- [ ] Mixed precision training (torch.amp)
