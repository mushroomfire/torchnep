# torchnep

PyTorch implementation of the NEP4 (Neuroevolution Potential) for molecular dynamics.

## Features

- **Training**: Train NEP4 models from `nep.in` + `train.xyz` with MACE-inspired training strategy
- **Prediction**: Load trained `nep.txt` models and compute energy, forces, virial, descriptors
- **Three compute modes**:
  - Autograd forces (cross-platform, gold-standard reference)
  - Analytical forces in pure PyTorch (cross-platform, fast)
  - CUDA kernel accelerated (NVIDIA GPU only, fastest)
- **MACE-style training**: ReduceLROnPlateau + Stage 2 energy-focused fine-tuning with SWA
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

### Training

All training parameters are configured in `nep.in`:

```python
from torchnep import train_nep

train_nep(
    config_file="nep.in",
    data_file="train.xyz",
    output_dir="output",
    device="cuda",
    pytorch_only=False,  # CUDA kernel acceleration
)
```

Output files:

- `nep.txt` -- best model (GPUMD-compatible)
- `nep_swa.txt` -- SWA averaged model (if Stage 2 enabled)
- `nep_final.txt` -- final epoch model
- `loss.out` -- training metrics per epoch
- `energy_predict.out`, `force_predict.out`, `virial_predict.out` -- predictions
- `checkpoint.pt` -- training state for restart

### Prediction from trained model

```python
from torchnep import NEPCalculator

calc = NEPCalculator("nep.txt")
result = calc.compute(species, positions, cell)
# result["energy"]  -- per-atom energy (N,)
# result["forces"]  -- forces (N, 3)
# result["virial"]  -- per-atom virial (N, 9)
```

### Full-dataset prediction

```python
from torchnep import predict_dataset

predict_dataset("nep.txt", "test.xyz", output_dir="results")
```

## Compute Modes

|   | Autograd | Analytical PyTorch | CUDA Kernel |
| --- | --- | --- | --- |
| **Settings** | `autograd=True, pytorch_only=True` | `autograd=False, pytorch_only=True` | `autograd=False, pytorch_only=False` |
| **Force method** | `torch.autograd.grad` | Explicit chain rule (PyTorch ops) | Explicit chain rule (CUDA kernels) |
| **CUDA code** | None | None | 4 `.cu` files (~1500 lines) |
| **Cross-platform** | CPU / CUDA / MPS | CPU / CUDA / MPS | CUDA only (auto-fallback) |
| **Speed** | Slowest | Fast | Fastest |

## Input Parameters (nep.in)

### Model parameters (GPUMD-compatible)

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

### Training parameters (torchnep extensions)

| Parameter            | Default  | Description                                      |
| -------------------- | -------- | ------------------------------------------------ |
| `lambda_e`           | 1.0      | Energy loss weight (Stage 1)                     |
| `lambda_f`           | 100.0    | Force loss weight (Stage 1)                      |
| `lambda_v`           | 1.0      | Virial loss weight (Stage 1)                     |
| `lambda_1`           | 0        | L1 regularization                                |
| `lambda_2`           | 0        | L2 regularization (weight decay)                 |
| `epoch`              | 200      | Number of training epochs                        |
| `batch`              | 32       | Batch size                                       |
| `lr`                 | 0.01     | Initial learning rate                            |
| `stop_lr`            | 1e-6     | Minimum learning rate                            |
| `scheduler_patience` | 50       | Epochs w/o improvement before LR reduction       |
| `scheduler_factor`   | 0.8      | LR reduction factor on plateau                   |
| `max_grad_norm`      | 10.0     | Gradient clipping threshold                      |
| `huber_delta`        | 0        | Huber loss delta (0 = use MSE)                   |
| `stage2`             | 0        | Enable Stage 2 fine-tuning (0/1)                 |
| `start_stage2`       | 75%      | Epoch to begin Stage 2                           |
| `stage2_lr`          | 1e-3     | Stage 2 learning rate                            |
| `stage2_lambda_e`    | 1000.0   | Stage 2 energy weight                            |
| `stage2_lambda_f`    | 100.0    | Stage 2 force weight                             |
| `stage2_lambda_v`    | 10.0     | Stage 2 virial weight                            |
| `use_swa`            | 1        | Stochastic Weight Averaging in Stage 2 (0/1)     |

### Example nep.in

```
type 3 Cr Co Ni
version    4
zbl        2.5
cutoff     6 4
n_max      8 8
basis_size 12 12
l_max      4 2 1
neuron     80

lambda_e   1.0
lambda_f   100.0
lambda_v   1.0

epoch      1000
batch      32
lr         0.01
stop_lr    1e-6

stage2           1
start_stage2     750
stage2_lambda_e  1000.0
stage2_lambda_f  100.0
stage2_lambda_v  10.0
```

## Output Files

| File | Contents |
|------|---------|
| `nep.txt` | Best-loss model weights (GPUMD-compatible text format) |
| `nep_final.txt` | Weights at final epoch |
| `nep_swa.txt` | SWA-averaged model (only when Stage 2 + `use_swa=True`) |
| `best_model.pt` | Best-loss model as PyTorch state dict |
| `checkpoint.pt` | Full training state for restart (weights + optimizer + scheduler + epoch) |
| `loss.out` | Per-epoch metrics: loss, RMSE_E, RMSE_F, RMSE_V, gnorm |
| `output.log` | Console log saved to file |
| `energy_predict.out` | Per-structure energy predictions (written at end of training) |
| `force_predict.out` | Per-atom force predictions |
| `virial_predict.out` | Per-structure virial predictions |

### Inspecting checkpoints

```python
import torch

# Best model weights only
state = torch.load("output/best_model.pt", map_location="cpu")
print(list(state.keys()))
# ['c_param_2', 'c_param_3', 'fitting_nets.0.w0', 'fitting_nets.0.b0', ...]

# Full checkpoint (weights + optimizer + epoch)
ckpt = torch.load("output/checkpoint.pt", map_location="cpu")
print(ckpt["epoch"], ckpt["best_loss"])
print(list(ckpt["model_state"].keys()))
```

## Restart and Fine-tuning

### Restarting a stopped run

Set `restart=True` (the default).  torchnep looks for `checkpoint.pt` in
`output_dir` and resumes from exactly where training stopped — epoch number,
learning rate, optimizer momentum, and scheduler state are all restored.

```python
train_nep("nep.in", "train.xyz", output_dir="output")  # restart=True by default
```

Works correctly regardless of which stage was active when training stopped.

**What you can change on restart:**

| Parameter | Effect |
|-----------|--------|
| `pref_e/f/v`, `lambda_1/2` | Takes effect immediately — loss weights change next epoch |
| `num_epochs` | Extend training: set higher than original |
| `batch_size` | Safe to change |
| `stage2`, `start_stage2` | Can add Stage 2 to a run that didn't have it originally |
| `lr` | **Ignored** — overridden by checkpoint's optimizer state |
| Architecture params | **Cannot change** — model dimensions are fixed |

To **force a new learning rate** when restarting (e.g., after LR has decayed to near `stop_lr`):

```python
train_nep("nep.in", "train.xyz", output_dir="output", reset_lr=1e-3)
```

### Fine-tuning from a pre-trained model

Load weights from a previous `nep.txt` or `best_model.pt` and train on new data.
The model architecture (element types, cutoffs, `neuron`, `n_max`, etc.) must be
identical — use the same `nep.in`.  Element types in the new dataset can be a
subset of the original types.

```python
# From nep.txt (GPUMD format)
train_nep(
    "nep.in", "new_data.xyz",
    output_dir="finetune_output",
    finetune_from="pretrained/nep.txt",
    lr=1e-3,          # lower LR for fine-tuning
    num_epochs=200,
)

# From best_model.pt (PyTorch format)
train_nep(
    "nep.in", "new_data.xyz",
    output_dir="finetune_output",
    finetune_from="pretrained/best_model.pt",
    lr=1e-3,
)
```

**What happens during fine-tuning:**
- All trainable weights (`c_param`, fitting net weights, `b1`) are loaded from the pre-trained model
- `q_scaler` (descriptor normalization) is **recomputed** from the new dataset — this is important because descriptor statistics change with new data
- A fresh optimizer is created (no momentum carryover from the original training)
- `checkpoint.pt` from a previous run in `output_dir` is **ignored** when `finetune_from` is set

### nep.txt format

`nep.txt` is a plain-text file readable by GPUMD.  The structure is:

```
nep4 3 Cr Co Ni          # architecture header
cutoff 6 4               # hyperparameter lines (several)
...
ANN 80 0
  -1.2345678901e-01      # from here: one float per line
  ...                    # order: per-type w0/b0/w1, b1, c_param_2, c_param_3, q_scaler
```

You can load weights from `nep.txt` directly into a model for custom use:

```python
from torchnep.model import NEPModel
from torchnep.data import parse_nep_in

config = parse_nep_in("nep.in")
model = NEPModel(config)
model.load_weights_from_nep_txt("nep.txt")
```

## Training Strategy

Training follows a MACE-inspired two-stage approach:

**Stage 1** -- Fixed loss weights + ReduceLROnPlateau scheduler.
- Forces dominate the loss (weight 100x energy) to learn atomic environments first.
- LR auto-reduces when loss plateaus (patience=50, factor=0.8).

**Stage 2** (optional) -- Energy-focused fine-tuning with SWA.
- Energy weight increases 1000x to push energy error down.
- Lower fixed LR (1e-3) for stable convergence.
- Stochastic Weight Averaging smooths the final model.
- Outputs both best-loss model (`nep.txt`) and SWA model (`nep_swa.txt`).

All function parameters can also be passed directly to `train_nep()` to override `nep.in`:

```python
train_nep("nep.in", "train.xyz", lr=0.005, pref_f=50.0)  # override specific params
```

## Project Structure

```text
torchnep/
  data.py        -- XYZ parser, nep.in parser
  ops.py         -- core operations (basis functions, descriptors, analytical forces)
  model.py       -- NEPModel (trainable nn.Module)
  train.py       -- training pipeline with GPU data pre-loading
  nep.py         -- NEPCalculator (prediction from nep.txt)
  predict.py     -- full-dataset prediction with output files
  cuda_ops.py    -- torch.autograd.Function CUDA wrappers
  constants.py   -- physical constants, element data, C3B/C4B/C5B coefficients
  csrc/          -- CUDA kernels (JIT-compiled at runtime)
    nep_cached.cu    -- type-contraction kernels for training
    nep_kernels.cu   -- fused radial+angular descriptor forward+backward
    nep_ops.cu       -- force/virial accumulation kernel
    nep_descriptor.cu -- fused radial descriptor + force kernel
```

## Performance

Training benchmark (1024 atoms, 66k radial pairs, 18k angular pairs):

| Backend                        | Time/step | Throughput       |
| ------------------------------ | --------- | ---------------- |
| Pure PyTorch (type-pair loops) | 9690 ms   | 3 structures/s   |
| CUDA kernels                   | 26.6 ms   | 1126 structures/s |
| **Speedup**                    | **363x**  |                  |

Prediction accuracy: < 1e-14 vs NEP_CPU reference.

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
- No `create_graph=True` needed -- stable for high-order descriptors
