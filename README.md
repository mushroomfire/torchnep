# torchnep

A PyTorch implementation of [NEP4](https://gpumd.org/theory/nep.html) (Neuroevolution Potential) for training machine-learning interatomic potentials.  Models are fully compatible with [GPUMD](https://github.com/brucefan1983/GPUMD) and can be used as drop-in replacements for GPUMD-trained `nep.txt` files.

## Features

- **GPUMD-compatible** — output `nep.txt` files load directly into GPUMD for MD simulation
- **Three compute backends** — pure PyTorch (CPU/CUDA/MPS), analytical PyTorch forces, CUDA kernel accelerated (up to 363× faster than naive PyTorch)
- **Two-stage training** — Stage 1: force-focused with ReduceLROnPlateau; Stage 2: energy fine-tuning with Stochastic Weight Averaging
- **Multi-GPU training** — replicated DDP (`train_nep`) or data-sharded DDP (`train_nep_sharded`) via `torchrun`
- **Fine-tuning** — load any `nep.txt` or `best_model.pt` as starting weights; optionally slim the model to only the element types present in the new dataset
- **Restart** — full training state (weights + optimizer + scheduler + epoch) saved to `checkpoint.pt`; resume with one flag
- **ZBL** — Universal ZBL repulsive potential with optional typewise cutoffs
- **Batched prediction** — full-dataset prediction using pre-cached GPU basis, typically 10–50× faster than per-frame evaluation

---

## Installation

```bash
pip install -e .
```

Requirements: `torch >= 2.0`, `numpy`.

Optional — `ninja` speeds up CUDA kernel compilation:

```bash
pip install ninja
```

Pre-compile CUDA kernels so the first training run starts immediately:

```bash
torchnep-build
```

---

## Quick Start

```python
from torchnep import train_nep

train_nep("nep.in", "train.xyz", output_dir="output")
```

That is all that is needed for a single-GPU run.  Output files appear in `output/`.

---

## Training

### Single GPU / CPU

```python
from torchnep import train_nep

train_nep(
    config_file="nep.in",
    data_file="train.xyz",
    output_dir="output",
    device="cuda",          # "cuda" | "cpu" | "mps" — auto-detected if omitted
    pytorch_only=False,     # use CUDA kernel acceleration
    use_compile=True,       # torch.compile (~10% extra speedup)
)
```

### Multi-GPU — replicated data (standard DDP)

Every GPU holds the full dataset.  Use this when the dataset fits comfortably in one GPU's memory.

```python
# run_train.py
from torchnep import train_nep
train_nep("nep.in", "train.xyz", output_dir="output")
```

```bash
torchrun --nproc_per_node=4 run_train.py
```

Each GPU processes a different subset of batches each epoch; gradients are all-reduced by DDP automatically.

### Multi-GPU — sharded data (large datasets)

Each GPU loads only `1/N` of the training structures, so total GPU memory for the data store scales as `1/N` instead of being replicated.  Use this when the dataset is too large to fit on a single GPU.

```python
# run_train.py
from torchnep import train_nep_sharded
train_nep_sharded("nep.in", "train.xyz", output_dir="output")
```

```bash
torchrun --nproc_per_node=4 run_train.py
```

`train_nep_sharded` requires `torchrun` (single-process launch is not supported).  The descriptor normalization (`q_scaler`) and energy shift (`b1`) are all-reduced across shards so every rank starts from a globally consistent state.

### Choosing between replicated and sharded

| | `train_nep` | `train_nep_sharded` |
|---|---|---|
| Single GPU / CPU | Yes | No |
| Multi-GPU launch | `torchrun` (auto-detected) | `torchrun` (required) |
| Dataset per GPU | Full copy | `1/N` shard |
| When to use | Dataset fits on one GPU | Dataset too large for one GPU |

---

## Training Parameters

All parameters can be set in `nep.in` or passed as keyword arguments to `train_nep` / `train_nep_sharded`.  Keyword arguments take precedence over `nep.in`.

### Model architecture (GPUMD-compatible)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `type` | required | `N name1 name2 ...` — number and names of element types |
| `cutoff` | `6.0 6.0` | Radial and angular cutoff (Å) |
| `n_max` | `4 4` | Radial and angular expansion orders |
| `basis_size` | `12 12` | Chebyshev basis functions per channel |
| `l_max` | `4 2 0` | Max L for 3-body, 4-body, 5-body terms |
| `neuron` | `40` | Hidden layer width |
| `zbl` | — | ZBL outer cutoff (Å); enables short-range repulsion |
| `use_typewise_cutoff_zbl` | — | Scale ZBL cutoffs by covalent radii |

### Training hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epoch` | `200` | Total training epochs |
| `batch` | `32` | Batch size (structures per step) |
| `lr` | `0.01` | Initial learning rate |
| `stop_lr` | `1e-6` | Minimum learning rate (scheduler floor) |
| `lambda_e` | `1.0` | Energy loss weight |
| `lambda_f` | `100.0` | Force loss weight |
| `lambda_v` | `1.0` | Virial loss weight |
| `lambda_1` | `0` | L1 regularisation |
| `lambda_2` | `0` | L2 regularisation (weight decay) |
| `max_grad_norm` | `10.0` | Gradient clipping threshold |
| `huber_delta` | `0` | Huber loss delta (0 = MSE) |
| `scheduler_patience` | `50` | Epochs without improvement before LR reduction |
| `scheduler_factor` | `0.8` | LR reduction factor |

### Stage 2 (energy fine-tuning)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stage2` | `0` | Enable Stage 2 (`1` = on) |
| `start_stage2` | 75 % of epochs | Epoch to switch to Stage 2 |
| `stage2_lr` | `1e-3` | Stage 2 learning rate |
| `stage2_lambda_e` | `1000.0` | Stage 2 energy weight |
| `stage2_lambda_f` | `100.0` | Stage 2 force weight |
| `stage2_lambda_v` | `10.0` | Stage 2 virial weight |
| `use_swa` | `1` | Stochastic Weight Averaging in Stage 2 |

### Example `nep.in`

```
type 3 Cr Co Ni
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

---

## Output Files

| File | Contents |
|------|----------|
| `nep.txt` | Best-loss model (GPUMD-compatible) |
| `nep_final.txt` | Model at final epoch |
| `nep_swa.txt` | SWA-averaged model (Stage 2 + `use_swa=True`) |
| `best_model.pt` | Best-loss weights as PyTorch state dict |
| `checkpoint.pt` | Full training state: weights + optimizer + scheduler + epoch |
| `loss.out` | Per-epoch: loss, RMSE_E (meV/atom), RMSE_F (eV/Å), RMSE_V, gnorm |
| `output.log` | Full console log |
| `energy_predict.out` | Per-structure energy predictions vs targets |
| `force_predict.out` | Per-atom force predictions vs targets |
| `virial_predict.out` | Per-structure virial predictions vs targets |

---

## Restart and Resume

Set `restart=True` (the default).  torchnep looks for `checkpoint.pt` in `output_dir` and resumes from exactly where training stopped — epoch, learning rate, optimizer momentum, and scheduler state are all restored.

```python
train_nep("nep.in", "train.xyz", output_dir="output")  # restart=True by default
```

Works correctly regardless of which stage was active when training stopped.

### What you can safely change on restart

| Parameter | Safe to change? | Notes |
|-----------|----------------|-------|
| `num_epochs` | Yes | Extend training by increasing this |
| `pref_e/f/v` | Yes | New loss weights take effect next epoch |
| `batch_size` | Yes | — |
| `stage2`, `start_stage2` | Yes | Add Stage 2 to a run that did not have it |
| `lr` | **No** | Overridden by saved optimizer state |
| Architecture (neuron, cutoff, …) | **No** | Dimensions are fixed |

To force a new learning rate after resuming (e.g. LR has decayed to `stop_lr`):

```python
train_nep("nep.in", "train.xyz", output_dir="output", reset_lr=1e-3)
```

### Inspecting checkpoint files

```python
import torch

# Best model weights only
state = torch.load("output/best_model.pt", map_location="cpu")
print(list(state.keys()))
# ['c_param_2', 'c_param_3', 'fitting_nets.0.w0', ..., 'b1', 'q_scaler', ...]

# Full checkpoint
ckpt = torch.load("output/checkpoint.pt", map_location="cpu")
print(ckpt["epoch"], ckpt["best_loss"])
```

---

## Fine-Tuning

Fine-tuning starts from a pre-trained model's weights instead of random initialisation.  The model architecture (`nep.in` parameters) must match the source model.  The element types in the new dataset can be a **subset** of the original types.

### Basic fine-tuning

```python
train_nep(
    "nep.in",
    "new_data.xyz",
    output_dir="finetune_output",
    finetune_from="pretrained/nep.txt",   # or "pretrained/best_model.pt"
    lr=1e-3,
    num_epochs=200,
)
```

`finetune_from` accepts:
- `nep.txt` — GPUMD text format (works with models trained by GPUMD or torchnep)
- `best_model.pt` — PyTorch state dict
- `checkpoint.pt` — full checkpoint (weights are extracted automatically)

What happens internally:
- All trainable weights (`c_param`, fitting nets, `b1`) are loaded from the source model
- `q_scaler` is **recomputed** from the new dataset (descriptor statistics change with new data)
- A fresh optimizer is created — no momentum carryover from the original training
- Any existing `checkpoint.pt` in `output_dir` is ignored when `finetune_from` is set

### Fine-tuning with model slimming

If the new dataset contains fewer element types than the original model, setting `slim_types=True` removes the unused types **before training begins**.  This reduces the model size and makes training faster, because smaller `c_param` matrices and fewer fitting networks mean less computation per batch.

```python
# Original model: Cr Co Ni (3 types)
# New data: only Cr and Ni structures

train_nep(
    "nep.in",            # still lists all 3 types (must match source arch)
    "new_data.xyz",
    output_dir="finetune_output",
    finetune_from="pretrained/nep.txt",
    slim_types=True,     # detect types from data, slim before training
    lr=1e-3,
)
# output nep.txt will contain only [Cr, Ni]
```

`slim_types=True` can also be used without `finetune_from` (training from scratch on a subset of the types listed in `nep.in`).

### Standalone model slimming (no retraining)

```python
from torchnep.model import NEPModel, slim_model
from torchnep.data import parse_nep_in

config = parse_nep_in("nep.in")
model = NEPModel(config)
model.load_weights_from_nep_txt("nep.txt")

slimmed = slim_model(model, ["Cr", "Ni"])
slimmed.save_nep_txt("nep_slim.txt")
```

---

## Prediction

### Single-structure prediction

```python
from torchnep import NEPCalculator
import numpy as np

calc = NEPCalculator("nep.txt")
result = calc.compute(
    species=["Cr", "Cr", "Ni"],
    positions=np.array([[0,0,0],[1.5,0,0],[3,0,0]]),
    cell=np.eye(3) * 6.0,
)
print(result["energy"])     # total energy (eV)
print(result["forces"])     # (N, 3) forces (eV/Å)
print(result["virial"])     # (N, 9) per-atom virial (eV)
```

### Full-dataset prediction

Runs batched GPU inference on an entire `.xyz` file and writes GPUMD-compatible output files.

```python
from torchnep import predict_dataset

predict_dataset(
    "nep.txt",
    "test.xyz",
    output_dir="results",
    dtype="float64",   # float32 or float64
    batch_size=64,
)
# writes: energy_predict.out, force_predict.out, virial_predict.out
```

---

## Compute Backends

| Backend | How to enable | Platforms | Speed |
|---------|--------------|-----------|-------|
| Autograd PyTorch | `use_autograd_forces=True` | CPU / CUDA / MPS | Slowest |
| Analytical PyTorch | default (`pytorch_only=True`) | CPU / CUDA / MPS | Fast |
| CUDA kernels | `pytorch_only=False` | CUDA only | Fastest |

The CUDA kernel backend uses three hand-written `.cu` files for fused descriptor computation, type contraction, and force/virial accumulation.  It falls back to analytical PyTorch automatically on non-CUDA devices.

In practice the CUDA kernel backend is **10–20% faster** than analytical PyTorch on typical training workloads — the bottleneck is the MLP forward/backward pass, not the descriptor computation.  The main benefit of `pytorch_only=False` is reduced descriptor memory traffic, not raw throughput.

---

## Architecture

### Descriptor

- **Radial (2-body):** Chebyshev polynomials with cosine cutoff, contracted with learnable `c_param_2 (nt, nt, n_max+1, basis_size+1)`
- **Angular (3-body):** Solid harmonic basis accumulated per atom, contracted via C3B coefficients; learnable `c_param_3`
- **4-body:** Cubic invariants of L=2 angular moments
- **5-body:** Quartic invariants of L=1 angular moments

Descriptor dimension = `(n_max_radial+1) + (n_max_angular+1)*l_max_3b + ...` — independent of the number of element types.

### Neural network

One per-type single-hidden-layer network: `tanh(q @ W₀ − b₀) @ w₁ − b₁`

GPUMD convention: bias is subtracted, not added; `b₁` is a single shared scalar across all types.

### Training loss

```
L = λ_e · MSE(E_pred/N, E_ref/N)
  + λ_f · MSE(F_pred, F_ref)
  + λ_v · MSE(V_pred/N, V_ref/N)
```

Huber loss can replace MSE via `huber_delta > 0`.

---

## Project Structure

```
torchnep/
  model.py        — NEPModel (nn.Module), FittingNet, slim_model
  train.py        — train_nep (single-GPU or torchrun DDP)
  train_sharded.py — train_nep_sharded (torchrun, data-sharded DDP)
  nep.py          — NEPCalculator (inference from nep.txt)
  predict.py      — predict_dataset (batched full-dataset prediction)
  data.py         — read_xyz, parse_nep_in, build_neighbor_list_np
  ops.py          — basis functions, descriptors, analytical forces (PyTorch)
  cuda_ops.py     — CUDA kernel wrappers (torch.autograd.Function)
  build_ext.py    — torchnep-build CLI for pre-compiling CUDA kernels
  constants.py    — physical constants, element data, C3B/C4B/C5B
  csrc/
    nep_kernels.cu      — fused radial+angular descriptor forward+backward
    nep_cached.cu       — type/scatter contraction kernels
    nep_ops.cu          — force/virial accumulation
```
