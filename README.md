# torchnep

A PyTorch implementation of [NEP4](https://gpumd.org/theory/nep.html) (Neuroevolution Potential) for training machine-learning interatomic potentials.  Models are fully compatible with [GPUMD](https://github.com/brucefan1983/GPUMD) and can be used as drop-in replacements for GPUMD-trained `nep.txt` files.

## Features

- **GPUMD-compatible** — output `nep.txt` files load directly into GPUMD for MD simulation
- **Two compute backends auto-picked by element count** — `"loop"` (Python type-pair loop, best for few types) and `"bmm"` (fancy-index + `torch.bmm`, best for ≥8 types). Both pure PyTorch; works on CPU / CUDA / MPS. Set `backend="auto"` and the trainer picks the right one.
- **Two-stage training** — Stage 1: force-focused with ReduceLROnPlateau; Stage 2: energy fine-tuning with Stochastic Weight Averaging
- **Multi-GPU training** — data-sharded DDP via `train_nep_sharded` + `torchrun`; each rank holds only `1/N` of the structures (single-node and multi-node SLURM launch snippets in the README)
- **Fine-tuning** — load any `nep.txt` or `nep_best.pt` as starting weights; optionally slim the model to only the element types present in the new dataset
- **Restart** — full training state (weights + optimizer + scheduler + epoch) saved to `checkpoint.pt`; resume with one flag
- **ZBL** — Universal ZBL repulsive potential with optional typewise cutoffs
- **Batched prediction** — full-dataset prediction using pre-cached GPU basis, typically 10–50× faster than per-frame evaluation

---

## Installation

```bash
pip install -e .
```

Requirements: `torch >= 2.0`, `numpy`.

Pure PyTorch — no native extensions to compile.

---

## Device backends

torchnep is written on top of stock PyTorch, so it runs on any device
backend that PyTorch itself supports. The four that are handled explicitly
(auto-detected in this order) are:

| Backend | Vendor / hardware | Status |
| --- | --- | --- |
| `cuda` | NVIDIA GPUs | tested, primary target (single GPU, DDP multi-GPU, multi-node) |
| `cuda` | AMD GPUs via PyTorch-ROCm | works unmodified — PyTorch-HIP exposes AMD GPUs through the `torch.cuda` namespace, so `dev.type == "cuda"` is already `True`. Not CI-tested by us |
| `xpu` | Intel GPUs | auto-detected if `torch.xpu.is_available()`. Not CI-tested by us |
| `mps` | Apple Silicon | tested |
| `cpu` | any | tested |

Any other PyTorch device that behaves like a standard stream-based
accelerator (exposes `.to(dev)` plus the usual tensor ops) should work if
you pass it explicitly, e.g. `train_nep(..., device="<name>")`. The only
device-specific code paths are `torch.cuda.synchronize()` /
`torch.cuda.empty_cache()` calls used for timing accuracy on CUDA/ROCm;
they are skipped on other backends, so correctness is unaffected but the
reported per-epoch seconds may include a small amount of async queueing
overhead.

No custom kernels: both compute backends (`"loop"` and `"bmm"`) are pure
PyTorch. `"bmm"` dispatches to whatever `torch.bmm` uses on your device
(cuBLAS on CUDA, rocBLAS on ROCm, oneDNN on XPU, MKL/Accelerate on CPU,
MPS on Apple).

---

## Quick Start

```python
from torchnep import train_nep

train_nep("nep.in", "train.xyz", output_dir="output")
```

That is all that is needed for a single-GPU run.  Output files appear in `output/`.

---

## Input XYZ format

torchnep reads extended-XYZ files. Every frame is two lines of header
(atom count + comment line with `key="value"` tags) followed by one line
per atom. The parser is strict — the following rules are enforced and
violations raise on load rather than silently producing wrong physics.

### Comment line tags

- `Lattice="ax ay az bx by bz cx cy cz"` — **mandatory**. Nine floats in
  Å giving the three lattice vectors as rows. Every frame is treated as
  fully periodic; if your data has isolated clusters / molecules, wrap
  them in a large vacuum box first (a standalone helper script is
  provided in [PeriodicTable/prepare_xyz.py](PeriodicTable/prepare_xyz.py)).
- `pbc=...` — ignored. Because `Lattice` is mandatory, frames are always
  treated as fully periodic. If you really need a non-periodic direction,
  use a vacuum box wider than the NEP cutoff in that direction.
- `energy=<value>` — optional, eV. The tag name is configurable via the
  `energy_key` argument of `train_nep` / `train_nep_sharded` /
  `predict_dataset` (default `"energy"`). For example, pass
  `energy_key="atomization_energy"` to train against atomization energies.
- `virial="vxx vxy vxz vyx vyy vyz vzx vzy vzz"` — optional, eV. Must
  have exactly 9 components. Positive values denote compressed states,
  negative denote stretched states (GPUMD convention).
- `stress="sxx sxy sxz syx syy syz szx szy szz"` — optional, eV/Å³.
  Must have exactly 9 components. Positive = stretched, negative =
  compressed — opposite sign to virial. Internally converted as
  `virial = -stress * det(Lattice)`. If both `virial` and `stress` are
  present, `virial` wins.

### Per-atom columns

The `Properties=...` schema declares column layout. torchnep reads only
three fields and silently ignores everything else (e.g. `Z:I:1`):

- `species:S:1` — chemical symbol (case-sensitive; must match the
  `type` list in `nep.in`).
- `pos:R:3` — Cartesian position in Å. Positions may lie outside the
  primary cell; the neighbor-list code wraps them back automatically
  (full PBC makes this exact).
- `force:R:3` or `forces:R:3` — reference force in eV/Å (optional).

Example frame header:

```text
Lattice="5.43 0 0 0 5.43 0 0 0 5.43" Properties=species:S:1:pos:R:3:forces:R:3 energy=-123.45 stress="0.001 0 0 0 0.001 0 0 0 0.001" pbc="T T T"
```

---

## Training

torchnep has **two entry points** with non-overlapping responsibilities:

| | `train_nep` | `train_nep_sharded` |
|---|---|---|
| Devices | 1 GPU / CPU / MPS | ≥ 1 GPU (multi-GPU only) |
| Launcher | `python run_train.py` | `torchrun … run_train.py` |
| Dataset per GPU | Full copy | `1/N` shard (linear scale-out) |
| Use it when | Dataset fits on one card | Dataset too large for one card, or you want the speedup |

Pick the one that matches your hardware. There is no `train_nep` multi-GPU mode any more — multi-GPU **must** go through `train_nep_sharded`.

### Single GPU / CPU / MPS — `train_nep`

```python
# run_train.py
from torchnep import train_nep

train_nep(
    config_file="nep.in",
    data_file="train.xyz",
    output_dir="output",
    device="cuda",          # "cuda" | "xpu" | "mps" | "cpu" — auto-detected if omitted
    backend="auto",         # "auto" | "loop" | "bmm" — see table below
    use_compile=True,       # torch.compile (~10% extra speedup)
)
```

```bash
python run_train.py
```

### Multi-GPU, single node — `train_nep_sharded`

Each rank loads only `1/N` of the structures, so total GPU memory for the data store scales as `1/N` instead of being replicated. The descriptor normalization (`q_scaler`) and energy shift (`b1`) are all-reduced across ranks, gradients are all-reduced via DDP.

```python
# run_train.py
from torchnep import train_nep_sharded
train_nep_sharded("nep.in", "train.xyz", output_dir="output")
```

```bash
torchrun --standalone --nproc_per_node=4 run_train.py    # 4 GPUs on this node
```

`--standalone` lets torchrun pick a free local port; no rendezvous configuration needed for one node. Plain `python run_train.py` will **not** work for the sharded entry point.

### Multi-GPU, multi-node (SLURM) — `train_nep_sharded` + sbatch

For 1 node × N GPUs, use the `torchrun --standalone --nproc_per_node=N run_train.py`
snippet above inside an `sbatch` script.  For M nodes × N GPUs each, the key
SLURM directives are:

```bash
#SBATCH --nodes=2                  # M nodes
#SBATCH --ntasks-per-node=1        # 1 srun task per node; torchrun fans out to N GPUs
#SBATCH --gpus-per-node=4          # N GPUs per node
#SBATCH --cpus-per-task=16         # CPU cores per node (preprocess workers share these)
```

and the launch line:

```bash
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

srun --nodes=$SLURM_NNODES --ntasks-per-node=1 bash -c "
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --node_rank=\$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    run_train.py
"
```

**Multi-node gotchas:**

- `--ntasks-per-node=1` (not `=N`): SLURM only launches torchrun once per node; torchrun handles the local fan-out. Mixing the two is a common deadlock source.
- `\$SLURM_NODEID` keeps the backslash so each node's child shell evaluates it (gives node rank 0, 1, …, M-1). Without it both nodes get the same rank → rendezvous hangs.
- `MASTER_ADDR` comes from `scontrol show hostnames` (expands SLURM's compressed nodelist like `gpu[01-02]`).
- If the cluster has multiple network interfaces, you may need `export NCCL_SOCKET_IFNAME=^lo,docker` to avoid loopback / docker bridges.
- No Infiniband? `export NCCL_IB_DISABLE=1` falls back to TCP.

**How sharding interacts with batch size and learning rate:**

- `batch` in `nep.in` is the **per-rank** batch size. With 4 GPUs and `batch 64`, the effective global batch is `4 × 64 = 256`.
- After scaling up the global batch, the learning rate often needs to scale too. Common rules of thumb:
  - **SGD-like optimizers** → linear: `lr_new = lr_single × world_size`
  - **Adam / AMSGrad** (what torchnep uses) → square-root or smaller: `lr_new ≈ lr_single × √world_size`, often even less.
- A safer recipe if unsure: keep `lr` the same as the single-GPU value, watch the first 5–10 epochs, and only bump it up if loss is descending steadily. Bumping it down if loss diverges or stalls.
- CPU cores per rank for preprocessing default to `cpu_count() / LOCAL_WORLD_SIZE`. On a 16-core node with 4 GPUs that's 4 workers per rank — already balanced. Override with `TORCHNEP_PREPROC_WORKERS=N` if needed.

### Verifying the multi-GPU launch worked

When you submit, the stdout banner should look like (rank 0 only prints):

```
Backend  : CUDA (DDP, 4 processes)        ← world_size matches your nproc × nnodes
Mode     : data-sharded DDP (4 ranks, each holds 1/4 of structures)
```

`nvidia-smi` on each node should show all GPUs busy. If after 2 minutes there is no progress past the banner, rendezvous is hanging — set `export NCCL_DEBUG=INFO` and resubmit to see where the handshake stalls.

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
| `batch` | `1000` | Structures per gradient step |
| `lr` | `0.01` | Initial learning rate |
| `stop_lr` | `1e-6` | Minimum learning rate (scheduler floor) |
| `lambda_e` | `1.0` | Energy loss weight |
| `lambda_f` | `100.0` | Force loss weight |
| `lambda_v` | `1.0` | Virial loss weight |
| `lambda_1` | `0.0` | L1 regularisation |
| `lambda_2` | `0.0` | L2 regularisation (weight decay) |
| `max_grad_norm` | `10.0` | Gradient clipping threshold |
| `scheduler_patience` | `50` | Epochs without improvement before LR reduction |
| `scheduler_factor` | `0.8` | LR reduction factor |

Loss is plain MSE. Hyperparameters are **read from `nep.in` only** — no function-argument override.

### Stage 2 (energy fine-tuning, optional)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stage2` | `0` | Enable Stage 2 (`1` = on) |
| `start_stage2` | 75 % of epochs | Epoch to switch to Stage 2 |
| `stage2_lr` | `1e-3` | Stage 2 learning rate |
| `stage2_lambda_e` | `1000.0` | Stage 2 energy weight |
| `stage2_lambda_f` | `100.0` | Stage 2 force weight |
| `stage2_lambda_v` | `10.0` | Stage 2 virial weight |

SWA (Stochastic Weight Averaging) is opt-in via the **function argument** `use_swa=True` (not a `nep.in` key, since it is an optional feature).

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
batch      64
lr         0.01
stop_lr    1e-6

stage2           1
start_stage2     750
stage2_lambda_e  1000.0
stage2_lambda_f  100.0
stage2_lambda_v  10.0
```

### Runtime arguments (function kwargs)

Everything that is not about hyperparameter *values* lives on the Python
function (`train_nep` / `train_nep_sharded`) — things the user flips at
launch time:

| Argument | Default | What it controls |
|---|---|---|
| `device` | auto | `"cuda"` / `"xpu"` / `"mps"` / `"cpu"` (see Device backends above) |
| `precision` | `"float32"` | dtype for training + store |
| `backend` | `"auto"` | `"loop"`, `"bmm"`, or `"auto"` (picks by num_types) |
| `use_autograd_forces` | `False` | autograd-through-rij (gold standard) vs analytical |
| `use_swa` | `False` | maintain SWA-averaged model and save `nep_average.*` |
| `use_compile` | `False` | wrap in `torch.compile` |
| `print_interval` | `10` | log to screen every N epochs (all epochs land in `output.log`) |
| `checkpoint_interval` | `100` | save `checkpoint.pt` every N epochs |
| `prediction_interval` | `20` | every N epochs run predict on the current `nep_best` and overwrite `{energy,force,virial}_predict.out` — live parity plot |
| `restart` | `True` | resume from `checkpoint.pt` if present |
| `finetune_from` | `None` | load weights from a `.pt` or `nep.txt` before training |
| `reset_lr` | `None` | override LR after resume/finetune |
| `slim_types` | `False` | drop element types absent from the dataset |
| `energy_key` | `"energy"` | comment-line tag read as reference energy (e.g. `"atomization_energy"`) |

---

## Output Files

| File | Contents |
|------|----------|
| `nep_best.txt`     | **Best-loss** model (GPUMD-compatible) — rewritten whenever `avg_loss < best_loss` |
| `nep_best.pt`      | Same weights as PyTorch state_dict |
| `nep_final.txt`    | Model at the **last** epoch (used for the end-of-training predict) |
| `nep_average.txt`  | **SWA-averaged** model — only when `use_swa=True` |
| `nep_average.pt`   | SWA weights as PyTorch state_dict |
| `checkpoint.pt`    | Full training state: weights + optimizer + scheduler + epoch |
| `output.log`       | Full training log |
| `loss.out`         | Per-epoch loss / RMSE (for plotting) |
| `energy_predict.out`, `force_predict.out`, `virial_predict.out` | Interim parity plot (every `prediction_interval` epochs, **nep_best weights**), replaced at end of training by final-epoch prediction |
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
state = torch.load("output/nep_best.pt", map_location="cpu")
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
    finetune_from="pretrained/nep.txt",   # or "pretrained/nep_best.pt"
    lr=1e-3,
    num_epochs=200,
)
```

`finetune_from` accepts:
- `nep.txt` — GPUMD text format (works with models trained by GPUMD or torchnep)
- `nep_best.pt` — PyTorch state dict
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

Two implementations of the same type-pair contraction
`q[i,n] = Σ_k c[t1,t2,n,k]·basis[p,k]` → scatter. Pick one with `backend=`:

| `backend=` | Implementation | Best for |
|---|---|---|
| `"loop"` | nested `for t1, t2` over type pairs, small matmul + scatter per pair | **few types** (≤ ~7) — outer loop runs few iterations |
| `"bmm"`  | fancy-index `c[t1, t2]` then `torch.bmm` (one batched GEMM) | **many types** (≥ 8) — one kernel launch beats the Python loop |
| `"auto"` (default) | picks `bmm` if `num_types ≥ 8` else `loop` | everything |

Both backends are pure PyTorch, fully autograd-differentiable, and work on CPU / CUDA / MPS (`torch.bmm` dispatches to cuBLAS / MKL / MPS respectively). They compute the same function — float64 output agrees to machine precision.

Measured one-epoch training wall-time on RTX A2000, float32 (single-GPU, see `probe/`):

| Dataset | num_types | loop | bmm | auto picks |
|---|---:|---:|---:|---|
| Si (2474f, BS=64) | 1 | **~2 s** | ~13 s | loop |
| AlO (2190f, BS=64) | 2 | **2.1 s** | 16.2 s | loop |
| CrCoNi (3030f, BS=64) | 3 | **2.2 s** | 11.1 s | loop |
| NEP89 (3000f, BS=100) | 53 | 26 s | **2.2 s** | bmm |

Orthogonal force toggle: `use_autograd_forces=True` switches to autograd-through-rij forces (slower, used only as a gold standard in tests — the analytical path matches it to float precision).

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

The banner `loss` column printed every epoch equals this same weighted-MSE
sum recomputed per-sample across the whole epoch (not a per-batch mean),
so it is self-consistent with the `RMSE_E / RMSE_F / RMSE_V` columns.

---

## Project Structure

```
torchnep/
  model.py        — NEPModel (nn.Module), FittingNet, slim_model
  train.py        — train_nep (single-GPU / CPU / MPS, plain python launcher)
  train_sharded.py — train_nep_sharded (multi-GPU only, torchrun launcher)
  nep.py          — NEPCalculator (inference from nep.txt)
  predict.py      — predict_dataset (batched full-dataset prediction)
  data.py         — read_xyz, parse_nep_in, build_neighbor_list_np
  ops.py          — basis functions, descriptors, analytical forces (pure PyTorch)
  constants.py    — physical constants, element data, C3B/C4B/C5B, Z_COEFFICIENT
example/
  run_train.py      — typical single-GPU entry point users edit
  run_fine_tune.py  — fine-tuning entry point (loads pretrained nep.txt)
  nep*.in           — example hyperparameter files
  train*.xyz        — example extxyz datasets
```
