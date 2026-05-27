# torchnep

A PyTorch implementation of [NEP4](https://gpumd.org/theory/nep.html) (Neuroevolution Potential) for training machine-learning interatomic potentials.  Models are fully compatible with [GPUMD](https://github.com/brucefan1983/GPUMD) and can be used as drop-in replacements for GPUMD-trained `nep.txt` files.

## Features

- **GPUMD-compatible** — output `nep.txt` files load directly into GPUMD for MD simulation
- **Two-stage training** — Stage 1: force-focused; Stage 2: energy fine-tuning
- **Two LR scheduler modes** — `plateau` (ReduceLROnPlateau — default, drops LR after N epochs with no improvement) or `step` (StepLR — drops LR every N epochs at a fixed rate). Stage 1 and Stage 2 share the mode
- **Multi-GPU training** — data-sharded DDP via `train_nep_sharded` + `torchrun`
- **Fine-tuning** — load any `nep.txt` or `nep_best.pt` as starting weights; optionally slim the model to only the element types present in the new dataset
- **Restart** — full training state (weights + optimizer + scheduler + epoch) saved to `checkpoint.pt`
- **ZBL** — Universal ZBL repulsive potential with optional typewise cutoffs

---

## Installation

```bash
pip install -e .
```

Requirements: `torch >= 2.0`, `numpy`.

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
you pass it explicitly, e.g. `train_nep(..., device="<name>")`.

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

---

## Training

torchnep has **two entry points** with non-overlapping responsibilities:

| | `train_nep` | `train_nep_sharded` |
|---|---|---|
| Devices | 1 GPU / CPU / MPS | ≥ 1 GPU (multi-GPU only) |
| Launcher | `python run_train.py` | `torchrun … run_train.py` |
| Dataset per GPU | Full copy | `1/N` shard (linear scale-out) |
| Use it when | Dataset fits on one card | Dataset too large for one card, or you want the speedup |

### Single GPU / CPU / MPS — `train_nep`

```python
# run_train.py
from torchnep import train_nep

train_nep(
    config_file="nep.in",
    data_file="train.xyz",
    output_dir="output",
)
```

```bash
python run_train.py
```

### Multi-GPU, single node — `train_nep_sharded`

Each rank loads only `1/N` of the structures, so total GPU memory for the data store scales as `1/N`.

```python
# run_train.py
from torchnep import train_nep_sharded
train_nep_sharded("nep.in", "train.xyz", output_dir="output")
```

```bash
torchrun --standalone --nproc_per_node=4 run_train.py    # 4 GPUs on this node
```

### Multi-GPU, multi-node (SLURM) — `train_nep_sharded` + sbatch

For 1 node × N GPUs, use the `torchrun --standalone --nproc_per_node=N run_train.py`
snippet above inside an `sbatch` script.  For M nodes × N GPUs each, the key
SLURM directives are:

```bash
#SBATCH --nodes=2                  # M nodes
#SBATCH --ntasks-per-node=1        # 1 srun task per node; torchrun fans out to all GPUs
#SBATCH --gpus-per-node=4          # N GPUs per node
#SBATCH --cpus-per-task=16         # CPU cores per node (XYZ preprocessing pool shares these)
```

and the launch line:

```bash
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

srun --nodes=$SLURM_NNODES --ntasks-per-node=1 bash -c "
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=\$SLURM_GPUS_ON_NODE \
    --node_rank=\$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    run_train.py
"
```

`\$SLURM_GPUS_ON_NODE` (set by `--gpus-per-node`) keeps `nproc_per_node` in
sync with the allocation without hard-coding the GPU count. Backslash-escape
the SLURM-task-local vars (`\$SLURM_NODEID`, `\$SLURM_GPUS_ON_NODE`) so they
expand inside each `srun` task; bare `$` would substitute at sbatch parse time.

On some clusters the NCCL transport needs help finding the right NIC
(e.g. `NCCL_SOCKET_IFNAME=ib0`) or fabric (`NCCL_IB_HCA=...`). torchnep
itself doesn't override these; set them in your sbatch script if multi-node
all-reduce hangs.

---

## Training Parameters

### Model architecture (GPUMD-compatible)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `type` | required | `N name1 name2 ...` — number and names of element types |
| `cutoff` | `8.0 4.0` | Radial and angular cutoff (Å) |
| `n_max` | `6 6` | Radial and angular expansion orders |
| `basis_size` | `6 6` | Chebyshev basis size per channel (radial / angular). Max 16 |
| `l_max` | `4 1 0 0 0` | `L_3b q_222 q_1111 q_112 q_1122 q_123 q_233` — max L of 3-body terms (1–8) plus up to six boolean flags (matching GPUMD PR #1517) enabling each higher-body invariant. `q_123`/`q_233` (4-body bispectrum, fields 6–7) need `L_3b ≥ 3`. `q_1111` is redundant (= const × 3-body L=1 squared) — kept for compatibility but warns if set. Legacy 3-field form `L l_max_4b l_max_5b` still accepted |
| `neuron` | `30` | Hidden layer width |
| `zbl` | — | ZBL outer cutoff (Å); enables short-range repulsion |
| `use_typewise_cutoff_zbl` | — | Scale ZBL cutoffs by covalent radii |

### Training hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epoch` | `300` | Total training epochs |
| `batch` | `32` | Structures per gradient step |
| `lr` | `0.01` | Initial learning rate |
| `stop_lr` | `1e-6` | Minimum learning rate (scheduler floor) |
| `lambda_e` | `1.0` | Energy loss weight |
| `lambda_f` | `100.0` | Force loss weight |
| `lambda_v` | `1.0` | Virial loss weight |
| `lambda_1` | `0.0` | L1 regularisation |
| `lambda_2` | `0.0` | L2 regularisation (weight decay) |
| `max_grad_norm` | `10.0` | Gradient clipping threshold |
| `lr_scheduler` | `plateau` | LR schedule — `plateau` (ReduceLROnPlateau) or `step` (StepLR). Stage 1 and Stage 2 share this mode |
| `scheduler_patience` | `15` | For `plateau`: epochs without improvement before LR reduction. For `step`: epoch interval between LR reductions |
| `scheduler_factor` | `0.7` | LR reduction factor — multiplied on each decay in both modes |

### Stage 2 (energy fine-tuning, optional)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stage2` | `0` | Enable Stage 2 (`1` = on) |
| `start_stage2` | 75 % of epochs | Epoch to switch to Stage 2 |
| `stage2_lr` | `1e-3` | Stage 2 learning rate |
| `stage2_scheduler_patience` | `scheduler_patience` | Stage 2 scheduler patience (overrides Stage 1's; same semantics — for `step` it is the epoch interval, for `plateau` it is the epochs-without-improvement window) |
| `stage2_scheduler_factor` | `scheduler_factor` | Stage 2 LR decay factor (overrides Stage 1's). Lets the two stages span different LR ranges — e.g. stage 1 `1e-2 → 1e-3` (factor `0.794`, 10 decays over 200 epochs) and stage 2 `1e-3 → 1e-5` (factor `0.631`). When unset, Stage 2 reuses the Stage 1 values. |
| `stage2_lambda_e` | `1.0` | Stage 2 energy weight |
| `stage2_lambda_f` | `100.0` | Stage 2 force weight |
| `stage2_lambda_v` | `1.0` | Stage 2 virial weight |

SWA (Stochastic Weight Averaging) is opt-in via the **function argument** `use_swa=True` (not a `nep.in` key, since it is an optional feature).

### Example `nep.in`

```
type 3 Cr Co Ni
stage2 1
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
| `prediction_interval` | `20` | every N epochs run predict on the current `nep_best` and overwrite `{energy,force,virial}_train.out` — live parity plot |
| `restart` | `True` | resume from `checkpoint.pt` if present |
| `finetune_from` | `None` | load weights from a `.pt` or `nep.txt` before training |
| `reset_lr` | `None` | override LR after resume/finetune |
| `slim_types` | `False` | drop element types absent from the dataset |
| `energy_key` | `"energy"` | comment-line tag read as reference energy (e.g. `"atomization_energy"`) |

---

## Output Files

| File | Contents |
|------|----------|
| `nep_best.txt` / `nep_best.pt` | **Best-loss** model (GPUMD-compatible / PyTorch state_dict). Rewritten whenever `avg_loss < best_loss` |
| `nep_stage1.txt` / `nep_stage1.pt` | **End-of-Stage-1** snapshot (only when `stage2=1`) — written the instant Stage 2 kicks in; lets you restart with different Stage-2 weights |
| `nep_final.txt`    | Model at the **last** epoch (used for the end-of-training predict) |
| `nep_average.txt` / `nep_average.pt` | **SWA-averaged** model — only when `use_swa=True` |
| `checkpoint.pt`    | Full training state: weights + optimizer + scheduler + epoch + loss weights |
| `output.log`       | Full console log |
| `loss.out`         | Per-epoch: epoch, loss, RMSE_E (eV/atom), RMSE_F (eV/Å), RMSE_V, RMSE_stress (GPa), gnorm |
| `energy_train.out` | Per-frame predicted vs reference E/atom (eV/atom) |
| `force_train.out` | Per-atom predicted vs reference Fx Fy Fz (eV/Å) |
| `virial_train.out` | Per-frame predicted vs reference virial xx yy zz xy yz zx (eV/atom) |
| `stress_train.out` | Per-frame predicted vs reference stress (GPa); negated wrt `virial_train.out` so input and output stress carry the same sign |
| `descriptor.out` | Scaled descriptor `q * q_scaler` — only when `output_descriptor` is passed to `predict_dataset` |

The `*_train.out` files are rewritten every `prediction_interval` epochs using **`nep_best` weights** as a live parity plot, then replaced at end of training by a final-epoch prediction. Format matches GPUMD's `*_train.out` so the two can be diffed column by column.

---

## Restart and Resume

Set `restart=True` (the default).  torchnep looks for `checkpoint.pt` in `output_dir` and resumes from exactly where training stopped — epoch, learning rate, optimizer momentum, and scheduler state are all restored.

```python
train_nep("nep.in", "train.xyz", output_dir="output")  # restart=True by default
```

Works correctly regardless of which stage was active when training stopped.

### What you can safely change on restart

`nep.in` is re-parsed every run, so editing it between runs just works for
value-only changes. Structural changes (architecture, shapes) are not safe.

| Parameter | Safe to change? | Notes |
|-----------|----------------|-------|
| `epoch` | Yes | Extend training by increasing this |
| `lambda_e` / `lambda_f` / `lambda_v` | Yes | New weights take effect next epoch. Changing any of them triggers an auto-reset of `best_loss` so the new scale can establish a new best (instead of staying gated by the old-scale number stored in `checkpoint.pt`). |
| `stage2_lambda_e` / `stage2_lambda_f` / `stage2_lambda_v` | Yes | Same auto-reset rule. Especially useful: restart from `nep_stage1.pt` with different Stage-2 weights by copying it to `nep_best.pt` (or passing it via `finetune_from`) and editing `nep.in`. |
| `batch` | Yes | — |
| `stage2`, `start_stage2` | Yes | Add Stage 2 to a run that did not have it, or push it later |
| `stage2_lr` | Only at the transition | Applied **once**, when training first crosses Stage 1 → Stage 2. If you resume from a checkpoint that was *already* in Stage 2, the checkpoint's current (possibly-decayed) LR is kept — editing `stage2_lr` then has no effect. Use `reset_lr` to force a new LR. |
| `lr_scheduler` (`plateau` ↔ `step`) | Yes | Scheduler state from the old mode is incompatible and silently discarded; the new scheduler starts fresh from the current LR |
| `scheduler_patience` / `scheduler_factor` | Yes | Applied immediately |
| `stage2_scheduler_patience` / `stage2_scheduler_factor` | Yes | Applied immediately to the Stage 2 scheduler |
| `lr` (Stage 1) | **No** directly | Overridden by saved optimizer state — pass `reset_lr=<new>` to override |
| Architecture (`neuron`, `cutoff`, `n_max`, `basis_size`, `l_max`, `type`) | **No** | Dimensions are fixed in the saved weights |

**LR on resume.** A restart always keeps the LR stored in `checkpoint.pt`
(optimizer + scheduler state are restored), so a run interrupted mid-decay
picks up exactly where it left off — in either stage. The only exceptions:
the one-time `stage2_lr` applied at the natural Stage 1 → Stage 2 crossing,
and an explicit `reset_lr=<value>` override (a float, not a flag). Use the
latter when you've edited `nep.in`'s LR and want it to take effect, e.g. when
the LR has decayed to `stop_lr`:

```python
train_nep("nep.in", "train.xyz", output_dir="output", reset_lr=1e-3)
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
    slim_types=True,
)
```

`finetune_from` accepts:
- `nep.txt` — GPUMD text format (works with models trained by GPUMD or torchnep)
- `nep_best.pt` — PyTorch state dict
- `checkpoint.pt` — full checkpoint (weights are extracted automatically)

If the new dataset contains fewer element types than the original model, setting `slim_types=True` removes the unused types **before training begins**.  This reduces the model size and makes training faster,

What happens internally:
- All trainable weights (`c_param`, fitting nets, `b1`) are loaded from the source model
- `q_scaler` is **recomputed** from the new dataset (descriptor statistics change with new data)
- A fresh optimizer is created — no momentum carryover from the original training
- Any existing `checkpoint.pt` in `output_dir` is ignored when `finetune_from` is set

### Standalone model slimming (no retraining)

```python
from torchnep.model import NEPModel, slim_model
from torchnep.data import parse_nep_in

config = parse_nep_in("nep.in")
model = NEPModel(config)
model.load_weights_from_nep_txt("nep.txt")

slimmed = slim_model(model, ["Cr", "Ni"])
# max_NN_radial / max_NN_angular are required (GPUMD's nep.txt format
# mandates both on the cutoff line). Read them from the source nep.txt's
# cutoff line — slimming element types doesn't change neighbor counts.
slimmed.save_nep_txt("nep_slim.txt", max_NN_radial=127, max_NN_angular=42)
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
print(result["energy"])         # (N,) per-atom energy (eV); sum for total
print(result["forces"])         # (N, 3) forces (eV/Å)
print(result["virial"])         # (N, 9) per-atom virial (eV)
```

### Full-dataset prediction

Runs batched GPU inference on an entire `.xyz` file and writes GPUMD-compatible output files.

```python
from torchnep import predict_dataset

predict_dataset(
    "nep.txt",
    "test.xyz",
    output_dir="results",
    dtype="float64",       # float32 or float64
    batch_size=64,
    output_descriptor=0,   # 0=off, 1=per-frame mean, 2=per-atom (matches GPUMD)
)
# writes energy_train.out, force_train.out, virial_train.out,
# stress_train.out, and (when output_descriptor != 0) descriptor.out
```
