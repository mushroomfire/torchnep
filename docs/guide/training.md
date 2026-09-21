# Training

`train_nep` trains on one device — a GPU, the CPU or Apple's MPS. Hyperparameter values come from [`nep.in`](nep-in.md); everything else is an argument of the function.

```python title="run_train.py"
from torchnep import train_nep

train_nep("nep.in", "train.xyz", output_dir="output")
```

```bash
python run_train.py
```

For several GPUs or nodes use [`train_nep_sharded`](distributed.md), which takes the same arguments.

## Two-stage training

Stage 1 trains with `lambda_e`, `lambda_f`, `lambda_v` — by default a force-dominated loss. With `stage2 1`, training switches at `start_stage2` to the stage-2 weights (`stage2_lambda_*`, energy-dominated by default) and restarts the learning rate from `stage2_lr`. The state at the switch is saved as `checkpoint_stage1.pt`, so stage 2 can be redone with other settings (see [Restart](restart-finetune.md#redo-stage-2)).

The global energy offset `b1` is not trained by gradient descent: it is solved exactly every epoch.

## Validation

```python
# a separate validation file
train_nep("nep.in", "train.xyz", output_dir="output", valid_file="valid.xyz")

# or hold out 10 % of the training file
train_nep("nep.in", "train.xyz", output_dir="output", valid_ratio=0.1)
```

With a validation set, `nep_best.txt` and the plateau learning-rate schedule follow the validation loss, `loss.out` gains the validation RMSE columns, and the end-of-training prediction also writes `*_test.out`.

`valid_ratio` draws the split from `run_seed` and keeps it on resume. `valid_strategy` chooses how:

- `"stratified"` (default) splits within groups of (element combination × cell size). Tiny cells (≤ 4 atoms) and groups of fewer than 20 frames stay in training, so rare compositions are never lost to the validation draw. Falls back to `"random"` when the validation set would be starved.
- `"random"` draws frames uniformly.

`export_valid_split` writes the same split as two GPUMD-ready files, to train the identical partition in GPUMD and compare the curves:

```python
from torchnep import export_valid_split

export_valid_split("train.xyz", valid_ratio=0.1, run_seed=42,
                   output_dir="split", strategy="stratified")   # split/train.xyz, split/test.xyz
```

## Averaged model (SWA)

`use_swa=True` keeps a running average of the weights over the last epochs (`swa_start`, by default the last 100) and saves it as `nep_average.txt`, next to `nep_best.txt` and `nep_final.txt`.

## Reproducibility

`run_seed` seeds the weight initialisation, the batch shuffle and the `valid_ratio` split. The seed is stored in `checkpoint.pt` and restored on resume. `run_seed=None` draws a new seed each run.

## Speed and memory

- **Compilation.** `use_compile=None` compiles the kernels with `torch.compile` on a GPU and runs eager on the CPU. The first epoch pays the compilation.
- **Precision.** `precision="float32"` (default) or `"float64"`.
- **Forces.** Forces use an analytical chain rule by default; `use_autograd_forces=True` differentiates through the pair vectors instead (slower, identical physics).
- **Host memory.** The dataset stays in host memory and batches are streamed to the GPU. `neighbor_mode` sets how neighbor lists are kept: `"cached"` (fastest), `"compact"` (~4× less memory) or `"on_the_fly"` (built on the GPU per batch, least memory); `"auto"` picks the first that fits.

## Runtime arguments

| Argument | Default | What it controls |
|---|---|---|
| `device` | auto | `"cuda"`, `"xpu"`, `"mps"` or `"cpu"`; any other stream-based PyTorch accelerator passed explicitly should also work. |
| `precision` | `"float32"` | Training and data dtype, `"float32"` or `"float64"`. |
| `use_autograd_forces` | `False` | Autograd forces through the pair vectors. |
| `use_swa` | `False` | Keep an averaged model and save `nep_average.txt`. |
| `swa_start` | last 100 epochs | First epoch included in the average. |
| `use_compile` | `None` | Auto: compile on GPU, eager on CPU; `True` / `False` force it. |
| `print_interval` | `1` | Log to screen every N epochs. |
| `checkpoint_interval` | `100` | Save `checkpoint.pt` every N epochs. |
| `prediction_interval` | `100` | Every N epochs, overwrite `*_train.out` with the current weights. |
| `restart` | `True` | Resume from `output_dir/checkpoint.pt` if it exists. |
| `resume_from` | `None` | Continue from a given checkpoint, e.g. `checkpoint_stage1.pt`. |
| `finetune_from` | `None` | Start a new training from the weights of a `nep.txt` or `.pt`. |
| `recompute_q_scaler` | `False` | With `finetune_from`: recompute the descriptor scaler on the new data. |
| `slim_types` | `False` | Drop element types absent from the dataset. |
| `energy_key` | `"energy"` | Comment-line tag read as the reference energy. |
| `use_gpumd_qscaler` | `False` | GPUMD-style initialisation, for comparison runs. |
| `run_seed` | `None` | Master random seed. |
| `valid_file` | `None` | Validation `.xyz` file. |
| `valid_ratio` | `None` | Fraction of the training file held out; excludes `valid_file`. |
| `valid_strategy` | `"stratified"` | `"stratified"` or `"random"` split. |
| `neighbor_mode` | `"auto"` | `"cached"`, `"compact"`, `"on_the_fly"` or `"auto"`. |

`train_nep_sharded` takes the same arguments except `device`.
