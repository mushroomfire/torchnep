# nep.in reference

`nep.in` sets the model architecture and the training hyperparameters, one keyword per line (`#` starts a comment). The architecture keywords follow GPUMD, so a GPUMD `nep.in` describes the same model. Keywords TorchNEP does not use (e.g. `lambda_1`, `population`, `generation`) are ignored.

```text title="nep.in"
type       3 Cr Co Ni
version    4
zbl        2.5
use_typewise_cutoff_zbl 0.7
cutoff     6 4
n_max      8 8
basis_size 12 12
l_max      4 2 0
neuron     80

epoch            600
batch            16
lr               5e-3
scheduler_factor 0.5
stage2           1
start_stage2     300
stage2_lr        5e-4
stage2_lambda_e  2.0
```

## Model architecture

| Keyword | Default | Description |
|---|---|---|
| `type` | required | `N name1 name2 ...` — number and names of the element types. |
| `version` | `4` | NEP version; only NEP4 is implemented. |
| `cutoff` | `8 4` | Radial and angular cutoff in Å. Per species, as in GPUMD: `cutoff rR1 rA1 rR2 rA2 …`, one radial/angular pair per element in `type` order; an element pair uses the mean of the two values. |
| `n_max` | `6 6` | Radial and angular expansion orders. |
| `basis_size` | `6 6` | Chebyshev basis size, radial and angular. |
| `l_max` | `4 1 0` | `L_3b q_222 q_1111 q_112 q_123 q_233 q_134`: the largest L of the 3-body terms (1–8), then up to six flags that switch on each higher-body invariant (GPUMD order). |
| `neuron` | `30` | Neurons in the hidden layer. |
| `zbl` | off | ZBL outer cutoff in Å; switches on the short-range repulsion. A file name instead of a number (`zbl zbl.in`) reads GPUMD's flexible ZBL table. |
| `use_typewise_cutoff_zbl` | off | `use_typewise_cutoff_zbl <factor>`: per-pair ZBL outer cutoff = min(factor × (R<sub>i</sub> + R<sub>j</sub>), `zbl`) with covalent radii R, inner cutoff 0. The factor is required (0.7 recommended, ≥ 0.5). |

### Cutoff rules

Checked when `nep.in` is read:

- the angular cutoff is ≥ 3 Å and ≤ the radial cutoff, for every species;
- the radial cutoff is ≤ 100 Å;
- the ZBL outer cutoff (`zbl`, and every row of `zbl.in`) is between 1 and 3 Å, so ZBL always lies inside the angular neighbor list.

### Flexible ZBL

`zbl zbl.in` reads one line per element pair, in the order 1-1, 1-2, …, 1-n, 2-2, …, n-n:

```text title="zbl.in"
rc_inner rc_outer a1 a2 a3 a4 a5 a6 a7 a8
```

To change only the cutoffs, keep the universal coefficients `0.18175 3.1998 0.50986 0.94229 0.28022 0.4029 0.02817 0.20162`. The table is stored in `nep.txt`, so GPUMD, NEP_CPU and LAMMPS use it without the file; `use_typewise_cutoff_zbl` is ignored.

## Training hyperparameters

| Keyword | Default | Description |
|---|---|---|
| `epoch` | `600` | Total training epochs. |
| `batch` | `32` | Structures per gradient step (per GPU with `train_nep_sharded`). |
| `lr` | `0.01` | Initial learning rate. |
| `stop_lr` | `1e-6` | Lower bound of the learning rate. |
| `lambda_e` | `0.01` | Energy loss weight. |
| `lambda_f` | `1.0` | Force loss weight. |
| `lambda_v` | `0.01` | Virial loss weight. |
| `weight_decay` | `1e-4` | AdamW decoupled weight decay on all trainable parameters; `0` uses plain Adam. |
| `max_grad_norm` | `10.0` | Gradient clipping threshold. |
| `lr_scheduler` | `plateau` | `plateau` (reduce on plateau) or `step` (fixed interval); both stages use this mode. |
| `scheduler_patience` | `15` | `plateau`: epochs without improvement before a reduction. `step`: epochs between reductions. |
| `scheduler_factor` | `0.7` | Factor applied at each reduction. |
| `early_stop` | `0` | Stop when the monitored loss has not improved for N epochs (`0` = off). Per stage: a stage-1 plateau moves on to stage 2. Use a value larger than `scheduler_patience`. |
| `stage2` | `0` | `1` switches on the second, energy-focused stage. |
| `start_stage2` | half of `epoch` | Epoch at which stage 2 starts. |
| `stage2_lr` | `1e-3` | Learning rate at the start of stage 2. |
| `stage2_lambda_e` | `1.0` | Stage 2 energy weight. |
| `stage2_lambda_f` | `0.05` | Stage 2 force weight. |
| `stage2_lambda_v` | `0.1` | Stage 2 virial weight. |
| `stage2_scheduler_patience` | `scheduler_patience` | Stage 2 scheduler patience. |
| `stage2_scheduler_factor` | `scheduler_factor` | Stage 2 reduction factor. |

The monitored loss is the validation loss when a validation set is used, otherwise the training loss. Options that are not hyperparameter values — device, precision, validation data, checkpoints — are arguments of the Python function; see [Training](training.md#runtime-arguments).
