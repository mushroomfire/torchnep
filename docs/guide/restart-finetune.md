# Restart and fine-tuning

## Resume a run

A run resumes automatically: with `restart=True` (the default), `train_nep` continues from `output_dir/checkpoint.pt` if it exists. Resuming is an exact continuation — learning rate, optimizer, scheduler and best-model state all come from the checkpoint.

```python
# continue output/checkpoint.pt
train_nep("nep.in", "train.xyz", output_dir="output")

# continue a specific checkpoint
train_nep("nep.in", "train.xyz", output_dir="output",
          resume_from="output/checkpoint_stage1.pt")
```

### Redo stage 2

`checkpoint_stage1.pt` holds the state at the end of stage 1. To try other stage-2 settings, edit `nep.in` and resume from it:

```python
train_nep("nep.in", "train.xyz", output_dir="output_stage2b",
          resume_from="output/checkpoint_stage1.pt")
```

### What can change on restart

| Parameter | Change on restart? | Notes |
|---|---|---|
| `epoch` | yes | Increase it to train longer. |
| `lambda_e` / `lambda_f` / `lambda_v` | yes | Take effect from the next epoch. |
| `stage2_lambda_e` / `_f` / `_v` | yes | Same. |
| `batch` | yes | |
| `stage2`, `start_stage2` | yes | Add stage 2 to a run without it, or move it later. |
| `stage2_lr` | only at the switch | Applied once, when training crosses from stage 1 to stage 2. Resuming a checkpoint already in stage 2 keeps its learning rate; resume from `checkpoint_stage1.pt` to use a new `stage2_lr`. |
| `lr_scheduler` | yes | The old scheduler state is dropped; the new one starts from the current learning rate. |
| `scheduler_patience` / `scheduler_factor` | yes | Applied immediately. |
| `stage2_scheduler_patience` / `_factor` | yes | Applied immediately to the stage-2 scheduler. |
| `lr` (stage 1) | no | The checkpoint's learning rate is kept. |
| `run_seed` | no (ignored) | The checkpoint's seed is kept, so the shuffle and the `valid_ratio` split stay the same. |
| `valid_file` / `valid_ratio` | not recommended | Changes the train/validation split; a warning is logged and the best-model tracking resets. |
| `type`, `cutoff`, `n_max`, `basis_size`, `l_max`, `neuron` | no | Fixed by the saved weights. |

## Fine-tuning

Fine-tuning starts a new training from the weights of a trained model. The architecture in `nep.in` must match the source model; the new dataset may contain a subset of its elements.

```python
train_nep(
    "nep.in",
    "new_data.xyz",
    output_dir="finetune",
    finetune_from="pretrained/nep.txt",   # or pretrained/checkpoint.pt
    slim_types=True,
)
```

- `finetune_from` takes a `nep.txt` (from GPUMD or TorchNEP) or a `checkpoint.pt`.
- `slim_types=True` removes the element types the new dataset does not contain before training starts, which shrinks the model and speeds up training.
- The source model's descriptor scaler is kept; `recompute_q_scaler=True` recomputes it on the new data.

## Slim a model without training

```python
from torchnep.data import parse_nep_in
from torchnep.model import NEPModel, slim_model

model = NEPModel(parse_nep_in("nep.in"))
model.load_weights_from_nep_txt("nep.txt")

slimmed = slim_model(model, ["Cr", "Ni"])
slimmed.save_nep_txt("nep_slim.txt", max_NN_radial, max_NN_angular)
```

`max_NN_radial` / `max_NN_angular` are the largest neighbor counts GPUMD should allocate for; `torchnep.train.compute_max_neighbors` computes them from preprocessed structures.
