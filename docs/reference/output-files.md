# Output files

A training run writes these files to `output_dir`. `predict_dataset` writes the prediction files (`*_train.out`, `descriptor.out`) in the same format.

## Models and checkpoints

| File | Contents |
|---|---|
| `nep_best.txt` | Best model (lowest validation loss, or training loss without a validation set). The end-of-training prediction uses it. |
| `nep_final.txt` | Model at the last epoch. |
| `nep_average.txt` | Averaged model; only with `use_swa=True`. |
| `checkpoint.pt` | Full training state, for resuming. |
| `checkpoint_stage1.pt` | Full state at the end of stage 1. |

All `nep*.txt` files are GPUMD `nep4` / `nep4_zbl` models.

## Logs

| File | Contents |
|---|---|
| `output.log` | The full console log, including the final E/F/V RMSE / MAE tables. |
| `loss.out` | One row per epoch: epoch, loss, RMSE of energy (eV/atom), force (eV/Å), virial (eV/atom) and stress (GPa). With a validation set, four more columns hold the validation RMSEs and the loss column is the validation loss. |

## Predictions

Written at the end of training with `nep_best.txt`, every `prediction_interval` epochs with the current weights, and by `predict_dataset`. Each row holds the prediction first and the reference second; missing references are written as GPUMD does.

| File | Rows | Columns |
|---|---|---|
| `energy_train.out` | one per frame | predicted, reference energy (eV/atom) |
| `force_train.out` | one per atom | predicted Fx Fy Fz, reference Fx Fy Fz (eV/Å) |
| `virial_train.out` | one per frame | predicted, reference xx yy zz xy yz zx (eV/atom) |
| `stress_train.out` | one per frame | predicted, reference xx yy zz xy yz zx (GPa) |
| `*_test.out` | | the same four files for the validation set |
| `descriptor.out` | per frame or per atom | scaled descriptors (`predict_dataset(..., output_descriptor=1 or 2)`) |
