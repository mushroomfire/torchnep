# Release Notes

## Unreleased

- **`stream_mode`** (single-GPU `train_nep`): keep the dataset in host memory
  and stream only the current batch to the GPU, computing the Chebyshev /
  angular basis on the fly (CPU batch assembly prefetched one batch ahead).
  GPU memory scales with `batch` instead of dataset size. Eager runs are
  bit-identical to the default preloaded mode; under `use_compile=True` the
  per-batch basis is compiled too.

## 1.0.1

- **GPUMD-consistent init.** `use_gpumd_qscaler=True`.
- **Analytical `b1`** — the global energy offset is solved exactly each epoch
  instead of by gradient descent.
- **GPUMD-form L1/L2 regularization** (`lambda_1` / `lambda_2`, global, default
  `0`); `loss.out` gains `L1`/`L2` columns.
- **6-component virial** convention; `loss.out` uses GPUMD's column layout with
  a `#` header (the `gnorm` column was dropped).
- **`run_seed`** for fully reproducible runs (weight init + batch shuffle).
- **Validation** via `valid_file` / `valid_ratio`: `nep_best` and the plateau
  LR schedule follow the validation loss; writes GPUMD-style `*_test.out`.
- **`early_stop`** — stop when the monitored loss (validation loss if a
  validation set is used, else training loss) has not improved for N epochs.
- Standalone project: file headers relicensed to TorchNEP (GPL-3.0 unchanged).

## 1.0.0

Initial release: two-stage NEP4 training, GPUMD-compatible `nep.txt` I/O, full
descriptor set (radial + 3/4/5-body angular invariants), ZBL, multi-GPU DDP,
fine-tuning with optional type slimming, batched inference, and an ASE
calculator.
