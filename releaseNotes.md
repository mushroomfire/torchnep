# Release Notes

## 1.0.2a2

- **`use_gpumd_qscaler` now defaults to `False`**: torch's default init
  with the self-consistent q_scaler converges to clearly better minima
  than the GPUMD-style start (600-epoch 4-seed PdCuNiP benchmark: ~12%
  lower E/V RMSE, ~3% lower F, on train and validation alike). `True`
  (the old default) remains available for GPUMD-comparison runs; the
  saved nep.txt is GPUMD-compatible either way.
- **`export_valid_split`**: write the exact `valid_ratio` split
  `train_nep` uses as verbatim GPUMD-ready `train.xyz` / `test.xyz`
  files, so the same data partition can be trained in GPUMD and the loss
  curves compared directly.

## 1.0.2a1

- **Streaming-only data path**: the preloaded GPU data store and the
  `stream_mode` option are removed — the dataset stays in host memory and
  batches are streamed to the device. Same speed, ~10–15x less GPU memory.
- **Automatic backend, `backend` option removed**: auto selected is best.
- **Compiled autograd forces**: `use_autograd_forces=True` +
  `use_compile=True` now works (first-order gradient materialized via
  `make_fx`) — ~4x faster than eager autograd.
- **Per-stage `early_stop`**: a stage-1 plateau jumps into Stage 2 instead
  of ending the run; only a final-stage plateau stops training (kept across
  resume).

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
