# Release Notes

## Unreleased

- **Streaming-only data path**: the preloaded GPU data store and the
  `stream_mode` option are removed — the dataset stays in host memory and
  batches are streamed to the device. Same speed, ~10–15x less GPU memory.
- **Automatic backend, `backend` option removed**: the type-pair
  contraction backend is now picked automatically from an 8-variant sweep
  over 1–87 element types on MI250X, V100 and A2000. New `mulsum`
  contraction (one-hot matmul + fused multiply/sum — no BLAS calls, no
  atomic `index_put` in the backward) is used under `use_compile` at every
  type count and eagerly up to 32 types; `bmm` above. On an AMD MI250X
  compiled training gets ~9x faster than before (44 → 5 ms/step, 4-type
  system); NVIDIA also gains. The compiled-autograd force path uses it
  too (and now works on ROCm — `torch.backends.opt_einsum` import fix).
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
