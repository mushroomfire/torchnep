# Release Notes

## Unreleased

- **`stage2_lambda_v` default 0.1 → 0.05** (same as `stage2_lambda_f`):
  the old value measurably overfits the virial in stage 2 on
  multi-element sets — lowering it improved validation E and V together
  in the unep16 sweeps.
- **Noisy-train labeling**: with `pos_noise` on, the per-epoch train
  RMSEs (screen and loss.out) are measured on the noise-augmented
  batches and sit above the true training error by the injected jitter;
  they are now marked `(noisy)` and loss.out carries an explanatory
  header line. Clean train error: the periodic `*_train.out`
  predictions.

- **`predict_dataset` streams from host memory**: the whole-dataset GPU
  upload is gone — only each batch's slice is shipped to the device, so
  prediction memory scales with `batch_size` like training does (a
  105k-frame set needed ~15 GB resident before; it now runs on any card).

- **`pos_noise`**: training-time coordinate jitter — per-atom Gaussian
  displacements (σ in Å) applied to each training batch's pair vectors;
  labels untouched, validation/eval passes stay clean, and the noise
  stream is reproducible from `run_seed` via a dedicated generator.
- **`weight_decay`**: > 0 switches the optimizer to AdamW (decoupled
  weight decay). Preferred over the GPUMD-form `lambda_1`/`lambda_2`
  when regularizing torchnep runs — Adam rescales in-loss L2 gradients
  per-parameter, AdamW does not.

## 1.0.2b1

- **Stratified validation split** (`valid_strategy="stratified"`, also in
  `export_valid_split`): frames are grouped by (element combination ×
  cell-size class) and split within each group; tiny cells (≤ 4 atoms —
  the pair-specific short-range scans) and groups with fewer than 20
  frames go entirely to training. Guarantees the short-range physics and
  rare compositions are always learned instead of being silently lost to
  a random validation draw.

- **Fix `predict_dataset` GPU OOM on multi-element models**: the mulsum
  contraction's one-hot matrix exists only for the backward pass; it is
  now skipped whenever gradients are off (prediction, q_scaler,
  frozen-weight eval) — the plain gather is bit-identical and avoids the
  (pairs × ntypes²) allocation that reached GiB at prediction batch
  sizes.
- **Fix `nep_average.txt` energy offset**: ``b1`` is solved analytically
  each epoch (not gradient-trained), so averaging it along the SWA
  trajectory left the saved SWA model with a stale global energy shift
  (~10 meV/atom on a 16-element benchmark). Both trainers now re-solve
  ``b1`` for the averaged weights before saving (sharded: from the
  all-reduced global residual).

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
