# Release Notes

## 1.0.3b2

- **Flexible ZBL** (GPUMD `zbl.in`): `zbl <file>` in nep.in reads per-element-pair
  cutoffs and screening coefficients; they are written into `nep.txt`, so GPUMD,
  NEP_CPU and LAMMPS use them without the file.

## 1.0.3b1

- **Multi-GPU scaling fix**: one all-reduce per epoch instead of one per step.
- **Lower host memory**: new `neighbor_mode` argument (`train_nep` /
  `train_nep_sharded`, default `"auto"`) — `"cached"`, `"compact"`
  (~4x less memory) or `"on_the_fly"` (neighbor lists built on the GPU
  per batch, least memory); `"auto"` picks the first that fits. The
  sharded data path (validation file, store build, final prediction) also
  uses less memory, and host RSS is printed at each data stage.

## 1.0.3a2

- **Streamed `predict_dataset`**: the xyz is indexed once and processed in
  chunks of ~`chunk_atoms` atoms (default 200k, env
  `TORCHNEP_PREDICT_CHUNK_ATOMS`) — read → neighbor lists → batches → rows
  appended — so host memory is bounded by the chunk and device memory by
  the batch; any dataset finishes on any machine. Outputs unchanged.
- Progress bar (tqdm when installed, plain otherwise) and a per-stage
  timing summary; one neighbor-list worker pool reused across chunks.
- **`predict_dataset_sharded`**: multi-GPU / multi-node prediction
  (torchrun / srun, one process per GPU): rank 0 indexes and broadcasts the
  frame index, contiguous atom-balanced ranges per rank, each rank streams
  its range, rank 0 merges the parts in order — identical files to the
  single-process call.

## 1.0.3a1

- **Streamed shard loading** (`train_nep_sharded`): each rank reads only
  its own byte range of the xyz file (auto for files ≥ 2 GiB,
  `TORCHNEP_STREAM_THRESHOLD` to override). Bit-identical results;
  stratified validation split falls back to random.
- **Fix multi-node DDP backend**: the GPU-sharing check now counts ranks
  per node (`LOCAL_WORLD_SIZE` / `SLURM_NTASKS_PER_NODE`) instead of
  globally, so multi-node jobs use NCCL/RCCL; gloo stays the CPU /
  GPU-sharing fallback.
- **`predict_dataset`**: `dtype` default `float32`; `batch_size=None`
  auto-sizes from free GPU memory with OOM retry; prints an E/F/V
  RMSE/MAE table.

## 1.0.2

Defaults changed:

- **`weight_decay` default `1e-4`** (AdamW), on all trainable parameters
  (`b1` is solved analytically and never decays). Set `0` for plain Adam.
- **`stage2_lambda_v` back to `0.1`** — independent-test benchmarks rank
  0.1 > 0.05 > 0.02 on both energy and virial; the 1.0.2b2 default of
  0.05 is reverted.
- **`valid_strategy` default `"stratified"`**; falls back to `"random"`
  automatically (with a log note) when stratification would starve the
  validation set (e.g. an all-tiny-cell dataset).
- **SWA averages only the run tail**: new `swa_start` function argument,
  default = the last 100 epochs (averaging all of stage 2 degraded
  energies).

New:

- **`use_compile` defaults to auto**: compile on GPU, eager on CPU;
  missing Triton / C++ toolchain degrades to eager instead of raising.
  `print_interval` default 10 → 1, `prediction_interval` 20 → 100.
- **`TORCHNEP_PROFILE=1`**: per-epoch phase breakdown (data-wait,
  step, validation, tail) plus peak allocated/reserved GPU memory.
  `=2` adds per-step syncs for GPU-attributed times.

Removed:

- **`pos_noise`** — no benefit on independent tests at any dose; the
  clean-metrics machinery it required is gone too. Unsupported nep.in
  keys (including `lambda_1` / `lambda_2` / `pos_noise`) are now silently
  ignored. These features remain available in ≤ 1.0.2b2.

Fixed:

- DDP deadlock at the end of `use_swa` runs (the SWA average is now
  maintained on every rank).
- A hidden per-step GPU sync in the batch pipeline (pageable-index
  gather) and ~10 host syncs per step in the metric/loss code — real
  epochs are moderately faster on every GPU tested.
- Preprocessing pool now respects the job's actual CPU allocation
  (slurm cgroups) instead of the node's core count.
- ROCm: the fused NN uses a multiply+reduce formulation instead of
  batched matmul (rocBLAS handles those shapes poorly) — 25% faster
  full step on MI250X, identical math.

## 1.0.2b2

- **Kernel fusion**: per-type NN → one gathered-weight batched matmul;
  ZBL moved inside the compiled graph (branch-free, analytic pair
  derivatives) for both the analytical and autograd force paths; fused
  Adam on CUDA. ~2x faster training step on GH200, correctness pinned
  by GPUMD-reference tests.
- **`use_autograd_forces` + `use_compile` now works in DDP**
  (`train_nep_sharded`).
- **`lambda_1` / `lambda_2` removed** — SNES-form L1/L2 has no usable
  meaning under Adam's per-parameter normalization; use `weight_decay`.

- **`stage2_lambda_v` default 0.1 → 0.05** (same as `stage2_lambda_f`):
  the old value measurably overfits the virial in stage 2 on
  multi-element sets — lowering it improved validation E and V together
  in the unep16 sweeps.
- **True train metrics under `pos_noise`**: the loss still trains on the
  noise-augmented geometry (that is the regularization), but the logged
  train RMSEs and the analytic `b1` residual now come from one extra
  no-grad forward on each batch's clean geometry (~+30% epoch time when
  noise is on). loss.out keeps its exact format and shows real errors —
  loss curves stay directly plottable and comparable.

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
