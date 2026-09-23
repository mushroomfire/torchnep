# Release Notes

## 1.0.7a1

- **GPU machines without a C compiler**: PyTorch 2.12+ runs some built-in CUDA operations through Triton even without `torch.compile`, and Triton needs a C compiler the first time it runs on a machine. TorchNEP now checks this at start-up: it warns, runs those operations with the regular CUDA kernels and keeps `torch.compile` off, instead of failing in the middle of training. See [Installation](getting-started/installation.md#a-c-compiler-on-the-gpu-machine).
- **Training on Apple GPUs (MPS) works again**; it had failed at the first epoch since 1.0.2.
- **Unknown nep.in keywords are an error**: a typo or a GPUMD-only option stops the run with the keyword and its line, instead of being ignored.
- **Exact restart without a validation set**: extending a finished run by raising `epoch` now gives exactly the longer run (the best-model check no longer changes the training state).
- **Fixes**:
    - `slim_types=True` failed for models with per-species cutoffs or a `zbl.in` table; the kept elements now keep their cutoffs and ZBL parameters.
    - Training in `float64` on a CUDA GPU failed at the first step.
    - Steps skipped for a non-finite gradient are now reported on every device, not only on CUDA.
    - The loss plot marks stage 2 where it really started when stage 1 stopped early.
    - Blank lines between frames of an xyz file are accepted everywhere, not only by the streamed reader.
- **Test coverage** reported on [Codecov](https://codecov.io/gh/mushroomfire/torchnep); new tests for fine-tuning, slimming, restart, the multi-GPU trainer and GPU training with `torch.compile`, and unused code removed.

## 1.0.6

- **Extrapolation grade** (`torchnep.extrapolation`): pick new training structures with one model instead of a committee, with multi-GPU variants and export to GPUMD's `compute_extrapolation`. See the [guide](https://mushroomfire.github.io/torchnep/guide/extrapolation/).
- **Documentation site** at <https://mushroomfire.github.io/torchnep/>.
- **Error figure redesigned** (`NEPPlotter.errors`): NEP − DFT histograms with training and validation overlaid; the `split` argument is removed.
- **Multi-GPU timeout**: 30 min by default for training, prediction and extrapolation (`TORCHNEP_DIST_TIMEOUT_MIN`).
- **Lint-clean code**: `ruff check .` reports no warnings.

## 1.0.5

- **Plots**: `torchnep.plot.NEPPlotter` (`pip install torchnep[plot]`) — loss curves, E/F/V parity plots (scatter or hexagonal density), error distributions per element and config type, a training dashboard and periodic-table error maps.
- **End of training**: the final `*_train.out` / `*_test.out` come from `nep_best.txt`, with an E/F/V RMSE/MAE table on screen and in `output.log`.
- **Cutoff checks** when nep.in is read: angular cutoff >= 3 Å and <= radial, radial <= 100 Å, ZBL outer cutoff between 1 and 3 Å.

## 1.0.4

- **Per-species cutoffs**: `cutoff rR1 rA1 rR2 rA2 ...` in nep.in, as in GPUMD.
- **Fix**: a flexible ZBL table was ignored with `use_autograd_forces` + `use_compile`.

## 1.0.3

- **Multi-node training** scales almost linearly (each rank streams its own part of the xyz file).
- **`neighbor_mode`** (`cached` / `compact` / `on_the_fly` / `auto`) for lower host memory.
- **Streamed prediction**: `predict_dataset` with bounded memory and automatic batch size; new `predict_dataset_sharded`.
- **Flexible ZBL**: `zbl zbl.in` reads per-pair cutoffs and coefficients, stored in `nep.txt`.

## 1.0.2

- **New defaults**: `weight_decay 1e-4` (AdamW), `stage2_lambda_v 0.1`, `valid_strategy "stratified"`, SWA over the last 100 epochs (`swa_start`), `use_compile` auto, `use_gpumd_qscaler False`.
- **Faster training**: fused kernels, fewer GPU syncs, faster NN on ROCm.
- **Compiled autograd forces**: `use_autograd_forces` + `use_compile`, also with DDP.
- **Streaming data path**: data stays in host memory and batches are streamed to the GPU; `stream_mode` and `backend` removed.
- **Per-stage `early_stop`**, **`export_valid_split`** and **`TORCHNEP_PROFILE=1`**.
- **Removed**: `pos_noise`, `lambda_1` / `lambda_2` (use `weight_decay`); unknown nep.in keys are ignored.
- **Fixes**: DDP deadlock with `use_swa`, energy offset of `nep_average.txt`, prediction OOM on multi-element models.

## 1.0.1

- GPUMD-consistent init, analytical `b1`, 6-component virial, `run_seed`, validation (`valid_file` / `valid_ratio`) and `early_stop`.

## 1.0.0

- Initial release: two-stage NEP4 training, GPUMD-compatible `nep.txt`, ZBL, multi-GPU DDP, fine-tuning, batched inference and an ASE calculator.
