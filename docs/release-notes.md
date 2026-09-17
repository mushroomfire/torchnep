# Release Notes

## 1.0.6a1

- **Documentation site**: the full documentation now lives at
  <https://mushroomfire.github.io/torchnep/> (sources in `docs/`, built with
  MkDocs Material); the README is a short overview and these release notes
  moved to `docs/release-notes.md`.
- **Error figure redesigned** (`NEPPlotter.errors`): one histogram of
  NEP − DFT per quantity with training and validation overlaid and annotated
  with RMSE / MAE / max, plus the force error against the force magnitude, in
  the style of the parity panels. New `quantities`, `force_magnitude` and
  `bins` arguments; the per-element and per-`config_type` bars are gone
  (`periodic_table` draws the per-element errors) and with them the `split`
  argument.

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
