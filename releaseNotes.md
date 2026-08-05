# Release Notes

## Unreleased

- **Streaming is now the only data path** — the preloaded GPU data store
  (and the `stream_mode` option) has been removed. The dataset stays in
  host memory and batches are streamed to the device; benchmarks across
  1–16 element types show speed parity with preloading under
  `use_compile` (and ≤ a few percent cost in eager mode) at ~10–15x less
  GPU memory, so the preload path had no remaining use case.
- **`backend="auto"` eager threshold raised**: eager mode now picks the
  `loop` contraction backend unless there are ≥20 element types (was ≥8) —
  benchmarked crossover is near ~20; under `use_compile` auto still picks
  `bmm`.

- **Compiled autograd forces** (`use_autograd_forces=True` +
  `use_compile=True`, single-GPU `train_nep`): the autograd force path can
  now be `torch.compile`d — the first-order dE/drij gradient is materialized
  into the graph with `make_fx`, so no runtime double
  backward remains. Outputs and parameter gradients match eager autograd to
  ~1e-5 (float32); ~4x faster per epoch, on par with the compiled
  analytical path.

- **Per-stage `early_stop`** (MACE-style): a stage-1 plateau with `stage2 1`
  configured now jumps straight into Stage 2 at the next epoch instead of
  terminating the run; only a plateau in the final stage stops training. The
  advanced stage-2 start epoch is saved in the checkpoint, so resumed runs
  stay in Stage 2.

- **`stream_mode`** (`train_nep` and `train_nep_sharded`): keep the dataset
  (or each rank's shard) in host memory and stream only the current batch to
  the GPU, computing the Chebyshev / angular basis on the fly (CPU batch
  assembly prefetched one batch ahead). GPU memory scales with `batch`
  instead of dataset size. Eager runs are bit-identical to the default
  preloaded mode; under `use_compile=True` the per-batch basis is compiled
  too.

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
