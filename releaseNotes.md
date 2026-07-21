# Release Notes

All notable changes to **TorchNEP** are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/), and the project
adheres to [Semantic Versioning](https://semver.org/).

TorchNEP is a pure-PyTorch implementation of the NEP4 (Neuroevolution
Potential) training framework. Trained models are written in GPUMD's `nep.txt`
format and load directly into GPUMD for MD simulation.

---

## [1.0.1]

First public open-source release of the standalone TorchNEP project.

### Added
- **GPUMD-consistent NN initialization.** `use_gpumd_qscaler=True` now
  re-initializes the fitting-network weights (`w0`/`b0`/`w1`) uniform(-1, 1)
  in addition to the descriptor coefficients, matching SNES's `mu` init
  (`snes.cu`). Previously only the descriptor coefficients were re-initialized,
  leaving the NN at torch's small-variance init whose shallow binding landscape
  does not match the one GPUMD's models inherit. Applies to fresh training only
  (ignored under `finetune_from`).
- **L1/L2 weight regularization** in the GPUMD form: the gradient of
  `λ₁·mean(|w|) + λ₂·RMS(w)` is fused directly into `.grad` after `backward()`
  (no autograd graph, no per-step sync). Global over all trainable weights.
  Both default to `0.0` — TorchNEP trains an MSE loss, so GPUMD's RMSE-tuned
  auto-λ formula is intentionally not applied. `loss.out` gains `L1`/`L2`
  columns.
- **Analytical `b1` energy shift** — the single global energy offset is solved
  analytically each epoch rather than by gradient descent.
- **SAM (Sharpness-Aware Minimization)** smoothing option (`sam_rho`) for
  flatter minima.
- **Validation support** (`valid_file` / `valid_ratio`): `nep_best` and the
  plateau LR schedule follow the validation loss; GPUMD-style `*_test.out`
  files are written.
- **Reproducibility** — `run_seed` makes a run fully deterministic (weight init
  and per-epoch shuffle); unset draws a fresh random seed per run.
- **Higher-body descriptors** `q_123` / `q_233` / `q_134`.
- **Stage-2 scheduler knobs** for the energy-focused second training stage.
- **Descriptor export** and GPUMD-compatible batched prediction output.
- **ASE calculator** with memory-bounded tiled inference for large MD cells.

### Changed
- **Standalone project.** Source headers and license notices now identify the
  code as part of the *TorchNEP* project rather than GPUMD. The license is
  unchanged: GPL-3.0-or-later.
- **GPUMD parity of outputs.** `q_scaler` matches GPUMD's `c=1` convention;
  virials use the 6-component tensor convention; `loss.out` uses GPUMD's exact
  column layout with a `#` header line (the `gnorm` column was removed).
- Default backend is batched-matmul (`bmm`); automatic `block_size` picks a
  memory-safe tile from measured free memory; TF32 is supported.

### Fixed
- MPS float32 force error — neighbor geometry now runs in float64 on MPS.
- ZBL evaluation in the ASE calculator.
- DDP double-training / print bugs on multi-card runs.

---

## [1.0.0]

Initial release. Core NEP4 training framework:

- Two-stage training (Stage 1 force-focused, Stage 2 energy-focused).
- Full NEP4 descriptor set (radial + angular 3/4/5-body invariants) with
  configurable `n_max`, `basis_size`, and `l_max`.
- GPUMD-compatible `nep.txt` read/write — models train in TorchNEP and run in
  GPUMD, or vice versa.
- ZBL universal repulsive potential with optional typewise cutoffs.
- Multi-GPU distributed data-parallel (DDP) training, single- or multi-node.
- Fine-tuning from any `nep.txt` / `checkpoint.pt`, with optional slimming to
  the element types present in the new dataset.
- Batched full-dataset inference and an ASE calculator for MD.

[1.0.1]: https://github.com/mushroomfire/torchnep/releases/tag/v1.0.1
[1.0.0]: https://github.com/mushroomfire/torchnep/releases/tag/v1.0.0
