# TorchNep tests

Pure pytest suite (`numpy` + `torch`; `ase` only for the ASE test).

```bash
pytest tests/                      # full suite
pytest tests/ -k float32           # one dtype
TEST_DEVICE=cpu pytest tests/      # restrict device (default: cpu + cuda if present)
TORCHNEP_TEST_DDP=1 pytest tests/  # also the multi-process (torchrun) tests; with 2 GPUs also multi-GPU training
```

CI runs on CPU. Before a release, run the whole suite with `TORCHNEP_TEST_DDP=1` on a machine with
two GPUs as well: training with `torch.compile` (the default on GPUs), the CUDA-only guards and the
multi-GPU paths are only exercised there.

| file | covers |
| --- | --- |
| `test_gpumd_parity.py` | E / F / V / descriptor vs the GPUMD reference (incl. compressed CrCoNi frames where ZBL forces reach ~120 eV/Å); analytical vs autograd; train path vs predict path on every frame (float64 to round-off). |
| `test_descriptors.py` | Angular basis L=1..8; gradient checks; the six higher-body channels (q_222, q_1111, q_112, q_123, q_233, q_134) — GPUMD-polynomial match and rotational invariance; backend auto-resolution and loop/bmm/mulsum numerical equivalence. |
| `test_neighbor.py` | Cell-list vs brute-force neighbor search; tiled / auto-block paths. |
| `test_parsing.py` | Legacy and current `l_max` nep.in / nep.txt parsing. |
| `test_ase_calculator.py` | Optional ASE calculator (energy/forces/stress, ZBL split). |
| `test_b1_and_gpumd_qscaler.py` | Analytical `b1` offset (residual → 0, `nep_best` ≤ `nep_final`); `use_gpumd_qscaler` reproduces GPUMD's `c=1` q_scaler; `gpumd_init_parameters` re-inits coeffs **and** NN weights uniform(−1,1); weight_decay (AdamW) shrinks the weights. |
| `test_run_seed_and_valid.py` | `run_seed` reproducibility; `valid_file` / `valid_ratio` deterministic split, best-model selection on validation loss, `*_test.out`, split preserved across resume; `early_stop` fires on a plateau (validation-loss branch), is off by default, is per-stage (a stage-1 plateau jumps into Stage 2, surviving resume). `export_valid_split` reproduces the internal split (verbatim frames, matches `energy_test.out`). |
| `test_stream_mode.py` | `StreamDataStore` (the training data store): `collate` is bit-exact vs an independently assembled reference (concatenation + offsets + basis straight from the ops functions); metadata/mask consistency incl. missing channels; prefetched `iter_collated` matches direct collate; 2-rank DDP same-seed reproducibility. The DDP case is local-only: set `TORCHNEP_TEST_DDP=1` (skipped in CI — multi-process rendezvous is unreliable on shared runners). |
| `test_nep_in.py` | nep.in parsing: every keyword reaches the trainer, unknown keywords and invalid values are rejected, the documented keywords are exactly the supported ones. |
| `test_xyz_reader.py` | Extended-XYZ reading: stress → virial (sign, volume), indexed (streamed) reader equals the plain one, blank lines between frames, malformed frames rejected. |
| `test_train_features.py` | `train_nep` features: fine-tuning from nep.txt / checkpoint (frozen weights keep their forces), `slim_types`, `recompute_q_scaler`, redoing stage 2, exact restart (after a kill, extending a finished run without validation), step scheduler and `stop_lr`, early stop, non-finite gradient steps skipped, random split, parallel preprocessing. |
| `test_sharded_features.py` | `train_nep_sharded` on 2 ranks (`TORCHNEP_TEST_DDP=1`): streamed loading equals in-memory, sharded validation, exact resume with and without validation, fine-tuning with `slim_types` (incl. zbl.in / per-species cutoffs), early stop, non-finite steps, autograd vs analytical forces. |
| `test_frame_weight.py` | Per-frame `weight=`: read on every path, invalid values (incl. 0) rejected, the logged loss equals the weighted objective recomputed from the saved model (train and validation, single process — also on GPU — and sharded), `b1` is the weighted mean residual, `weight=1` equals no weight bit for bit. |
| `test_slim.py` | Slimming a model to a subset of elements is exact for every ZBL variant (typewise, universal, zbl.in table, per-species cutoffs), via `slim_model` + nep.txt and via fine-tuning with `slim_types`. |
| `test_gpu_training.py` | GPU only: training with `torch.compile` matches eager (loss curve through stage 2, final model) for analytical / autograd forces, per-species cutoffs + ZBL, zbl.in, float32 and float64; also on 2 GPUs with `TORCHNEP_TEST_DDP=1`. |
| `test_predict_features.py` | `predict_dataset`: descriptor.out vs GPUMD, the out-of-memory retry and small chunks leave the results unchanged, progress line without tqdm. |
| `test_cutoff_per_type.py` | Per-species cutoffs: parsing, pair table, equal values reproduce the uniform cutoff, agreement of the four compute paths. |
| `test_cutoff_rules.py` | Cutoff rules checked on nep.in and every model (angular/radial/ZBL limits, typewise factor). |
| `test_zbl_matrix.py` | Every ZBL variant on every path, against NEP_CPU (double precision, `data/zbl/*.nepcpu.npz`) and GPUMD (`data/zbl/*.gpumd.npz`): universal, typewise with the cap active / inactive, `zbl.in`, `zbl.in` + typewise line, each with per-species cutoffs, one type, CrCoNi with `zbl 2`. Paths: the four training paths (float64 whole model and ZBL term; float32 ZBL forces), calculator `compute` / `compute_batch` / `compute_tiled`, ASE, `predict_dataset`, nep.txt header and round trip, checkpoints, `slim_model`, the end-of-training `*_train.out` (single process and sharded). The structures sample both sides of every cutoff; a test checks that they do. |
| `test_typewise_zbl_cap.py` | The typewise pair cutoff is capped at the `zbl` value (PR #29). |
| `test_zbl_flexible.py` | Flexible ZBL (`zbl.in`): universal-parameter file equals the universal path, per-pair parameters vs a numpy reference, nep.txt round trip. |
| `test_neighbor_modes.py` | The `cached` / `compact` / `on_the_fly` neighbor layouts give the same batches; training runs in every mode. |
| `test_noise_and_wd.py` | weight_decay with bias-exempt groups, the gathered NN, the compiled ZBL path. |
| `test_extrapolation.py` | Extrapolation grade: parameter gradients, MaxVol, build / grade / select, multi-process variants. |
| `test_runtime.py` | GPU machines without a C compiler (`_runtime.py`); the last test reproduces it on a real GPU. |
| `test_plot.py` | `torchnep.plot`: readers (incl. GPUMD fixed-width files), energy shifts, stage-2 marker, every figure builds. |
| `test_compiled_autograd.py` | `CompiledAutogradForce` (make_fx-materialized autograd forces): outputs and parameter gradients (second-order path through the force loss) match eager autograd across batch shapes on one dynamic graph; energy-only calls fall back to eager. CUDA-only — auto-skipped on CPU hosts/CI. |

**Tolerance vs GPUMD:** `rtol=1e-5, atol=2e-4`.

**Re-baking the reference** (only if `nep_CrCoNi.txt` / `CrCoNi.xyz` change):

```bash
GPUMD_NEP=/path/to/GPUMD/src/nep python tests/bake_fixtures.py
```

**ZBL fixtures** (`data/zbl/`, defined in `zbl_cases.py`): `python tests/make_zbl_fixtures.py inputs` writes the
structures, nep.in / zbl.in files and nep.txt files; `NEP_CPU=/path/to/NEP_CPU python tests/make_zbl_fixtures.py nepcpu`
compiles a small driver against NEP_CPU and writes the double-precision references; `gpumd-prepare` /
`gpumd-collect` do the same with GPUMD's `nep` on a GPU machine. `zbl_oracle.py` is an independent numpy port of
GPUMD's ZBL that reads the nep.in text.

**Mutation audit** of the ZBL tests (minutes, not part of the pytest run): `python tests/mutate_zbl.py` plants each
listed ZBL bug in a scratch copy and runs the ZBL tests; every mutant must be killed.
