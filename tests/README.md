# TorchNep tests

Pure pytest suite (`numpy` + `torch`; `ase` only for the ASE test).

```bash
pytest tests/                      # full suite
pytest tests/ -k float32           # one dtype
TEST_DEVICE=cpu pytest tests/      # restrict device (default: cpu + cuda if present)
```

| file | covers |
| --- | --- |
| `test_gpumd_parity.py` | E / F / V / descriptor vs the GPUMD reference (incl. compressed CrCoNi frames where ZBL forces reach ~120 eV/Å); analytical vs autograd; train path vs predict path. |
| `test_descriptors.py` | Angular basis L=1..8; gradient checks; the six higher-body channels (q_222, q_1111, q_112, q_123, q_233, q_134) — GPUMD-polynomial match and rotational invariance. |
| `test_neighbor.py` | Cell-list vs brute-force neighbor search; tiled / auto-block paths. |
| `test_parsing.py` | Legacy and current `l_max` nep.in / nep.txt parsing. |
| `test_ase_calculator.py` | Optional ASE calculator (energy/forces/stress, ZBL split). |
| `test_b1_and_gpumd_qscaler.py` | Analytical `b1` offset (residual → 0, `nep_best` ≤ `nep_final`); `use_gpumd_qscaler` reproduces GPUMD's `c=1` q_scaler; `gpumd_init_parameters` re-inits coeffs **and** NN weights uniform(−1,1); L2 (`lambda_2`) shrinks the weights. |
| `test_run_seed_and_valid.py` | `run_seed` reproducibility; `valid_file` / `valid_ratio` deterministic split, best-model selection on validation loss, `*_test.out`, split preserved across resume; `early_stop` fires on a plateau (validation-loss branch), is off by default, is per-stage (a stage-1 plateau jumps into Stage 2, surviving resume). |
| `test_stream_mode.py` | `stream_mode`: `StreamDataStore.collate` is bit-exact vs `GPUDataStore` (incl. the on-the-fly basis); full CPU-float64 training runs (single-GPU, `valid_ratio`, 2-rank DDP) reproduce the preloaded mode (numeric token comparison — BLAS alignment costs ~1 ULP on some hosts). The DDP case is local-only: set `TORCHNEP_TEST_DDP=1` (skipped in CI — multi-process rendezvous is unreliable on shared runners). |
| `test_compiled_autograd.py` | `CompiledAutogradForce` (make_fx-materialized autograd forces): outputs and parameter gradients (second-order path through the force loss) match eager autograd across batch shapes on one dynamic graph; energy-only calls fall back to eager. CUDA-only — auto-skipped on CPU hosts/CI. |

**Tolerance vs GPUMD:** `rtol=1e-5, atol=2e-4`.

**Re-baking the reference** (only if `nep_CrCoNi.txt` / `CrCoNi.xyz` change):

```bash
GPUMD_NEP=/path/to/GPUMD/src/nep python tests/bake_fixtures.py
```
