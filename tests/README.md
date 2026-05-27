# torchnep tests

End-to-end pytest suite. Zero third-party dependencies beyond what
torchnep itself uses (`numpy`, `torch`); GPUMD comparisons read frozen
reference fixtures in [data/](data/) rather than invoking the GPUMD
binary at test time.

## Layout

```
tests/
  conftest.py                    sys.path wiring (no install needed)
  _common.py                     fixture catalogue, helpers, dtypes/devices

  test_forward.py                E / F / V / Descriptor vs baked GPUMD
  test_backward.py               analytical vs autograd; train vs predict
  test_descriptor_gradient.py    q_112 / q_1122 dq/ds vs autograd
  test_angular_lmax8.py          L = 1..8 angular basis self-consistency
  test_q123_q233.py              q_123 / q_233 bispectrum vs GPUMD PR #1517

  bake_fixtures.py               regenerates data/<name>.gpumd.npz via GPUMD

  data/
    CrCoNi.xyz / nep_CrCoNi.txt        typewise ZBL,    l_max 4 2 1
    PdCuNiP.xyz / nep_PdCuNiP.txt      fixed ZBL,       l_max 4 2 0
    mixed.xyz / nep_mixed.txt          typewise ZBL,    l_max 4 1 1 1 1
                                       (q_222 + q_1111 + q_112 + q_1122)
    *.gpumd.npz                        frozen GPUMD reference (E/F/V/D)
```

## Running

From the repo root:

```bash
pytest tests/
```

Common subsets:

```bash
# one file
pytest tests/test_forward.py -v

# specific fixture / device / dtype
pytest tests/ -k "mixed and cuda and float64"

# restrict device or dtype globally
TEST_DEVICE=cpu pytest tests/
TEST_DTYPE=float64 pytest tests/
```

## What each test covers

| file | covers |
| --- | --- |
| `test_forward.py` | Per-atom E, F, V (6 comp.), and **scaled descriptor** against frozen GPUMD outputs for three fixtures (typewise ZBL / fixed ZBL / full mixed-body). Three fixtures * 2 dtypes * 2 devices. |
| `test_backward.py` | (A) analytical force / virial vs autograd-on-rij; (B) `NEPModel.compute_properties_cached` (training path) vs `NEPCalculator.compute_batch` (predict path). Same fixture matrix. |
| `test_descriptor_gradient.py` | `_angular_weight` (the hand-derived dEi/d(sum_fxyz) for every body order) matches `torch.autograd.grad` on the explicit q-vs-s polynomial. Pins down the new `q_112` / `q_1122` analytical gradients introduced for the mixed-body invariants. |
| `test_angular_lmax8.py` | Solid-harmonics angular basis: L = 1..4 regression vs the old hand-coded formula; `_compute_dblm_dhat` matches autograd and finite differences for L = 1..8. |
| `test_q123_q233.py` | The q_123 / q_233 higher-L 4-body bispectrum channels (GPUMD PR #1517 `has_q_123` / `has_q_233`): rotational invariance, **bit-identical match to GPUMD's find_q polynomial**, `ops._extra_grad` vs autograd, end-to-end analytical-force vs autograd, and 7-field-l_max nep.txt round-trip. |

## Tolerances

- **vs GPUMD fixtures** (forward, descriptor): `1e-3 .. 1e-4` abs.
  GPUMD writes `%g` (~6 sig figs) and computes in float32 internally.
- **Analytical vs autograd** (same context): `1e-10` in float64, `5e-3`
  in float32 — only floating-point reorder.
- **Train vs predict** (different autograd contexts): `5e-6` in float64
  due to kernel-dispatch differences; not a correctness issue.
- **Descriptor gradient vs autograd**: `1e-12` (float64, hand-derived).

## Regenerating GPUMD reference fixtures

Only needed if a fixture `nep.txt` is updated or if the GPUMD output format
changes. Requires a working GPUMD `nep` binary:

```bash
GPUMD_NEP=/path/to/GPUMD/src/nep python tests/bake_fixtures.py
```

The default path is hard-wired to `/u/22/wuy33/unix/Study/GPUMD/src/nep`.

> Note: GPUMD's parameter parser enforces `basis_size <= 8` by default
> (internal `MAX_NUM_N` actually allows 16). The `mixed` fixture uses
> `basis_size 12 12`, so re-baking it requires lifting that cap
> (`src/main_nep/parameters.cu :: parse_basis_size`).
