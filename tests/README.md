# torchnep tests

Two correctness tests validated against `mdapy.NEP` (the reference NEP_CPU
C++ backend). Every (device × dtype × model) combination is exercised.

## Files

| file | purpose |
| --- | --- |
| `test_vs_mdapy.py` | forward pass: E / F / V vs NEP_CPU |
| `test_backward.py` | backward pass: autograd force vs analytical force, and training-path vs predict-path consistency |
| `nep_PdCuNiP.txt` | reference NEP4 model, **fixed-cutoff ZBL** (`zbl 0.9 1.8`) |
| `nep_CrCoNi.txt` | reference NEP4 model, **typewise ZBL** (`zbl 1.25 2.5 0.7`) |
| `PdCuNiP.xyz` | 500-atom configuration paired with `nep_PdCuNiP.txt` |
| `CrCoNi.xyz` | 108-atom configuration paired with `nep_CrCoNi.txt` |
| `test_angular_lmax8.py` | separate self-contained test for the extended L-up-to-8 angular basis |

## test_vs_mdapy.py — forward

Max-abs tolerances vs NEP_CPU:

| dtype | E (eV/atom) | F (eV/Å) | V (eV) |
| --- | --- | --- | --- |
| float64 | 1e-10 | 1e-10 | 1e-10 |
| float32 | 5e-4 | 1e-3 | 1e-3 |

```bash
python test_vs_mdapy.py                         # full matrix
TEST_DEVICE=cpu     python test_vs_mdapy.py     # pin device
TEST_DTYPE=float64  python test_vs_mdapy.py     # pin dtype
python test_vs_mdapy.py <nep.txt> <xyz>         # custom single case
```

## test_backward.py — backward

Autograd is correct by construction once the forward is correct, so only
the closed-form analytical paths need verification. Two checks per cell:

**A. `compute_batch` (analytical) ≡ `compute` (autograd on rij)**
Tight: float64 ≤ 1e-10, float32 ≤ 5e-3.

**B. `NEPModel.compute_properties_cached` (training) vs
`NEPCalculator.compute_batch` (predict)**
Loose in float64 (≤ 5e-6) because the two call sites differ only in
`requires_grad` state, and PyTorch dispatches to different matmul/einsum
kernels for the two cases; their accumulation order gives ~1e-7 per element.
Both results match NEP_CPU to float64 round-off (verified in Part A and in
`test_vs_mdapy.py`).

```bash
python test_backward.py
```
