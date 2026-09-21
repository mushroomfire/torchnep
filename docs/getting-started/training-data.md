# Training data

TorchNEP reads extended-XYZ files, the same format as GPUMD's `train.xyz`. The parser is strict: the rules below are enforced, and a file that breaks them raises an error when it is loaded.

```text title="train.xyz (one frame)"
4
Lattice="3.61 0 0 0 3.61 0 0 0 3.61" energy=-14.93 virial="0.12 0 0 0 0.12 0 0 0 0.12" Properties=species:S:1:pos:R:3:forces:R:3
Cu 0.000 0.000 0.000  0.012 -0.003  0.000
Cu 1.805 1.805 0.000 -0.012  0.003  0.000
Cu 1.805 0.000 1.805  0.000  0.001 -0.004
Cu 0.000 1.805 1.805  0.000 -0.001  0.004
```

## Comment line

| Tag | Required | Meaning |
|---|---|---|
| `Lattice="ax ay az bx by bz cx cy cz"` | yes | The three lattice vectors as rows, in Å. |
| `energy=<value>` | no | Total energy of the frame, in eV. |
| `virial="vxx vxy vxz vyx vyy vyz vzx vzy vzz"` | no | Virial in eV, exactly 9 components. Positive values mean compressed states (GPUMD convention). |
| `stress="sxx sxy sxz syx syy syz szx szy szz"` | no | Stress in eV/Å³, exactly 9 components. Positive values mean stretched states — the opposite sign of the virial. |

- Every frame is treated as fully periodic; `pbc=...` is ignored. For clusters, molecules or surfaces, use a vacuum gap wider than the cutoff.
- If a frame has both `virial` and `stress`, `virial` is used.
- A different energy tag (e.g. `atomization_energy`) is read with the `energy_key` argument of `train_nep`.

## Per-atom columns

`Properties=...` declares the column layout. TorchNEP reads three fields and ignores every other column (e.g. `Z:I:1`, `magmom:R:1`):

| Field | Meaning |
|---|---|
| `species:S:1` | Chemical symbol, case-sensitive; must be one of the elements on the `type` line of `nep.in`. |
| `pos:R:3` | Cartesian position in Å. |
| `force:R:3` or `forces:R:3` | Reference force in eV/Å (optional). |

## Validation data

A validation set is either a second file in the same format (`valid_file=`) or a fraction of the training file (`valid_ratio=`); see [Validation](../guide/training.md#validation).
