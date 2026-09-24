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
| `weight=<value>` | no | Weight of the frame in the training loss, `0 < weight <= 100`, default 1. See [Frame weights](#frame-weights). |

- Every frame is treated as fully periodic; `pbc=...` is ignored. For clusters, molecules or surfaces, use a vacuum gap wider than the cutoff.
- If a frame has both `virial` and `stress`, `virial` is used.
- A different energy tag (e.g. `atomization_energy`) is read with the `energy_key` argument of `train_nep`.

## Frame weights

`weight=w` makes the frame count `w` times in the loss: its energy, the forces of its atoms and its virial are all weighted by `w`, while `lambda_e`, `lambda_f` and `lambda_v` in `nep.in` still set the balance between energy, forces and virial. The loss stays a mean over the frames (atoms for forces), so `weight=2` counts about like two copies of the frame, and a file where every frame has `weight=1` trains exactly as one without weights.

$$
L = \frac{\lambda_e}{N_E} \sum_{i} w_i \, \Delta E_i^2
  + \frac{\lambda_f}{3 N_F} \sum_{i} w_i \sum_{j \in i} \lvert \Delta \mathbf{F}_j \rvert^2
  + \frac{\lambda_v}{6 N_V} \sum_{i} w_i \, \lvert \Delta \mathbf{W}_i \rvert^2
$$

with the sums over the frames $i$ that carry the label and the atoms $j$ of a frame: $\Delta E_i$ the energy error per atom, $\Delta \mathbf{F}_j$ the force error of an atom, $\Delta \mathbf{W}_i$ the six independent virial components per atom, and $N_E$, $N_F$, $N_V$ the numbers of energy labels, force-labelled atoms and virial labels.

- The energy offset `b1` is the weighted mean energy residual, the optimum of this loss.
- The loss that selects `nep_best.txt` and drives the learning-rate schedule is weighted, on the validation set too if its frames carry weights.
- The errors TorchNEP reports (RMSE columns of `loss.out`, `*_train.out`, `*_test.out`) are not weighted: they stay the physical errors.
- The weight must be larger than 0. To leave a frame out, remove it from the file.

## Per-atom columns

`Properties=...` declares the column layout. TorchNEP reads three fields and ignores every other column (e.g. `Z:I:1`, `magmom:R:1`):

| Field | Meaning |
|---|---|
| `species:S:1` | Chemical symbol, case-sensitive; must be one of the elements on the `type` line of `nep.in`. |
| `pos:R:3` | Cartesian position in Å. |
| `force:R:3` or `forces:R:3` | Reference force in eV/Å (optional). |

## Validation data

A validation set is either a second file in the same format (`valid_file=`) or a fraction of the training file (`valid_ratio=`); see [Validation](../guide/training.md#validation).
