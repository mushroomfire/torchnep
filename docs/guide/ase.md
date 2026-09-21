# ASE calculator

With ASE installed (`pip install torchnep[ase]`), a trained model drives any ASE workflow — relaxations, molecular dynamics, equations of state, phonons.

```python
from ase.io import read
from torchnep.ase_calculator import NEP

atoms = read("POSCAR")
atoms.calc = NEP("nep.txt", dtype="float32", device="cuda")

atoms.get_potential_energy()   # eV
atoms.get_forces()             # (N, 3) eV/Å
atoms.get_stress()             # Voigt 6-vector, eV/Å³ (periodic cells)
```

Non-periodic structures (molecules, clusters) work too: the calculator puts them in a box wide enough for the cutoff.

## Example: vacancy formation energy in nickel

The calculator is reused for every structure, so the model is loaded once. This example finds the equilibrium lattice constant of fcc Ni, then relaxes a 3×3×3 supercell with one atom removed:

```python
import numpy as np
from ase.build import bulk
from ase.optimize import BFGS
from torchnep.ase_calculator import NEP

calc = NEP("nep.txt", device="cuda")

# equilibrium lattice constant of this model
best_a, best_e = None, None
for a in np.arange(3.44, 3.60, 0.005):
    cell = bulk("Ni", "fcc", a=a, cubic=True)
    cell.calc = calc
    e = cell.get_potential_energy() / len(cell)
    if best_e is None or e < best_e:
        best_a, best_e = a, e

perfect = bulk("Ni", "fcc", a=best_a, cubic=True).repeat(3)
perfect.calc = calc
e_perfect = perfect.get_potential_energy()

atoms = perfect.copy()
del atoms[0]                       # one vacancy
atoms.calc = calc
n = len(atoms)
e_unrelaxed = atoms.get_potential_energy()
BFGS(atoms).run(fmax=0.01)

print(f"a0 = {best_a:.3f} A, E/atom = {best_e:.4f} eV")
print(f"unrelaxed E_vac = {e_unrelaxed - n / (n + 1) * e_perfect:.3f} eV")
print(f"relaxed   E_vac = {atoms.get_potential_energy() - n / (n + 1) * e_perfect:.3f} eV")
```

With a Cr-Co-Ni model trained on 3030 frames this prints:

```text
      Step     Time          Energy          fmax
BFGS:    0 21:10:33     -583.710611        0.243060
BFGS:    1 21:10:33     -583.721211        0.211663
BFGS:    2 21:10:33     -583.754787        0.047080
BFGS:    3 21:10:33     -583.755732        0.041422
BFGS:    4 21:10:33     -583.759054        0.014542
BFGS:    5 21:10:33     -583.759159        0.012571
BFGS:    6 21:10:33     -583.759458        0.004755
a0 = 3.515 A, E/atom = -5.4695 eV
unrelaxed E_vac = 1.523 eV
relaxed   E_vac = 1.474 eV
```

An equilibrium lattice constant of 3.515 Å (experiment: 3.524 Å) and a relaxed vacancy formation energy of 1.47 eV, both in the range expected for nickel.

## NEP and ZBL parts

```python
parts = atoms.calc.get_components()
parts["nep"]["energy"], parts["zbl"]["energy"], parts["total"]["energy"]
```

Each entry holds the energy, forces and, for periodic cells, the stress of that part; the NEP and ZBL parts add up to the total. `get_energy_components()` returns only the energies. For a model without ZBL, and for any structure whose atoms all sit beyond the ZBL cutoff, the ZBL part is zero.
