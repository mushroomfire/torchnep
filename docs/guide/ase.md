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

## NEP and ZBL parts

```python
parts = atoms.calc.get_components()
parts["nep"]["energy"], parts["zbl"]["energy"], parts["total"]["energy"]
```

Each entry holds the energy, forces and, for periodic cells, the stress of that part; the NEP and ZBL parts add up to the total. `get_energy_components()` returns only the energies. For a model without ZBL the ZBL part is zero.

## Example: relax a vacancy

```python
from ase.build import bulk
from ase.optimize import BFGS
from torchnep.ase_calculator import NEP

atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(3)
del atoms[0]
atoms.calc = NEP("nep.txt", device="cuda")
BFGS(atoms).run(fmax=0.01)
print(atoms.get_potential_energy())
```

Non-periodic structures (molecules, clusters) work too: the calculator places them in a box large enough for the cutoff.
