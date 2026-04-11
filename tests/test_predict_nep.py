import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import mdapy as mp
import numpy as np
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, '..'))
from torchnep import NEPCalculator


def test_nep(nepname=None, xyzname=None):
    if nepname is None:
        nepname = os.path.join(_DIR, 'input_files', 'nep.txt')
    if xyzname is None:
        xyzname = os.path.join(_DIR, 'input_files', 'test.xyz')
    # Reference values from mdapy (NEP_CPU)
    nep = mp.NEP(nepname)
    system = mp.System(xyzname)
    system.calc = nep

    e_mda = system.get_energies()
    f_mda = system.get_force()
    v_mda = system.get_virials()
    d_mda = nep.get_descriptor(system.data, system.box)

    # Our implementation
    from torchnep.data import read_xyz
    frames = read_xyz(xyzname)
    frame = frames[0]

    calc = NEPCalculator(nepname, device='cuda')
    result = calc.compute(
        frame['species'], frame['positions'], frame['cell'],
        compute_descriptor=True,
    )

    e_ours = result['energy'].cpu().numpy()
    f_ours = result['forces'].cpu().numpy()
    v_ours = result['virial'].cpu().numpy()
    d_ours = result['descriptor'].cpu().numpy()

    # Compare energies (per-atom)
    e_err = np.max(np.abs(e_ours - e_mda))
    print(f"Energy max error: {e_err:.2e}")
    assert e_err < 1e-6, f"Energy error too large: {e_err}"

    # Compare forces
    f_err = np.max(np.abs(f_ours - f_mda))
    print(f"Force max error:  {f_err:.2e}")
    assert f_err < 1e-6, f"Force error too large: {f_err}"

    # Compare virial (per-atom)
    v_err = np.max(np.abs(v_ours - v_mda))
    print(f"Virial max error: {v_err:.2e}")
    assert v_err < 1e-6, f"Virial error too large: {v_err}"

    # Compare descriptors
    d_err = np.max(np.abs(d_ours - d_mda))
    print(f"Descriptor max error: {d_err:.2e}")
    assert d_err < 1e-6, f"Descriptor error too large: {d_err}"

    print("All tests passed!")


if __name__ == '__main__':
    test_nep()
