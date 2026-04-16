"""
NEP4 calculator for PyTorch.

Loads trained models from nep.txt (GPUMD NEP4 format) and computes
energy, forces, virial, and descriptors.

The descriptor/force computation is implemented in ops.py (pure PyTorch
by default, with optional CUDA kernel acceleration).
"""

import torch
import numpy as np
from typing import Dict

from .constants import ELEMENTS, COVALENT_RADIUS, C3B, C4B, C5B
from . import ops


class NEPCalculator:
    """NEP4 calculator: loads a trained model and computes atomic properties.

    Parameters
    ----------
    model_file : str
        Path to nep.txt (GPUMD NEP4 format).
    dtype : torch.dtype
        Precision (default: float64 for accuracy).
    device : str or torch.device
        Compute device (default: 'cpu').
    """

    def __init__(self, model_file: str, dtype=torch.float64, device="cpu"):
        self.dtype = dtype
        self.device = torch.device(device)
        self._load_model(model_file)

    def _load_model(self, path: str):
        with open(path) as f:
            lines = f.readlines()

        idx = 0

        # Line 1: version and types
        header = lines[idx].split()
        version_str = header[0]  # "nep4", "nep4_zbl", etc.
        self.has_zbl = "zbl" in version_str
        self.num_types = int(header[1])
        self.type_names = header[2 : 2 + self.num_types]
        # Real atomic numbers (Z = 1 for H, 6 for C, ...). ELEMENTS is
        # 0-indexed, so we add 1. ZBL uses these directly (Z*Z', Z^0.23).
        self.atomic_numbers = [ELEMENTS.index(n) + 1 for n in self.type_names]
        idx += 1

        # ZBL line
        if self.has_zbl:
            parts = lines[idx].split()
            self.zbl_rc_inner = float(parts[1])
            self.zbl_rc_outer = float(parts[2])
            self.zbl_typewise_factor = float(parts[3]) if len(parts) > 3 else None
            idx += 1

            if self.zbl_typewise_factor is not None:
                # Per-type cutoffs. COVALENT_RADIUS is 0-indexed (H at 0),
                # while self.atomic_numbers holds real Z (H=1), hence z-1.
                self.zbl_rc_inner_per_type = torch.tensor(
                    [self.zbl_typewise_factor * COVALENT_RADIUS[z - 1]
                     for z in self.atomic_numbers],
                    dtype=self.dtype, device=self.device)
                self.zbl_rc_outer_per_type = 2.0 * self.zbl_rc_inner_per_type
        else:
            self.zbl_rc_inner = None
            self.zbl_rc_outer = None
            self.zbl_typewise_factor = None

        # Cutoff
        parts = lines[idx].split()
        self.rc_radial = float(parts[1])
        self.rc_angular = float(parts[2])
        idx += 1

        # n_max, basis_size, l_max
        parts = lines[idx].split()
        self.n_max_radial = int(parts[1])
        self.n_max_angular = int(parts[2])
        idx += 1

        parts = lines[idx].split()
        self.basis_size_radial = int(parts[1])
        self.basis_size_angular = int(parts[2])
        idx += 1

        parts = lines[idx].split()
        self.l_max_3b = int(parts[1])
        self.l_max_4b = int(parts[2])
        self.l_max_5b = int(parts[3])
        idx += 1

        # ANN
        parts = lines[idx].split()
        self.num_neurons = int(parts[1])
        idx += 1

        # Descriptor dimension
        self.dim_radial = self.n_max_radial + 1
        self.dim_angular_3b = (self.n_max_angular + 1) * self.l_max_3b
        self.dim_angular_4b = (self.n_max_angular + 1) if self.l_max_4b > 0 else 0
        self.dim_angular_5b = (self.n_max_angular + 1) if self.l_max_5b > 0 else 0
        self.dim = (self.dim_radial + self.dim_angular_3b +
                    self.dim_angular_4b + self.dim_angular_5b)
        self.num_lm = sum(2 * ll + 1 for ll in range(1, self.l_max_3b + 1))

        # Parse data
        data = [float(l) for l in lines[idx:] if l.strip()]
        di = 0
        n = self.num_neurons
        d = self.dim

        # Per-type NN weights
        self.w0, self.b0, self.w1 = [], [], []
        for t in range(self.num_types):
            w0 = np.array(data[di:di + n * d]).reshape(n, d)
            di += n * d
            b0 = np.array(data[di:di + n])
            di += n
            w1 = np.array(data[di:di + n])
            di += n
            self.w0.append(torch.tensor(w0, dtype=self.dtype, device=self.device))
            self.b0.append(torch.tensor(b0, dtype=self.dtype, device=self.device))
            self.w1.append(torch.tensor(w1, dtype=self.dtype, device=self.device))

        # Common output bias
        self.b1 = torch.tensor(data[di], dtype=self.dtype, device=self.device)
        di += 1

        # c parameters
        nt2 = self.num_types ** 2
        c2_size = (self.n_max_radial + 1) * (self.basis_size_radial + 1) * nt2
        c2 = np.array(data[di:di + c2_size]).reshape(
            self.n_max_radial + 1, self.basis_size_radial + 1,
            self.num_types, self.num_types)
        # save_nep_txt stores c2 transposed as (n_max+1, basis+1, nt, nt);
        # ops.compute_descriptors expects (nt, nt, n_max+1, basis+1).
        self.c2 = torch.tensor(c2, dtype=self.dtype, device=self.device).permute(2, 3, 0, 1).contiguous()
        di += c2_size

        if self.l_max_3b > 0:
            c3_size = (self.n_max_angular + 1) * (self.basis_size_angular + 1) * nt2
            c3 = np.array(data[di:di + c3_size]).reshape(
                self.n_max_angular + 1, self.basis_size_angular + 1,
                self.num_types, self.num_types)
            self.c3 = torch.tensor(c3, dtype=self.dtype, device=self.device).permute(2, 3, 0, 1).contiguous()
            di += c3_size

        # q_scaler
        self.q_scaler = torch.tensor(
            data[di:di + self.dim], dtype=self.dtype, device=self.device)
        di += self.dim

        # Pre-build constants
        self._c3b = torch.tensor(C3B[:self.num_lm], dtype=self.dtype, device=self.device)
        self._c4b = torch.tensor(C4B, dtype=self.dtype, device=self.device)
        self._c5b = torch.tensor(C5B, dtype=self.dtype, device=self.device)

    def compute(
        self,
        species: list,
        positions: np.ndarray,
        cell: np.ndarray,
        compute_descriptor: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute energy, forces, and per-atom virial.

        Parameters
        ----------
        species : list of str
            Element symbols for each atom.
        positions : (N, 3) array
            Atomic positions in Angstrom.
        cell : (3, 3) array
            Lattice vectors (row-major).
        compute_descriptor : bool
            Also return scaled descriptors.

        Returns
        -------
        dict with 'energy', 'forces', 'virial', optionally 'descriptor'.
        """
        atom_types = torch.tensor(
            [self.type_names.index(s) for s in species],
            dtype=torch.long, device=self.device)
        pos = torch.tensor(positions, dtype=self.dtype, device=self.device)
        cell_t = torch.tensor(cell, dtype=self.dtype, device=self.device)
        N = pos.shape[0]

        # Neighbor list
        max_rc = max(self.rc_radial, self.rc_angular)
        pair_i, pair_j, rij, dij = ops.build_neighbor_list(
            pos, cell_t, max_rc, self.device, self.dtype)

        rad_mask = dij < self.rc_radial
        ang_mask = dij < self.rc_angular

        rij_rad = rij[rad_mask].detach().requires_grad_(True)
        rij_ang = rij[ang_mask].detach().requires_grad_(True)
        pi_rad, pj_rad = pair_i[rad_mask], pair_j[rad_mask]
        pi_ang, pj_ang = pair_i[ang_mask], pair_j[ang_mask]

        # Descriptors
        q = ops.compute_descriptors(
            rij_rad, rij_ang, pi_rad, pj_rad, pi_ang, pj_ang,
            atom_types, N, self.c2, self.c3,
            self.rc_radial, self.rc_angular,
            self.basis_size_radial, self.basis_size_angular,
            self.n_max_radial, self.n_max_angular,
            self.l_max_3b, self.l_max_4b, self.l_max_5b,
            self.num_lm, self._c3b, self._c4b, self._c5b,
            self.dtype, self.device,
        )

        descriptor = (q * self.q_scaler).detach() if compute_descriptor else None
        q_scaled = q * self.q_scaler

        # NN
        Ei = ops.apply_ann(q_scaled, atom_types, self.num_types,
                           self.w0, self.b0, self.w1, self.b1,
                           self.dtype, self.device)

        # ZBL
        if self.has_zbl:
            Ei_zbl = ops.compute_zbl(
                atom_types, pi_ang, pj_ang, rij_ang, N,
                self.atomic_numbers, self.zbl_rc_inner, self.zbl_rc_outer,
                self.zbl_typewise_factor,
                getattr(self, "zbl_rc_inner_per_type", None),
                getattr(self, "zbl_rc_outer_per_type", None),
                self.dtype, self.device,
            )
            Ei = Ei + Ei_zbl

        E_total = Ei.sum()

        # Forces & virial via autograd
        grads = torch.autograd.grad(
            E_total, [rij_rad, rij_ang], allow_unused=True)
        g_rad = grads[0] if grads[0] is not None else torch.zeros_like(rij_rad)
        g_ang = grads[1] if grads[1] is not None else torch.zeros_like(rij_ang)

        forces, virial = ops.accumulate_forces_virial(
            N, pi_rad, pj_rad, rij_rad.detach(), g_rad.detach(),
            pi_ang, pj_ang, rij_ang.detach(), g_ang.detach(),
            self.dtype, self.device,
        )

        result = {"energy": Ei.detach(), "forces": forces, "virial": virial}
        if descriptor is not None:
            result["descriptor"] = descriptor
        return result

    def compute_batch(self, batch: Dict) -> Dict:
        """Compute energy, forces, virial for a pre-built batch dict.

        The batch dict must contain pre-cached basis tensors on the device:
        fk_rad, fkp_rad, d12inv_rad, fk_ang, fkp_ang, d12inv_ang, blm,
        pair_i_rad, pair_j_rad, rij_rad, pair_i_ang, pair_j_ang, rij_ang,
        atom_types, struct_idx, N, num_structures.
        """
        dtype, device = self.dtype, self.device
        N = batch["N"]

        # Descriptors from pre-cached basis
        q, s, gn_ang = ops.compute_descriptors_cached(
            batch["fk_rad"], batch["fk_ang"], batch["blm"],
            batch["pair_i_rad"], batch["pair_j_rad"],
            batch["pair_i_ang"], batch["pair_j_ang"],
            batch["atom_types"], N,
            self.c2, getattr(self, "c3", None),
            self.n_max_radial, self.n_max_angular,
            self.l_max_3b, self.l_max_4b, self.l_max_5b,
            self.num_lm, self._c3b, self._c4b, self._c5b,
            dtype, device,
            return_intermediates=True,
            pytorch_only=True,
        )

        q_scaled = q * self.q_scaler

        # NN forward: compute Ei and Fp = dEi/dq_scaled
        Ei = torch.zeros(N, dtype=dtype, device=device)
        Fp = torch.zeros(N, self.dim, dtype=dtype, device=device)
        for t in range(self.num_types):
            mask = batch["atom_types"] == t
            if not mask.any():
                continue
            qt = q_scaled[mask]
            # w0[t]: (neurons, dim), b0[t]: (neurons,), w1[t]: (neurons,)
            z = qt @ self.w0[t].T - self.b0[t]
            h = torch.tanh(z)
            Ei[mask] = h @ self.w1[t]
            tanh_der = 1.0 - h * h
            Fp[mask] = (self.w1[t] * tanh_der) @ self.w0[t]

        Fp = Fp * self.q_scaler
        Ei = Ei - self.b1

        # ZBL correction (energy + forces/virial via local autograd on rij_ang).
        # enable_grad: predict_dataset wraps this call in torch.no_grad(),
        # but ZBL forces need a local autograd pass.
        zbl_forces = None
        zbl_virial = None
        if self.has_zbl:
            with torch.enable_grad():
                rij_zbl = batch["rij_ang"].detach().requires_grad_(True)
                Ei_zbl = ops.compute_zbl(
                    batch["atom_types"], batch["pair_i_ang"],
                    batch["pair_j_ang"], rij_zbl, N,
                    self.atomic_numbers,
                    self.zbl_rc_inner, self.zbl_rc_outer,
                    self.zbl_typewise_factor,
                    getattr(self, "zbl_rc_inner_per_type", None),
                    getattr(self, "zbl_rc_outer_per_type", None),
                    dtype, device,
                )
                if Ei_zbl.requires_grad:
                    g_zbl = torch.autograd.grad(
                        Ei_zbl.sum(), rij_zbl, allow_unused=True)[0]
                else:
                    g_zbl = None
            Ei = Ei + Ei_zbl.detach()
            if g_zbl is not None:
                empty_i = torch.zeros(0, dtype=torch.long, device=device)
                empty_r = torch.zeros(0, 3, dtype=dtype, device=device)
                zbl_forces, zbl_virial = ops.accumulate_forces_virial(
                    N, empty_i, empty_i, empty_r, empty_r,
                    batch["pair_i_ang"], batch["pair_j_ang"],
                    batch["rij_ang"].detach(), g_zbl.detach(),
                    dtype, device,
                )

        # Per-structure totals
        Etot = torch.zeros(batch["num_structures"], dtype=dtype, device=device)
        Etot.scatter_add_(0, batch["struct_idx"], Ei)

        # Analytical forces + virial
        forces, virial = ops.compute_analytical_forces(
            Fp, batch["atom_types"], N,
            self.c2, getattr(self, "c3", None),
            batch["fkp_rad"], batch["fkp_ang"], batch["blm"],
            batch["pair_i_rad"], batch["pair_j_rad"],
            batch["rij_rad"], batch["d12inv_rad"],
            batch["pair_i_ang"], batch["pair_j_ang"],
            batch["rij_ang"], batch["d12inv_ang"],
            s, gn_ang,
            self.n_max_radial, self.n_max_angular,
            self.l_max_3b, self.l_max_4b, self.l_max_5b,
            self.num_lm, self._c3b, self._c4b, self._c5b,
            dtype, device,
            compute_virial=True,
            pytorch_only=True,
        )
        if zbl_forces is not None:
            forces = forces + zbl_forces
            if zbl_virial is not None:
                virial = virial + zbl_virial

        return {"Ei": Ei, "Etot": Etot, "forces": forces, "virial": virial}

    def get_descriptor(self, species, positions, cell):
        """Compute scaled descriptors. Returns (N, dim) numpy."""
        return self.compute(species, positions, cell,
                            compute_descriptor=True)["descriptor"].cpu().numpy()
