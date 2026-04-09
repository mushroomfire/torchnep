"""
Trainable NEP4 model as a PyTorch nn.Module.

Supports per-type NN architecture with ZBL (including typewise cutoffs).
Uses ops module for core computations (pure PyTorch or CUDA).
"""

import torch
import torch.nn as nn
import numpy as np

from .constants import (
    ELEMENTS, PI, C3B, C4B, C5B, K_C_SP, ZBL_PARA, COVALENT_RADIUS,
)
from . import ops


class FittingNet(nn.Module):
    """Single-hidden-layer network: descriptor -> atomic energy.

    GPUMD convention: tanh(x @ W - b).
    """

    def __init__(self, input_dim: int, num_neurons: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.empty(input_dim, num_neurons))
        self.b0 = nn.Parameter(torch.zeros(num_neurons))
        self.w1 = nn.Parameter(torch.empty(num_neurons))
        nn.init.normal_(self.w0, std=1.0 / np.sqrt(input_dim + num_neurons))
        nn.init.normal_(self.w1, std=1.0 / np.sqrt(num_neurons + 1))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """q: (N, dim) -> (N,) per-atom energy (no bias; bias is shared in NEPModel)."""
        return torch.tanh(q @ self.w0 - self.b0) @ self.w1


class NEPModel(nn.Module):
    """Trainable NEP4 model.

    Parameters
    ----------
    config : dict
        From parse_nep_in(). Keys: cutoff_radial/angular, n_max_radial/angular,
        basis_size_radial/angular, l_max, neuron, num_types, type_names.
    energy_shift : array-like (num_types,)
        Per-type energy shift from training data.
    """

    def __init__(self, config: dict, energy_shift: np.ndarray = None):
        super().__init__()
        self.num_types = config["num_types"]
        self.type_names = config["type_names"]
        self.rc_radial = config["cutoff_radial"]
        self.rc_angular = config["cutoff_angular"]
        self.n_max_radial = config["n_max_radial"]
        self.n_max_angular = config["n_max_angular"]
        self.basis_size_radial = config["basis_size_radial"]
        self.basis_size_angular = config["basis_size_angular"]
        self.l_max_3b = config["l_max"][0]
        self.l_max_4b = config["l_max"][1]
        self.l_max_5b = config["l_max"][2] if len(config["l_max"]) > 2 else 0
        self.num_neurons = config["neuron"]

        # ZBL
        self.zbl = config.get("zbl", None)
        if self.zbl is not None:
            atomic_numbers = [ELEMENTS.index(n) for n in self.type_names]
            self.register_buffer("atomic_numbers",
                                 torch.tensor(atomic_numbers, dtype=torch.long))
            tw = config.get("typewise_cutoff_zbl_factor", None)
            if tw is not None:
                rc_i = [tw * COVALENT_RADIUS[z] for z in atomic_numbers]
                self.register_buffer("zbl_rc_inner_per_type", torch.tensor(rc_i))
                self.register_buffer("zbl_rc_outer_per_type",
                                     torch.tensor([2.0 * r for r in rc_i]))
                self.zbl_rc_inner = min(rc_i)
                self.zbl_rc_outer = max(2.0 * r for r in rc_i)
                self.zbl_typewise_factor = tw
            else:
                self.zbl_rc_inner = self.zbl / 2.0
                self.zbl_rc_outer = self.zbl
                self.zbl_typewise_factor = None

        # Dimensions
        self.dim_radial = self.n_max_radial + 1
        self.dim_angular_3b = (self.n_max_angular + 1) * self.l_max_3b
        self.dim_angular_4b = (self.n_max_angular + 1) if self.l_max_4b > 0 else 0
        self.dim_angular_5b = (self.n_max_angular + 1) if self.l_max_5b > 0 else 0
        self.dim = (self.dim_radial + self.dim_angular_3b +
                    self.dim_angular_4b + self.dim_angular_5b)
        self.num_lm = sum(2 * ll + 1 for ll in range(1, self.l_max_3b + 1))

        # Learnable c parameters
        nt = self.num_types
        self.c_param_2 = nn.Parameter(torch.empty(
            nt, nt, self.n_max_radial + 1, self.basis_size_radial + 1))
        self.c_param_3 = nn.Parameter(torch.empty(
            nt, nt, self.n_max_angular + 1, self.basis_size_angular + 1)
        ) if self.l_max_3b > 0 else None
        nn.init.normal_(self.c_param_2, std=0.1)
        if self.c_param_3 is not None:
            nn.init.normal_(self.c_param_3, std=0.1)

        # Per-type fitting networks (no per-type bias — shared b1 below)
        self.fitting_nets = nn.ModuleList([
            FittingNet(self.dim, self.num_neurons)
            for _ in range(self.num_types)
        ])
        # One shared output bias (GPUMD convention: single common b1)
        self.b1 = nn.Parameter(torch.tensor(0.0))

        # q_scaler (computed from data, not learned)
        self.register_buffer("q_scaler", torch.ones(self.dim))
        self.register_buffer("_c3b", torch.tensor(C3B[:self.num_lm]))
        self.register_buffer("_c4b", torch.tensor(C4B))
        self.register_buffer("_c5b", torch.tensor(C5B))

    @torch.no_grad()
    def set_q_scaler(self, q_min: torch.Tensor, q_max: torch.Tensor):
        diff = torch.clamp(q_max - q_min, min=1e-10)
        self.q_scaler.copy_(1.0 / diff)

    def compute_descriptors(self, rij_rad, rij_ang, pi_rad, pj_rad,
                            pi_ang, pj_ang, atom_types, N,
                            pytorch_only: bool = False):
        """Compute descriptors. Returns (N, dim)."""
        return ops.compute_descriptors(
            rij_rad, rij_ang, pi_rad, pj_rad, pi_ang, pj_ang,
            atom_types, N, self.c_param_2, self.c_param_3,
            self.rc_radial, self.rc_angular,
            self.basis_size_radial, self.basis_size_angular,
            self.n_max_radial, self.n_max_angular,
            self.l_max_3b, self.l_max_4b, self.l_max_5b,
            self.num_lm, self._c3b, self._c4b, self._c5b,
            rij_rad.dtype, rij_rad.device,
            pytorch_only=pytorch_only,
        )

    def forward(self, rij_rad, rij_ang, pi_rad, pj_rad,
                pi_ang, pj_ang, atom_types, N,
                pytorch_only: bool = False):
        """Forward pass: returns per-atom energy Ei (N,)."""
        q = self.compute_descriptors(
            rij_rad, rij_ang, pi_rad, pj_rad,
            pi_ang, pj_ang, atom_types, N,
            pytorch_only=pytorch_only)
        q_scaled = q * self.q_scaler
        Ei = torch.zeros(N, dtype=q.dtype, device=q.device)
        for t in range(self.num_types):
            mask = atom_types == t
            if mask.any():
                Ei[mask] = self.fitting_nets[t](q_scaled[mask])
        return Ei - self.b1  # subtract shared output bias

    def compute_properties(self, rij_rad, rij_ang, pi_rad, pj_rad,
                           pi_ang, pj_ang, atom_types, N,
                           struct_idx, num_structures,
                           need_forces=True, need_virial=False):
        """Compute energy, forces, virial."""
        dtype = rij_rad.dtype
        device = rij_rad.device

        if need_forces:
            rij_rad = rij_rad.detach().requires_grad_(True)
            rij_ang = rij_ang.detach().requires_grad_(True)

        # During training: pure PyTorch descriptors so gradients flow through
        # both rij (for forces) and c_param_2/c_param_3 (for descriptor params).
        # CUDA autograd.Functions are buggy for c3 gradients; PyTorch fallback
        # is fully differentiable and correct.
        use_pytorch_only = self.training
        Ei = self.forward(rij_rad, rij_ang, pi_rad, pj_rad,
                          pi_ang, pj_ang, atom_types, N,
                          pytorch_only=use_pytorch_only)

        if self.zbl is not None:
            Ei = Ei + ops.compute_zbl(
                atom_types, pi_ang, pj_ang, rij_ang, N,
                self.atomic_numbers.tolist(),
                self.zbl_rc_inner, self.zbl_rc_outer,
                self.zbl_typewise_factor,
                getattr(self, "zbl_rc_inner_per_type", None),
                getattr(self, "zbl_rc_outer_per_type", None),
                dtype, device)

        Etot = torch.zeros(num_structures, dtype=dtype, device=device)
        Etot.scatter_add_(0, struct_idx, Ei)

        result = {"Ei": Ei, "Etot": Etot}

        if need_forces:
            grads = torch.autograd.grad(
                Ei.sum(), [rij_rad, rij_ang],
                create_graph=self.training, allow_unused=True)
            g_rad = grads[0] if grads[0] is not None else torch.zeros_like(rij_rad)
            g_ang = grads[1] if grads[1] is not None else torch.zeros_like(rij_ang)

            rr = rij_rad if self.training else rij_rad.detach()
            ra = rij_ang if self.training else rij_ang.detach()
            gr = g_rad
            ga = g_ang

            if self.training:
                # Must use PyTorch scatter_add (differentiable) during training.
                # CUDA kernel breaks the autograd graph: forces would have no
                # gradient, so force/virial loss would give zero gradients.
                forces, virial = ops.accumulate_forces_virial(
                    N, pi_rad, pj_rad, rr, gr,
                    pi_ang, pj_ang, ra, ga, dtype, device)
            else:
                forces, virial = ops.accumulate_forces_virial_cuda(
                    N, pi_rad, pj_rad, rr, gr,
                    pi_ang, pj_ang, ra, ga, dtype, device)
            result["forces"] = forces
            if need_virial:
                result["virial"] = virial

        return result

    def compute_properties_cached(self, batch, need_forces=True, need_virial=False):
        """Compute energy, forces, virial using precomputed basis.

        Uses fully analytical force computation — no create_graph=True needed.
        Forces are differentiable through c2, c3 (via Fp→NN weights and via s→c3).
        """
        dtype = self.q_scaler.dtype
        device = self.q_scaler.device
        N = batch["N"]

        # Descriptors from cached basis.
        # When forces needed: return intermediates (s, gn_ang) for analytical force.
        if need_forces:
            q, s, gn_ang = ops.compute_descriptors_cached(
                batch["fk_rad"], batch["fk_ang"], batch["blm"],
                batch["pair_i_rad"], batch["pair_j_rad"],
                batch["pair_i_ang"], batch["pair_j_ang"],
                batch["atom_types"], N, self.c_param_2, self.c_param_3,
                self.n_max_radial, self.n_max_angular,
                self.l_max_3b, self.l_max_4b, self.l_max_5b,
                self.num_lm, self._c3b, self._c4b, self._c5b,
                dtype, device,
                return_intermediates=True,
            )
        else:
            q = ops.compute_descriptors_cached(
                batch["fk_rad"], batch["fk_ang"], batch["blm"],
                batch["pair_i_rad"], batch["pair_j_rad"],
                batch["pair_i_ang"], batch["pair_j_ang"],
                batch["atom_types"], N, self.c_param_2, self.c_param_3,
                self.n_max_radial, self.n_max_angular,
                self.l_max_3b, self.l_max_4b, self.l_max_5b,
                self.num_lm, self._c3b, self._c4b, self._c5b,
                dtype, device,
            )
            s = gn_ang = None

        q_scaled = q * self.q_scaler

        # NN forward + Fp computation (differentiable through NN weights)
        Ei = torch.zeros(N, dtype=dtype, device=device)
        Fp = torch.zeros(N, self.dim, dtype=dtype, device=device)

        for t in range(self.num_types):
            mask = batch["atom_types"] == t
            if not mask.any():
                continue
            net = self.fitting_nets[t]
            qt = q_scaled[mask]
            z = qt @ net.w0 - net.b0          # (Nt, neurons)
            h = torch.tanh(z)
            Ei[mask] = h @ net.w1
            # Fp = dEi/dq_scaled: backprop through tanh
            tanh_der = 1.0 - h * h            # (Nt, neurons)
            Fp[mask] = (net.w1 * tanh_der) @ net.w0.T  # (Nt, dim)

        Fp = Fp * self.q_scaler  # absorb q_scaler into Fp
        Ei = Ei - self.b1  # subtract shared output bias

        # ZBL (energy only — ZBL force not yet supported in cached path)
        if self.zbl is not None:
            Ei_zbl = ops.compute_zbl(
                batch["atom_types"], batch["pair_i_ang"], batch["pair_j_ang"],
                batch["rij_ang"], N, self.atomic_numbers.tolist(),
                self.zbl_rc_inner, self.zbl_rc_outer, self.zbl_typewise_factor,
                getattr(self, "zbl_rc_inner_per_type", None),
                getattr(self, "zbl_rc_outer_per_type", None), dtype, device)
            Ei = Ei + Ei_zbl

        Etot = torch.zeros(batch["num_structures"], dtype=dtype, device=device)
        Etot.scatter_add_(0, batch["struct_idx"], Ei)

        result = {"Ei": Ei, "Etot": Etot}

        if need_forces:
            # Analytical forces: fully differentiable through c2/c3 and NN weights (Fp).
            # No create_graph=True needed — chain rule is computed explicitly.
            forces, virial = ops.compute_analytical_forces(
                Fp, batch["atom_types"], N,
                self.c_param_2, self.c_param_3,
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
                compute_virial=need_virial,
            )
            result["forces"] = forces
            if need_virial and virial is not None:
                result["virial"] = virial

        return result

    def save_nep_txt(self, path: str, max_NN_radial: int = 0,
                     max_NN_angular: int = 0):
        """Save model to GPUMD nep4 nep.txt format."""
        lines = []
        zbl_suffix = "_zbl" if self.zbl is not None else ""
        lines.append(f"nep4{zbl_suffix} {self.num_types} "
                     + " ".join(self.type_names))

        if self.zbl is not None:
            tw = self.zbl_typewise_factor
            rc_inner_out = self.zbl / 2.0
            rc_outer_out = self.zbl
            if tw is not None:
                lines.append(f"zbl {rc_inner_out} {rc_outer_out} {tw}")
            else:
                lines.append(f"zbl {rc_inner_out} {rc_outer_out}")

        # Format cutoff: integer if whole number (matches GPUMD style)
        def _fmt(v):
            return str(int(v)) if v == int(v) else str(v)
        rc_str = f"cutoff {_fmt(self.rc_radial)} {_fmt(self.rc_angular)}"
        if max_NN_radial > 0:
            rc_str += f" {max_NN_radial} {max_NN_angular}"
        lines.append(rc_str)
        lines.append(f"n_max {self.n_max_radial} {self.n_max_angular}")
        lines.append(f"basis_size {self.basis_size_radial} "
                     f"{self.basis_size_angular}")
        lines.append(f"l_max {self.l_max_3b} {self.l_max_4b} {self.l_max_5b}")
        lines.append(f"ANN {self.num_neurons} 0")

        # Per-type NN weights
        for t in range(self.num_types):
            net = self.fitting_nets[t]
            w0 = net.w0.detach().cpu().numpy().T  # (neurons, dim)
            for v in w0.flat:
                lines.append(f"  {v:.10e}")
            for v in net.b0.detach().cpu().numpy():
                lines.append(f"  {v:.10e}")
            for v in net.w1.detach().cpu().numpy():
                lines.append(f"  {v:.10e}")

        # Common output bias (shared across all types — GPUMD convention)
        lines.append(f"  {self.b1.item():.10e}")

        # c2: stored as (n_max+1, basis+1, nt, nt)
        c2 = self.c_param_2.detach().cpu().numpy().transpose(2, 3, 0, 1)
        for v in c2.flat:
            lines.append(f"  {v:.10e}")
        if self.c_param_3 is not None:
            c3 = self.c_param_3.detach().cpu().numpy().transpose(2, 3, 0, 1)
            for v in c3.flat:
                lines.append(f"  {v:.10e}")

        for v in self.q_scaler.detach().cpu().numpy():
            lines.append(f"  {v:.10e}")

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
