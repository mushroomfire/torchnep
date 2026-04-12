"""
Core NEP operations.

Pure PyTorch implementations with optional CUDA kernel acceleration.
On CUDA-capable systems, fused kernels are JIT-compiled on first use
for force/virial accumulation. The descriptor computation uses the
PyTorch implementation (supports autograd for training).

Falls back to pure PyTorch on CPU/Mac — no CUDA required.
"""

import os
import sys
import torch
from typing import List, Optional, Tuple

from .constants import PI, K_C_SP, ZBL_PARA


# ---------------------------------------------------------------------------
# Neighbor list
# ---------------------------------------------------------------------------

def build_neighbor_list(
    positions: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build periodic neighbor list.

    Returns: pair_i, pair_j, rij (P,3), dij (P,).
    """
    N = positions.shape[0]
    inv_cell = torch.linalg.inv(cell)

    n_rep = [int(torch.ceil(cutoff / (1.0 / torch.norm(inv_cell[i]))).item())
             for i in range(3)]

    ranges = [torch.arange(-n, n + 1, device=device, dtype=dtype) for n in n_rep]
    grid = torch.stack(torch.meshgrid(*ranges, indexing="ij"), dim=-1)
    shifts_frac = grid.reshape(-1, 3)
    shifts_cart = shifts_frac @ cell

    S = shifts_cart.shape[0]
    disp = (positions[None, :, None, :] + shifts_cart[None, None, :, :]
            - positions[:, None, None, :])
    dist = torch.norm(disp, dim=-1)

    zero_shift = shifts_frac.abs().sum(-1) == 0
    self_mask = torch.eye(N, dtype=torch.bool, device=device)[:, :, None]
    exclude = self_mask & zero_shift[None, None, :]
    valid = (dist < cutoff) & (dist > 1e-10) & ~exclude

    idx_i, idx_j, idx_s = torch.where(valid)
    return idx_i, idx_j, disp[idx_i, idx_j, idx_s], dist[idx_i, idx_j, idx_s]


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------

def chebyshev_basis(dij: torch.Tensor, rc: float,
                    basis_size: int) -> torch.Tensor:
    """Chebyshev radial basis: f_k(r) = 0.5*(T_k(x)+1)*fc(r).

    Returns: (P, basis_size+1).
    """
    rcinv = 1.0 / rc
    fc = 0.5 * torch.cos(PI * dij * rcinv) + 0.5
    x = 2.0 * (dij * rcinv - 1.0) ** 2 - 1.0

    T = [torch.ones_like(dij), x]
    for k in range(2, basis_size + 1):
        T.append(2.0 * x * T[-1] - T[-2])
    T = torch.stack(T, dim=-1)
    return 0.5 * (T + 1.0) * fc.unsqueeze(-1)


def chebyshev_basis_and_deriv(dij: torch.Tensor, rc: float,
                              basis_size: int):
    """Compute Chebyshev basis AND its derivative wrt distance.

    Returns: fk (P, basis_size+1), fkp (P, basis_size+1).
    fkp[k] = d(fk[k])/d(rij).
    """
    rcinv = 1.0 / rc
    fc = 0.5 * torch.cos(PI * dij * rcinv) + 0.5
    fcp = -0.5 * PI * rcinv * torch.sin(PI * dij * rcinv)
    x = 2.0 * (dij * rcinv - 1.0) ** 2 - 1.0
    dxdr = 4.0 * (dij * rcinv - 1.0) * rcinv  # dx/dr

    # Chebyshev T and U (second kind for derivatives)
    T = [torch.ones_like(dij)]
    U = [torch.ones_like(dij)]  # U_{-1} = 1, but we use U for derivative
    T.append(x)
    U.append(2.0 * x)
    for k in range(2, basis_size + 1):
        T.append(2.0 * x * T[-1] - T[-2])
        U.append(2.0 * x * U[-1] - U[-2])

    fk_list, fkp_list = [], []
    for k in range(basis_size + 1):
        fn_core = 0.5 * (T[k] + 1.0)
        fk_list.append(fn_core * fc)
        if k == 0:
            fkp_list.append(fcp)  # d(fc)/dr
        elif k == 1:
            # d/dr [0.5*(x+1)*fc] = 0.5*dx/dr*fc + 0.5*(x+1)*fcp
            fkp_list.append(0.5 * dxdr * fc + fn_core * fcp)
        else:
            # d/dr [fn_core * fc] = dT/dx * dx/dr * 0.5 * fc + fn_core * fcp
            # dT_k/dx = k * U_{k-1} (Chebyshev derivative formula)
            dTdx = k * U[k - 1]
            fkp_list.append(0.5 * dTdx * dxdr * fc + fn_core * fcp)

    return torch.stack(fk_list, dim=-1), torch.stack(fkp_list, dim=-1)


def angular_basis(x: torch.Tensor, y: torch.Tensor,
                  z: torch.Tensor, l_max_3b: int) -> torch.Tensor:
    """Solid harmonics basis Y_{lm}. Returns: (P, num_lm)."""
    x2, y2, z2 = x * x, y * y, z * z
    x2my2 = x2 - y2
    blm = []
    if l_max_3b >= 1:
        blm.extend([z, x, y])
    if l_max_3b >= 2:
        blm.extend([3.0*z2 - 1.0, x*z, y*z, x2my2, 2.0*x*y])
    if l_max_3b >= 3:
        blm.extend([
            (5.0*z2 - 3.0)*z, (5.0*z2 - 1.0)*x, (5.0*z2 - 1.0)*y,
            x2my2*z, 2.0*x*y*z, x*(x2 - 3.0*y2), y*(3.0*x2 - y2),
        ])
    if l_max_3b >= 4:
        blm.extend([
            (35.0*z2 - 30.0)*z2 + 3.0, (7.0*z2 - 3.0)*x*z,
            (7.0*z2 - 3.0)*y*z, (7.0*z2 - 1.0)*x2my2,
            (7.0*z2 - 1.0)*2.0*x*y, z*x*(x2 - 3.0*y2),
            z*y*(3.0*x2 - y2), x2my2*x2my2 - 4.0*x2*y2,
            4.0*x*y*x2my2,
        ])
    return torch.stack(blm, dim=-1)


# ---------------------------------------------------------------------------
# Descriptor computation
# ---------------------------------------------------------------------------

def compute_descriptors(
    rij_rad, rij_ang, pi_rad, pj_rad, pi_ang, pj_ang,
    atom_types, N, c2, c3,
    rc_radial, rc_angular, basis_size_radial, basis_size_angular,
    n_max_radial, n_max_angular, l_max_3b, l_max_4b, l_max_5b,
    num_lm, c3b_coeffs, c4b_coeffs, c5b_coeffs,
    dtype, device,
    pytorch_only: bool = False,
) -> torch.Tensor:
    """Compute NEP4 descriptors. Returns (N, dim).

    pytorch_only=True forces pure-PyTorch path (fully differentiable w.r.t.
    both rij and c params — required for force/virial training).
    """
    ntypes = int(c2.shape[0]) if c2.shape[0] != n_max_radial + 1 else int(c2.shape[2])

    # --- Radial (CUDA-accelerated when available) ---
    use_cuda_rad = (not pytorch_only and rij_rad.is_cuda and c2.shape[0] == ntypes
                    and c2.shape[0] != n_max_radial + 1)
    if use_cuda_rad:
        try:
            from .cuda_ops import radial_descriptor_cuda
            q_rad = radial_descriptor_cuda(
                rij_rad, pi_rad, pj_rad, atom_types, c2,
                N, n_max_radial, basis_size_radial, ntypes, rc_radial)
        except Exception:
            use_cuda_rad = False

    if not use_cuda_rad:
        dij_rad = torch.norm(rij_rad, dim=-1)
        fk_rad = chebyshev_basis(dij_rad, rc_radial, basis_size_radial)
        t1r = atom_types[pi_rad]
        t2r = atom_types[pj_rad]
        # Type-pair matmul loop: avoids atomic scatter in backward (38x faster)
        q_rad = torch.zeros(N, n_max_radial + 1, dtype=dtype, device=device)
        for _t1 in range(ntypes):
            for _t2 in range(ntypes):
                _m = (t1r == _t1) & (t2r == _t2)
                if not _m.any():
                    continue
                _gn = fk_rad[_m] @ c2[_t1, _t2].T
                q_rad.scatter_add_(0, pi_rad[_m].unsqueeze(-1).expand_as(_gn), _gn)

    parts = [q_rad]

    # --- Angular (CUDA-accelerated when available) ---
    n_ap1 = n_max_angular + 1
    num_L_total = l_max_3b + (1 if l_max_4b > 0 else 0) + (1 if l_max_5b > 0 else 0)

    if l_max_3b > 0 and rij_ang.shape[0] == 0:
        # No angular neighbors (e.g. isolated atom / dimer). Emit zero-filled
        # angular blocks so output dim still matches q_scaler.
        parts.append(torch.zeros(N, n_ap1 * l_max_3b, dtype=dtype, device=device))
        if l_max_4b > 0:
            parts.append(torch.zeros(N, n_ap1, dtype=dtype, device=device))
        if l_max_5b > 0:
            parts.append(torch.zeros(N, n_ap1, dtype=dtype, device=device))

    if l_max_3b > 0 and rij_ang.shape[0] > 0:
        # Try CUDA angular descriptor (autograd.Function with CUDA forward+backward)
        use_cuda_ang = (not pytorch_only and rij_ang.is_cuda and c3 is not None
                        and c3.shape[0] == ntypes)
        if use_cuda_ang:
            try:
                from .cuda_ops import angular_descriptor_cuda
                q_ang = angular_descriptor_cuda(
                    rij_ang, pi_ang, pj_ang, atom_types, c3,
                    N, n_ap1, basis_size_angular, ntypes, num_lm,
                    l_max_3b, num_L_total, rc_angular,
                    c3b_coeffs, c4b_coeffs, c5b_coeffs)
                parts.append(q_ang)
                use_cuda_ang = True
            except Exception:
                use_cuda_ang = False

        if not use_cuda_ang:
            # PyTorch fallback
            dij_ang = torch.norm(rij_ang, dim=-1)
            fk_ang = chebyshev_basis(dij_ang, rc_angular, basis_size_angular)
            t1a = atom_types[pi_ang]
            t2a = atom_types[pj_ang]
            # Type-pair matmul loop (same optimization as radial)
            gn_ang = torch.zeros(rij_ang.shape[0], n_ap1, dtype=dtype, device=device)
            for _t1 in range(ntypes):
                for _t2 in range(ntypes):
                    _m = (t1a == _t1) & (t2a == _t2)
                    if not _m.any():
                        continue
                    gn_ang[_m] = fk_ang[_m] @ c3[_t1, _t2].T
            d12inv = 1.0 / torch.clamp(dij_ang, min=1e-10)
            blm = angular_basis(rij_ang[:, 0]*d12inv, rij_ang[:, 1]*d12inv,
                                rij_ang[:, 2]*d12inv, l_max_3b)
            gn_blm = gn_ang.unsqueeze(-1) * blm.unsqueeze(1)
            s = torch.zeros(N, n_ap1, num_lm, dtype=dtype, device=device)
            s.scatter_add_(0, pi_ang.unsqueeze(-1).unsqueeze(-1).expand_as(gn_blm),
                           gn_blm)

            # 3-body q (PyTorch fallback path)
            q_3b_list = []
            for li in range(l_max_3b):
                L = li + 1
                nt = 2 * L + 1
                st = L * L - 1
                c = c3b_coeffs[st:st + nt]
                sb2 = s[:, :, st:st + nt] ** 2
                ql = c[0] * sb2[:, :, 0]
                if nt > 1:
                    ql = ql + 2.0 * (c[1:] * sb2[:, :, 1:]).sum(-1)
                q_3b_list.append(ql)
            q_3b = torch.stack(q_3b_list, dim=-1).transpose(1, 2).reshape(N, -1)
            parts.append(q_3b)

            # 4-body
            if l_max_4b > 0:
                s20, s21r, s21i = s[:, :, 3], s[:, :, 4], s[:, :, 5]
                s22r, s22i = s[:, :, 6], s[:, :, 7]
                cb = c4b_coeffs
                q4 = (cb[0]*s20**3
                      + cb[1]*s20*(s21r**2 + s21i**2)
                      + cb[2]*s20*(s22r**2 + s22i**2)
                      + cb[3]*s22r*(s21i**2 - s21r**2)
                      + cb[4]*s21r*s21i*s22i)
                parts.append(q4)

            # 5-body
            if l_max_5b > 0:
                s0sq = s[:, :, 0] ** 2
                s1sq = s[:, :, 1] ** 2 + s[:, :, 2] ** 2
                cb = c5b_coeffs
                q5 = cb[0]*s0sq**2 + cb[1]*s0sq*s1sq + cb[2]*s1sq**2
                parts.append(q5)

    return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

def apply_ann(
    q: torch.Tensor,
    atom_types: torch.Tensor,
    num_types: int,
    w0_list: List[torch.Tensor],
    b0_list: List[torch.Tensor],
    w1_list: List[torch.Tensor],
    b1: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Per-type NN: tanh(q @ W0^T - b0) @ w1 - b1. Returns (N,)."""
    N = q.shape[0]
    Ei = torch.zeros(N, dtype=dtype, device=device)
    for t in range(num_types):
        mask = atom_types == t
        if not mask.any():
            continue
        qt = q[mask]
        hidden = torch.tanh(qt @ w0_list[t].T - b0_list[t])
        Ei[mask] = hidden @ w1_list[t] - b1
    return Ei


# ---------------------------------------------------------------------------
# ZBL
# ---------------------------------------------------------------------------

def compute_zbl(
    atom_types, pair_i, pair_j, rij, N,
    atomic_numbers_list, rc_inner_default, rc_outer_default,
    typewise_factor, rc_inner_per_type, rc_outer_per_type,
    dtype, device,
) -> torch.Tensor:
    """ZBL repulsive energy with optional typewise cutoffs. Returns (N,)."""
    dij = torch.norm(rij, dim=-1)
    use_tw = typewise_factor is not None and rc_inner_per_type is not None

    if use_tw:
        t1 = atom_types[pair_i]
        t2 = atom_types[pair_j]
        rc_inner = (rc_inner_per_type[t1] + rc_inner_per_type[t2]) * 0.5
        rc_outer = (rc_outer_per_type[t1] + rc_outer_per_type[t2]) * 0.5
        max_rc = rc_outer_per_type.max().item()
    else:
        rc_inner = rc_inner_default
        rc_outer = rc_outer_default
        max_rc = rc_outer_default

    mask = dij < max_rc
    if not mask.any():
        return torch.zeros(N, dtype=dtype, device=device)

    pi, pj = pair_i[mask], pair_j[mask]
    d = dij[mask]

    an = torch.tensor(atomic_numbers_list, dtype=dtype, device=device)
    zi = an[atom_types[pi]]
    zj = an[atom_types[pj]]

    a_inv = (zi ** 0.23 + zj ** 0.23) / 0.46850
    zizj = K_C_SP * zi * zj
    x = d * a_inv
    phi = (ZBL_PARA[0] * torch.exp(-ZBL_PARA[1] * x)
           + ZBL_PARA[2] * torch.exp(-ZBL_PARA[3] * x)
           + ZBL_PARA[4] * torch.exp(-ZBL_PARA[5] * x)
           + ZBL_PARA[6] * torch.exp(-ZBL_PARA[7] * x))

    if use_tw:
        rc_i = rc_inner[mask]
        rc_o = rc_outer[mask]
    else:
        rc_i = rc_inner
        rc_o = rc_outer

    fc = torch.zeros_like(d)
    m1 = d < rc_i
    fc[m1] = 1.0
    m2 = (~m1) & (d < rc_o)
    if m2.any():
        ri = rc_i[m2] if use_tw else rc_i
        ro = rc_o[m2] if use_tw else rc_o
        fc[m2] = 0.5 * torch.cos(PI / (ro - ri) * (d[m2] - ri)) + 0.5

    e_pair = zizj * phi / d * fc
    e_atom = torch.zeros(N, dtype=dtype, device=device)
    e_atom.scatter_add_(0, pi, 0.5 * e_pair)
    e_atom.scatter_add_(0, pj, 0.5 * e_pair)
    return e_atom


# ---------------------------------------------------------------------------
# Precomputation for analytical force training
# ---------------------------------------------------------------------------

def precompute_basis(rij, dij, rc, basis_size, l_max_3b=0):
    """Precompute Chebyshev basis, derivative, and angular basis.

    Call once at startup for fixed geometry. Returns dict of cached tensors.
    """
    fk, fkp = chebyshev_basis_and_deriv(dij, rc, basis_size)
    result = {"fk": fk, "fkp": fkp, "dij": dij, "d12inv": 1.0 / dij}

    if l_max_3b > 0:
        d12inv = result["d12inv"]
        x_hat = rij[:, 0] * d12inv
        y_hat = rij[:, 1] * d12inv
        z_hat = rij[:, 2] * d12inv
        result["blm"] = angular_basis(x_hat, y_hat, z_hat, l_max_3b)
        result["x_hat"] = x_hat
        result["y_hat"] = y_hat
        result["z_hat"] = z_hat

    return result


def _scatter_contraction_pytorch(basis, pair_i, pair_j, atom_types, c, N):
    """Pure PyTorch: type-contraction + scatter_add. Fully differentiable via autograd."""
    ntypes = c.shape[0]
    t1 = atom_types[pair_i]
    t2 = atom_types[pair_j]
    q = torch.zeros(N, c.shape[2], dtype=basis.dtype, device=basis.device)
    for _t1 in range(ntypes):
        for _t2 in range(ntypes):
            _m = (t1 == _t1) & (t2 == _t2)
            if not _m.any():
                continue
            _gn = basis[_m] @ c[_t1, _t2].T
            q.scatter_add_(0, pair_i[_m].unsqueeze(-1).expand_as(_gn), _gn)
    return q


def _type_contraction_pytorch(basis, pair_i, pair_j, atom_types, c):
    """Pure PyTorch: type-pair contraction (no scatter). Fully differentiable via autograd."""
    ntypes = c.shape[0]
    t1 = atom_types[pair_i]
    t2 = atom_types[pair_j]
    gn = torch.zeros(basis.shape[0], c.shape[2], dtype=basis.dtype, device=basis.device)
    for _t1 in range(ntypes):
        for _t2 in range(ntypes):
            _m = (t1 == _t1) & (t2 == _t2)
            if not _m.any():
                continue
            gn[_m] = basis[_m] @ c[_t1, _t2].T
    return gn


def compute_descriptors_cached(
    fk_rad, fk_ang, blm,
    pi_rad, pj_rad, pi_ang, pj_ang,
    atom_types, N, c2, c3,
    n_max_radial, n_max_angular, l_max_3b, l_max_4b, l_max_5b,
    num_lm, c3b_coeffs, c4b_coeffs, c5b_coeffs,
    dtype, device,
    return_intermediates: bool = False,
    pytorch_only: bool = False,
):
    """Compute descriptors using precomputed basis functions.

    Faster than compute_descriptors because Chebyshev/angular basis are cached.
    Differentiable through c2, c3 only (not rij).

    When pytorch_only=True, uses pure PyTorch ops (no custom autograd.Function).
    When pytorch_only=False, uses CUDA-accelerated type-pair contraction kernels
    when available, falling back to vectorized PyTorch otherwise.

    If return_intermediates=True, returns (q, s, gn_ang) where:
      s      : (N, n_ap1, num_lm) sum_fxyz moments — needed for analytical forces
      gn_ang : (P_ang, n_ap1) pair-level angular radial factor
    """
    if pytorch_only:
        _scatter_fn = _scatter_contraction_pytorch
        _type_fn = _type_contraction_pytorch
    else:
        from .cuda_ops import scatter_contraction, type_contraction
        _scatter_fn = scatter_contraction
        _type_fn = type_contraction

    ntypes = c2.shape[0]

    # Radial descriptor: type-lookup + contraction + scatter
    q_rad = _scatter_fn(fk_rad, pi_rad, pj_rad, atom_types, c2, N)

    parts = [q_rad]
    s_out = None
    gn_ang_out = None

    if l_max_3b > 0 and fk_ang.shape[0] > 0:
        n_ap1 = n_max_angular + 1

        # Angular: type contraction (no scatter yet — need gn for blm product)
        gn_ang = _type_fn(fk_ang, pi_ang, pj_ang, atom_types, c3)

        gn_blm = gn_ang.unsqueeze(-1) * blm.unsqueeze(1)
        s = torch.zeros(N, n_ap1, num_lm, dtype=dtype, device=device)
        s.scatter_add_(0, pi_ang.unsqueeze(-1).unsqueeze(-1).expand_as(gn_blm),
                       gn_blm)

        if return_intermediates:
            s_out = s
            gn_ang_out = gn_ang

        q_3b_list = []
        for li in range(l_max_3b):
            L = li + 1
            nt = 2 * L + 1
            st = L * L - 1
            c = c3b_coeffs[st:st + nt]
            sb2 = s[:, :, st:st + nt] ** 2
            ql = c[0] * sb2[:, :, 0]
            if nt > 1:
                ql = ql + 2.0 * (c[1:] * sb2[:, :, 1:]).sum(-1)
            q_3b_list.append(ql)
        q_3b = torch.stack(q_3b_list, dim=-1).transpose(1, 2).reshape(N, -1)
        parts.append(q_3b)

        if l_max_4b > 0:
            s20, s21r, s21i = s[:, :, 3], s[:, :, 4], s[:, :, 5]
            s22r, s22i = s[:, :, 6], s[:, :, 7]
            cb = c4b_coeffs
            q4 = (cb[0]*s20**3 + cb[1]*s20*(s21r**2 + s21i**2)
                  + cb[2]*s20*(s22r**2 + s22i**2)
                  + cb[3]*s22r*(s21i**2 - s21r**2)
                  + cb[4]*s21r*s21i*s22i)
            parts.append(q4)

        if l_max_5b > 0:
            s0sq = s[:, :, 0] ** 2
            s1sq = s[:, :, 1] ** 2 + s[:, :, 2] ** 2
            cb = c5b_coeffs
            q5 = cb[0]*s0sq**2 + cb[1]*s0sq*s1sq + cb[2]*s1sq**2
            parts.append(q5)

    q = torch.cat(parts, dim=-1)
    if return_intermediates:
        return q, s_out, gn_ang_out
    return q


def _angular_weight(Fp, s, dim_r, n_ap1, l_max_3b, l_max_4b, l_max_5b,
                    c3b_coeffs, c4b_coeffs, c5b_coeffs):
    """Compute dEi/d(sum_fxyz)[N, n_ap1, num_lm] for ALL body orders.

    This is the "effective Fp" in sum_fxyz space needed for the analytical
    angular force chain rule. Differentiable through s (→ c3) and Fp (→ NN weights).
    """
    N = s.shape[0]
    weight = torch.zeros_like(s)  # (N, n_ap1, num_lm)

    # --- 3-body: q3b = sum_l c[m=0]*s[l,m=0]² + 2*sum_{m>0} c[m]*s[l,m]²
    # dq3b_l/ds[n,st+m] = 2*c3b[st]   *s[n,st]   for m=0
    #                    = 4*c3b[st+m] *s[n,st+m] for m>0
    Fp_3b = Fp[:, dim_r:dim_r + l_max_3b * n_ap1].reshape(N, l_max_3b, n_ap1)
    for li in range(l_max_3b):
        L = li + 1
        nt = 2 * L + 1
        st = L * L - 1
        c = c3b_coeffs[st:st + nt]       # (nt,)
        s_lm = s[:, :, st:st + nt]        # (N, n_ap1, nt)
        Fp_l = Fp_3b[:, li, :].unsqueeze(-1)  # (N, n_ap1, 1)
        dq_ds = 2.0 * c * s_lm            # m=0: 2c[0]*s; m>0: 2c[m]*s (×2 below)
        dq_ds[:, :, 1:] = dq_ds[:, :, 1:] * 2.0  # m>0 gets extra factor 2
        weight[:, :, st:st + nt] = weight[:, :, st:st + nt] + Fp_l * dq_ds

    # --- 4-body: q4 = cb[0]*s20³ + cb[1]*s20*(s21r²+s21i²) + ...
    if l_max_4b > 0 and s.shape[2] >= 8:
        off4 = dim_r + l_max_3b * n_ap1
        Fp_4b = Fp[:, off4:off4 + n_ap1]  # (N, n_ap1)
        s20 = s[:, :, 3]; s21r = s[:, :, 4]; s21i = s[:, :, 5]
        s22r = s[:, :, 6]; s22i = s[:, :, 7]
        cb = c4b_coeffs
        weight[:, :, 3] = weight[:, :, 3] + Fp_4b * (
            3*cb[0]*s20**2 + cb[1]*(s21r**2 + s21i**2) + cb[2]*(s22r**2 + s22i**2))
        weight[:, :, 4] = weight[:, :, 4] + Fp_4b * (
            2*cb[1]*s20*s21r - 2*cb[3]*s22r*s21r + cb[4]*s21i*s22i)
        weight[:, :, 5] = weight[:, :, 5] + Fp_4b * (
            2*cb[1]*s20*s21i + 2*cb[3]*s22r*s21i + cb[4]*s21r*s22i)
        weight[:, :, 6] = weight[:, :, 6] + Fp_4b * (
            2*cb[2]*s20*s22r + cb[3]*(s21i**2 - s21r**2))
        weight[:, :, 7] = weight[:, :, 7] + Fp_4b * (
            2*cb[2]*s20*s22i + cb[4]*s21r*s21i)

    # --- 5-body: q5 = cb5[0]*s0sq² + cb5[1]*s0sq*s1sq + cb5[2]*s1sq²
    if l_max_5b > 0:
        off5 = (dim_r + l_max_3b * n_ap1
                + (n_ap1 if l_max_4b > 0 else 0))
        Fp_5b = Fp[:, off5:off5 + n_ap1]  # (N, n_ap1)
        s0 = s[:, :, 0]; s1 = s[:, :, 1]; s2 = s[:, :, 2]
        s0sq = s0**2; s1sq = s1**2 + s2**2
        cb5 = c5b_coeffs
        factor_1sq = cb5[1]*s0sq + 2*cb5[2]*s1sq
        weight[:, :, 0] = weight[:, :, 0] + Fp_5b * 2*s0*(2*cb5[0]*s0sq + cb5[1]*s1sq)
        weight[:, :, 1] = weight[:, :, 1] + Fp_5b * 2*s1*factor_1sq
        weight[:, :, 2] = weight[:, :, 2] + Fp_5b * 2*s2*factor_1sq

    return weight  # (N, n_ap1, num_lm)


def compute_analytical_forces(
    Fp, atom_types, N,
    c2, c3, fkp_rad, fkp_ang, blm,
    pi_rad, pj_rad, rij_rad, d12inv_rad,
    pi_ang, pj_ang, rij_ang, d12inv_ang,
    s, gn_ang,
    n_max_radial, n_max_angular, l_max_3b, l_max_4b, l_max_5b,
    num_lm, c3b_coeffs, c4b_coeffs, c5b_coeffs,
    dtype, device,
    compute_virial: bool = True,
    pytorch_only: bool = False,
):
    """Compute forces analytically — no create_graph needed, fully differentiable
    through c2, c3 and NN weights (via Fp).

    Fp: (N, dim) = dEi/dq * q_scaler, includes q_scaler already.
    s: (N, n_ap1, num_lm) sum_fxyz from descriptor forward.
    gn_ang: (P_ang, n_ap1) pair-level angular radial basis.

    Geometry tensors (rij, fkp, d12inv, blm) are detached so PyTorch does not
    track gradients through them — they are not trainable parameters.  Only Fp
    (→ NN weights) and c2/c3 carry gradient information.

    When pytorch_only=True, uses pure PyTorch ops (no custom autograd.Function).
    """
    if pytorch_only:
        _type_fn = _type_contraction_pytorch
    else:
        from .cuda_ops import type_contraction
        _type_fn = type_contraction

    forces = torch.zeros(N, 3, dtype=dtype, device=device)
    virial = torch.zeros(N, 9, dtype=dtype, device=device) if compute_virial else None
    dim_r = n_max_radial + 1

    # Detach all geometry — these are precomputed from fixed atom positions
    rij_rad   = rij_rad.detach()
    d12inv_rad = d12inv_rad.detach()
    fkp_rad   = fkp_rad.detach()
    if rij_ang is not None:
        rij_ang    = rij_ang.detach()
        d12inv_ang = d12inv_ang.detach()
        fkp_ang    = fkp_ang.detach()
        blm        = blm.detach()

    def _exp(idx, t):
        return idx.unsqueeze(-1).expand_as(t)

    # ---- Radial force ----
    Fp_rad = Fp[:, :dim_r]
    gnp_rad = _type_fn(fkp_rad, pi_rad, pj_rad, atom_types, c2)
    tmp_rad = (Fp_rad[pi_rad] * gnp_rad).sum(-1, keepdim=True) * d12inv_rad.unsqueeze(-1)
    f12_rad = tmp_rad * rij_rad

    forces.scatter_add_(0, _exp(pi_rad, f12_rad), f12_rad)
    forces.scatter_add_(0, _exp(pj_rad, f12_rad), -f12_rad)
    if compute_virial:
        v9 = -(rij_rad.unsqueeze(-1) * f12_rad.unsqueeze(-2)).reshape(-1, 9)
        virial.scatter_add_(0, pj_rad.unsqueeze(-1).expand_as(v9), v9)

    # ---- Angular force ----
    if l_max_3b > 0 and pi_ang.shape[0] > 0 and s is not None:
        n_ap1 = n_max_angular + 1

        # gnp_ang: radial distance derivative term (differentiable via c3)
        gnp_ang_v = _type_fn(fkp_ang, pi_ang, pj_ang, atom_types, c3)

        # weight = dEi/d(sum_fxyz): all body orders, differentiable via s→c3 and Fp→NN
        w_atom = _angular_weight(Fp, s, dim_r, n_ap1, l_max_3b, l_max_4b, l_max_5b,
                                 c3b_coeffs, c4b_coeffs, c5b_coeffs)  # (N, n_ap1, num_lm)
        w_i = w_atom[pi_ang]   # (P, n_ap1, num_lm) — atom_i's weight per pair

        # Term 1: distance derivative — f12 = (sum_n,lm w_i * gnp * blm) * rij/dij
        gnp_blm = gnp_ang_v.unsqueeze(-1) * blm.unsqueeze(1)  # (P, n_ap1, num_lm)
        scalar_gnp = (w_i * gnp_blm).sum(dim=(1, 2))           # (P,)
        f12_gnp = (scalar_gnp * d12inv_ang).unsqueeze(-1) * rij_ang

        # Term 2: direction derivative — f12 = sum_n,lm w_i * gn * dblm/drij
        # dblm/drij = (dblm/dhat - hat*(hat·dblm/dhat)) / dij
        x_hat = rij_ang[:, 0] * d12inv_ang
        y_hat = rij_ang[:, 1] * d12inv_ang
        z_hat = rij_ang[:, 2] * d12inv_ang
        # dblm_dhat depends only on geometry → no-grad context
        with torch.no_grad():
            dblm_dhat = _compute_dblm_dhat(x_hat, y_hat, z_hat, l_max_3b)  # (P, num_lm, 3)
        hat = torch.stack([x_hat, y_hat, z_hat], dim=-1)

        w_gn = (w_i * gn_ang.unsqueeze(-1)).sum(dim=1)  # (P, num_lm)
        term1 = (w_gn.unsqueeze(-1) * dblm_dhat).sum(1) * d12inv_ang.unsqueeze(-1)
        hat_dot_dblm = (hat.unsqueeze(1) * dblm_dhat).sum(-1)  # (P, num_lm)
        t2_sc = (w_gn * hat_dot_dblm).sum(1) * d12inv_ang
        f12_ang = f12_gnp + term1 - t2_sc.unsqueeze(-1) * hat

        forces.scatter_add_(0, _exp(pi_ang, f12_ang), f12_ang)
        forces.scatter_add_(0, _exp(pj_ang, f12_ang), -f12_ang)
        if compute_virial:
            v9_a = -(rij_ang.unsqueeze(-1) * f12_ang.unsqueeze(-2)).reshape(-1, 9)
            virial.scatter_add_(0, pj_ang.unsqueeze(-1).expand_as(v9_a), v9_a)

    return forces, virial


def _compute_dblm_dhat(x, y, z, l_max_3b):
    """Compute derivatives of angular basis wrt unit direction (x_hat,y_hat,z_hat).

    Returns: (P, num_lm, 3) where [..., 0/1/2] = d/d(x_hat/y_hat/z_hat).
    """
    P = x.shape[0]
    device, dtype = x.device, x.dtype
    x2, y2, z2 = x*x, y*y, z*z

    derivs = []  # list of (P, 3) tensors

    if l_max_3b >= 1:
        # blm[0] = z  → dz/d(x,y,z) = (0,0,1)
        derivs.append(torch.stack([torch.zeros(P,device=device,dtype=dtype),
                                   torch.zeros(P,device=device,dtype=dtype),
                                   torch.ones(P,device=device,dtype=dtype)], -1))
        # blm[1] = x  → (1,0,0)
        derivs.append(torch.stack([torch.ones(P,device=device,dtype=dtype),
                                   torch.zeros(P,device=device,dtype=dtype),
                                   torch.zeros(P,device=device,dtype=dtype)], -1))
        # blm[2] = y  → (0,1,0)
        derivs.append(torch.stack([torch.zeros(P,device=device,dtype=dtype),
                                   torch.ones(P,device=device,dtype=dtype),
                                   torch.zeros(P,device=device,dtype=dtype)], -1))

    if l_max_3b >= 2:
        # 3z²-1 → (0, 0, 6z)
        derivs.append(torch.stack([torch.zeros(P,device=device,dtype=dtype),
                                   torch.zeros(P,device=device,dtype=dtype),
                                   6*z], -1))
        # xz → (z, 0, x)
        derivs.append(torch.stack([z, torch.zeros(P,device=device,dtype=dtype), x], -1))
        # yz → (0, z, y)
        derivs.append(torch.stack([torch.zeros(P,device=device,dtype=dtype), z, y], -1))
        # x²-y² → (2x, -2y, 0)
        derivs.append(torch.stack([2*x, -2*y, torch.zeros(P,device=device,dtype=dtype)], -1))
        # 2xy → (2y, 2x, 0)
        derivs.append(torch.stack([2*y, 2*x, torch.zeros(P,device=device,dtype=dtype)], -1))

    if l_max_3b >= 3:
        z0 = torch.zeros(P, device=device, dtype=dtype)
        # (5z²-3)z → (0, 0, 15z²-3)
        derivs.append(torch.stack([z0, z0, 15*z2-3], -1))
        # (5z²-1)x → (5z²-1, 0, 10xz)
        derivs.append(torch.stack([5*z2-1, z0, 10*x*z], -1))
        # (5z²-1)y → (0, 5z²-1, 10yz)
        derivs.append(torch.stack([z0, 5*z2-1, 10*y*z], -1))
        # (x²-y²)z → (2xz, -2yz, x²-y²)
        derivs.append(torch.stack([2*x*z, -2*y*z, x2-y2], -1))
        # 2xyz → (2yz, 2xz, 2xy)
        derivs.append(torch.stack([2*y*z, 2*x*z, 2*x*y], -1))
        # x(x²-3y²) → (3x²-3y², -6xy, 0)
        derivs.append(torch.stack([3*x2-3*y2, -6*x*y, z0], -1))
        # y(3x²-y²) → (6xy, 3x²-3y², 0)
        derivs.append(torch.stack([6*x*y, 3*x2-3*y2, z0], -1))

    if l_max_3b >= 4:
        z0 = torch.zeros(P, device=device, dtype=dtype)
        # (35z⁴-30z²+3) → (0, 0, 140z³-60z)
        derivs.append(torch.stack([z0, z0, 140*z2*z-60*z], -1))
        # (7z²-3)xz → ((7z²-3)z, 0, x(21z²-3))
        derivs.append(torch.stack([(7*z2-3)*z, z0, x*(21*z2-3)], -1))
        # (7z²-3)yz → (0, (7z²-3)z, y(21z²-3))
        derivs.append(torch.stack([z0, (7*z2-3)*z, y*(21*z2-3)], -1))
        # (7z²-1)(x²-y²) → (2x(7z²-1), -2y(7z²-1), 14z(x²-y²))
        derivs.append(torch.stack([2*x*(7*z2-1), -2*y*(7*z2-1), 14*z*(x2-y2)], -1))
        # (7z²-1)*2xy → (2y(7z²-1), 2x(7z²-1), 28xyz)
        derivs.append(torch.stack([2*y*(7*z2-1), 2*x*(7*z2-1), 28*x*y*z], -1))
        # zx(x²-3y²) → (z(3x²-3y²), -6xyz, x(x²-3y²))
        derivs.append(torch.stack([z*(3*x2-3*y2), -6*x*y*z, x*(x2-3*y2)], -1))
        # zy(3x²-y²) → (6xyz, z*(3x²-3y²), y*(3*x2-y2))
        derivs.append(torch.stack([6*x*y*z, z*(3*x2-3*y2), y*(3*x2-y2)], -1))
        # (x²-y²)²-4x²y² → (4x(x²-y²)-8xy², -4y(x²-y²)-8x²y, 0)
        #  = (4x³-4xy²-8xy², -4x²y+4y³-8x²y, 0) = (4x³-12xy², 4y³-12x²y, 0)
        derivs.append(torch.stack([4*x*(x2-3*y2), 4*y*(y2-3*x2), z0], -1))
        # 4xy(x²-y²) → (4y(3x²-y²), 4x(x²-3y²), 0)
        derivs.append(torch.stack([4*y*(3*x2-y2), 4*x*(x2-3*y2), z0], -1))

    return torch.stack(derivs, dim=1)  # (P, num_lm, 3)


# ---------------------------------------------------------------------------
# Force / virial accumulation
# ---------------------------------------------------------------------------

def accumulate_forces_virial(
    N, pi_rad, pj_rad, rij_rad, g_rad,
    pi_ang, pj_ang, rij_ang, g_ang,
    dtype, device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Accumulate pair gradients into per-atom forces and virial.

    F_k = sum_{pairs i=k} grad - sum_{pairs j=k} grad
    virial_ab(j) = -rij_a * grad_b  (accumulated on j)
    """
    forces = torch.zeros(N, 3, dtype=dtype, device=device)
    virial = torch.zeros(N, 9, dtype=dtype, device=device)

    def _acc(pi, pj, r, g):
        e = lambda idx: idx.unsqueeze(-1).expand_as(g)
        forces.scatter_add_(0, e(pi), g)
        forces.scatter_add_(0, e(pj), -g)
        v9 = -(r.unsqueeze(-1) * g.unsqueeze(-2)).reshape(-1, 9)
        virial.scatter_add_(0, pj.unsqueeze(-1).expand_as(v9), v9)

    _acc(pi_rad, pj_rad, rij_rad, g_rad)
    _acc(pi_ang, pj_ang, rij_ang, g_ang)
    return forces, virial


# ---------------------------------------------------------------------------
# CUDA kernel loading (optional — falls back to PyTorch on CPU/Mac)
# ---------------------------------------------------------------------------

_cuda_ops = None
_cuda_desc_ops = None


def _get_platform_flags():
    """Get platform-specific compiler flags for CUDA JIT compilation."""
    import sys
    extra_cflags = []
    extra_cuda = ["-O3", "--use_fast_math"]
    if sys.platform == "win32":
        extra_cflags = ["/permissive-"]
        extra_cuda.append("-Xcompiler=/permissive-")
    return extra_cflags, extra_cuda


def _load_cuda_ops():
    """JIT-compile CUDA kernels on first use."""
    global _cuda_ops
    if _cuda_ops is not None:
        return _cuda_ops
    if not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import load
        csrc = os.path.join(os.path.dirname(__file__), "csrc", "nep_ops.cu")
        if not os.path.exists(csrc):
            return None
        extra_cflags, extra_cuda = _get_platform_flags()
        _cuda_ops = load(name="nep_cuda_ops", sources=[csrc], verbose=False,
                         extra_cflags=extra_cflags,
                         extra_cuda_cflags=extra_cuda)
        return _cuda_ops
    except Exception:
        return None


def _load_cuda_desc_ops():
    """JIT-compile descriptor+force CUDA kernels."""
    global _cuda_desc_ops
    if _cuda_desc_ops is not None:
        return _cuda_desc_ops
    if not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import load
        csrc = os.path.join(os.path.dirname(__file__), "csrc", "nep_descriptor.cu")
        if not os.path.exists(csrc):
            return None
        extra_cflags, extra_cuda = _get_platform_flags()
        _cuda_desc_ops = load(name="nep_cuda_desc", sources=[csrc], verbose=False,
                              extra_cflags=extra_cflags,
                              extra_cuda_cflags=extra_cuda)
        return _cuda_desc_ops
    except Exception:
        return None


def compute_radial_force_cuda(
    rij_rad, pair_i, pair_j, atom_types, c2,
    Fp_rad, N, dim_r, bs1, num_types, rc_radial,
    compute_virial=False,
):
    """CUDA-accelerated radial force computation. Falls back to PyTorch."""
    cuda = _load_cuda_desc_ops()
    if cuda is not None and rij_rad.is_cuda:
        # c2 needs to be (nt, nt, dim_r, bs1) contiguous
        c2_c = c2.contiguous()
        return cuda.radial_force(
            rij_rad.contiguous(), pair_i, pair_j, atom_types,
            c2_c, Fp_rad.contiguous(), N, dim_r, bs1, num_types,
            rc_radial, compute_virial)
    # PyTorch fallback
    return _radial_force_pytorch(
        rij_rad, pair_i, pair_j, atom_types, c2,
        Fp_rad, N, dim_r, bs1, num_types, rc_radial, compute_virial)


def _radial_force_pytorch(
    rij_rad, pair_i, pair_j, atom_types, c2,
    Fp_rad, N, dim_r, bs1, num_types, rc_radial, compute_virial,
):
    """Pure PyTorch radial force (fallback for CPU/Mac)."""
    dtype, device = rij_rad.dtype, rij_rad.device
    dij = torch.norm(rij_rad, dim=-1)
    _, fkp = chebyshev_basis_and_deriv(dij, rc_radial, bs1 - 1)
    t1, t2 = atom_types[pair_i], atom_types[pair_j]
    c2_p = c2[t1, t2] if c2.shape[0] != dim_r else c2[:, :, t1, t2].permute(2, 0, 1)
    gnp = (c2_p * fkp.unsqueeze(1)).sum(-1)
    d12inv = 1.0 / dij
    Fp_i = Fp_rad[pair_i]
    tmp = (Fp_i * gnp).sum(-1, keepdim=True) * d12inv.unsqueeze(-1)
    f12 = tmp * rij_rad
    forces = torch.zeros(N, 3, dtype=dtype, device=device)
    virial = torch.zeros(N, 9, dtype=dtype, device=device) if compute_virial else None

    def _e(idx, t):
        return idx.unsqueeze(-1).expand_as(t)
    forces.scatter_add_(0, _e(pair_i, f12), f12)
    forces.scatter_add_(0, _e(pair_j, f12), -f12)
    if compute_virial:
        v9 = -(rij_rad.unsqueeze(-1) * f12.unsqueeze(-2)).reshape(-1, 9)
        virial.scatter_add_(0, pair_j.unsqueeze(-1).expand_as(v9), v9)
    return forces, virial


def accumulate_forces_virial_cuda(
    N, pi_rad, pj_rad, rij_rad, g_rad,
    pi_ang, pj_ang, rij_ang, g_ang,
    dtype, device,
):
    """CUDA-accelerated force/virial accumulation."""
    cuda = _load_cuda_ops()
    if cuda is None or device.type != "cuda":
        return accumulate_forces_virial(
            N, pi_rad, pj_rad, rij_rad, g_rad,
            pi_ang, pj_ang, rij_ang, g_ang, dtype, device)

    # Concatenate radial + angular pairs
    rij_all = torch.cat([rij_rad, rij_ang], dim=0)
    g_all = torch.cat([g_rad, g_ang], dim=0)
    pi_all = torch.cat([pi_rad, pi_ang], dim=0)
    pj_all = torch.cat([pj_rad, pj_ang], dim=0)

    forces, virial = cuda.force_virial(rij_all, g_all, pi_all, pj_all, N)
    return forces, virial
