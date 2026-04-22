"""
Core NEP operations — pure PyTorch on CPU / CUDA / MPS.
"""

import torch
from typing import List, Literal, Optional, Tuple

from .constants import PI, K_C_SP, ZBL_PARA, Z_COEFFICIENT, MAX_L3B


# ---------------------------------------------------------------------------
# Backend selection
#
# Two concrete implementations of the type-pair contraction plus one meta:
#   "loop" : pure PyTorch, nested for-loop over (t1, t2) type pairs. Wins when
#            ntypes is small (≤ ~5) — the outer loop runs few times.
#   "bmm"  : pure PyTorch, fancy index + torch.bmm (dispatched to cuBLAS on
#            CUDA, MKL on CPU, MPS on Apple). Wins when ntypes ≥ ~8 — one
#            batched GEMM replaces the O(ntypes²) python-level loop.
#   "auto" : picks by num_types (≥ 8 → bmm, else loop).
#
# Both backends are autograd-compatible and run on any PyTorch backend.
# ---------------------------------------------------------------------------

Backend = Literal["auto", "loop", "bmm"]


def resolve_backend(backend: str = "auto",
                     num_types: Optional[int] = None) -> str:
    """Resolve "auto" into a concrete backend.

    ntypes ≥ 8  → "bmm"   (fancy-index + batched GEMM wins)
    otherwise   → "loop"  (few-types; inline Python loop is fastest)

    Any non-"auto" string is returned unchanged (for explicit overrides).
    """
    if backend != "auto":
        return backend
    if num_types is not None and num_types >= 8:
        return "bmm"
    return "loop"


def _select_contraction_funcs(backend: str):
    """Return (scatter_fn, type_fn) for the concrete backend."""
    if backend == "bmm":
        return _scatter_contraction_bmm, _type_contraction_bmm
    # "loop" — default
    return _scatter_contraction_loop, _type_contraction_loop


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

    Writes directly into a preallocated (P, basis_size+1) output buffer and
    keeps only 4 sliding-window scalars (T_{k-2}, T_{k-1}, U_{k-2}, U_{k-1})
    in memory, instead of materializing all 2*(basis_size+1) intermediate
    tensors + a final torch.stack copy. On 19M pairs this avoids ~2 GB of
    GPU→GPU traffic per call.
    """
    rcinv = 1.0 / rc
    arg = PI * dij * rcinv
    fc = 0.5 * torch.cos(arg) + 0.5
    fcp = -0.5 * PI * rcinv * torch.sin(arg)
    dij_m1 = dij * rcinv - 1.0
    x = 2.0 * dij_m1 * dij_m1 - 1.0
    dxdr = 4.0 * dij_m1 * rcinv

    P = dij.shape[0]
    B = basis_size + 1
    fk  = torch.empty(P, B, dtype=dij.dtype, device=dij.device)
    fkp = torch.empty(P, B, dtype=dij.dtype, device=dij.device)

    # k = 0: T_0 = 1, so fn_core = 1 → fk[..,0] = fc; fkp[..,0] = fcp
    fk[:, 0] = fc
    fkp[:, 0] = fcp

    if B >= 2:
        # k = 1: T_1 = x, U_0 = 1 → dT_1/dx = 1
        fn_core1 = 0.5 * (x + 1.0)
        fk[:, 1] = fn_core1 * fc
        fkp[:, 1] = 0.5 * dxdr * fc + fn_core1 * fcp

    # Sliding window: T_prev2 = T_{k-2}, T_prev1 = T_{k-1},
    #                 U_prev2 = U_{k-2}, U_prev1 = U_{k-1}
    T_prev2 = None  # T_0 = 1 (not needed as tensor; reuse ones trick below)
    T_prev1 = x     # T_1
    U_prev2 = None  # U_0 = 1
    U_prev1 = 2.0 * x   # U_1

    # We need tensors for T_prev2 / U_prev2. Use explicit ones_like just once.
    T_prev2 = torch.ones_like(dij)   # T_0
    U_prev2 = torch.ones_like(dij)   # U_0

    for k in range(2, B):
        T_next = 2.0 * x * T_prev1 - T_prev2     # T_k
        U_next = 2.0 * x * U_prev1 - U_prev2     # U_k
        fn_core = 0.5 * (T_next + 1.0)
        fk[:, k] = fn_core * fc
        # dT_k/dx = k * U_{k-1} = k * U_prev1  (before we shift)
        fkp[:, k] = 0.5 * (k * U_prev1) * dxdr * fc + fn_core * fcp
        T_prev2, T_prev1 = T_prev1, T_next
        U_prev2, U_prev1 = U_prev1, U_next

    return fk, fkp


def _build_xy_powers(x: torch.Tensor, y: torch.Tensor, n_max: int):
    """Real and imaginary parts of (x + iy)^n for n = 0..n_max.

    Returns two lists of tensors (each shape (P,)): Re[n], Im[n].
    """
    Re = [torch.ones_like(x)]
    Im = [torch.zeros_like(x)]
    for _ in range(n_max):
        r_prev, i_prev = Re[-1], Im[-1]
        # (x + iy) * (r + ii) = (xr - yi) + i(xi + yr)
        Re.append(x * r_prev - y * i_prev)
        Im.append(x * i_prev + y * r_prev)
    return Re, Im


def _build_z_powers(z: torch.Tensor, n_max: int):
    """z^0..z^n_max as a list of tensors."""
    zp = [torch.ones_like(z)]
    for _ in range(n_max):
        zp.append(zp[-1] * z)
    return zp


def _z_factor(z_pow, coeff_row, start, stop):
    """Sum_{n2 = start, start+2, ..., stop-1} coeff_row[n2] * z^n2.

    Skips zero coefficients to save ops. ``start`` matches (L+n1) parity.
    """
    out = None
    n2 = start
    while n2 < stop:
        c = coeff_row[n2]
        if c != 0.0:
            term = z_pow[n2] if c == 1.0 else (c * z_pow[n2])
            out = term if out is None else out + term
        n2 += 2
    if out is None:
        return torch.zeros_like(z_pow[0])
    return out


def angular_basis(x: torch.Tensor, y: torch.Tensor,
                  z: torch.Tensor, l_max_3b: int) -> torch.Tensor:
    """Solid-harmonics-style angular basis Y_{Ln1} for L = 1..l_max_3b.

    Each basis element is z_factor(z) · {Re or Im}[(x + iy)^n1], using the
    polynomial coefficients in ``Z_COEFFICIENT``. Bit-identical to the
    previous hand-coded formulas for l_max_3b ≤ 4.

    Returns: (P, num_lm) with num_lm = sum_{L=1}^{l_max_3b}(2L + 1).
    """
    if l_max_3b < 1:
        return torch.zeros(x.shape[0], 0, dtype=x.dtype, device=x.device)
    if l_max_3b > MAX_L3B:
        raise ValueError(f"l_max_3b={l_max_3b} exceeds MAX_L3B={MAX_L3B}")

    z_pow = _build_z_powers(z, l_max_3b)
    Re, Im = _build_xy_powers(x, y, l_max_3b)

    blm = []
    for L in range(1, l_max_3b + 1):
        Z = Z_COEFFICIENT[L]
        for n1 in range(L + 1):
            parity = (L + n1) % 2
            start = parity               # 0 if L+n1 even, 1 if odd
            stop = L - n1 + 1
            zf = _z_factor(z_pow, Z[n1], start, stop)
            if n1 == 0:
                blm.append(zf)
            else:
                blm.append(zf * Re[n1])
                blm.append(zf * Im[n1])
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
    backend: str = "auto",
) -> torch.Tensor:
    """Compute NEP4 descriptors from raw pair geometry. Returns (N, dim).

    This builds the Chebyshev/angular basis internally, unlike
    ``compute_descriptors_cached`` which takes them as inputs. Used by the
    non-training ASE-like path (NEPCalculator.compute).
    """
    backend = resolve_backend(backend, num_types=int(c2.shape[0]))
    scatter_fn, type_fn = _select_contraction_funcs(backend)

    # --- Radial ---
    dij_rad = torch.norm(rij_rad, dim=-1)
    fk_rad = chebyshev_basis(dij_rad, rc_radial, basis_size_radial)
    q_rad = scatter_fn(fk_rad, pi_rad, pj_rad, atom_types, c2, N)

    parts = [q_rad]

    # --- Angular ---
    n_ap1 = n_max_angular + 1

    if l_max_3b > 0 and rij_ang.shape[0] == 0:
        # No angular neighbors (e.g. isolated atom / dimer). Emit zero-filled
        # angular blocks so output dim still matches q_scaler.
        parts.append(torch.zeros(N, n_ap1 * l_max_3b, dtype=dtype, device=device))
        if l_max_4b > 0:
            parts.append(torch.zeros(N, n_ap1, dtype=dtype, device=device))
        if l_max_5b > 0:
            parts.append(torch.zeros(N, n_ap1, dtype=dtype, device=device))

    if l_max_3b > 0 and rij_ang.shape[0] > 0:
        dij_ang = torch.norm(rij_ang, dim=-1)
        fk_ang = chebyshev_basis(dij_ang, rc_angular, basis_size_angular)
        gn_ang = type_fn(fk_ang, pi_ang, pj_ang, atom_types, c3)
        d12inv = 1.0 / torch.clamp(dij_ang, min=1e-10)
        blm = angular_basis(rij_ang[:, 0]*d12inv, rij_ang[:, 1]*d12inv,
                            rij_ang[:, 2]*d12inv, l_max_3b)
        gn_blm = gn_ang.unsqueeze(-1) * blm.unsqueeze(1)
        s = torch.zeros(N, n_ap1, num_lm, dtype=dtype, device=device)
        s.scatter_add_(0, pi_ang.unsqueeze(-1).unsqueeze(-1).expand_as(gn_blm),
                       gn_blm)

        # 3-body q
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

    # Coarse cutoff for the initial distance mask. For typewise, the actual
    # per-pair cutoff may be smaller, so we evaluate tighter cutoffs later.
    if use_tw:
        max_rc = min(float(rc_outer_per_type.max().item()), rc_outer_default)
    else:
        max_rc = rc_outer_default

    mask = dij < max_rc
    if not mask.any():
        return torch.zeros(N, dtype=dtype, device=device)

    pi, pj = pair_i[mask], pair_j[mask]
    d = dij[mask]

    if use_tw:
        # NEP_CPU typewise convention (nep.cpp:1795-1801):
        #   rc_outer_pair = min((cov_i + cov_j) * typewise_factor, rc_outer_default)
        #   rc_inner      = 0
        # rc_outer_per_type was built as 2*typewise_factor*cov, so its pair
        # average equals (cov_i + cov_j) * typewise_factor.
        t1 = atom_types[pi]
        t2 = atom_types[pj]
        rc_outer_pair = (rc_outer_per_type[t1] + rc_outer_per_type[t2]) * 0.5
        rc_outer = torch.clamp(rc_outer_pair, max=rc_outer_default)
        rc_inner = torch.zeros_like(rc_outer)
        # Drop pairs outside the tightened per-pair cutoff
        tw_mask = d < rc_outer
        if not tw_mask.all():
            pi, pj = pi[tw_mask], pj[tw_mask]
            d = d[tw_mask]
            rc_inner = rc_inner[tw_mask]
            rc_outer = rc_outer[tw_mask]
    else:
        rc_inner = rc_inner_default
        rc_outer = rc_outer_default

    an = torch.tensor(atomic_numbers_list, dtype=dtype, device=device)
    zi = an[atom_types[pi]]
    zj = an[atom_types[pj]]

    # Constant must match NEP_CPU's `2.134563` literal to get bit-identical
    # forward output; 1/0.46850 = 2.1344717… differs at the 4e-5 level,
    # which accumulates to meV-scale errors on strong ZBL pairs.
    a_inv = (zi ** 0.23 + zj ** 0.23) * 2.134563
    zizj = K_C_SP * zi * zj
    x = d * a_inv
    phi = (ZBL_PARA[0] * torch.exp(-ZBL_PARA[1] * x)
           + ZBL_PARA[2] * torch.exp(-ZBL_PARA[3] * x)
           + ZBL_PARA[4] * torch.exp(-ZBL_PARA[5] * x)
           + ZBL_PARA[6] * torch.exp(-ZBL_PARA[7] * x))

    rc_i = rc_inner  # per-pair tensor in typewise, scalar otherwise
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
    # Neighbor list is directed: every physical pair (i,j) appears twice
    # (once as (i,j) and once as (j,i)). Halving the pair energy and
    # scattering only to pi gives each atom 0.5*e_pair per neighbour and
    # a total system energy of 1*e_pair per physical pair.
    e_atom = torch.zeros(N, dtype=dtype, device=device)
    e_atom.scatter_add_(0, pi, 0.5 * e_pair)
    return e_atom


def _scatter_contraction_loop(basis, pair_i, pair_j, atom_types, c, N):
    """Type-pair loop: Σ_k c[t1, t2, n, k]·basis[p, k] then scatter_add into q.

    The outer loop is O(ntypes²) python iterations. Preferred when ntypes is
    small (few kernel launches, each matmul is fat) and peak memory matters."""
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


def _type_contraction_loop(basis, pair_i, pair_j, atom_types, c):
    """Type-pair loop (no scatter). See ``_scatter_contraction_loop``."""
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


def _scatter_contraction_bmm(basis, pair_i, pair_j, atom_types, c, N):
    """Vectorised: gather c[t1, t2] per pair, one torch.bmm, scatter_add.

    Allocates a (P, N_out, K) intermediate so peak memory is higher, but
    launches ~1 kernel instead of O(ntypes²). Wins when ntypes ≥ ~8."""
    t1 = atom_types[pair_i]
    t2 = atom_types[pair_j]
    c_p = c[t1, t2]                                            # (P, N_out, K)
    gn = torch.bmm(c_p, basis.unsqueeze(-1)).squeeze(-1)        # (P, N_out)
    q = torch.zeros(N, c.shape[2], dtype=basis.dtype, device=basis.device)
    q.scatter_add_(0, pair_i.unsqueeze(-1).expand_as(gn), gn)
    return q


def _type_contraction_bmm(basis, pair_i, pair_j, atom_types, c):
    """Vectorised counterpart of ``_type_contraction_loop``."""
    t1 = atom_types[pair_i]
    t2 = atom_types[pair_j]
    c_p = c[t1, t2]                                            # (P, N_out, K)
    return torch.bmm(c_p, basis.unsqueeze(-1)).squeeze(-1)      # (P, N_out)


def compute_descriptors_cached(
    fk_rad, fk_ang, blm,
    pi_rad, pj_rad, pi_ang, pj_ang,
    atom_types, N, c2, c3,
    n_max_radial, n_max_angular, l_max_3b, l_max_4b, l_max_5b,
    num_lm, c3b_coeffs, c4b_coeffs, c5b_coeffs,
    dtype, device,
    return_intermediates: bool = False,
    backend: str = "loop",
):
    """Compute descriptors using precomputed basis functions.

    Faster than compute_descriptors because Chebyshev/angular basis are cached.
    Differentiable through c2, c3 only (not rij).

    ``backend`` selects the type-pair contraction implementation:
      "loop" — pure-PyTorch ntypes² loop (few types)
      "bmm"  — fancy-index + torch.bmm (many types)

    If return_intermediates=True, returns (q, s, gn_ang) where:
      s      : (N, n_ap1, num_lm) sum_fxyz moments — needed for analytical forces
      gn_ang : (P_ang, n_ap1) pair-level angular radial factor
    """
    _scatter_fn, _type_fn = _select_contraction_funcs(backend)

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
    backend: str = "loop",
):
    """Compute forces analytically — no create_graph needed, fully differentiable
    through c2, c3 and NN weights (via Fp).

    Fp: (N, dim) = dEi/dq * q_scaler, includes q_scaler already.
    s: (N, n_ap1, num_lm) sum_fxyz from descriptor forward.
    gn_ang: (P_ang, n_ap1) pair-level angular radial basis.

    Geometry tensors (rij, fkp, d12inv, blm) are detached so PyTorch does not
    track gradients through them — they are not trainable parameters.  Only Fp
    (→ NN weights) and c2/c3 carry gradient information.

    ``backend`` — see ``compute_descriptors_cached`` for options.
    """
    _, _type_fn = _select_contraction_funcs(backend)

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
    """Derivatives of ``angular_basis`` wrt unit direction (x̂, ŷ, ẑ).

    For a basis element blm = z_factor(z) · C(x, y) where
      C = 1              (n1 = 0)
      C = Re[(x+iy)^n1]  (n1 > 0, real  component)
      C = Im[(x+iy)^n1]  (n1 > 0, imag component)
    the gradients are:
      ∂blm/∂x = z_factor(z)        · ∂C/∂x
      ∂blm/∂y = z_factor(z)        · ∂C/∂y
      ∂blm/∂z = z_factor'(z)       · C
    with ∂Re_n/∂x =  n·Re_{n-1},  ∂Im_n/∂x =  n·Im_{n-1},
         ∂Re_n/∂y = -n·Im_{n-1},  ∂Im_n/∂y =  n·Re_{n-1}.

    Returns: (P, num_lm, 3) where [..., 0/1/2] = d/d(x̂, ŷ, ẑ).
    Bit-identical to the old hand-coded implementation for l_max_3b ≤ 4.
    """
    if l_max_3b < 1:
        return torch.zeros(x.shape[0], 0, 3, dtype=x.dtype, device=x.device)
    if l_max_3b > MAX_L3B:
        raise ValueError(f"l_max_3b={l_max_3b} exceeds MAX_L3B={MAX_L3B}")

    z_pow = _build_z_powers(z, l_max_3b)
    # Need (x+iy)^n for n = 0..l_max_3b-1 (max n1-1) + (x+iy)^n1 for n1 up to l_max_3b
    Re, Im = _build_xy_powers(x, y, l_max_3b)

    zeros = torch.zeros_like(x)
    derivs = []  # list of (P, 3)

    for L in range(1, l_max_3b + 1):
        Z = Z_COEFFICIENT[L]
        for n1 in range(L + 1):
            parity = (L + n1) % 2
            start = parity
            stop = L - n1 + 1
            zf = _z_factor(z_pow, Z[n1], start, stop)
            # z_factor'(z) = sum n2 * coeff[n2] * z^{n2-1}, iterating the same
            # parity as z_factor (start, start+2, ...); only n2>=1 contributes.
            zfp = None
            n2 = start
            while n2 < stop:
                if n2 >= 1:
                    c = Z[n1][n2]
                    if c != 0.0:
                        term = (n2 * c) * z_pow[n2 - 1]
                        zfp = term if zfp is None else zfp + term
                n2 += 2
            if zfp is None:
                zfp = zeros

            if n1 == 0:
                # blm = zf(z). dx=0, dy=0, dz=zf'(z)
                derivs.append(torch.stack([zeros, zeros, zfp], dim=-1))
            else:
                # real component
                dRe_dx = n1 * Re[n1 - 1]
                dRe_dy = -n1 * Im[n1 - 1]
                derivs.append(torch.stack(
                    [zf * dRe_dx, zf * dRe_dy, zfp * Re[n1]], dim=-1))
                # imag component
                dIm_dx = n1 * Im[n1 - 1]
                dIm_dy = n1 * Re[n1 - 1]
                derivs.append(torch.stack(
                    [zf * dIm_dx, zf * dIm_dy, zfp * Im[n1]], dim=-1))

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
