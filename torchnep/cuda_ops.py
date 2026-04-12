"""
CUDA-accelerated descriptor functions for NEP.

Design:
  - Radial: CUDA kernel for forward (fused scatter-add).
            PyTorch backward for grad_rij (uses saved gnp) and grad_c2 (recomputes fk).
  - Angular: CUDA kernel for sum_fxyz accumulation when no gradient needed.
             PyTorch path when c3 gradient is required (computes c3→gn→sum_fxyz in PyTorch).

This gives correct parameter gradients while keeping CUDA speedup for inference /
the q_scaler computation path (pytorch_only=False).
"""

import os
import sys
import torch
from typing import Optional


_kernels = None


def _load_kernels():
    """JIT-compile CUDA kernels on first use.

    Set TORCHNEP_NO_CUDA_KERNELS=1 to force pure-PyTorch fallback.
    Set TORCHNEP_VERBOSE_BUILD=1 to see nvcc output during JIT compile.
    """
    global _kernels
    if _kernels is not None:
        return _kernels
    if os.environ.get("TORCHNEP_NO_CUDA_KERNELS", "0") == "1":
        return None
    if not torch.cuda.is_available():
        return None
    verbose = os.environ.get("TORCHNEP_VERBOSE_BUILD", "0") == "1"
    try:
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(__file__), "csrc", "nep_kernels.cu")
        if not os.path.exists(src):
            return None
        extra_cflags = []
        extra_cuda = ["-O3", "--use_fast_math"]
        if sys.platform == "win32":
            extra_cflags = ["/permissive-"]
            extra_cuda.append("-Xcompiler=/permissive-")
        _kernels = load(
            name="nep_kernels", sources=[src], verbose=verbose,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda)
        return _kernels
    except Exception:
        if verbose:
            import traceback
            traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Pure-PyTorch helpers (used in backward)
# ---------------------------------------------------------------------------

def _radial_forward_pytorch(rij, pair_i, pair_j, atom_types, c2,
                            N, n_max_r, basis_r, ntypes, rc):
    """PyTorch radial forward. Returns (q, gnp) where gnp = dfeat_2b."""
    from .ops import chebyshev_basis_and_deriv
    dim_r = n_max_r + 1
    dtype, device = rij.dtype, rij.device

    dij = torch.norm(rij, dim=-1)
    fk, fkp = chebyshev_basis_and_deriv(dij, rc, basis_r)
    t1, t2 = atom_types[pair_i], atom_types[pair_j]
    c2_p = c2[t1, t2]                          # (P, dim_r, bs1)
    gn = (c2_p * fk.unsqueeze(1)).sum(-1)       # (P, dim_r)
    gnp = (c2_p * fkp.unsqueeze(1)).sum(-1)     # (P, dim_r)

    q = torch.zeros(N, dim_r, dtype=dtype, device=device)
    q.scatter_add_(0, pair_i.unsqueeze(-1).expand_as(gn), gn)
    return q, gnp


# ---------------------------------------------------------------------------
# Radial descriptor: CUDA forward, PyTorch backward
# ---------------------------------------------------------------------------

class NEPRadialDescriptor(torch.autograd.Function):
    """Radial descriptor with CUDA-accelerated forward.

    Forward:  CUDA kernel (fused scatter-add) for speed.
    Backward: PyTorch — grad_rij via saved gnp (dfeat_2b), grad_c2 via recomputed fk.
              This avoids the --use_fast_math trig-accumulation error in the CUDA
              radial_feat_backward_c2 kernel (~0.25 absolute diff on 30k-pair batches).
    """

    @staticmethod
    def forward(ctx, rij, pair_i, pair_j, atom_types, c2,
                N, n_max_r, basis_r, ntypes, rc):
        k = _load_kernels()
        if k is not None and rij.is_cuda:
            result = k.radial_forward(
                rij.contiguous(), pair_i, pair_j, atom_types,
                c2.contiguous(), N, n_max_r, basis_r, ntypes, rc)
            q, dfeat_2b = result[0], result[1]
        else:
            q, dfeat_2b = _radial_forward_pytorch(
                rij, pair_i, pair_j, atom_types, c2,
                N, n_max_r, basis_r, ntypes, rc)

        ctx.save_for_backward(rij, pair_i, pair_j, atom_types, dfeat_2b)
        ctx.N = N
        ctx.n_max_r = n_max_r
        ctx.basis_r = basis_r
        ctx.ntypes = ntypes
        ctx.rc = rc
        return q

    @staticmethod
    def backward(ctx, grad_q):
        rij, pair_i, pair_j, atom_types, dfeat_2b = ctx.saved_tensors
        n_max_r, basis_r, ntypes, rc = ctx.n_max_r, ctx.basis_r, ctx.ntypes, ctx.rc

        # --- grad_rij: chain rule through gn → d12 → rij ---
        # dfeat_2b[p,n] = gnp[p,n] = sum_k c2[t1,t2,n,k]*fnp[k]  (saved from forward)
        # d(loss)/d(rij[p]) = (sum_n grad_q[i,n]*gnp[p,n]) * rij[p]/d12[p]
        d12 = torch.norm(rij, dim=-1, keepdim=True).clamp(min=1e-10)
        gq_i = grad_q[pair_i]                               # (P, dim_r)
        scalar = (gq_i * dfeat_2b).sum(-1, keepdim=True)   # (P, 1)
        grad_rij = scalar / d12 * rij                       # (P, 3)

        # --- grad_c2: recompute fk (basis values) from rij ---
        # d(loss)/d(c2[t1,t2,n,k]) = sum_{p:t1p=t1,t2p=t2} grad_q[pair_i[p],n] * fk[p,k]
        # Recomputing fk ensures exact match with the PyTorch compute_descriptors path.
        from .ops import chebyshev_basis
        fk = chebyshev_basis(torch.norm(rij, dim=-1), rc, basis_r)  # (P, bs1)
        t1_pairs = atom_types[pair_i]
        t2_pairs = atom_types[pair_j]
        grad_c2 = torch.zeros(ntypes, ntypes, n_max_r + 1, basis_r + 1,
                              dtype=rij.dtype, device=rij.device)
        for _t1 in range(ntypes):
            for _t2 in range(ntypes):
                _m = (t1_pairs == _t1) & (t2_pairs == _t2)
                if _m.any():
                    # (dim_r, P_m) @ (P_m, bs1) = (dim_r, bs1)
                    grad_c2[_t1, _t2] = gq_i[_m].T @ fk[_m]

        return grad_rij, None, None, None, grad_c2, None, None, None, None, None


def radial_descriptor_cuda(rij, pair_i, pair_j, atom_types, c2,
                           N, n_max_r, basis_r, ntypes, rc):
    """Radial descriptors with CUDA-accelerated forward, correct PyTorch backward."""
    return NEPRadialDescriptor.apply(
        rij, pair_i, pair_j, atom_types, c2,
        N, n_max_r, basis_r, ntypes, rc)


# ---------------------------------------------------------------------------
# Angular descriptor: PyTorch (for correct c3 gradient) with optional CUDA sum_fxyz
# ---------------------------------------------------------------------------

def angular_descriptor_cuda(rij, pair_i, pair_j, atom_types, c3,
                             N, n_ap1, basis_a, ntypes, num_lm,
                             L_max, num_L, rc,
                             c3b_coeffs, c4b_coeffs, c5b_coeffs):
    """Angular descriptors — differentiable w.r.t. c3.

    The c3 → gn → sum_fxyz → q_ang chain runs entirely in PyTorch so that
    c3.grad is computed correctly.  (The CUDA angular_feat_forward kernel
    accumulates sum_fxyz without tracking c3 through the computation graph,
    so c3.grad would always be None/zero if we used that kernel here.)
    """
    from .ops import chebyshev_basis, angular_basis

    dij = torch.norm(rij, dim=-1)
    fk = chebyshev_basis(dij, rc, basis_a)                  # (P, bs1)
    t1_pairs = atom_types[pair_i]
    t2_pairs = atom_types[pair_j]

    # c3 contraction: use type-pair matmul loop (matches compute_descriptors exactly
    # so gradients agree bit-for-bit with the pytorch_only=True reference path)
    gn = torch.zeros(rij.shape[0], n_ap1, dtype=rij.dtype, device=rij.device)
    for _t1 in range(ntypes):
        for _t2 in range(ntypes):
            _m = (t1_pairs == _t1) & (t2_pairs == _t2)
            if _m.any():
                gn[_m] = fk[_m] @ c3[_t1, _t2].T          # (P_m, n_ap1)

    d12inv = 1.0 / dij.clamp(min=1e-10)
    blm = angular_basis(rij[:, 0] * d12inv, rij[:, 1] * d12inv,
                        rij[:, 2] * d12inv, L_max)          # (P, num_lm)

    # Accumulate: sum_fxyz[i, n, m] = sum_{pairs p: pair_i[p]=i} gn[p,n] * blm[p,m]
    gn_blm = gn.unsqueeze(-1) * blm.unsqueeze(1)            # (P, n_ap1, num_lm)
    sum_fxyz = torch.zeros(N, n_ap1, num_lm,
                           dtype=rij.dtype, device=rij.device)
    sum_fxyz.scatter_add_(
        0,
        pair_i[:, None, None].expand_as(gn_blm),
        gn_blm,
    )

    # q_angular from sum_fxyz (same formula as before)
    s = sum_fxyz
    parts = []

    # 3-body terms (L=1..L_max)
    q_3b_list = []
    for li in range(L_max):
        L = li + 1
        nt_lm = 2 * L + 1
        st = L * L - 1
        c = c3b_coeffs[st:st + nt_lm]
        sb2 = s[:, :, st:st + nt_lm] ** 2
        ql = c[0] * sb2[:, :, 0]
        if nt_lm > 1:
            ql = ql + 2.0 * (c[1:] * sb2[:, :, 1:]).sum(-1)
        q_3b_list.append(ql)
    q_3b = torch.stack(q_3b_list, dim=-1).transpose(1, 2).reshape(N, -1)
    parts.append(q_3b)

    # 4-body
    if num_L >= L_max + 1:
        s20, s21r, s21i = s[:, :, 3], s[:, :, 4], s[:, :, 5]
        s22r, s22i = s[:, :, 6], s[:, :, 7]
        cb = c4b_coeffs
        q4 = (cb[0] * s20 ** 3
              + cb[1] * s20 * (s21r ** 2 + s21i ** 2)
              + cb[2] * s20 * (s22r ** 2 + s22i ** 2)
              + cb[3] * s22r * (s21i ** 2 - s21r ** 2)
              + cb[4] * s21r * s21i * s22i)
        parts.append(q4)

    # 5-body
    if num_L >= L_max + 2:
        s0sq = s[:, :, 0] ** 2
        s1sq = s[:, :, 1] ** 2 + s[:, :, 2] ** 2
        cb5 = c5b_coeffs
        q5 = cb5[0] * s0sq ** 2 + cb5[1] * s0sq * s1sq + cb5[2] * s1sq ** 2
        parts.append(q5)

    return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# Cached type-pair contraction: CUDA-accelerated for training path
# ---------------------------------------------------------------------------

_cached_kernels = None


def _load_cached_kernels():
    """JIT-compile cached contraction CUDA kernels."""
    global _cached_kernels
    if _cached_kernels is not None:
        return _cached_kernels
    if os.environ.get("TORCHNEP_NO_CUDA_KERNELS", "0") == "1":
        return None
    if not torch.cuda.is_available():
        return None
    verbose = os.environ.get("TORCHNEP_VERBOSE_BUILD", "0") == "1"
    try:
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(__file__), "csrc", "nep_cached.cu")
        if not os.path.exists(src):
            return None
        extra_cflags = []
        extra_cuda = ["-O3"]
        if sys.platform == "win32":
            extra_cflags = ["/permissive-"]
            extra_cuda.append("-Xcompiler=/permissive-")
        _cached_kernels = load(
            name="nep_cached", sources=[src], verbose=verbose,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda)
        return _cached_kernels
    except Exception:
        if verbose:
            import traceback
            traceback.print_exc()
        return None


class ScatterContraction(torch.autograd.Function):
    """Fused type-lookup + contraction + scatter for cached descriptor path.

    Forward:  CUDA kernel computes q AND pre-accumulates dfeat_c for backward.
    Backward: CUDA kernel uses dfeat_c (per-atom, not per-pair) for grad_c.

    Only differentiable w.r.t. c (basis is precomputed/detached).
    """

    @staticmethod
    def forward(ctx, basis, pair_i, pair_j, atom_types, c, N):
        k = _load_cached_kernels()
        ntypes = c.shape[0]
        N_out = c.shape[2]
        K = c.shape[3]

        if k is not None and basis.is_cuda:
            result = k.scatter_contraction_forward(
                basis.contiguous(), pair_i, pair_j, atom_types,
                c.contiguous(), N, N_out, ntypes)
            q, dfeat_c = result[0], result[1]
        else:
            # Vectorized PyTorch fallback
            t1 = atom_types[pair_i]
            t2 = atom_types[pair_j]
            c_p = c[t1, t2]  # (P, N_out, K)
            gn = (c_p * basis.unsqueeze(1)).sum(-1)  # (P, N_out)
            q = torch.zeros(N, N_out, dtype=basis.dtype, device=basis.device)
            q.scatter_add_(0, pair_i.unsqueeze(-1).expand_as(gn), gn)
            # Build dfeat_c for backward: dfeat_c[i, t2, k] = sum_{p: pair_i=i, type_j=t2} basis[p, k]
            dfeat_c = torch.zeros(N, ntypes, K, dtype=basis.dtype, device=basis.device)
            for p_idx in range(basis.shape[0]):
                i = pair_i[p_idx].item()
                tj = atom_types[pair_j[p_idx]].item()
                dfeat_c[i, tj] += basis[p_idx]

        ctx.save_for_backward(dfeat_c, atom_types)
        ctx.N_out = N_out
        ctx.K = K
        ctx.ntypes = ntypes
        return q

    @staticmethod
    def backward(ctx, grad_q):
        dfeat_c, atom_types = ctx.saved_tensors
        ntypes = ctx.ntypes
        N_out = ctx.N_out
        K = ctx.K

        k = _load_cached_kernels()
        if k is not None and grad_q.is_cuda:
            grad_c = k.scatter_contraction_backward(
                grad_q.contiguous(), dfeat_c.contiguous(),
                atom_types, N_out, K, ntypes)
        else:
            # PyTorch fallback: per-atom outer product
            N = grad_q.shape[0]
            grad_c = torch.zeros(ntypes, ntypes, N_out, K,
                                 dtype=grad_q.dtype, device=grad_q.device)
            for i in range(N):
                t1 = atom_types[i].item()
                for t2 in range(ntypes):
                    # grad_c[t1, t2, n, k] += grad_q[i, n] * dfeat_c[i, t2, k]
                    grad_c[t1, t2] += grad_q[i].unsqueeze(-1) * dfeat_c[i, t2].unsqueeze(0)

        return None, None, None, None, grad_c, None


class TypeContraction(torch.autograd.Function):
    """Type-lookup + contraction (no scatter) for cached angular path.

    Forward:  CUDA kernel for gn[p,n] = Σ_k c[t1,t2,n,k] * basis[p,k].
    Backward: CUDA kernel for grad_c[t1,t2,n,k] += grad_gn[p,n] * basis[p,k].

    Only differentiable w.r.t. c.
    """

    @staticmethod
    def forward(ctx, basis, pair_i, pair_j, atom_types, c):
        k = _load_cached_kernels()
        ntypes = c.shape[0]
        N_out = c.shape[2]
        K = c.shape[3]

        if k is not None and basis.is_cuda:
            gn = k.type_contraction_forward(
                basis.contiguous(), pair_i, pair_j, atom_types,
                c.contiguous(), N_out, ntypes)
        else:
            t1 = atom_types[pair_i]
            t2 = atom_types[pair_j]
            c_p = c[t1, t2]
            gn = (c_p * basis.unsqueeze(1)).sum(-1)

        ctx.save_for_backward(basis, pair_i, pair_j, atom_types)
        ctx.c_shape = c.shape
        ctx.ntypes = ntypes
        ctx.N_out = N_out
        ctx.K = K
        return gn

    @staticmethod
    def backward(ctx, grad_gn):
        basis, pair_i, pair_j, atom_types = ctx.saved_tensors
        ntypes = ctx.ntypes
        N_out = ctx.N_out
        K = ctx.K

        k = _load_cached_kernels()
        if k is not None and grad_gn.is_cuda:
            grad_c = k.type_contraction_backward(
                grad_gn.contiguous(), basis.contiguous(),
                pair_i, pair_j, atom_types,
                N_out, K, ntypes)
        else:
            # PyTorch fallback: loop-based matmul
            t1 = atom_types[pair_i]
            t2 = atom_types[pair_j]
            grad_c = torch.zeros(ctx.c_shape, dtype=basis.dtype, device=basis.device)
            for _t1 in range(ntypes):
                for _t2 in range(ntypes):
                    _m = (t1 == _t1) & (t2 == _t2)
                    if _m.any():
                        grad_c[_t1, _t2] = grad_gn[_m].T @ basis[_m]

        return None, None, None, None, grad_c


def scatter_contraction(basis, pair_i, pair_j, atom_types, c, N):
    """Fused type-contraction + scatter_add. Differentiable w.r.t. c."""
    return ScatterContraction.apply(basis, pair_i, pair_j, atom_types, c, N)


def type_contraction(basis, pair_i, pair_j, atom_types, c):
    """Type-pair contraction without scatter. Differentiable w.r.t. c."""
    return TypeContraction.apply(basis, pair_i, pair_j, atom_types, c)
