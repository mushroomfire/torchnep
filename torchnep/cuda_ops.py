"""CUDA-accelerated ops for the training (cached) path.

Only the two ops actually used on the training hot path are kept here:
  ScatterContraction : fused type-contraction + scatter_add  (radial descriptor)
  TypeContraction    : type-contraction (no scatter)         (angular descriptor, gnp_*)

Both are autograd.Function with CUDA forward + CUDA-or-matmul backward and
are fully differentiable with respect to ``c`` (the learnable 3-body / 4-body
coupling parameters).

The eval-only CUDA paths (radial/angular descriptor via autograd + rij,
force/virial accumulator) were removed — eval uses pure PyTorch now.
"""

import os
import sys
import traceback
import torch
from torch.utils.cpp_extension import load as _cpp_extension_load


_cached_kernels = None


def _ensure_ninja_in_path():
    """Add the current Python env's bin dir to PATH so ninja is found."""
    python_bin = os.path.dirname(sys.executable)
    path = os.environ.get("PATH", "")
    if python_bin not in path.split(os.pathsep):
        os.environ["PATH"] = python_bin + os.pathsep + path


def _load_cached_kernels():
    """Compile (first call) or load (cached) the CUDA contraction kernels.

    In DDP / torchrun setups, ONLY rank 0 should call this first. Other ranks
    must wait on a ``dist.barrier()`` and then call afterwards so they pick
    up the cached ``.so`` without racing on the torch-extensions cache.

    Set ``TORCHNEP_NO_CUDA_KERNELS=1`` to force pure-PyTorch fallback.
    Set ``TORCHNEP_VERBOSE_BUILD=1`` to stream nvcc output during compilation.
    """
    global _cached_kernels
    if _cached_kernels is not None:
        return _cached_kernels
    if os.environ.get("TORCHNEP_NO_CUDA_KERNELS", "0") == "1":
        return None
    if not torch.cuda.is_available():
        return None

    src = os.path.join(os.path.dirname(__file__), "csrc", "nep_cached.cu")
    if not os.path.exists(src):
        return None

    _ensure_ninja_in_path()
    verbose = os.environ.get("TORCHNEP_VERBOSE_BUILD", "0") == "1"

    # Detect whether a JIT compile will actually happen, so we only emit the
    # "compiling" message on a cold cache. torch's extension dir layout:
    #   <cache>/nep_cached/nep_cached.so
    from torch.utils.cpp_extension import _get_build_directory
    build_dir = _get_build_directory("nep_cached", verbose=False)
    will_compile = not any(f.endswith(".so") for f in os.listdir(build_dir)) \
                   if os.path.isdir(build_dir) else True
    if will_compile:
        print("Compiling torchnep CUDA kernels (nep_cached.cu) — "
              "this takes ~30–90 s the first time, then cached forever.",
              flush=True)

    extra_cflags = []
    extra_cuda = ["-O3"]
    if sys.platform == "win32":
        extra_cflags = ["/permissive-"]
        extra_cuda.append("-Xcompiler=/permissive-")

    try:
        _cached_kernels = _cpp_extension_load(
            name="nep_cached", sources=[src], verbose=verbose,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda)
        if will_compile:
            print("  → nep_cached compiled and cached.", flush=True)
        return _cached_kernels
    except Exception:
        if verbose:
            traceback.print_exc()
        return None


class ScatterContraction(torch.autograd.Function):
    """Fused type-lookup + contraction + scatter.

    Forward:  CUDA kernel computes q AND pre-accumulates dfeat_c for backward.
    Backward: CUDA kernel uses dfeat_c (per-atom) for grad_c (no atomicAdd).

    Only differentiable w.r.t. c (basis is precomputed / detached).
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
            # Vectorized PyTorch fallback (CPU / kernel unavailable)
            t1 = atom_types[pair_i]
            t2 = atom_types[pair_j]
            c_p = c[t1, t2]                                   # (P, N_out, K)
            gn = (c_p * basis.unsqueeze(1)).sum(-1)            # (P, N_out)
            q = torch.zeros(N, N_out, dtype=basis.dtype, device=basis.device)
            q.scatter_add_(0, pair_i.unsqueeze(-1).expand_as(gn), gn)
            # dfeat_c[i, t2, k] = sum_{p: pair_i=i, type_j=t2} basis[p, k]
            dfeat_c = torch.zeros(N, ntypes, K,
                                  dtype=basis.dtype, device=basis.device)
            for _t2 in range(ntypes):
                _m = t2 == _t2
                if _m.any():
                    dfeat_c[:, _t2].index_add_(0, pair_i[_m], basis[_m])

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
            # PyTorch fallback: per-type outer product
            grad_c = torch.zeros(ntypes, ntypes, N_out, K,
                                 dtype=grad_q.dtype, device=grad_q.device)
            for _t1 in range(ntypes):
                _m = atom_types == _t1
                if not _m.any():
                    continue
                # grad_c[t1, t2, n, k] = sum_{i: at[i]=t1} grad_q[i, n] * dfeat_c[i, t2, k]
                gq = grad_q[_m]                    # (M, N_out)
                df = dfeat_c[_m]                   # (M, ntypes, K)
                for _t2 in range(ntypes):
                    grad_c[_t1, _t2] = gq.T @ df[:, _t2]

        return None, None, None, None, grad_c, None


class TypeContraction(torch.autograd.Function):
    """Type-lookup + contraction (no scatter).

    Forward:  CUDA kernel for gn[p, n] = Σ_k c[t1,t2,n,k] * basis[p, k].
    Backward: Type-grouped cuBLAS matmul  grad_c[t1,t2] = grad_gn[mask].T @ basis[mask].
              (No atomicAdd → much faster than a naive per-pair CUDA backward.)

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
        return gn

    @staticmethod
    def backward(ctx, grad_gn):
        basis, pair_i, pair_j, atom_types = ctx.saved_tensors
        ntypes = ctx.ntypes

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
