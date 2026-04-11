/**
 * CUDA kernels for cached descriptor path (training).
 *
 * Key design:
 *   Forward: fuses type-lookup + contraction + scatter, AND pre-accumulates
 *            dfeat_c[atom_i, type_j, k] = Σ_{pairs} basis[p,k] for backward.
 *   Backward: per-atom outer product using dfeat_c (N atoms, not P pairs),
 *             dramatically reducing atomicAdd conflicts.
 *
 * Operations:
 *   scatter_contraction: q[i,n] = Σ_{p:pair_i=i} Σ_k c[t1,t2,n,k] * basis[p,k]
 *   type_contraction:    gn[p,n] = Σ_k c[t1,t2,n,k] * basis[p,k]
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

// ---------------------------------------------------------------------------
// Scatter contraction forward: 1 thread per pair
// Computes q via atomicAdd AND accumulates dfeat_c for backward
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void scatter_contraction_forward_kernel(
    const scalar_t* __restrict__ basis,       // (P, K)
    const int64_t* __restrict__ pair_i,       // (P,)
    const int64_t* __restrict__ pair_j,       // (P,)
    const int64_t* __restrict__ atom_types,   // (N,)
    const scalar_t* __restrict__ c,           // (nt, nt, N_out, K)
    int P, int N_out, int K, int ntypes,
    scalar_t* __restrict__ q_out,             // (N, N_out) — pre-zeroed
    scalar_t* __restrict__ dfeat_c            // (N, ntypes, K) — pre-zeroed
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;

    int atom_i = pair_i[p];
    int t1 = atom_types[atom_i];
    int t2 = atom_types[pair_j[p]];
    int c_base = (t1 * ntypes + t2) * N_out * K;

    for (int n = 0; n < N_out; n++) {
        scalar_t val = 0;
        for (int k = 0; k < K; k++) {
            val += c[c_base + n * K + k] * basis[p * K + k];
        }
        atomicAdd(&q_out[atom_i * N_out + n], val);
    }

    // Accumulate dfeat_c: sum of basis values grouped by (atom_i, type_j)
    // dfeat_c[atom_i, t2, k] += basis[p, k]
    int dfeat_base = atom_i * ntypes * K + t2 * K;
    for (int k = 0; k < K; k++) {
        atomicAdd(&dfeat_c[dfeat_base + k], basis[p * K + k]);
    }
}

// ---------------------------------------------------------------------------
// Scatter contraction backward: 1 thread per atom
// Uses pre-accumulated dfeat_c (N atoms, not P pairs)
// grad_c[t1, t2, n, k] += grad_q[i, n] * dfeat_c[i, t2, k]
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void scatter_contraction_backward_kernel(
    const scalar_t* __restrict__ grad_q,      // (N, N_out)
    const scalar_t* __restrict__ dfeat_c,     // (N, ntypes, K)
    const int64_t* __restrict__ atom_types,   // (N,)
    int N, int N_out, int K, int ntypes,
    scalar_t* __restrict__ grad_c             // (nt, nt, N_out, K) — pre-zeroed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int t1 = atom_types[i];
    for (int t2 = 0; t2 < ntypes; t2++) {
        int dfeat_base = i * ntypes * K + t2 * K;
        int c_base = (t1 * ntypes + t2) * N_out * K;
        for (int n = 0; n < N_out; n++) {
            scalar_t gq = grad_q[i * N_out + n];
            for (int k = 0; k < K; k++) {
                atomicAdd(&grad_c[c_base + n * K + k],
                          gq * dfeat_c[dfeat_base + k]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Type contraction forward: 1 thread per pair (no scatter)
// gn[p, n] = sum_k c[t1, t2, n, k] * basis[p, k]
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void type_contraction_forward_kernel(
    const scalar_t* __restrict__ basis,       // (P, K)
    const int64_t* __restrict__ pair_i,       // (P,)
    const int64_t* __restrict__ pair_j,       // (P,)
    const int64_t* __restrict__ atom_types,   // (N,)
    const scalar_t* __restrict__ c,           // (nt, nt, N_out, K)
    int P, int N_out, int K, int ntypes,
    scalar_t* __restrict__ gn_out             // (P, N_out)
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;

    int t1 = atom_types[pair_i[p]];
    int t2 = atom_types[pair_j[p]];
    int c_base = (t1 * ntypes + t2) * N_out * K;

    for (int n = 0; n < N_out; n++) {
        scalar_t val = 0;
        for (int k = 0; k < K; k++) {
            val += c[c_base + n * K + k] * basis[p * K + k];
        }
        gn_out[p * N_out + n] = val;
    }
}

// ---------------------------------------------------------------------------
// Type contraction backward: 1 thread per pair
// grad_c[t1, t2, n, k] += grad_gn[p, n] * basis[p, k]
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void type_contraction_backward_kernel(
    const scalar_t* __restrict__ grad_gn,     // (P, N_out)
    const scalar_t* __restrict__ basis,       // (P, K)
    const int64_t* __restrict__ pair_i,       // (P,)
    const int64_t* __restrict__ pair_j,       // (P,)
    const int64_t* __restrict__ atom_types,   // (N,)
    int P, int N_out, int K, int ntypes,
    scalar_t* __restrict__ grad_c             // (nt, nt, N_out, K) — pre-zeroed
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;

    int t1 = atom_types[pair_i[p]];
    int t2 = atom_types[pair_j[p]];
    int c_base = (t1 * ntypes + t2) * N_out * K;

    for (int n = 0; n < N_out; n++) {
        scalar_t g = grad_gn[p * N_out + n];
        for (int k = 0; k < K; k++) {
            atomicAdd(&grad_c[c_base + n * K + k], g * basis[p * K + k]);
        }
    }
}

} // anonymous namespace


// ===========================================================================
// Python-visible functions
// ===========================================================================

// Returns (q, dfeat_c) — dfeat_c saved for backward
std::vector<torch::Tensor> cuda_scatter_contraction_forward(
    torch::Tensor basis, torch::Tensor pair_i, torch::Tensor pair_j,
    torch::Tensor atom_types, torch::Tensor c,
    int N_atoms, int N_out, int ntypes
) {
    int P = basis.size(0);
    int K = basis.size(1);
    auto opts = basis.options();
    auto q = torch::zeros({N_atoms, N_out}, opts);
    auto dfeat_c = torch::zeros({N_atoms, ntypes, K}, opts);

    int threads = 256, blocks = (P + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(basis.scalar_type(), "scatter_contract_fwd", [&] {
        scatter_contraction_forward_kernel<scalar_t><<<blocks, threads>>>(
            basis.data_ptr<scalar_t>(), pair_i.data_ptr<int64_t>(),
            pair_j.data_ptr<int64_t>(), atom_types.data_ptr<int64_t>(),
            c.contiguous().data_ptr<scalar_t>(),
            P, N_out, K, ntypes,
            q.data_ptr<scalar_t>(), dfeat_c.data_ptr<scalar_t>());
    });
    return {q, dfeat_c};
}

// Backward using pre-accumulated dfeat_c (N atoms, not P pairs)
torch::Tensor cuda_scatter_contraction_backward(
    torch::Tensor grad_q, torch::Tensor dfeat_c,
    torch::Tensor atom_types,
    int N_out, int K, int ntypes
) {
    int N = grad_q.size(0);
    auto grad_c = torch::zeros({ntypes, ntypes, N_out, K}, grad_q.options());

    int threads = 256, blocks = (N + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(grad_q.scalar_type(), "scatter_contract_bwd", [&] {
        scatter_contraction_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_q.data_ptr<scalar_t>(), dfeat_c.data_ptr<scalar_t>(),
            atom_types.data_ptr<int64_t>(),
            N, N_out, K, ntypes,
            grad_c.data_ptr<scalar_t>());
    });
    return grad_c;
}

torch::Tensor cuda_type_contraction_forward(
    torch::Tensor basis, torch::Tensor pair_i, torch::Tensor pair_j,
    torch::Tensor atom_types, torch::Tensor c,
    int N_out, int ntypes
) {
    int P = basis.size(0);
    int K = basis.size(1);
    auto gn = torch::zeros({P, N_out}, basis.options());

    int threads = 256, blocks = (P + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(basis.scalar_type(), "type_contract_fwd", [&] {
        type_contraction_forward_kernel<scalar_t><<<blocks, threads>>>(
            basis.data_ptr<scalar_t>(), pair_i.data_ptr<int64_t>(),
            pair_j.data_ptr<int64_t>(), atom_types.data_ptr<int64_t>(),
            c.contiguous().data_ptr<scalar_t>(),
            P, N_out, K, ntypes, gn.data_ptr<scalar_t>());
    });
    return gn;
}

torch::Tensor cuda_type_contraction_backward(
    torch::Tensor grad_gn, torch::Tensor basis,
    torch::Tensor pair_i, torch::Tensor pair_j,
    torch::Tensor atom_types,
    int N_out, int K, int ntypes
) {
    int P = basis.size(0);
    auto grad_c = torch::zeros({ntypes, ntypes, N_out, K}, grad_gn.options());

    int threads = 256, blocks = (P + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(basis.scalar_type(), "type_contract_bwd", [&] {
        type_contraction_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_gn.data_ptr<scalar_t>(), basis.data_ptr<scalar_t>(),
            pair_i.data_ptr<int64_t>(), pair_j.data_ptr<int64_t>(),
            atom_types.data_ptr<int64_t>(),
            P, N_out, K, ntypes, grad_c.data_ptr<scalar_t>());
    });
    return grad_c;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scatter_contraction_forward", &cuda_scatter_contraction_forward);
    m.def("scatter_contraction_backward", &cuda_scatter_contraction_backward);
    m.def("type_contraction_forward", &cuda_type_contraction_forward);
    m.def("type_contraction_backward", &cuda_type_contraction_backward);
}
