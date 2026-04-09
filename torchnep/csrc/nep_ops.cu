/**
 * CUDA kernels for NEP descriptor computation and force accumulation.
 *
 * Fused kernels that eliminate Python-level scatter_add overhead:
 *   1. radial_descriptor_kernel  — Chebyshev basis + c_param contraction + scatter
 *   2. angular_descriptor_kernel — angular basis + gn + scatter + q computation
 *   3. force_accumulation_kernel — pair gradients → per-atom forces + virial
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr double NEP_PI = 3.141592653589793;
constexpr int MAX_BASIS = 20;   // max basis_size + 1
constexpr int MAX_NMAX = 20;    // max n_max + 1
constexpr int NUM_LM = 24;     // max angular components for l_max=4

// ---- Chebyshev basis f_k(r) = 0.5*(T_k(x)+1)*fc(r) ----

template <typename scalar_t>
__device__ void compute_chebyshev(
    scalar_t d, scalar_t rcinv, int basis_size, scalar_t* fk
) {
    scalar_t fc = 0.5 * cos(NEP_PI * d * rcinv) + 0.5;
    scalar_t x = 2.0 * (d * rcinv - 1.0) * (d * rcinv - 1.0) - 1.0;

    scalar_t t0 = 1.0, t1 = x;
    fk[0] = 0.5 * (t0 + 1.0) * fc;
    if (basis_size >= 1) fk[1] = 0.5 * (t1 + 1.0) * fc;
    for (int k = 2; k <= basis_size; k++) {
        scalar_t t2 = 2.0 * x * t1 - t0;
        fk[k] = 0.5 * (t2 + 1.0) * fc;
        t0 = t1; t1 = t2;
    }
}

// ---- Angular basis (solid harmonics, l=1..4) ----

template <typename scalar_t>
__device__ void compute_angular_basis(
    scalar_t xh, scalar_t yh, scalar_t zh, int l_max_3b, scalar_t* blm
) {
    scalar_t x2 = xh*xh, y2 = yh*yh, z2 = zh*zh;
    scalar_t x2my2 = x2 - y2;
    int idx = 0;
    if (l_max_3b >= 1) {
        blm[idx++] = zh;
        blm[idx++] = xh;
        blm[idx++] = yh;
    }
    if (l_max_3b >= 2) {
        blm[idx++] = 3.0*z2 - 1.0;
        blm[idx++] = xh*zh;
        blm[idx++] = yh*zh;
        blm[idx++] = x2my2;
        blm[idx++] = 2.0*xh*yh;
    }
    if (l_max_3b >= 3) {
        blm[idx++] = (5.0*z2 - 3.0)*zh;
        blm[idx++] = (5.0*z2 - 1.0)*xh;
        blm[idx++] = (5.0*z2 - 1.0)*yh;
        blm[idx++] = x2my2*zh;
        blm[idx++] = 2.0*xh*yh*zh;
        blm[idx++] = xh*(x2 - 3.0*y2);
        blm[idx++] = yh*(3.0*x2 - y2);
    }
    if (l_max_3b >= 4) {
        blm[idx++] = (35.0*z2 - 30.0)*z2 + 3.0;
        blm[idx++] = (7.0*z2 - 3.0)*xh*zh;
        blm[idx++] = (7.0*z2 - 3.0)*yh*zh;
        blm[idx++] = (7.0*z2 - 1.0)*x2my2;
        blm[idx++] = (7.0*z2 - 1.0)*2.0*xh*yh;
        blm[idx++] = zh*xh*(x2 - 3.0*y2);
        blm[idx++] = zh*yh*(3.0*x2 - y2);
        blm[idx++] = x2my2*x2my2 - 4.0*x2*y2;
        blm[idx++] = 4.0*xh*yh*x2my2;
    }
}


// ---- Kernel: Radial descriptor per atom ----
// One thread per atom. Loops over radial neighbors.

template <typename scalar_t>
__global__ void radial_descriptor_kernel(
    const scalar_t* __restrict__ rij,       // (P_rad, 3)
    const int64_t* __restrict__ pair_i,     // (P_rad,)
    const int64_t* __restrict__ pair_j,     // (P_rad,)
    const int64_t* __restrict__ atom_types, // (N,)
    const scalar_t* __restrict__ c2,        // (nt, nt, n_max+1, basis+1)
    int N, int P_rad,
    int n_max_r, int basis_size_r, int num_types,
    scalar_t rc_radial,
    scalar_t* __restrict__ q_out            // (N, n_max+1)
) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= N) return;

    int dim_r = n_max_r + 1;
    int bs1 = basis_size_r + 1;
    scalar_t rcinv = 1.0 / rc_radial;

    // Zero output
    for (int n = 0; n < dim_r; n++) q_out[atom * dim_r + n] = 0;

    // Loop over all pairs (not ideal but works for now)
    for (int p = 0; p < P_rad; p++) {
        if (pair_i[p] != atom) continue;

        scalar_t dx = rij[p*3+0], dy = rij[p*3+1], dz = rij[p*3+2];
        scalar_t d = sqrt(dx*dx + dy*dy + dz*dz);
        if (d >= rc_radial || d < 1e-10) continue;

        int t1 = atom_types[atom];
        int t2 = atom_types[pair_j[p]];

        scalar_t fk[MAX_BASIS];
        compute_chebyshev(d, rcinv, basis_size_r, fk);

        // c2 layout: (nt, nt, n_max+1, basis+1) → index [t1, t2, n, k]
        int c_base = (t1 * num_types + t2) * dim_r * bs1;
        for (int n = 0; n < dim_r; n++) {
            scalar_t gn = 0;
            for (int k = 0; k < bs1; k++) {
                gn += c2[c_base + n * bs1 + k] * fk[k];
            }
            q_out[atom * dim_r + n] += gn;
        }
    }
}


// ---- Kernel: Force + virial accumulation ----
// One thread per atom. Accumulates pair gradients.

template <typename scalar_t>
__global__ void force_virial_kernel(
    const scalar_t* __restrict__ rij,       // (P, 3)
    const scalar_t* __restrict__ grad,      // (P, 3)
    const int64_t* __restrict__ pair_i,     // (P,)
    const int64_t* __restrict__ pair_j,     // (P,)
    int N, int P,
    scalar_t* __restrict__ forces,          // (N, 3)
    scalar_t* __restrict__ virial           // (N, 9)
) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= N) return;

    scalar_t fx = 0, fy = 0, fz = 0;

    for (int p = 0; p < P; p++) {
        int pi = pair_i[p], pj = pair_j[p];
        scalar_t gx = grad[p*3+0], gy = grad[p*3+1], gz = grad[p*3+2];

        if (pi == atom) { fx += gx; fy += gy; fz += gz; }
        if (pj == atom) { fx -= gx; fy -= gy; fz -= gz; }

        // Virial on j: v_ab -= rij_a * grad_b
        if (pj == atom) {
            scalar_t rx = rij[p*3+0], ry = rij[p*3+1], rz = rij[p*3+2];
            virial[atom*9+0] -= rx * gx;
            virial[atom*9+1] -= rx * gy;
            virial[atom*9+2] -= rx * gz;
            virial[atom*9+3] -= ry * gx;
            virial[atom*9+4] -= ry * gy;
            virial[atom*9+5] -= ry * gz;
            virial[atom*9+6] -= rz * gx;
            virial[atom*9+7] -= rz * gy;
            virial[atom*9+8] -= rz * gz;
        }
    }
    forces[atom*3+0] = fx;
    forces[atom*3+1] = fy;
    forces[atom*3+2] = fz;
}

} // anonymous namespace


// ---- Python-visible functions ----

torch::Tensor cuda_radial_descriptor(
    torch::Tensor rij,        // (P_rad, 3)
    torch::Tensor pair_i,     // (P_rad,)
    torch::Tensor pair_j,     // (P_rad,)
    torch::Tensor atom_types, // (N,)
    torch::Tensor c2,         // (nt, nt, n_max+1, basis+1)
    int N, int n_max_r, int basis_size_r, int num_types,
    double rc_radial
) {
    int dim_r = n_max_r + 1;
    auto q = torch::zeros({N, dim_r}, rij.options());
    int P = rij.size(0);

    // Reorder c2 from (nt,nt,n,k) to be contiguous for access pattern
    auto c2_c = c2.permute({0, 1, 2, 3}).contiguous();

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(rij.scalar_type(), "radial_descriptor", [&] {
        radial_descriptor_kernel<scalar_t><<<blocks, threads>>>(
            rij.data_ptr<scalar_t>(),
            pair_i.data_ptr<int64_t>(),
            pair_j.data_ptr<int64_t>(),
            atom_types.data_ptr<int64_t>(),
            c2_c.data_ptr<scalar_t>(),
            N, P, n_max_r, basis_size_r, num_types,
            static_cast<scalar_t>(rc_radial),
            q.data_ptr<scalar_t>()
        );
    });

    return q;
}


std::tuple<torch::Tensor, torch::Tensor> cuda_force_virial(
    torch::Tensor rij,         // (P, 3)
    torch::Tensor grad,        // (P, 3)
    torch::Tensor pair_i,      // (P,)
    torch::Tensor pair_j,      // (P,)
    int N
) {
    auto forces = torch::zeros({N, 3}, rij.options());
    auto virial = torch::zeros({N, 9}, rij.options());
    int P = rij.size(0);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(rij.scalar_type(), "force_virial", [&] {
        force_virial_kernel<scalar_t><<<blocks, threads>>>(
            rij.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            pair_i.data_ptr<int64_t>(),
            pair_j.data_ptr<int64_t>(),
            N, P,
            forces.data_ptr<scalar_t>(),
            virial.data_ptr<scalar_t>()
        );
    });

    return {forces, virial};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("radial_descriptor", &cuda_radial_descriptor,
          "Fused radial descriptor computation (CUDA)");
    m.def("force_virial", &cuda_force_virial,
          "Force and virial accumulation (CUDA)");
}
