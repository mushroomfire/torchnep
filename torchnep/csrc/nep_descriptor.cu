/**
 * Fused NEP descriptor + analytical force CUDA kernel.
 *
 * Computes descriptors and forces in a single pass, avoiding PyTorch's
 * create_graph=True overhead (which causes 89% of training time).
 *
 * Key operations fused:
 *   1. Chebyshev basis + derivative (fk, fkp)
 *   2. c_param contraction → gn, gnp
 *   3. scatter_add → per-atom descriptors q
 *   4. Analytical radial forces from Fp
 *
 * The angular force is computed in PyTorch (vectorized) since it requires
 * the accumulated s values which are shared across pairs.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {
constexpr double NEP_PI = 3.141592653589793;
constexpr int MAX_BASIS = 20;
constexpr int MAX_NMAX = 20;

template <typename scalar_t>
__device__ void compute_chebyshev_and_deriv(
    scalar_t d, scalar_t rcinv, int basis_size,
    scalar_t* __restrict__ fk, scalar_t* __restrict__ fkp
) {
    scalar_t fc = 0.5 * cos(NEP_PI * d * rcinv) + 0.5;
    scalar_t fcp = -0.5 * NEP_PI * rcinv * sin(NEP_PI * d * rcinv);
    scalar_t x = 2.0 * (d * rcinv - 1.0) * (d * rcinv - 1.0) - 1.0;
    scalar_t dxdr = 4.0 * (d * rcinv - 1.0) * rcinv;

    // T and U (Chebyshev first and second kind)
    scalar_t T[MAX_BASIS], U[MAX_BASIS];
    T[0] = 1.0; T[1] = x;
    U[0] = 1.0; U[1] = 2.0 * x;
    for (int k = 2; k <= basis_size; k++) {
        T[k] = 2.0 * x * T[k-1] - T[k-2];
        U[k] = 2.0 * x * U[k-1] - U[k-2];
    }
    for (int k = 0; k <= basis_size; k++) {
        scalar_t fn_core = 0.5 * (T[k] + 1.0);
        fk[k] = fn_core * fc;
        if (k == 0)
            fkp[k] = fcp;
        else if (k == 1)
            fkp[k] = 0.5 * dxdr * fc + fn_core * fcp;
        else
            fkp[k] = 0.5 * k * U[k-1] * dxdr * fc + fn_core * fcp;
    }
}

// Fused radial descriptor + force contribution per atom
template <typename scalar_t>
__global__ void radial_descriptor_and_force_kernel(
    const scalar_t* __restrict__ rij,        // (P, 3)
    const int64_t* __restrict__ pair_i,      // (P,)
    const int64_t* __restrict__ pair_j,      // (P,)
    const int64_t* __restrict__ atom_types,  // (N,)
    const scalar_t* __restrict__ c2,         // (nt, nt, dim_r, bs1)
    const scalar_t* __restrict__ Fp_rad,     // (N, dim_r) — NULL if no force
    int N, int P, int dim_r, int bs1, int num_types,
    scalar_t rc_radial,
    scalar_t* __restrict__ q_out,            // (N, dim_r) — output descriptors
    scalar_t* __restrict__ f_out             // (N, 3) — output forces (NULL if no force)
) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= N) return;

    scalar_t rcinv = 1.0 / rc_radial;
    int t1 = atom_types[atom];

    // Zero output for this atom
    for (int n = 0; n < dim_r; n++) q_out[atom * dim_r + n] = 0;

    scalar_t fx = 0, fy = 0, fz = 0;

    for (int p = 0; p < P; p++) {
        if (pair_i[p] != atom) continue;

        scalar_t dx = rij[p*3], dy = rij[p*3+1], dz = rij[p*3+2];
        scalar_t d = sqrt(dx*dx + dy*dy + dz*dz);
        if (d >= rc_radial || d < 1e-10) continue;

        int t2 = atom_types[pair_j[p]];
        scalar_t fk[MAX_BASIS], fkp[MAX_BASIS];
        compute_chebyshev_and_deriv(d, rcinv, bs1 - 1, fk, fkp);

        int c_base = (t1 * num_types + t2) * dim_r * bs1;
        scalar_t d12inv = 1.0 / d;

        for (int n = 0; n < dim_r; n++) {
            scalar_t gn = 0, gnp = 0;
            for (int k = 0; k < bs1; k++) {
                scalar_t c_val = c2[c_base + n * bs1 + k];
                gn += c_val * fk[k];
                gnp += c_val * fkp[k];
            }
            q_out[atom * dim_r + n] += gn;

            // Force contribution
            if (Fp_rad != nullptr && f_out != nullptr) {
                scalar_t tmp = Fp_rad[atom * dim_r + n] * gnp * d12inv;
                fx += tmp * dx;
                fy += tmp * dy;
                fz += tmp * dz;
            }
        }
    }

    if (Fp_rad != nullptr && f_out != nullptr) {
        atomicAdd(&f_out[atom * 3 + 0], fx);
        atomicAdd(&f_out[atom * 3 + 1], fy);
        atomicAdd(&f_out[atom * 3 + 2], fz);
    }
}

// Kernel to accumulate force on neighbor atoms (Newton's 3rd law)
template <typename scalar_t>
__global__ void radial_force_neighbor_kernel(
    const scalar_t* __restrict__ rij,
    const int64_t* __restrict__ pair_i,
    const int64_t* __restrict__ pair_j,
    const int64_t* __restrict__ atom_types,
    const scalar_t* __restrict__ c2,
    const scalar_t* __restrict__ Fp_rad,
    int N, int P, int dim_r, int bs1, int num_types,
    scalar_t rc_radial,
    scalar_t* __restrict__ f_out,
    scalar_t* __restrict__ virial_out  // (N, 9)
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;

    int atom_i = pair_i[p];
    int atom_j = pair_j[p];

    scalar_t dx = rij[p*3], dy = rij[p*3+1], dz = rij[p*3+2];
    scalar_t d = sqrt(dx*dx + dy*dy + dz*dz);
    if (d >= rc_radial || d < 1e-10) return;

    scalar_t rcinv = 1.0 / rc_radial;
    int t1 = atom_types[atom_i];
    int t2 = atom_types[atom_j];

    scalar_t fk[MAX_BASIS], fkp[MAX_BASIS];
    compute_chebyshev_and_deriv(d, rcinv, bs1 - 1, fk, fkp);

    int c_base = (t1 * num_types + t2) * dim_r * bs1;
    scalar_t d12inv = 1.0 / d;

    scalar_t fx = 0, fy = 0, fz = 0;
    for (int n = 0; n < dim_r; n++) {
        scalar_t gnp = 0;
        for (int k = 0; k < bs1; k++)
            gnp += c2[c_base + n * bs1 + k] * fkp[k];
        scalar_t tmp = Fp_rad[atom_i * dim_r + n] * gnp * d12inv;
        fx += tmp * dx;
        fy += tmp * dy;
        fz += tmp * dz;
    }

    // Newton's 3rd law: subtract from neighbor
    atomicAdd(&f_out[atom_j * 3 + 0], -fx);
    atomicAdd(&f_out[atom_j * 3 + 1], -fy);
    atomicAdd(&f_out[atom_j * 3 + 2], -fz);

    // Virial on j: v_ab -= rij_a * f12_b
    if (virial_out != nullptr) {
        atomicAdd(&virial_out[atom_j * 9 + 0], -dx * fx);
        atomicAdd(&virial_out[atom_j * 9 + 1], -dx * fy);
        atomicAdd(&virial_out[atom_j * 9 + 2], -dx * fz);
        atomicAdd(&virial_out[atom_j * 9 + 3], -dy * fx);
        atomicAdd(&virial_out[atom_j * 9 + 4], -dy * fy);
        atomicAdd(&virial_out[atom_j * 9 + 5], -dy * fz);
        atomicAdd(&virial_out[atom_j * 9 + 6], -dz * fx);
        atomicAdd(&virial_out[atom_j * 9 + 7], -dz * fy);
        atomicAdd(&virial_out[atom_j * 9 + 8], -dz * fz);
    }
}

} // namespace


// Python-visible: fused radial descriptor
torch::Tensor cuda_radial_descriptor_v2(
    torch::Tensor rij, torch::Tensor pair_i, torch::Tensor pair_j,
    torch::Tensor atom_types, torch::Tensor c2,
    int N, int dim_r, int bs1, int num_types, double rc_radial
) {
    auto q = torch::zeros({N, dim_r}, rij.options());
    int P = rij.size(0);
    int threads = 256, blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(rij.scalar_type(), "radial_desc_v2", [&] {
        radial_descriptor_and_force_kernel<scalar_t><<<blocks, threads>>>(
            rij.data_ptr<scalar_t>(), pair_i.data_ptr<int64_t>(),
            pair_j.data_ptr<int64_t>(), atom_types.data_ptr<int64_t>(),
            c2.data_ptr<scalar_t>(), nullptr,
            N, P, dim_r, bs1, num_types, static_cast<scalar_t>(rc_radial),
            q.data_ptr<scalar_t>(), nullptr);
    });
    return q;
}


// Python-visible: fused radial force (both i and j atoms)
std::tuple<torch::Tensor, torch::Tensor> cuda_radial_force(
    torch::Tensor rij, torch::Tensor pair_i, torch::Tensor pair_j,
    torch::Tensor atom_types, torch::Tensor c2, torch::Tensor Fp_rad,
    int N, int dim_r, int bs1, int num_types, double rc_radial,
    bool compute_virial
) {
    auto forces = torch::zeros({N, 3}, rij.options());
    auto virial = compute_virial ? torch::zeros({N, 9}, rij.options()) : torch::Tensor();
    int P = rij.size(0);

    // Force on center atoms (pair_i)
    int threads = 256, blocks = (N + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(rij.scalar_type(), "rad_force_i", [&] {
        radial_descriptor_and_force_kernel<scalar_t><<<blocks, threads>>>(
            rij.data_ptr<scalar_t>(), pair_i.data_ptr<int64_t>(),
            pair_j.data_ptr<int64_t>(), atom_types.data_ptr<int64_t>(),
            c2.data_ptr<scalar_t>(), Fp_rad.data_ptr<scalar_t>(),
            N, P, dim_r, bs1, num_types, static_cast<scalar_t>(rc_radial),
            torch::zeros({N, dim_r}, rij.options()).data_ptr<scalar_t>(),
            forces.data_ptr<scalar_t>());
    });

    // Force on neighbor atoms (Newton's 3rd) + virial
    blocks = (P + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(rij.scalar_type(), "rad_force_j", [&] {
        radial_force_neighbor_kernel<scalar_t><<<blocks, threads>>>(
            rij.data_ptr<scalar_t>(), pair_i.data_ptr<int64_t>(),
            pair_j.data_ptr<int64_t>(), atom_types.data_ptr<int64_t>(),
            c2.data_ptr<scalar_t>(), Fp_rad.data_ptr<scalar_t>(),
            N, P, dim_r, bs1, num_types, static_cast<scalar_t>(rc_radial),
            forces.data_ptr<scalar_t>(),
            compute_virial ? virial.data_ptr<scalar_t>() : nullptr);
    });

    return {forces, virial};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("radial_descriptor", &cuda_radial_descriptor_v2,
          "Fused radial descriptor (CUDA)");
    m.def("radial_force", &cuda_radial_force,
          "Fused radial force + virial (CUDA)");
}
