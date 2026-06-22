/**
 * Solver.cu
 * Explicit FEM elasticity solver using CUDA.
 *
 * Pipeline per substep:
 *   1. ComputeForces  �?compute elastic + gravity forces on every vertex
 *   2. Integrate      �?symplectic-Euler velocity/position update
 *   3. BoundaryCheck  �?enforce axis-aligned bounding-box collisions
 *
 * Supported energy models (EnergyType):
 *   STVK        �?Saint-Venant Kirchhoff
 *   COROTATED   �?Corotated linear elasticity
 *   NEOHOOKEAN  �?Stable Neo-Hookean (Smith et al. 2018)
 */

#include "Solver.h"
#include "Solver/FemEnergy.cuh"
#include "math/culib_helper.hpp"
#include "math/decomposition.hpp"
#include "iostream"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

 // ─────────────────────────────────────────────────────────────────────────────
 // Helpers
 // ─────────────────────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error in " << __FILE__ << " at line "           \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "\n";  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t err = call;                                              \
        if (err != CUBLAS_STATUS_SUCCESS) {                                     \
            std::cerr << "cuBLAS error in " << __FILE__ << " at line "         \
                      << __LINE__ << ": " << static_cast<int>(err) << "\n";   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUSPARSE_CHECK(call)                                                    \
    do {                                                                        \
        cusparseStatus_t err = call;                                            \
        if (err != CUSPARSE_STATUS_SUCCESS) {                                   \
            std::cerr << "cuSPARSE error in " << __FILE__ << " at line "       \
                      << __LINE__ << ": " << cusparseGetErrorString(err)       \
                      << " (" << static_cast<int>(err) << ")\n";              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Convenience: launch with 1-D grid covering n elements
static dim3 grid1D(int n, int block = 256) {
    return dim3((n + block - 1) / block);
}

template <typename Real>
struct DenseCublasOps;

template <>
struct DenseCublasOps<float> {
    static cublasStatus_t gemv(
        cublasHandle_t handle,
        cublasOperation_t trans,
        int m,
        int n,
        const float* alpha,
        const float* A,
        int lda,
        const float* x,
        int incx,
        const float* beta,
        float* y,
        int incy)
    {
        return cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }
};

template <>
struct DenseCublasOps<double> {
    static cublasStatus_t gemv(
        cublasHandle_t handle,
        cublasOperation_t trans,
        int m,
        int n,
        const double* alpha,
        const double* A,
        int lda,
        const double* x,
        int incx,
        const double* beta,
        double* y,
        int incy)
    {
        return cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }
};

template <typename Real>
static int DenseCG(
    cublasHandle_t cublasH,
    const Real* DnA,
    Real* delta_x,
    const Real* d_b,
    Real* d_r,
    Real* d_p,
    Real* d_q,
    int dof,
    int maxIters,
    Real tolerance)
{
    const Real zero = static_cast<Real>(0);
    const Real one = static_cast<Real>(1);

    CUDA_CHECK(cudaMemset(delta_x, 0, sizeof(Real) * static_cast<size_t>(dof)));
    CUBLAS_CHECK(CublasOps<Real>::copy(cublasH, dof, d_b, 1, d_r, 1));
    CUBLAS_CHECK(CublasOps<Real>::copy(cublasH, dof, d_r, 1, d_p, 1));

    Real rr = static_cast<Real>(0);
    CUBLAS_CHECK(CublasOps<Real>::dot(cublasH, dof, d_r, 1, d_r, 1, &rr));
    if (std::sqrt(rr) <= tolerance) {
        return 0;
    }

    int iter = 0;
    for (; iter < maxIters; ++iter) {
        CUBLAS_CHECK(DenseCublasOps<Real>::gemv(
            cublasH,
            CUBLAS_OP_T,
            dof,
            dof,
            &one,
            DnA,
            dof,
            d_p,
            1,
            &zero,
            d_q,
            1));

        Real pAp = static_cast<Real>(0);
        CUBLAS_CHECK(CublasOps<Real>::dot(cublasH, dof, d_p, 1, d_q, 1, &pAp));
        if (std::abs(pAp) <= static_cast<Real>(1e-20)) {
            break;
        }

        const Real alpha = rr / pAp;
        CUBLAS_CHECK(CublasOps<Real>::axpy(cublasH, dof, &alpha, d_p, 1, delta_x, 1));
        const Real negAlpha = -alpha;
        CUBLAS_CHECK(CublasOps<Real>::axpy(cublasH, dof, &negAlpha, d_q, 1, d_r, 1));

        Real rrNew = static_cast<Real>(0);
        CUBLAS_CHECK(CublasOps<Real>::dot(cublasH, dof, d_r, 1, d_r, 1, &rrNew));
        if (std::sqrt(rrNew) <= tolerance) {
            ++iter;
            break;
        }

        const Real beta = rrNew / rr;
        CUBLAS_CHECK(CublasOps<Real>::scale(cublasH, dof, &beta, d_p, 1));
        CUBLAS_CHECK(CublasOps<Real>::axpy(cublasH, dof, &one, d_r, 1, d_p, 1));
        rr = rrNew;
    }

    return iter;
}

// ─────────────────────────────────────────────────────────────────────────────
// Device-side parameter struct (constant memory)
// ─────────────────────────────────────────────────────────────────────────────

template <typename Real>
struct DevParams {
    Real lambda;
    Real mu;
    Real damping;
    Real dt;
    Real density;
    Real _alpha;
    Vector<Real, 3> gravity;
    Vector<Real, 3> boundary_min;
    Vector<Real, 3> boundary_max;
    Real barrier_distance;
    Real barrier_stiffness;
    int energyType; // 0=STVK, 1=COROTATED, 2=NEOHOOKEAN, 3=ARAP
};

__constant__ DevParams<float> c_params_f;
__constant__ DevParams<double> c_params_d;

template <typename Real>
__device__ __forceinline__ const DevParams<Real>& GetDevParams();

template <>
__device__ __forceinline__ const DevParams<float>& GetDevParams<float>() { return c_params_f; }

template <>
__device__ __forceinline__ const DevParams<double>& GetDevParams<double>() { return c_params_d; }

// ─────────────────────────────────────────────────────────────────────────────
// Deformation gradient  F = Ds * Dm_inv
// Ds = [x1-x0 | x2-x0 | x3-x0]  (current edge matrix)
// ─────────────────────────────────────────────────────────────────────────────

template <typename Real>
__device__ __forceinline__ mat3<Real> computeF(
    const Tetrahedron<Real>& tet,
    const Vector<Real, 3>* vertex)
{
    using Vec3 = Vector<Real, 3>;
    using Mat3 = mat3<Real>;

    int i0 = tet.verticesIndex.x;
    int i1 = tet.verticesIndex.y;
    int i2 = tet.verticesIndex.z;
    int i3 = tet.verticesIndex.w;

    Vec3 x0 = vertex[i0];
    Vec3 x1 = vertex[i1];
    Vec3 x2 = vertex[i2];
    Vec3 x3 = vertex[i3];

    // Ds: columns are edge vectors from vertex 0
    Mat3 Ds(x1 - x0, x2 - x0, x3 - x0);
    Mat3 DmInv;
    for (int i = 0; i < 9; ++i) {
        DmInv[i] = static_cast<Real>(tet.Dm_inv[i]);
    }
    return Ds * DmInv;
}

// ─────────────────────────────────────────────────────────────────────────────
// First Piola-Kirchhoff stress  P(F)
// ─────────────────────────────────────────────────────────────────────────────

// ---------- StVK ----------
// E = 0.5*(F'F - I)
// P = F*(2*mu*E + lambda*tr(E)*I)
// ---------- Corotated ----------
// F = R * S  (polar decomp)
// P = 2*mu*(F - R) + lambda*(J-1)*J * F^{-T}
// Simplified linear form: P = 2*mu*(F-R) + lambda*tr(S-I)*R
// ---------- Stable Neo-Hookean (Smith et al. 2018) ----------
// Psi = mu/2*(I_C - 3) - mu*log(J) + lambda/2*(J-1)^2
// P   = mu*(F - F^{-T}) + lambda*(J-1)*J*F^{-T}
// The return results is vectorized.
// Reference code: https://github.com/theodorekim/HOBAKv1/blob/main/src/Geometry/TET_MESH.cpp#L197
template <typename Real>
__inline__ __device__
Mat9x12<Real> computepFpx(const mat3<Real>& DmInv)
{
    Mat9x12<Real> PFPu(Real(0));
    const Real m = DmInv(0, 0);
    const Real n = DmInv(0, 1);
    const Real o = DmInv(0, 2);
    const Real p = DmInv(1, 0);
    const Real q = DmInv(1, 1);
    const Real r = DmInv(1, 2);
    const Real s = DmInv(2, 0);
    const Real t = DmInv(2, 1);
    const Real u = DmInv(2, 2);

    const Real t1 = -m - p - s;
    const Real t2 = -n - q - t;
    const Real t3 = -o - r - u;

    PFPu(0, 0)  = t1;
    PFPu(0, 3)  = m;
    PFPu(0, 6)  = p;
    PFPu(0, 9)  = s;
    PFPu(1, 1)  = t1;
    PFPu(1, 4)  = m;
    PFPu(1, 7)  = p;
    PFPu(1, 10) = s;
    PFPu(2, 2)  = t1;
    PFPu(2, 5)  = m;
    PFPu(2, 8)  = p;
    PFPu(2, 11) = s;
    PFPu(3, 0)  = t2;
    PFPu(3, 3)  = n;
    PFPu(3, 6)  = q;
    PFPu(3, 9)  = t;
    PFPu(4, 1)  = t2;
    PFPu(4, 4)  = n;
    PFPu(4, 7)  = q;
    PFPu(4, 10) = t;
    PFPu(5, 2)  = t2;
    PFPu(5, 5)  = n;
    PFPu(5, 8)  = q;
    PFPu(5, 11) = t;
    PFPu(6, 0)  = t3;
    PFPu(6, 3)  = o;
    PFPu(6, 6)  = r;
    PFPu(6, 9)  = u;
    PFPu(7, 1)  = t3;
    PFPu(7, 4)  = o;
    PFPu(7, 7)  = r;
    PFPu(7, 10) = u;
    PFPu(8, 2)  = t3;
    PFPu(8, 5)  = o;
    PFPu(8, 8)  = r;
    PFPu(8, 11) = u;

    return PFPu;
}

template <typename Real>
__global__ void k_AddGravity(
    Vector<Real, 3>* __restrict__ force,
    Real* __restrict__ mass,
    int numVerts
)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVerts)
        return;
    const DevParams<Real>& params = GetDevParams<Real>();
    force[vid] += mass[vid] * params.gravity;
}


// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1   compute elastic forces and mass from tetrahedra
//            Uses atomic adds to scatter forces/masses to vertices
// ─────────────────────────────────────────────────────────────────────────────

template <typename Real>
__global__ void k_AddSkeletonCouplingForces(
    const Vector<Real, 3>* __restrict__ vertex,
    const Vector<Real, 3>* __restrict__ velocity,
    Vector<Real, 3>* __restrict__ force,
    const Vector<Real, 3>* __restrict__ targets,
    const Vector<Real, 3>* __restrict__ targetVelocities,
    const Real* __restrict__ weights,
    Real stiffness,
    Real damping,
    int numVerts)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVerts) return;

    const Real weight = weights[vid];
    if (weight <= static_cast<Real>(0) || stiffness <= static_cast<Real>(0)) {
        return;
    }

    const Vector<Real, 3> dx = targets[vid] - vertex[vid];
    const Vector<Real, 3> dv = targetVelocities[vid] - velocity[vid];
    force[vid] += weight * (stiffness * dx + damping * dv);
}

template <typename Real>
__global__ void k_AddSkeletonCouplingImplicitForces(
    const Vector<Real, 3>* __restrict__ vertex,
    Vector<Real, 3>* __restrict__ force,
    const Vector<Real, 3>* __restrict__ targets,
    const Vector<Real, 3>* __restrict__ targetVelocities,
    const Real* __restrict__ weights,
    Real stiffness,
    Real damping,
    int numVerts)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVerts) return;

    const Real weight = weights[vid];
    if (weight <= static_cast<Real>(0) || stiffness <= static_cast<Real>(0)) {
        return;
    }

    const Vector<Real, 3> dx = targets[vid] - vertex[vid];
    force[vid] += weight * (stiffness * dx + damping * targetVelocities[vid]);
}

template <typename Real>
__global__ void k_AddSkeletonCouplingDenseDiagonal(
    Real* __restrict__ A,
    const Real* __restrict__ weights,
    Real stiffness,
    Real damping,
    int numVerts)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVerts) return;

    const Real weight = weights[vid];
    if (weight <= static_cast<Real>(0) || stiffness <= static_cast<Real>(0)) {
        return;
    }

    const int dof = 3 * numVerts;
    const Real dt = GetDevParams<Real>().dt;
    const Real diagonal = weight * (stiffness + damping / dt);
    for (int c = 0; c < 3; ++c) {
        const int row = 3 * vid + c;
        A[static_cast<size_t>(row) * dof + row] += diagonal;
    }
}

template <typename Real>
__global__ void k_AddSkeletonCouplingCsrDiagonal(
    Real* __restrict__ A_values,
    const int* __restrict__ A_diag_indices,
    const Real* __restrict__ weights,
    Real stiffness,
    Real damping,
    int numVerts)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVerts) return;

    const Real weight = weights[vid];
    if (weight <= static_cast<Real>(0) || stiffness <= static_cast<Real>(0)) {
        return;
    }

    const Real dt = GetDevParams<Real>().dt;
    const Real diagonal = weight * (stiffness + damping / dt);
    for (int c = 0; c < 3; ++c) {
        const int dof = 3 * vid + c;
        A_values[A_diag_indices[dof]] += diagonal;
    }
}

template <typename Real>
__global__ void k_ComputeForces(
    const Tetrahedron<Real>* __restrict__ tets,
    const Vector<Real, 3>* __restrict__ vertex,
    Vector<Real, 3>* __restrict__ force,
    mat3<Real>* __restrict__ d_F,
    int                             numTets)
{
    using Vec3 = Vector<Real, 3>;
    using Mat3 = mat3<Real>;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numTets) return;

    const Tetrahedron<Real>& tet = tets[tid];

    int i0 = tet.verticesIndex[0];
    int i1 = tet.verticesIndex[1];
    int i2 = tet.verticesIndex[2];
    int i3 = tet.verticesIndex[3];

    Real V0 = tet.volume; // rest volume

    // ─── Deformation gradient ──────────────────────────────────────────────
    Mat3 F = computeF<Real>(tet, vertex);
    d_F[tid] = F;

    // ─── First Piola-Kirchhoff stress ──────────────────────────────────────
    const DevParams<Real>& params = GetDevParams<Real>();
    Real mu = params.mu;
    Real lambda = params.lambda;
    Mat3 P = computeStressForEnergy(F, mu, lambda, params._alpha, params.energyType);

    // ─── Nodal forces from the stress ─────────────────────────────────────
    // f = -V0 * P * Dm^{-T}   (distributed to the four nodes)
    // The force on node 1,2,3 = -V0 * P * col(Dm^{-T}, 0/1/2)
    // Force on node 0 = -(f1+f2+f3)
    Mat3 DmInv = tet.Dm_inv;
    // for (int i = 0; i < 9; ++i) DmInv[i] = static_cast<Real>(tet.Dm_inv[i]);
    Mat3 DmInvT = Mat3::transpose(DmInv);
    Mat3 H = P * DmInvT * (-V0);   // 3x3, columns = forces on nodes 1,2,3

    Vec3 f1 = H.column(0);
    Vec3 f2 = H.column(1);
    Vec3 f3 = H.column(2);
    Vec3 f0{ -f1.x - f2.x - f3.x,
             -f1.y - f2.y - f3.y,
             -f1.z - f2.z - f3.z };

    // Atomic scatter to vertex force array
    atomicAdd(&force[i0][0], f0[0]);
    atomicAdd(&force[i0][1], f0[1]);
    atomicAdd(&force[i0][2], f0[2]);

    atomicAdd(&force[i1][0], f1[0]);
    atomicAdd(&force[i1][1], f1[1]);
    atomicAdd(&force[i1][2], f1[2]);

    atomicAdd(&force[i2][0], f2[0]);
    atomicAdd(&force[i2][1], f2[1]);
    atomicAdd(&force[i2][2], f2[2]);

    atomicAdd(&force[i3][0], f3[0]);
    atomicAdd(&force[i3][1], f3[1]);
    atomicAdd(&force[i3][2], f3[2]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2 symplectic Euler integration + gravity + damping
// ─────────────────────────────────────────────────────────────────────────────

template <typename Real>
__global__ void k_Integrate(
    Vector<Real, 3>* __restrict__ vertex,
    Vector<Real, 3>* __restrict__ velocity,
    Vector<Real, 3>* __restrict__ force,
    Real* __restrict__ mass,
    int numVerts)
{
    using Vec3 = Vector<Real, 3>;
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVerts) return;

    Real m = mass[vid];
    if (m < static_cast<Real>(1e-12)) return;   // guard against zero-mass vertices

    const DevParams<Real>& params = GetDevParams<Real>();
    Real dt = params.dt;
    Real damping = params.damping;

    Vec3 f = force[vid];

    // Acceleration
    Vec3 a = f / m;

    // Symplectic Euler
    Vec3 v = velocity[vid];
    v = v * (static_cast<Real>(1) - damping) + a * dt;

    velocity[vid] = v;
    vertex[vid] += v * dt;

    // Reset force and mass for next substep
    force[vid] = Vec3{ static_cast<Real>(0), static_cast<Real>(0), static_cast<Real>(0) };
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3 AABB boundary collision (simple position projection + restitution)
// ─────────────────────────────────────────────────────────────────────────────

template <typename Real>
__global__ void k_BoundaryCheck(
    Vector<Real, 3>* __restrict__ vertex,
    Vector<Real, 3>* __restrict__ velocity,
    int numVerts)
{
    using Vec3 = Vector<Real, 3>;
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVerts) return;

    Vec3 x = vertex[vid];
    Vec3 v = velocity[vid];

    const DevParams<Real>& params = GetDevParams<Real>();
    Vec3 bmin = params.boundary_min;
    Vec3 bmax = params.boundary_max;

    const Real restitution = static_cast<Real>(0.3);
    const Real friction = static_cast<Real>(0.6);

    // X
    if (x.x < bmin.x) { x.x = bmin.x; if (v.x < 0) { v.x = -v.x * restitution; v.y *= friction; v.z *= friction; } }
    if (x.x > bmax.x) { x.x = bmax.x; if (v.x > 0) { v.x = -v.x * restitution; v.y *= friction; v.z *= friction; } }

    // Y
    if (x.y < bmin.y) { x.y = bmin.y; if (v.y < 0) { v.y = -v.y * restitution; v.x *= friction; v.z *= friction; } }
    if (x.y > bmax.y) { x.y = bmax.y; if (v.y > 0) { v.y = -v.y * restitution; v.x *= friction; v.z *= friction; } }

    // Z
    if (x.z < bmin.z) { x.z = bmin.z; if (v.z < 0) { v.z = -v.z * restitution; v.x *= friction; v.y *= friction; } }
    if (x.z > bmax.z) { x.z = bmax.z; if (v.z > 0) { v.z = -v.z * restitution; v.x *= friction; v.y *= friction; } }

    vertex[vid] = x;
    velocity[vid] = v;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 4 initialise Dm_inv and rest volume for each tetrahedron
// ─────────────────────────────────────────────────────────────────────────────
template <typename Real>
__global__ void K_PreCompute(
    Tetrahedron<Real>* tets,
    const Vector<Real, 3>* __restrict__ vertex,
    Real* __restrict__ mass,
    int numTets)
{
    using Vec3 = Vector<Real, 3>;
    using Mat3 = mat3<Real>;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numTets) return;

    const DevParams<Real>& params = GetDevParams<Real>();
    Tetrahedron<Real>& tet = tets[tid];

    int i0 = tet.verticesIndex[0];
    int i1 = tet.verticesIndex[1];
    int i2 = tet.verticesIndex[2];
    int i3 = tet.verticesIndex[3];

    Vec3 x0 = vertex[i0];
    Vec3 x1 = vertex[i1];
    Vec3 x2 = vertex[i2];
    Vec3 x3 = vertex[i3];

    Mat3 Dm(x1 - x0, x2 - x0, x3 - x0);

    // Volume = |det(Dm)| / 6
    Real det = Mat3::determinant(Dm);
    tet.volume = fabs(det / 6.0);

    Real tetMass = params.density * tet.volume * static_cast<Real>(0.25);
    atomicAdd(&mass[i0], tetMass);
    atomicAdd(&mass[i1], tetMass);
    atomicAdd(&mass[i2], tetMass);
    atomicAdd(&mass[i3], tetMass);

    auto dminv = Mat3::inverse(Dm);
    for (int i = 0; i < 9; ++i) {
        tet.Dm_inv[i] = static_cast<Real>(dminv[i]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 5 Compute stiffness matrix for each tetrahedron. Then scatter the stiffness matrix to the global matrix.
// 
// stiffness matrix:
// df / dx = vec(dF / dx)^T * vec(dp / dF) * vec(dF / dx)
// 
// Linear system:  (I - M^(-1) * (df/dx)*dt^2) \delta x = v_n * dt + M^(-1) * f * dt^2
// Ax = b
// A = (I - M^(-1) * (df/dx)*dt^2)
// b = v_n * dt + M^(-1) * f * dt^2
// ─────────────────────────────────────────────────────────────────────────────
template <typename Real>
__global__ void k_computeK(
    const Tetrahedron<Real>* __restrict__ tets,
    const mat3<Real>*      __restrict__ d_F,  // deformation gradient
    const Real* __restrict__ mass,
    Real* __restrict__ DnA,
    int numTets,
    int numVerts
)
{
    using Vec3 = Vector<Real, 3>;
    using Mat3 = mat3<Real>;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numTets) return;

    const Tetrahedron<Real>& tet = tets[tid]; 

    const DevParams<Real>& params = GetDevParams<Real>();
    Real mu = params.mu;
    Real lambda = params.lambda;

    const Mat3& F = d_F[tid];
    Real J = Mat3::determinant(F);
    // Clamp J to avoid singularity
    //J = (J > static_cast<Real>(1e-4)) ? J : static_cast<Real>(1e-4);

    //Compute dF / dx = (dD_s / dx) * (Dm_Inv)
    Mat3 Dm_inv = tet.Dm_inv;
    Mat9x12<Real> pFpx = computepFpx(Dm_inv);
    Mat9x9<Real>  hessian = -tet.volume * computeEnergyHessian(
        F, mu, lambda, params._alpha, params.energyType, false);

    Mat12x12<Real> Ke = (transpose(pFpx) * hessian) * pFpx;
    
    Vec4i ids = tet.verticesIndex;
    for (int y = 0; y < 4; y++)
    {
        int yVertex = ids[y];
        for (int x = 0; x < 4; x++)
        {
            int xVertex = ids[x];
            for (int b = 0; b < 3; b++)
                for (int a = 0; a < 3; a++)
                {
                    const Real entry = -Ke(3 * x + a, 3 * y + b);
                    const size_t row = static_cast<size_t>(3 * xVertex + a);
                    const size_t col = static_cast<size_t>(3 * yVertex + b);
                    atomicAdd(&DnA[row * (3 * numVerts) + col], entry);
                }
        }
    }
}

template <typename Real>
__global__ void k_Assemble(
    Real* __restrict__ DnA,
    Real* __restrict__ b,
    Vector<Real, 3>* __restrict__ vn,
    Vector<Real, 3>* __restrict__ force,
    const Real* __restrict__ mass,
    int numVerts
)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVerts) return;

    const DevParams<Real>& params = GetDevParams<Real>();
    const Real invDt = Real(1) / params.dt;
    const Real invDt2 = invDt * invDt;
    const Real m = mass[vid];

    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        const size_t diag = static_cast<size_t>(3 * vid + c);

        DnA[diag * (3 * numVerts) + diag] += invDt2 * m;
        b[diag] = invDt * m * vn[vid][c] + force[vid][c];
    }
}

template <typename Real>
__global__ void k_integrateImplicit(
    Vector<Real, 3>* __restrict__ vertex,
    Vector<Real, 3>* __restrict__ velocity,
    Real* __restrict__ delta_x,
    int numVerts)
{
    using Vec3 = Vector<Real, 3>;
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVerts)
        return;
    
    Vec3 dx = Vec3{ delta_x[vid * 3], delta_x[vid * 3 + 1], delta_x[vid * 3 + 2] };
    vertex[vid] += dx;
    velocity[vid] = dx / GetDevParams<Real>().dt; // Update velocity based on position change
}

// ─────────────────────────────────────────────────────────────────────────────
// CSR-mode kernels for the IMPLICIT_SPARSE pipeline
//
// The CSR topology (row offsets, column indices, diag indices, element->csr
// scatter map) is built once on the host in BuildGlobalCsrFromTetMesh and
// uploaded to the device.  Each substep we:
//   1. zero d_A_values
//   2. scatter element 12x12 stiffness matrices into d_A_values
//   3. add diagonal mass term and form b
//   4. extract a Jacobi preconditioner (M_inv) from the diagonal entries
//   5. solve A delta_x = b with cuSPARSE SpMV based PCG
// ─────────────────────────────────────────────────────────────────────────────

template <typename Real>
__global__ void k_zero_real(Real* __restrict__ data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] = static_cast<Real>(0);
}

template <typename Real>
__global__ void k_applyKinematicTargets(
    Vector<Real, 3>* __restrict__ vertex,
    Vector<Real, 3>* __restrict__ velocity,
    const int* __restrict__ constraintFlags,
    const Real* __restrict__ constraintTargets,
    int numVerts)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVerts) return;

    const int base = 3 * vid;
    if (constraintFlags[base] == 0 &&
        constraintFlags[base + 1] == 0 &&
        constraintFlags[base + 2] == 0) {
        return;
    }

    Vector<Real, 3> x = vertex[vid];
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        if (constraintFlags[base + c]) {
            x[c] = constraintTargets[base + c];
            velocity[vid][c] = static_cast<Real>(0);
        }
    }
    vertex[vid] = x;
}

template <typename Real>
__global__ void k_applyDenseDirichletConstraints(
    Real* __restrict__ A,
    Real* __restrict__ b,
    const int* __restrict__ constraintFlags,
    int dof)
{
    const size_t total = static_cast<size_t>(dof) * static_cast<size_t>(dof);
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int row = static_cast<int>(idx / dof);
    const int col = static_cast<int>(idx - static_cast<size_t>(row) * dof);
    const bool rowConstrained = constraintFlags[row] != 0;
    const bool colConstrained = constraintFlags[col] != 0;
    if (rowConstrained || colConstrained) {
        A[idx] = (row == col && rowConstrained) ? static_cast<Real>(1) : static_cast<Real>(0);
    }
}

template <typename Real>
__global__ void k_applyDenseDirichletRhs(
    Real* __restrict__ b,
    const int* __restrict__ constraintFlags,
    int dof)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dof) return;
    if (constraintFlags[i]) {
        b[i] = static_cast<Real>(0);
    }
}

template <typename Real>
__global__ void k_applyCsrDirichletConstraints(
    Real* __restrict__ A_values,
    const int* __restrict__ A_row_offsets,
    const int* __restrict__ A_col_indices,
    const int* __restrict__ A_diag_indices,
    Real* __restrict__ b,
    const int* __restrict__ constraintFlags,
    int dof)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dof) return;

    const bool rowConstrained = constraintFlags[row] != 0;
    for (int idx = A_row_offsets[row]; idx < A_row_offsets[row + 1]; ++idx) {
        const int col = A_col_indices[idx];
        if (rowConstrained || constraintFlags[col]) {
            A_values[idx] = (rowConstrained && idx == A_diag_indices[row])
                ? static_cast<Real>(1)
                : static_cast<Real>(0);
        }
    }

    if (rowConstrained) {
        b[row] = static_cast<Real>(0);
    }
}

template <typename Real>
__global__ void k_computeK_csr(
    const Tetrahedron<Real>* __restrict__ tets,
    const mat3<Real>*        __restrict__ d_F,
    const int*               __restrict__ elem_to_A_csr,
    Real*                    __restrict__ A_values,
    int numTets)
{
    using Mat3 = mat3<Real>;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numTets) return;

    const Tetrahedron<Real>& tet = tets[tid];

    const DevParams<Real>& params = GetDevParams<Real>();
    Real mu = params.mu;
    Real lambda = params.lambda;

    const Mat3& F = d_F[tid];

    Mat3 Dm_inv = tet.Dm_inv;
    Mat9x12<Real> pFpx = computepFpx(Dm_inv);
    Mat9x9<Real>  hessian = -tet.volume * computeEnergyHessian(
        F, mu, lambda, params._alpha, params.energyType, true);

    Mat12x12<Real> Ke = (transpose(pFpx) * hessian) * pFpx;

    const int base = tid * 144;
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            for (int b = 0; b < 3; ++b) {
                for (int a = 0; a < 3; ++a) {
                    const int local_r = 3 * x + a;
                    const int local_c = 3 * y + b;
                    const Real entry  = -Ke(local_r, local_c);
                    const int csr_idx = elem_to_A_csr[base + local_r * 12 + local_c];
                    atomicAdd(&A_values[csr_idx], entry);
                }
            }
        }
    }
}

template <typename Real>
__global__ void k_assemble_csr(
    Real*                          __restrict__ A_values,
    const int*                     __restrict__ A_diag_indices,
    Real*                          __restrict__ b,
    const Vector<Real, 3>*         __restrict__ vn,
    const Vector<Real, 3>*         __restrict__ force,
    const Real*                    __restrict__ mass,
    int numVerts)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVerts) return;

    const DevParams<Real>& params = GetDevParams<Real>();
    const Real invDt  = Real(1) / params.dt;
    const Real invDt2 = invDt * invDt;
    const Real m      = mass[vid];

    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        const int dof = 3 * vid + c;
        const int diagIdx = A_diag_indices[dof];
        A_values[diagIdx] += invDt2 * m;
        b[dof] = invDt * m * vn[vid][c] + force[vid][c];
    }
}

template <typename Real>
__global__ void k_extract_diag_inv(
    const Real* __restrict__ A_values,
    const int*  __restrict__ A_diag_indices,
    Real*       __restrict__ M_inv,
    int dof)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dof) return;
    Real d = A_values[A_diag_indices[i]];
    M_inv[i] = (d > static_cast<Real>(1e-30) || d < -static_cast<Real>(1e-30))
        ? (static_cast<Real>(1) / d)
        : static_cast<Real>(1);
}

template <typename Real>
__global__ void k_elementwise_mul(
    const Real* __restrict__ a,
    const Real* __restrict__ b,
    Real*       __restrict__ out,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] * b[i];
}

// ─────────────────────────────────────────────────────────────────────────────
// Barrier-based AABB collision for the IMPLICIT_SPARSE pipeline.
//
// Energy per plane:  E(d) = 0.5 * k * (1/d - 1/dhat)^2
// Force (along normal):  f = k * (1/d^3 - 1/(dhat*d^2))
// Diagonal hessian:     h = k * (3/d^4 - 2/(dhat*d^3))
//
// Six AABB planes contribute independently per vertex.
// ─────────────────────────────────────────────────────────────────────────────

template <typename Real>
__device__ __forceinline__ void addPlaneBarrier(
    Real signedDistance,
    int axis,
    Real normalSign,
    Real barrierDist,
    Real barrierK,
    Vector<Real, 3>& outForce,
    Real* outDiagHess)
{
    if (signedDistance >= barrierDist)
        return;

    const Real eps  = barrierDist * static_cast<Real>(1e-4);
    const Real dMin = (eps > static_cast<Real>(1e-8)) ? eps : static_cast<Real>(1e-8);
    const Real d    = (signedDistance > dMin) ? signedDistance : dMin;
    const Real dhat = barrierDist;
    const Real k    = barrierK;

    const Real d2 = d * d;
    const Real d3 = d2 * d;
    const Real d4 = d3 * d;

    const Real forceMag = k * (static_cast<Real>(1) / d3
                             - static_cast<Real>(1) / (dhat * d2));
    const Real hessMag  = k * (static_cast<Real>(3) / d4
                             - static_cast<Real>(2) / (dhat * d3));

    outForce[axis] += normalSign * forceMag;
    outDiagHess[axis] += hessMag;
}

template <typename Real>
__global__ void k_addBarrierForces(
    const Vector<Real, 3>* __restrict__ vertex,
    Vector<Real, 3>*       __restrict__ force,
    Real*                  __restrict__ A_values,
    const int*             __restrict__ A_diag_indices,
    int numVerts)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVerts) return;

    const DevParams<Real>& params = GetDevParams<Real>();
    const Vector<Real, 3> x  = vertex[vid];
    const Real bDist = params.barrier_distance;
    const Real bK    = params.barrier_stiffness;

    Vector<Real, 3> bf{ static_cast<Real>(0), static_cast<Real>(0), static_cast<Real>(0) };
    Real bh[3] = { static_cast<Real>(0), static_cast<Real>(0), static_cast<Real>(0) };

    // 6 AABB planes: min/max for each axis
    addPlaneBarrier<Real>(x[0] - params.boundary_min[0],  0, static_cast<Real>(+1), bDist, bK, bf, bh);
    addPlaneBarrier<Real>(params.boundary_max[0] - x[0],  0, static_cast<Real>(-1), bDist, bK, bf, bh);
    addPlaneBarrier<Real>(x[1] - params.boundary_min[1],  1, static_cast<Real>(+1), bDist, bK, bf, bh);
    addPlaneBarrier<Real>(params.boundary_max[1] - x[1],  1, static_cast<Real>(-1), bDist, bK, bf, bh);
    addPlaneBarrier<Real>(x[2] - params.boundary_min[2],  2, static_cast<Real>(+1), bDist, bK, bf, bh);
    addPlaneBarrier<Real>(params.boundary_max[2] - x[2],  2, static_cast<Real>(-1), bDist, bK, bf, bh);

    force[vid] += bf;

    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        const int dof     = 3 * vid + c;
        const int diagIdx = A_diag_indices[dof];
        A_values[diagIdx] += bh[c];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Jacobi-preconditioned CG using cuSPARSE SpMV.
// Solves  A * delta_x = b   with x0 = 0  (so r0 = b).
// ─────────────────────────────────────────────────────────────────────────────
template <typename Real>
static int SparsePCG(
    cublasHandle_t        cublasH,
    cusparseHandle_t      cusparseH,
    cusparseSpMatDescr_t  A,
    cusparseDnVecDescr_t  vecP,
    cusparseDnVecDescr_t  vecQ,
    void*                 d_spmv_buffer,
    Real*       delta_x,
    const Real* d_b,
    Real*       d_r,
    Real*       d_p,
    Real*       d_q,
    Real*       d_z,
    const Real* d_M_inv,
    int dof,
    int maxIters,
    Real tolerance)
{
    const Real zero = static_cast<Real>(0);
    const Real one  = static_cast<Real>(1);
    //const cudaDataType valType =
    //    std::is_same<Real, float>::value ? CUDA_R_32F : CUDA_R_64F;

    // x = 0  =>  r = b
    CUDA_CHECK(cudaMemset(delta_x, 0, sizeof(Real) * static_cast<size_t>(dof)));
    CUBLAS_CHECK(CublasOps<Real>::copy(cublasH, dof, d_b, 1, d_r, 1));

    // z = M^{-1} r,  p = z
    k_elementwise_mul<Real><<<grid1D(dof), 256>>>(d_M_inv, d_r, d_z, dof);
    CUBLAS_CHECK(CublasOps<Real>::copy(cublasH, dof, d_z, 1, d_p, 1));

    Real rz = static_cast<Real>(0);
    CUBLAS_CHECK(CublasOps<Real>::dot(cublasH, dof, d_r, 1, d_z, 1, &rz));

    Real rr = static_cast<Real>(0);
    CUBLAS_CHECK(CublasOps<Real>::dot(cublasH, dof, d_r, 1, d_r, 1, &rr));
    if (std::sqrt(rr) <= tolerance) {
        return 0;
    }

    int iter = 0;
    for (; iter < maxIters; ++iter) {
        // q = A * p
        CUSPARSE_CHECK(CusparseOps<Real>::SpMV(
            cusparseH,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one,
            A,
            vecP,
            &zero,
            vecQ,
            CUSPARSE_SPMV_ALG_DEFAULT,
            d_spmv_buffer));

        Real pAp = static_cast<Real>(0);
        CUBLAS_CHECK(CublasOps<Real>::dot(cublasH, dof, d_p, 1, d_q, 1, &pAp));
        if (std::abs(pAp) <= static_cast<Real>(1e-20)) {
            break;
        }

        const Real alpha    = rz / pAp;
        const Real negAlpha = -alpha;
        CUBLAS_CHECK(CublasOps<Real>::axpy(cublasH, dof, &alpha,    d_p, 1, delta_x, 1));
        CUBLAS_CHECK(CublasOps<Real>::axpy(cublasH, dof, &negAlpha, d_q, 1, d_r,     1));

        Real rrNew = static_cast<Real>(0);
        CUBLAS_CHECK(CublasOps<Real>::dot(cublasH, dof, d_r, 1, d_r, 1, &rrNew));
        if (std::sqrt(rrNew) <= tolerance) {
            ++iter;
            break;
        }

        // z = M^{-1} r,  rz_new = <r, z>
        k_elementwise_mul<Real><<<grid1D(dof), 256>>>(d_M_inv, d_r, d_z, dof);
        Real rzNew = static_cast<Real>(0);
        CUBLAS_CHECK(CublasOps<Real>::dot(cublasH, dof, d_r, 1, d_z, 1, &rzNew));

        const Real beta = rzNew / rz;
        // p = z + beta * p
        CUBLAS_CHECK(CublasOps<Real>::scale(cublasH, dof, &beta, d_p, 1));
        CUBLAS_CHECK(CublasOps<Real>::axpy(cublasH, dof, &one, d_z, 1, d_p, 1));
        rz = rzNew;
    }

    return iter;
}

// ─────────────────────────────────────────────────────────────────────────────
// ElasticitySolver member implementations
// ─────────────────────────────────────────────────────────────────────────────
template <typename Real>
void ElasticitySolverT<Real>::SetParams()
{
    // Compute Lame parameters from Young's modulus and Poisson's ratio.
    const Real E = h_params.youngs_modulus;
    const Real nu = h_params.poisson_ratio;

    h_params.mu = E / (static_cast<Real>(2.0) * (static_cast<Real>(1.0) + nu));
    h_params.lambda = E * nu / ((static_cast<Real>(1.0) + nu) * (static_cast<Real>(1.0) - static_cast<Real>(2.0) * nu));

    // Upload to constant memory.
    DevParams<Real> dp{};
    dp.lambda = h_params.lambda;
    dp.mu = h_params.mu;
    dp.damping = h_params.damping;
    dp.dt = h_params.dt;
    dp.density = h_params.density;
    dp.gravity = h_params.gravity;
    dp.boundary_min = h_params.boundary_min;
    dp.boundary_max = h_params.boundary_max;
    dp.barrier_distance = h_params.barrier_distance;
    dp.barrier_stiffness = h_params.barrier_stiffness;
    dp.energyType = static_cast<int>(h_params.energyType);

    if (h_params.energyType == NEOHOOKEAN)
        dp._alpha = Real(1) + h_params.mu / h_params.lambda;
    else
        dp._alpha = Real(0);

    if constexpr (std::is_same_v<Real, float>) {
        CUDA_CHECK(cudaMemcpyToSymbol(c_params_f, &dp, sizeof(DevParams<float>)));
    } else {
        CUDA_CHECK(cudaMemcpyToSymbol(c_params_d, &dp, sizeof(DevParams<double>)));
    }
}

template <typename Real>
void ElasticitySolverT<Real>::PreCompute()
{
    int numTets = static_cast<int>(h_tet.size());

    K_PreCompute<Real><<<grid1D(numTets), 256>>>(d_tet, d_vertex, d_mass, numTets);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_tet.data(), d_tet,
        numTets * sizeof(Tetrahedron<Real>),
        cudaMemcpyDeviceToHost));

    Real minVol = 1e30f;
    for (auto& t : h_tet) minVol = std::min(minVol, t.volume);
    std::cout << "Min tet rest-volume: " << minVol << std::endl;
}

template <typename Real>
void ElasticitySolverT<Real>::SetInitialOffset(const Vec3& offset)
{
    for (auto& v : h_vertex) {
        v += offset;
    }
    CUDA_CHECK(cudaMemcpy(d_vertex,
        h_vertex.data(),
        h_vertex.size() * sizeof(Vec3),
        cudaMemcpyHostToDevice));
}

template <typename Real>
void ElasticitySolverT<Real>::RotateVerticesByEulerAngles(const Vec3& euler_angles)
{
    const Real cx = std::cos(euler_angles.x);
    const Real sx = std::sin(euler_angles.x);
    const Real cy = std::cos(euler_angles.y);
    const Real sy = std::sin(euler_angles.y);
    const Real cz = std::cos(euler_angles.z);
    const Real sz = std::sin(euler_angles.z);

    const mat3<Real> rot_x(
        static_cast<Real>(1), static_cast<Real>(0), static_cast<Real>(0),
        static_cast<Real>(0), cx, -sx,
        static_cast<Real>(0), sx, cx);

    const mat3<Real> rot_y(
        cy, static_cast<Real>(0), sy,
        static_cast<Real>(0), static_cast<Real>(1), static_cast<Real>(0),
        -sy, static_cast<Real>(0), cy);

    const mat3<Real> rot_z(
        cz, -sz, static_cast<Real>(0),
        sz, cz, static_cast<Real>(0),
        static_cast<Real>(0), static_cast<Real>(0), static_cast<Real>(1));

    const mat3<Real> rotation = rot_z * rot_y * rot_x;
    for (auto& v : h_vertex) {
        v = rotation * v;
    }

    CUDA_CHECK(cudaMemcpy(
        d_vertex,
        h_vertex.data(),
        h_vertex.size() * sizeof(Vec3),
        cudaMemcpyHostToDevice));
}

template <typename Real>
void ElasticitySolverT<Real>::RotateVerticesAroundCentroidByEulerAngles(const Vec3& euler_angles)
{
    if (h_vertex.empty()) {
        return;
    }

    Vec3 centroid{ static_cast<Real>(0), static_cast<Real>(0), static_cast<Real>(0) };
    for (const auto& v : h_vertex) {
        centroid += v;
    }
    centroid /= static_cast<Real>(h_vertex.size());

    const Real cx = std::cos(euler_angles.x);
    const Real sx = std::sin(euler_angles.x);
    const Real cy = std::cos(euler_angles.y);
    const Real sy = std::sin(euler_angles.y);
    const Real cz = std::cos(euler_angles.z);
    const Real sz = std::sin(euler_angles.z);

    const mat3<Real> rot_x(
        static_cast<Real>(1), static_cast<Real>(0), static_cast<Real>(0),
        static_cast<Real>(0), cx, -sx,
        static_cast<Real>(0), sx, cx);

    const mat3<Real> rot_y(
        cy, static_cast<Real>(0), sy,
        static_cast<Real>(0), static_cast<Real>(1), static_cast<Real>(0),
        -sy, static_cast<Real>(0), cy);

    const mat3<Real> rot_z(
        cz, -sz, static_cast<Real>(0),
        sz, cz, static_cast<Real>(0),
        static_cast<Real>(0), static_cast<Real>(0), static_cast<Real>(1));

    const mat3<Real> rotation = rot_z * rot_y * rot_x;
    for (auto& v : h_vertex) {
        v = rotation * (v - centroid) + centroid;
    }

    CUDA_CHECK(cudaMemcpy(
        d_vertex,
        h_vertex.data(),
        h_vertex.size() * sizeof(Vec3),
        cudaMemcpyHostToDevice));
}

template <typename Real>
void ElasticitySolverT<Real>::Step_Explicit()
{
    int numTets = static_cast<int>(h_tet.size());
    int numVerts = static_cast<int>(h_vertex.size());

    k_applyKinematicTargets<Real><<<grid1D(numVerts), 256>>>(
        d_vertex, d_vertex_velocity, d_constraint_dof_flags, d_constraint_dof_targets, numVerts);
    CUDA_CHECK(cudaGetLastError());

    k_AddGravity<<<grid1D(numVerts), 256>>>(d_force, d_mass, numVerts);
    CUDA_CHECK(cudaGetLastError());

    k_ComputeForces<Real><<<grid1D(numTets), 256>>>(
        d_tet, d_vertex, d_force, d_F, numTets);
    CUDA_CHECK(cudaGetLastError());

    if (d_skeleton_weights && h_params.muscle_coupling_stiffness > static_cast<Real>(0)) {
        k_AddSkeletonCouplingForces<Real><<<grid1D(numVerts), 256>>>(
            d_vertex,
            d_vertex_velocity,
            d_force,
            d_skeleton_targets,
            d_skeleton_target_velocities,
            d_skeleton_weights,
            h_params.muscle_coupling_stiffness,
            h_params.muscle_coupling_damping,
            numVerts);
        CUDA_CHECK(cudaGetLastError());
    }

    k_Integrate<Real><<<grid1D(numVerts), 256>>>(
        d_vertex, d_vertex_velocity, d_force, d_mass, numVerts);
    CUDA_CHECK(cudaGetLastError());

    k_applyKinematicTargets<Real><<<grid1D(numVerts), 256>>>(
        d_vertex, d_vertex_velocity, d_constraint_dof_flags, d_constraint_dof_targets, numVerts);
    CUDA_CHECK(cudaGetLastError());

    k_BoundaryCheck<Real><<<grid1D(numVerts), 256>>>(
        d_vertex, d_vertex_velocity, numVerts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename Real>
void ElasticitySolverT<Real>::Step_Implicit()
{
    if (h_params.energyType == COROTATED) {
        std::cerr << "Implicit solver supports STVK, Neo-Hookean, and ARAP energy. Use ARAP instead of COROTATED for SVD-based implicit stiffness.\n";
        exit(EXIT_FAILURE);
    }
    const int numTets = static_cast<int>(h_tet.size());
    const int numVerts = static_cast<int>(h_vertex.size());
    const int dof = 3 * numVerts;

    if (cublasH == nullptr) {
        InitCUDALib();
    }

    k_applyKinematicTargets<Real><<<grid1D(numVerts), 256>>>(
        d_vertex, d_vertex_velocity, d_constraint_dof_flags, d_constraint_dof_targets, numVerts);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemset(DnA, 0, dof * dof * sizeof(Real)));
    CUDA_CHECK(cudaMemset(d_force, 0, sizeof(Vec3) * numVerts));
    CUDA_CHECK(cudaMemset(d_b, 0, dof * sizeof(Real)));

    k_AddGravity<<<grid1D(numVerts), 256>>>(d_force, d_mass, numVerts);
    CUDA_CHECK(cudaGetLastError());

    k_ComputeForces<Real><<<grid1D(numTets), 256>>>(
        d_tet, d_vertex, d_force, d_F, numTets);
    CUDA_CHECK(cudaGetLastError());

    if (d_skeleton_weights && h_params.muscle_coupling_stiffness > static_cast<Real>(0)) {
        k_AddSkeletonCouplingImplicitForces<Real><<<grid1D(numVerts), 256>>>(
            d_vertex,
            d_force,
            d_skeleton_targets,
            d_skeleton_target_velocities,
            d_skeleton_weights,
            h_params.muscle_coupling_stiffness,
            h_params.muscle_coupling_damping,
            numVerts);
        CUDA_CHECK(cudaGetLastError());
    }

    // Assemble linear system
    k_computeK<Real><<<grid1D(numTets), 256>>>(
        d_tet, d_F, d_mass, DnA, numTets, numVerts);
    CUDA_CHECK(cudaGetLastError());

    k_Assemble<Real><<<grid1D(numVerts), 256>>>(
        DnA, d_b, d_vertex_velocity, d_force, d_mass, numVerts);
    CUDA_CHECK(cudaGetLastError());

    if (d_skeleton_weights && h_params.muscle_coupling_stiffness > static_cast<Real>(0)) {
        k_AddSkeletonCouplingDenseDiagonal<Real><<<grid1D(numVerts), 256>>>(
            DnA,
            d_skeleton_weights,
            h_params.muscle_coupling_stiffness,
            h_params.muscle_coupling_damping,
            numVerts);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t denseEntries = static_cast<size_t>(dof) * static_cast<size_t>(dof);
    k_applyDenseDirichletConstraints<Real><<<grid1D(static_cast<int>(denseEntries), 256), 256>>>(
        DnA, d_b, d_constraint_dof_flags, dof);
    CUDA_CHECK(cudaGetLastError());
    k_applyDenseDirichletRhs<Real><<<grid1D(dof), 256>>>(
        d_b, d_constraint_dof_flags, dof);
    CUDA_CHECK(cudaGetLastError());

    int iter = DenseCG<Real>(
        cublasH,
        DnA,
        delta_x,
        d_b,
        d_r,
        d_p,
        d_q,
        dof,
        cg_max_iters > 0 ? cg_max_iters : dof,
        cg_tolerance);
    
    printf("CG converged in %d iterations.\n", iter);

    // Implicit integrate.
    k_integrateImplicit<Real><<<grid1D(numVerts), 256>>>(
        d_vertex, d_vertex_velocity, delta_x, numVerts);
    CUDA_CHECK(cudaGetLastError());

    k_applyKinematicTargets<Real><<<grid1D(numVerts), 256>>>(
        d_vertex, d_vertex_velocity, d_constraint_dof_flags, d_constraint_dof_targets, numVerts);
    CUDA_CHECK(cudaGetLastError());

    //Boundary check.
    k_BoundaryCheck<Real><<<grid1D(numVerts), 256>>>(
        d_vertex, d_vertex_velocity, numVerts);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename Real>
void ElasticitySolverT<Real>::Step_Implicit_Sparse()
{
    if (h_params.energyType == COROTATED) {
        std::cerr << "Sparse implicit solver supports STVK, Neo-Hookean, and ARAP energy. Use ARAP instead of COROTATED for SVD-based implicit stiffness.\n";
        exit(EXIT_FAILURE);
    }
    const int numTets  = static_cast<int>(h_tet.size());
    const int numVerts = static_cast<int>(h_vertex.size());
    const int dof      = 3 * numVerts;
    const int nnz      = static_cast<int>(h_A_values.size());

    if (cublasH == nullptr) {
        InitCUDALib();
    }

    k_applyKinematicTargets<Real><<<grid1D(numVerts), 256>>>(
        d_vertex, d_vertex_velocity, d_constraint_dof_flags, d_constraint_dof_targets, numVerts);
    CUDA_CHECK(cudaGetLastError());

    // Reset working buffers for this substep.
    k_zero_real<Real><<<grid1D(nnz), 256>>>(d_A_values, nnz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemset(d_force, 0, sizeof(Vec3) * numVerts));
    CUDA_CHECK(cudaMemset(d_b, 0, dof * sizeof(Real)));

    k_AddGravity<<<grid1D(numVerts), 256>>>(d_force, d_mass, numVerts);
    CUDA_CHECK(cudaGetLastError());

    k_ComputeForces<Real><<<grid1D(numTets), 256>>>(
        d_tet, d_vertex, d_force, d_F, numTets);
    CUDA_CHECK(cudaGetLastError());

    if (d_skeleton_weights && h_params.muscle_coupling_stiffness > static_cast<Real>(0)) {
        k_AddSkeletonCouplingImplicitForces<Real><<<grid1D(numVerts), 256>>>(
            d_vertex,
            d_force,
            d_skeleton_targets,
            d_skeleton_target_velocities,
            d_skeleton_weights,
            h_params.muscle_coupling_stiffness,
            h_params.muscle_coupling_damping,
            numVerts);
        CUDA_CHECK(cudaGetLastError());
    }

    // Scatter element 12x12 stiffness blocks into the global CSR values.
    k_computeK_csr<Real><<<grid1D(numTets), 256>>>(
        d_tet, d_F, d_elem_to_A_csr, d_A_values, numTets);
    CUDA_CHECK(cudaGetLastError());

    // Barrier-based AABB collision: add barrier force to d_force and
    // barrier diagonal hessian to d_A_values (must run before k_assemble_csr).
    k_addBarrierForces<Real><<<grid1D(numVerts), 256>>>(
        d_vertex, d_force, d_A_values, d_A_diag_indices, numVerts);
    CUDA_CHECK(cudaGetLastError());

    // Add inertia M/dt^2 to diagonal and form b.
    k_assemble_csr<Real><<<grid1D(numVerts), 256>>>(
        d_A_values, d_A_diag_indices, d_b, d_vertex_velocity, d_force, d_mass, numVerts);
    CUDA_CHECK(cudaGetLastError());

    if (d_skeleton_weights && h_params.muscle_coupling_stiffness > static_cast<Real>(0)) {
        k_AddSkeletonCouplingCsrDiagonal<Real><<<grid1D(numVerts), 256>>>(
            d_A_values,
            d_A_diag_indices,
            d_skeleton_weights,
            h_params.muscle_coupling_stiffness,
            h_params.muscle_coupling_damping,
            numVerts);
        CUDA_CHECK(cudaGetLastError());
    }

    k_applyCsrDirichletConstraints<Real><<<grid1D(dof), 256>>>(
        d_A_values,
        d_A_row_offsets,
        d_A_col_indices,
        d_A_diag_indices,
        d_b,
        d_constraint_dof_flags,
        dof);
    CUDA_CHECK(cudaGetLastError());

    // Build Jacobi preconditioner from the assembled diagonal.
    k_extract_diag_inv<Real><<<grid1D(dof), 256>>>(
        d_A_values, d_A_diag_indices, d_M_inv, dof);
    CUDA_CHECK(cudaGetLastError());

    int iter = SparsePCG<Real>(
        cublasH,
        cusparseH,
        A,
        vecP,
        vecQ,
        d_spmv_buffer,
        delta_x,
        d_b,
        d_r,
        d_p,
        d_q,
        d_z,
        d_M_inv,
        dof,
        cg_max_iters > 0 ? cg_max_iters : dof,
        cg_tolerance);

    printf("Sparse PCG converged in %d iterations.\n", iter);

    // Implicit integrate.
    k_integrateImplicit<Real><<<grid1D(numVerts), 256>>>(
        d_vertex, d_vertex_velocity, delta_x, numVerts);
    CUDA_CHECK(cudaGetLastError());

    k_applyKinematicTargets<Real><<<grid1D(numVerts), 256>>>(
        d_vertex, d_vertex_velocity, d_constraint_dof_flags, d_constraint_dof_targets, numVerts);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename Real>
void ElasticitySolverT<Real>::Step()
{
    if (h_params.solverType == EXPLICIT) Step_Explicit();
    else if (h_params.solverType == IMPLICIT) Step_Implicit();
    else if (h_params.solverType == IMPLICIT_SPARSE) Step_Implicit_Sparse();
    else {
        std::cerr << "Unknown solver type!\n";
        exit(EXIT_FAILURE);
    }
}

template void ElasticitySolverT<float>::SetParams();
template void ElasticitySolverT<double>::SetParams();

template void ElasticitySolverT<float>::PreCompute();
template void ElasticitySolverT<double>::PreCompute();

template void ElasticitySolverT<float>::SetInitialOffset(const ElasticitySolverT<float>::Vec3&);
template void ElasticitySolverT<double>::SetInitialOffset(const ElasticitySolverT<double>::Vec3&);

template void ElasticitySolverT<float>::RotateVerticesByEulerAngles(const ElasticitySolverT<float>::Vec3&);
template void ElasticitySolverT<double>::RotateVerticesByEulerAngles(const ElasticitySolverT<double>::Vec3&);

template void ElasticitySolverT<float>::RotateVerticesAroundCentroidByEulerAngles(const ElasticitySolverT<float>::Vec3&);
template void ElasticitySolverT<double>::RotateVerticesAroundCentroidByEulerAngles(const ElasticitySolverT<double>::Vec3&);

template void ElasticitySolverT<float>::Step_Explicit();
template void ElasticitySolverT<double>::Step_Explicit();

template void ElasticitySolverT<float>::Step_Implicit();
template void ElasticitySolverT<double>::Step_Implicit();

template void ElasticitySolverT<float>::Step_Implicit_Sparse();
template void ElasticitySolverT<double>::Step_Implicit_Sparse();

template void ElasticitySolverT<float>::Step();
template void ElasticitySolverT<double>::Step();
