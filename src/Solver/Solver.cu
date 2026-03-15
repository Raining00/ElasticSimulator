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
#include "math/decomposition.hpp"
#include "iostream"
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

// Convenience: launch with 1-D grid covering n elements
static dim3 grid1D(int n, int block = 256) {
    return dim3((n + block - 1) / block);
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
    Vector<Real, 3> gravity;
    Vector<Real, 3> boundary_min;
    Vector<Real, 3> boundary_max;
    int energyType; // 0=STVK, 1=COROTATED, 2=NEOHOOKEAN
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
template <typename Real>
__device__ mat3<Real> P_STVK(const mat3<Real>& F, Real mu, Real lambda)
{
    using Mat3 = mat3<Real>;
    Mat3 FtF = Mat3::multiplyAtB(F, F); // F^T * F
    Mat3 E = (FtF - Mat3(static_cast<Real>(1))) * static_cast<Real>(0.5); // Green strain
    Real trE = Mat3::trace(E);
    Mat3 S = E * (static_cast<Real>(2) * mu) + Mat3(lambda * trE); // 2nd PK
    return F * S;
}

// ---------- Corotated ----------
// F = R * S  (polar decomp)
// P = 2*mu*(F - R) + lambda*(J-1)*J * F^{-T}
// Simplified linear form: P = 2*mu*(F-R) + lambda*tr(S-I)*R
template <typename Real>
__device__ mat3<Real> P_Corotated(const mat3<Real>& F, Real mu, Real lambda)
{
    using Mat3 = mat3<Real>;
    Mat3 R;
    computePD<Real>(F, R);

    // tr(R^T F - I) = tr(S - I)  where S is symmetric part
    Mat3 RtF = Mat3::multiplyAtB(R, F);
    Real tr = Mat3::trace(RtF) - static_cast<Real>(3);

    return (F - R) * (static_cast<Real>(2) * mu) + R * (lambda * tr);
}

// ---------- Stable Neo-Hookean (Smith et al. 2018) ----------
// Psi = mu/2*(I_C - 3) - mu*log(J) + lambda/2*(J-1)^2
// P   = mu*(F - F^{-T}) + lambda*(J-1)*J*F^{-T}
template <typename Real>
__device__ mat3<Real> P_NeoHookean(const mat3<Real>& F, Real mu, Real lambda)
{
    using Mat3 = mat3<Real>;
    using Vec3 = Vector<Real, 3>;
    Real J = Mat3::determinant(F);
    // Clamp J to avoid singularity
    J = (J > static_cast<Real>(1e-4)) ? J : static_cast<Real>(1e-4);

    // Mat3 Finvt = Mat3::transpose(Mat3::inverse(F));
    Vec3 col0 = F.column(0);
    Vec3 col1 = F.column(1);
    Vec3 col2 = F.column(2);
    Mat3 adjFT = Mat3(cross(col1, col2), cross(col2, col0), cross(col0, col1));

    // mu*(F - F^{-T}) + lambda*(J-1)*J * F^{-T}
    // Mat3 P = (F - Finvt) * mu + Finvt * (lambda * (J - static_cast<Real>(1)) * J);
    Mat3 P = mu * F + (lambda * (J -1) - mu) * adjFT;
    return P;
}

// The return results is vectorized.
template <typename Real>
__host__ __device__
Mat9x12<Real> BuildBMatrixFromdFdx(const mat3<Real> dF_dx[12])
{
    Mat9x12<Real> B(Real(0));

    for (int a = 0; a < 12; ++a)
    {
        const mat3<Real>& M = dF_dx[a];

        B(0, a) = M(0, 0);
        B(1, a) = M(1, 0);
        B(2, a) = M(2, 0);
        B(3, a) = M(0, 1);
        B(4, a) = M(1, 1);
        B(5, a) = M(2, 1);
        B(6, a) = M(0, 2);
        B(7, a) = M(1, 2);
        B(8, a) = M(2, 2);
    }

    return B;
}

template <typename Real>
__host__ __device__
Mat9x9<Real> BuildMatrixGFromdJdF(const mat3<Real> dJdF)
{
    Mat9x9<Real> G(Real(0));
    Real g[9];

    g[0] = dJdF(0,0);
    g[1] = dJdF(1,0);
    g[2] = dJdF(2,0);
    g[3] = dJdF(0,1);
    g[4] = dJdF(1,1);
    g[5] = dJdF(2,1);
    g[6] = dJdF(0,2);
    g[7] = dJdF(1,2);
    g[8] = dJdF(2,2);

    #pragma unroll
    for (int r = 0; r < 9; ++r) {
        #pragma unroll
        for (int c = 0; c < 9; ++c) {
            G(r, c) = g[r] * g[c];
        }
    }

    return G;
}

template <typename Real>
__host__ __device__
Mat9x9<Real> BuildHessianMatrix(const mat3<Real>& F)
{
    BlockMat3<Real, 3, 3> H;
    Vector<Real, 3> f0 = F.column(0);
    Vector<Real, 3> f1 = F.column(1);
    Vector<Real, 3> f2 = F.column(2);

    mat3<Real> f0hat = crossProductMatrix(f0);
    mat3<Real> f1hat = crossProductMatrix(f1);
    mat3<Real> f2hat = crossProductMatrix(f2);
    mat3<Real> zeroMat(Real(0));

    H(0, 0) = zeroMat;
    H(0, 1) = -f2hat;
    H(0, 2) = f1hat;
    H(1, 0) = f2hat;
    H(1, 1) = zeroMat;
    H(1, 2) = -f0hat;
    H(2, 0) = -f1hat;
    H(2, 1) = f0hat;
    H(2, 2) = zeroMat;

    return FlattenBlockMat3(H);
}

template <typename Real>
__host__ __device__
Mat9x9<Real> BuildHessianMatrix(
    const Vector<Real, 3>& f0, 
    const Vector<Real, 3>& f1, 
    const Vector<Real, 3>& f2)
{
    BlockMat3<Real, 3, 3> H;

    mat3<Real> f0hat = crossProductMatrix(f0);
    mat3<Real> f1hat = crossProductMatrix(f1);
    mat3<Real> f2hat = crossProductMatrix(f2);
    mat3<Real> zeroMat(Real(0));

    H(0, 0) = zeroMat;
    H(0, 1) = -f2hat;
    H(0, 2) = f1hat;
    H(1, 0) = f2hat;
    H(1, 1) = zeroMat;
    H(1, 2) = -f0hat;
    H(2, 0) = -f1hat;
    H(2, 1) = f0hat;
    H(2, 2) = zeroMat;

    return FlattenBlockMat3(H);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1   compute elastic forces and mass from tetrahedra
//            Uses atomic adds to scatter forces/masses to vertices
// ─────────────────────────────────────────────────────────────────────────────

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
    Mat3 P;
    const DevParams<Real>& params = GetDevParams<Real>();
    int etype = params.energyType;
    Real mu = params.mu;
    Real lambda = params.lambda;

    if (etype == 0)      P = P_STVK<Real>(F, mu, lambda);
    else if (etype == 1) P = P_Corotated<Real>(F, mu, lambda);
    else                 P = P_NeoHookean<Real>(F, mu, lambda);

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
    Vec3 g = params.gravity;

    // Add gravity
    Vec3 f = force[vid];
    f += m * g;

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
__global__ void k_ComputeTetInitVolume(
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
        tet.Dm_inv[i] = static_cast<float>(dminv[i]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 5 Compute stiffness matrix for each tetrahedron
// df / dx = vec(dF / dx)^T * vec(dp / dF) * vec(dF / dx)
// ─────────────────────────────────────────────────────────────────────────────
template <typename Real>
__global__ void K_ComputeK(
    const Tetrahedron<Real>* __restrict__ tets,
    const mat3<Real>*      __restrict__ d_F,
    int numTets
)
{
    using Vec3 = Vector<Real, 3>;
    using Mat3 = mat3<Real>;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numTets) return;

    const Tetrahedron<Real>& tet = tets[tid]; 
    const Vec4i ids = tet.verticesIndex;
    Real volume = tet.volume;

    const DevParams<Real>& params = GetDevParams<Real>();
    Real mu = params.mu;
    Real lambda = params.lambda;

    Mat3 F = d_F[tid];
    Real J = Mat3::determinant(F);
    // Clamp J to avoid singularity
    J = (J > static_cast<Real>(1e-4)) ? J : static_cast<Real>(1e-4);

    //Compute dF / dx = (dD_s / dx) * (Dm_Inv)
    Mat3 dF_dx[12];
    Mat3 Dm_inv = tet.Dm_inv;
    Mat3 Dm_invT = Mat3::transpose(Dm_inv);
    Vec3 g[4]; // gradients of shape functions
    // g0, g1, g2 from Dm_inv^T columns
    // g3 = -g0 - g1 - g2
    g[1] = Dm_invT.column(0);
    g[2] = Dm_invT.column(1);
    g[3] = Dm_invT.column(2);
    g[0] = - (g[1] + g[2] + g[3]);
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            Mat3 dF(Real(0));
            for (int j = 0; j < 3; ++j) {
                dF(c, j) = g[a][j];
            }
            dF_dx[a * 3 + c] = dF;
        }
    }
    Mat9x12<Real> B = BuildBMatrixFromdFdx(dF_dx);

    /**
     * 
     * Compute dP / dF = mu * (dF / dFi) + lambda * (dJ / dFi) * (dJ / dF) + [lambda(J - 1) - mu] * (d^2 j / dFdFi)
     * 
     * vec(dF / dFi) = I9x9
     * vec((dJ / dFi) * (dJ / dF)) = vec(dJ / dF) * vec(dJ / dF)^T
     * dJ_dF = [f1 x f2 | f2 x f0) | f0 x f1];
     * vec(d^2 j / dFdFi) = crossproduct[        0           crossproduct(-f2)    crossproduct(f1)
     *                                   crossproduct(-f2)         0              crossproduct(-f0)
     *                                   crossproduct(-f1)   crossproduct(f0)           0          ]
     * Here f0, f1, f2 are the columes of deformation gradient matrix F.
    */
    // material Hessian
    Vec3 f0 = F.column(0);
    Vec3 f1 = F.column(1);
    Vec3 f2 = F.column(2);

    Mat3 dJ_dF(
        cross(f1, f2),
        cross(f2, f0),
        cross(f0, f1)
    );

    Mat9x9<Real> G        = BuildMatrixGFromdJdF(dJ_dF);
    Mat9x9<Real> hessianJ = BuildHessianMatrix(f0, f1, f2);
    Mat9x9<Real> I9       = Mat9x9<Real>::Identity();

    Mat9x9<Real> dP_dF =
          mu * I9
        + lambda * G
        + (lambda * (J - Real(1)) - mu) * hessianJ;

    Mat12x12<Real> Ke = volume * (transpose(B) * dP_dF * B);

   //K[tid] = volume * (transpose(B) * dP_dF * B);
}

template <typename Real>
__global__ void k_integrateImplicit(
    Vector<Real, 3>* __restrict__ vertex,
    Vector<Real, 3>* __restrict__ velocity,
    Vector<Real, 3>* __restrict__ delta_x,
    int numVerts)
{
    // Placeholder for implicit solver integration step
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVerts)
        return;
    
    Vector<Real, 3> dx = delta_x[idx];
    vertex[idx] += dx;
    velocity[idx] = dx / GetDevParams<Real>().dt; // Update velocity based on position change

    return;
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
    dp.energyType = static_cast<int>(h_params.energyType);

    if constexpr (std::is_same_v<Real, float>) {
        CUDA_CHECK(cudaMemcpyToSymbol(c_params_f, &dp, sizeof(DevParams<float>)));
    } else {
        CUDA_CHECK(cudaMemcpyToSymbol(c_params_d, &dp, sizeof(DevParams<double>)));
    }
}

template <typename Real>
void ElasticitySolverT<Real>::ComputeTetInitVolume()
{
    int numTets = static_cast<int>(h_tet.size());

    k_ComputeTetInitVolume<Real><<<grid1D(numTets), 256>>>(d_tet, d_vertex, d_mass, numTets);
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
void ElasticitySolverT<Real>::Step_Explicit()
{
    int numTets = static_cast<int>(h_tet.size());
    int numVerts = static_cast<int>(h_vertex.size());

    k_ComputeForces<Real><<<grid1D(numTets), 256>>>(
        d_tet, d_vertex, d_force, d_F, numTets);
    CUDA_CHECK(cudaGetLastError());

    k_Integrate<Real><<<grid1D(numVerts), 256>>>(
        d_vertex, d_vertex_velocity, d_force, d_mass, numVerts);
    CUDA_CHECK(cudaGetLastError());

    k_BoundaryCheck<Real><<<grid1D(numVerts), 256>>>(
        d_vertex, d_vertex_velocity, numVerts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename Real>
void ElasticitySolverT<Real>::Step_Implicit()
{
    if (h_params.energyType != NEOHOOKEAN) {
        std::cerr << "Implicit solver currently only supports Neo-Hookean energy.\n";
        exit(EXIT_FAILURE);
    }
    int numTets = static_cast<int>(h_tet.size());
    int numVerts = static_cast<int>(h_vertex.size());

    k_ComputeForces<Real><<<grid1D(numTets), 256>>>(
        d_tet, d_vertex, d_force, d_F, numTets);
    CUDA_CHECK(cudaGetLastError());

    // Compute stiffness matrix
    //K_ComputeK<Real> << <grid1D(numTets), 256 >> > (d_tet, d_F, K, numTets);

    // Assembel Linear system.

    // CG solver.

    // Implicit integrate.
    k_integrateImplicit<Real><<<grid1D(numVerts), 256>>>(
        d_vertex, d_vertex_velocity, delta_x, numVerts);
    CUDA_CHECK(cudaGetLastError());

    //Boundary check.
    k_BoundaryCheck<Real><<<grid1D(numVerts), 256>>>(
        d_vertex, d_vertex_velocity, numVerts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename Real>
void ElasticitySolverT<Real>::Step()
{
    if (h_params.solverType == EXPLICIT) Step_Explicit();
    else if (h_params.solverType == IMPLICIT) Step_Implicit();
    else {
        std::cerr << "Unknown solver type!\n";
        exit(EXIT_FAILURE);
    }
}

template void ElasticitySolverT<float>::SetParams();
template void ElasticitySolverT<double>::SetParams();

template void ElasticitySolverT<float>::ComputeTetInitVolume();
template void ElasticitySolverT<double>::ComputeTetInitVolume();

template void ElasticitySolverT<float>::SetInitialOffset(const ElasticitySolverT<float>::Vec3&);
template void ElasticitySolverT<double>::SetInitialOffset(const ElasticitySolverT<double>::Vec3&);

template void ElasticitySolverT<float>::Step_Explicit();
template void ElasticitySolverT<double>::Step_Explicit();

template void ElasticitySolverT<float>::Step_Implicit();
template void ElasticitySolverT<double>::Step_Implicit();

template void ElasticitySolverT<float>::Step();
template void ElasticitySolverT<double>::Step();
