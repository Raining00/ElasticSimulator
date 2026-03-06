/**
 * Solver.cu
 * Explicit FEM elasticity solver using CUDA.
 *
 * Pipeline per substep:
 *   1. ComputeForces  – compute elastic + gravity forces on every vertex
 *   2. Integrate      – symplectic-Euler velocity/position update
 *   3. BoundaryCheck  – enforce axis-aligned bounding-box collisions
 *
 * Supported energy models (EnergyType):
 *   STVK        – Saint-Venant Kirchhoff
 *   COROTATED   – Corotated linear elasticity
 *   NEOHOOKEAN  – Stable Neo-Hookean (Smith et al. 2018)
 */

#include "Solver.h"
#include "math/decomposition.hpp"
#include "iostream"

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

struct DevParams {
    float lambda;
    float mu;
    float damping;
    float dt;
    float density;
    float3 gravity;
    float3 boundary_min;
    float3 boundary_max;
    int   energyType;   // 0=STVK, 1=COROTATED, 2=NEOHOOKEAN
};

__constant__ DevParams c_params;

// ─────────────────────────────────────────────────────────────────────────────
// Deformation gradient  F = Ds * Dm_inv
// Ds = [x1-x0 | x2-x0 | x3-x0]  (current edge matrix)
// ─────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__ mat3 computeF(
    const Tetrahedron& tet,
    const float3* vertex)
{
    int i0 = tet.verticesIndex.x;
    int i1 = tet.verticesIndex.y;
    int i2 = tet.verticesIndex.z;
    int i3 = tet.verticesIndex.w;

    float3 x0 = vertex[i0];
    float3 x1 = vertex[i1];
    float3 x2 = vertex[i2];
    float3 x3 = vertex[i3];

    // Ds: columns are edge vectors from vertex 0
    mat3 Ds(x1 - x0, x2 - x0, x3 - x0);
    return Ds * tet.Dm_inv;
}

// ─────────────────────────────────────────────────────────────────────────────
// First Piola-Kirchhoff stress  P(F)
// ─────────────────────────────────────────────────────────────────────────────

// ---------- StVK ----------
// E = 0.5*(F'F - I)
// P = F*(2*mu*E + lambda*tr(E)*I)
__device__ mat3 P_STVK(const mat3& F, float mu, float lambda)
{
    mat3 FtF = mat3::multiplyAtB(F, F);            // F^T * F
    mat3 E = (FtF - mat3(1.f)) * 0.5f;           // Green strain
    float trE = mat3::trace(E);
    mat3 S = E * (2.f * mu) + mat3(lambda * trE); // 2nd PK
    return F * S;
}

// ---------- Corotated ----------
// F = R * S  (polar decomp)
// P = 2*mu*(F - R) + lambda*(J-1)*J * F^{-T}
// Simplified linear form: P = 2*mu*(F-R) + lambda*tr(S-I)*R
__device__ mat3 P_Corotated(const mat3& F, float mu, float lambda)
{
    mat3 R;
    computePD(F, R);

    // tr(R^T F - I) = tr(S - I)  where S is symmetric part
    mat3 RtF = mat3::multiplyAtB(R, F);
    float tr = mat3::trace(RtF) - 3.f;

    return (F - R) * (2.f * mu) + R * (lambda * tr);
}

// ---------- Stable Neo-Hookean (Smith et al. 2018) ----------
// Psi = mu/2*(I_C - 3) - mu*log(J) + lambda/2*(J-1)^2
// P   = mu*(F - F^{-T}) + lambda*(J-1)*J*F^{-T}
__device__ mat3 P_NeoHookean(const mat3& F, float mu, float lambda)
{
    float J = mat3::determinant(F);
    // Clamp J to avoid singularity
    J = fmaxf(J, 1e-4f);

    mat3 Finvt = mat3::transpose(mat3::inverse(F));

    // mu*(F - F^{-T}) + lambda*(J-1)*J * F^{-T}
    mat3 P = (F - Finvt) * mu + Finvt * (lambda * (J - 1.f) * J);
    return P;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1 – compute elastic forces and mass from tetrahedra
//            Uses atomic adds to scatter forces/masses to vertices
// ─────────────────────────────────────────────────────────────────────────────

__global__ void k_ComputeForcesAndMass(
    const Tetrahedron* __restrict__ tets,
    const float3* __restrict__ vertex,
    float3* force,
    float* mass,
    int                             numTets)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numTets) return;

    const Tetrahedron& tet = tets[tid];

    int i0 = tet.verticesIndex.x;
    int i1 = tet.verticesIndex.y;
    int i2 = tet.verticesIndex.z;
    int i3 = tet.verticesIndex.w;

    float V0 = tet.volume;          // rest volume

    // ─── Deformation gradient ──────────────────────────────────────────────
    mat3 F = computeF(tet, vertex);

    // ─── First Piola-Kirchhoff stress ──────────────────────────────────────
    mat3 P;
    int etype = c_params.energyType;
    float mu = c_params.mu;
    float lambda = c_params.lambda;

    if (etype == 0)      P = P_STVK(F, mu, lambda);
    else if (etype == 1) P = P_Corotated(F, mu, lambda);
    else                 P = P_NeoHookean(F, mu, lambda);

    // ─── Nodal forces from the stress ─────────────────────────────────────
    // f = -V0 * P * Dm^{-T}   (distributed to the four nodes)
    // The force on node 1,2,3 = -V0 * P * col(Dm^{-T}, 0/1/2)
    // Force on node 0 = -(f1+f2+f3)
    mat3 DmInvT = mat3::transpose(tet.Dm_inv);
    mat3 H = P * DmInvT * (-V0);   // 3x3, columns = forces on nodes 1,2,3

    float3 f1 = H.column(0);
    float3 f2 = H.column(1);
    float3 f3 = H.column(2);
    float3 f0 = make_float3(-f1.x - f2.x - f3.x,
        -f1.y - f2.y - f3.y,
        -f1.z - f2.z - f3.z);

    // Atomic scatter to vertex force array
    atomicAdd(&force[i0].x, f0.x);
    atomicAdd(&force[i0].y, f0.y);
    atomicAdd(&force[i0].z, f0.z);

    atomicAdd(&force[i1].x, f1.x);
    atomicAdd(&force[i1].y, f1.y);
    atomicAdd(&force[i1].z, f1.z);

    atomicAdd(&force[i2].x, f2.x);
    atomicAdd(&force[i2].y, f2.y);
    atomicAdd(&force[i2].z, f2.z);

    atomicAdd(&force[i3].x, f3.x);
    atomicAdd(&force[i3].y, f3.y);
    atomicAdd(&force[i3].z, f3.z);

    // ─── Mass distribution (1/4 of tet mass to each vertex) ───────────────
    float tetMass = c_params.density * V0 * 0.25f;
    atomicAdd(&mass[i0], tetMass);
    atomicAdd(&mass[i1], tetMass);
    atomicAdd(&mass[i2], tetMass);
    atomicAdd(&mass[i3], tetMass);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2 – symplectic Euler integration + gravity + damping
// ─────────────────────────────────────────────────────────────────────────────

__global__ void k_Integrate(
    float3* __restrict__ vertex,
    float3* __restrict__ velocity,
    float3* __restrict__ force,
    float* __restrict__ mass,
    int numVerts)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVerts) return;

    float m = mass[vid];
    if (m < 1e-12f) return;   // guard against zero-mass vertices

    float dt = c_params.dt;
    float damping = c_params.damping;
    float3 g = c_params.gravity;

    // Add gravity
    float3 f = force[vid];
    f.x += m * g.x;
    f.y += m * g.y;
    f.z += m * g.z;

    // Acceleration
    float3 a = make_float3(f.x / m, f.y / m, f.z / m);

    // Symplectic Euler
    float3 v = velocity[vid];
    v.x = v.x * (1.f - damping) + a.x * dt;
    v.y = v.y * (1.f - damping) + a.y * dt;
    v.z = v.z * (1.f - damping) + a.z * dt;

    float3 x = vertex[vid];
    x.x += v.x * dt;
    x.y += v.y * dt;
    x.z += v.z * dt;

    velocity[vid] = v;
    vertex[vid] = x;

    // Reset force and mass for next substep
    force[vid] = make_float3(0.f, 0.f, 0.f);
    mass[vid] = 0.f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3 – AABB boundary collision (simple position projection + restitution)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void k_BoundaryCheck(
    float3* __restrict__ vertex,
    float3* __restrict__ velocity,
    int numVerts)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVerts) return;

    float3 x = vertex[vid];
    float3 v = velocity[vid];

    float3 bmin = c_params.boundary_min;
    float3 bmax = c_params.boundary_max;

    const float restitution = 0.3f;
    const float friction = 0.6f;

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
// Kernel 4 – initialise Dm_inv and rest volume for each tetrahedron
// ─────────────────────────────────────────────────────────────────────────────

__global__ void k_ComputeTetInitVolume(
    Tetrahedron* tets,
    const float3* __restrict__ vertex,
    int numTets)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numTets) return;

    Tetrahedron& tet = tets[tid];

    int i0 = tet.verticesIndex.x;
    int i1 = tet.verticesIndex.y;
    int i2 = tet.verticesIndex.z;
    int i3 = tet.verticesIndex.w;

    float3 x0 = vertex[i0];
    float3 x1 = vertex[i1];
    float3 x2 = vertex[i2];
    float3 x3 = vertex[i3];

    mat3 Dm(x1 - x0, x2 - x0, x3 - x0);

    // Volume = |det(Dm)| / 6
    float det = mat3::determinant(Dm);
    tet.volume = fabsf(det) / 6.f;
    tet.Dm_inv = mat3::inverse(Dm);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 5 – Compute stiffness matrix for each tetrahedron
// ─────────────────────────────────────────────────────────────────────────────
__global__ void K_ComputeK(
    const Tetrahedron* __restrict__ tets,
    const float3* __restrict__ vertex,
    const float3* __restrict__ velocity,
    const float3* __restrict__ force,
    const float* __restrict__ mass,
    const float* __restrict__ K, // global stiffness matrix in COO format (preallocated)
    int numTets, int numVerts)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numTets) return;

    const Tetrahedron& tet = tets[tid];
    const int ids[4] = {
        tet.verticesIndex.x,
        tet.verticesIndex.y,
        tet.verticesIndex.z,
        tet.verticesIndex.w
    };

    const float k_edge = (c_params.lambda + 2.f * c_params.mu) * tet.volume * 0.25f;
    const float k_diag = 3.f * k_edge;
    float* __restrict__ K_out = const_cast<float*>(K);
    const int ndof = numVerts * 3;

    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int ia3 = ids[a] * 3;
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int ib3 = ids[b] * 3;
            const float kab = (a == b) ? k_diag : -k_edge;

            atomicAdd(&K_out[(ia3 + 0) * ndof + (ib3 + 0)], kab);
            atomicAdd(&K_out[(ia3 + 1) * ndof + (ib3 + 1)], kab);
            atomicAdd(&K_out[(ia3 + 2) * ndof + (ib3 + 2)], kab);
        }
    }
}

__global__ void k_integrateImplicit(
    float3* __restrict__ vertex,
    float3* __restrict__ velocity,
    float3* __restrict__ delta_x,
    int numVerts)
{
    // Placeholder for implicit solver integration step
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVerts)
        return;
    
    float3 dx = delta_x[idx];
    vertex[idx] += dx;
    velocity[idx] = dx / c_params.dt; // Update velocity based on position change

    return;
}

// ─────────────────────────────────────────────────────────────────────────────
// ElasticitySolver member implementations
// ─────────────────────────────────────────────────────────────────────────────

void ElasticitySolver::SetParams()
{
    // Compute Lamé parameters from Young's modulus and Poisson's ratio
    float E = h_params.youngs_modulus;
    float nu = h_params.poisson_ratio;

    h_params.mu = E / (2.f * (1.f + nu));
    h_params.lambda = E * nu / ((1.f + nu) * (1.f - 2.f * nu));

    // Upload to constant memory
    DevParams dp;
    dp.lambda = h_params.lambda;
    dp.mu = h_params.mu;
    dp.damping = h_params.damping;
    dp.dt = h_params.dt;
    dp.density = h_params.density;
    dp.gravity = h_params.gravity;
    dp.boundary_min = h_params.boundary_min;
    dp.boundary_max = h_params.boundary_max;
    dp.energyType = static_cast<int>(h_params.energyType);

    CUDA_CHECK(cudaMemcpyToSymbol(c_params, &dp, sizeof(DevParams)));
}

void ElasticitySolver::ComputeTetInitVolume()
{
    int numTets = static_cast<int>(h_tet.size());

    k_ComputeTetInitVolume << <grid1D(numTets), 256 >> > (d_tet, d_vertex, numTets);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back so host-side h_tet also has Dm_inv and volume
    CUDA_CHECK(cudaMemcpy(h_tet.data(), d_tet,
        numTets * sizeof(Tetrahedron),
        cudaMemcpyDeviceToHost));

    // Verify no degenerate tets
    float minVol = 1e30f;
    for (auto& t : h_tet) minVol = std::min(minVol, t.volume);
    std::cout << "Min tet rest-volume: " << minVol << std::endl;
}

void ElasticitySolver::SetInitialOffset(const float3& offset)
{
    for (auto& v : h_vertex) {
        v.x += offset.x;
        v.y += offset.y;
        v.z += offset.z;
    }
    CUDA_CHECK(cudaMemcpy(d_vertex,
        h_vertex.data(),
        h_vertex.size() * sizeof(float3),
        cudaMemcpyHostToDevice));
}

void ElasticitySolver::Step_Explicit()
{
    int numTets = static_cast<int>(h_tet.size());
    int numVerts = static_cast<int>(h_vertex.size());

    // Forces need to start at zero before accumulation
    // (reset is done at the end of k_Integrate; first substep is zeroed in DataTransfer)

    // 1. Accumulate elastic forces and lumped mass
    k_ComputeForcesAndMass << <grid1D(numTets), 256 >> > (
        d_tet, d_vertex, d_force, d_mass, numTets);
    CUDA_CHECK(cudaGetLastError());

    // 2. Integrate (symplectic Euler) + reset buffers
    k_Integrate << <grid1D(numVerts), 256 >> > (
        d_vertex, d_vertex_velocity, d_force, d_mass, numVerts);
    CUDA_CHECK(cudaGetLastError());

    // 3. Boundary enforcement
    k_BoundaryCheck << <grid1D(numVerts), 256 >> > (
        d_vertex, d_vertex_velocity, numVerts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void ElasticitySolver::Step_Implicit()
{
    // Placeholder for implicit solver implementation
    if(h_params.energyType != NEOHOOKEAN) {
        std::cerr << "Implicit solver currently only supports Neo-Hookean energy.\n";
        exit(EXIT_FAILURE);
    }
    int numTets = static_cast<int>(h_tet.size());
    int numVerts = static_cast<int>(h_vertex.size());

    // 1. Compute forces
    k_ComputeForcesAndMass << <grid1D(numTets), 256 >> > (
        d_tet, d_vertex, d_force, d_mass, numTets);
    CUDA_CHECK(cudaGetLastError());

    // 2. Assmble linear systm. (Each thread computes contribution from one tet to the global stiffness matrix and force vector)

    // 3. Solve linear system. matrix-free CG or PCG with implicit mat-vec product kernel.

    // 4. Integrate positions/velocities
    k_integrateImplicit << <grid1D(numVerts), 256 >> > (
        d_vertex, d_vertex_velocity, d_vertex, numVerts);
    CUDA_CHECK(cudaGetLastError());

    // 5. Boundary enforcement
    k_BoundaryCheck << <grid1D(numVerts), 256 >> > (
        d_vertex, d_vertex_velocity, numVerts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ─── Single substep ────────────────────────────────────────────────────────

void ElasticitySolver::Step()
{
    if (h_params.solverType == EXPLICIT) Step_Explicit();
    else if (h_params.solverType == IMPLICIT) Step_Implicit();
    else {
        std::cerr << "Unknown solver type!\n";
        exit(EXIT_FAILURE);
    }
}
