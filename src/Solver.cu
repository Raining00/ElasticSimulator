/**
 * Solver.cu
 * Explicit and implicit FEM elasticity solver using CUDA.
 *
 * Explicit path:
 *   1. Compute elastic forces
 *   2. Symplectic Euler integration
 *   3. Boundary projection
 *
 * Implicit path (Neo-Hookean only for now):
 *   1. Predictor x* = x_n + dt v_n
 *   2. Linearize implicit Euler around x*
 *   3. Solve (M / dt^2 - dF/dx) dx = f(x*) with matrix-free CG
 *   4. Update x_{n+1} = x* + dx, v_{n+1} = (x_{n+1} - x_n) / dt
 */

#include "Solver.h"
#include "math/decomposition.hpp"
#include "iostream"
#include <algorithm>
#include <cmath>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error in " << __FILE__ << " at line "           \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "\n";  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

static dim3 grid1D(int n, int block = 256) {
    return dim3((n + block - 1) / block);
}

struct DevParams {
    float lambda;
    float mu;
    float damping;
    float dt;
    float density;
    float implicit_fd_epsilon;
    float3 gravity;
    float3 boundary_min;
    float3 boundary_max;
    int energyType;
};

__constant__ DevParams c_params;

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

    mat3 Ds(x1 - x0, x2 - x0, x3 - x0);
    return Ds * tet.Dm_inv;
}

__device__ mat3 P_STVK(const mat3& F, float mu, float lambda)
{
    mat3 FtF = mat3::multiplyAtB(F, F);
    mat3 E = (FtF - mat3(1.f)) * 0.5f;
    float trE = mat3::trace(E);
    mat3 S = E * (2.f * mu) + mat3(lambda * trE);
    return F * S;
}

__device__ mat3 P_Corotated(const mat3& F, float mu, float lambda)
{
    mat3 R;
    computePD(F, R);

    mat3 RtF = mat3::multiplyAtB(R, F);
    float tr = mat3::trace(RtF) - 3.f;

    return (F - R) * (2.f * mu) + R * (lambda * tr);
}

__device__ mat3 P_NeoHookean(const mat3& F, float mu, float lambda)
{
    float J = mat3::determinant(F);
    J = fmaxf(J, 1e-4f);

    mat3 Finvt = mat3::transpose(mat3::inverse(F));
    return (F - Finvt) * mu + Finvt * (lambda * (J - 1.f) * J);
}

__device__ __forceinline__ mat3 computeStress(const mat3& F)
{
    int etype = c_params.energyType;
    float mu = c_params.mu;
    float lambda = c_params.lambda;

    if (etype == 0) return P_STVK(F, mu, lambda);
    if (etype == 1) return P_Corotated(F, mu, lambda);
    return P_NeoHookean(F, mu, lambda);
}

__global__ void k_ZeroVec(float3* vec, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    vec[tid] = make_float3(0.f, 0.f, 0.f);
}

__global__ void k_ComputeForces(
    const Tetrahedron* __restrict__ tets,
    const float3* __restrict__ vertex,
    float3* force,
    int numTets)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numTets) return;

    const Tetrahedron& tet = tets[tid];
    int i0 = tet.verticesIndex.x;
    int i1 = tet.verticesIndex.y;
    int i2 = tet.verticesIndex.z;
    int i3 = tet.verticesIndex.w;

    float V0 = tet.volume;
    mat3 F = computeF(tet, vertex);
    mat3 P = computeStress(F);

    mat3 DmInvT = mat3::transpose(tet.Dm_inv);
    mat3 H = P * DmInvT * (-V0);

    float3 f1 = H.column(0);
    float3 f2 = H.column(1);
    float3 f3 = H.column(2);
    float3 f0 = make_float3(-f1.x - f2.x - f3.x,
        -f1.y - f2.y - f3.y,
        -f1.z - f2.z - f3.z);

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
}

__global__ void k_AddGravity(float3* force, const float* mass, int numVerts)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVerts) return;

    float m = mass[vid];
    float3 g = c_params.gravity;
    force[vid].x += m * g.x;
    force[vid].y += m * g.y;
    force[vid].z += m * g.z;
}

__global__ void k_IntegrateExplicit(
    float3* __restrict__ vertex,
    float3* __restrict__ velocity,
    const float3* __restrict__ force,
    const float* __restrict__ mass,
    int numVerts)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVerts) return;

    float m = mass[vid];
    if (m < 1e-12f) return;

    float dt = c_params.dt;
    float damping = c_params.damping;

    float3 a = force[vid] / m;
    float3 v = velocity[vid];
    v = v * (1.f - damping) + a * dt;

    float3 x = vertex[vid];
    x += v * dt;

    velocity[vid] = v;
    vertex[vid] = x;
}

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

    if (x.x < bmin.x) { x.x = bmin.x; if (v.x < 0) { v.x = -v.x * restitution; v.y *= friction; v.z *= friction; } }
    if (x.x > bmax.x) { x.x = bmax.x; if (v.x > 0) { v.x = -v.x * restitution; v.y *= friction; v.z *= friction; } }

    if (x.y < bmin.y) { x.y = bmin.y; if (v.y < 0) { v.y = -v.y * restitution; v.x *= friction; v.z *= friction; } }
    if (x.y > bmax.y) { x.y = bmax.y; if (v.y > 0) { v.y = -v.y * restitution; v.x *= friction; v.z *= friction; } }

    if (x.z < bmin.z) { x.z = bmin.z; if (v.z < 0) { v.z = -v.z * restitution; v.x *= friction; v.y *= friction; } }
    if (x.z > bmax.z) { x.z = bmax.z; if (v.z > 0) { v.z = -v.z * restitution; v.x *= friction; v.y *= friction; } }

    vertex[vid] = x;
    velocity[vid] = v;
}

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
    float det = mat3::determinant(Dm);
    tet.volume = fabsf(det) / 6.f;
    tet.Dm_inv = mat3::inverse(Dm);
}

__global__ void k_CopyVec(float3* dst, const float3* src, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    dst[tid] = src[tid];
}

__global__ void k_SetPredictor(float3* vertex, float3* prev, const float3* velocity, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    prev[tid] = vertex[tid];
    vertex[tid] = vertex[tid] + velocity[tid] * c_params.dt;
}

__global__ void k_BuildTrialPositions(float3* trial, const float3* base, const float3* dir, float alpha, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    trial[tid] = base[tid] + dir[tid] * alpha;
}

__global__ void k_MassScaledCopy(float3* out, const float3* in, const float* mass, float scale, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = in[tid] * (mass[tid] * scale);
}

__global__ void k_AddForceDifferenceToAp(float3* Ap, const float3* forceTrial, const float3* forceBase, float invEps, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    Ap[tid] -= (forceTrial[tid] - forceBase[tid]) * invEps;
}

__global__ void k_CopyNegated(float3* dst, const float3* src, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    dst[tid] = src[tid] * -1.f;
}

__global__ void k_InitCG(float3* solution, float3* residual, float3* direction, const float3* rhs, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    solution[tid] = make_float3(0.f, 0.f, 0.f);
    residual[tid] = rhs[tid];
    direction[tid] = rhs[tid];
}

__global__ void k_CGStepSolutionResidual(float3* solution, float3* residual, const float3* direction, const float3* Ap, float alpha, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    solution[tid] += direction[tid] * alpha;
    residual[tid] -= Ap[tid] * alpha;
}

__global__ void k_CGStepDirection(float3* direction, const float3* residual, float beta, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    direction[tid] = residual[tid] + direction[tid] * beta;
}

__global__ void k_Dot(const float3* a, const float3* b, float* out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    atomicAdd(out, dot(a[tid], b[tid]));
}

__global__ void k_FinalizeImplicitUpdate(float3* vertex, const float3* prev, float3* velocity, const float3* dx, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float3 x = vertex[tid] + dx[tid];
    velocity[tid] = (x - prev[tid]) / c_params.dt;
    vertex[tid] = x;
}

void ElasticitySolver::SetParams()
{
    float E = h_params.youngs_modulus;
    float nu = h_params.poisson_ratio;

    h_params.mu = E / (2.f * (1.f + nu));
    h_params.lambda = E * nu / ((1.f + nu) * (1.f - 2.f * nu));

    DevParams dp;
    dp.lambda = h_params.lambda;
    dp.mu = h_params.mu;
    dp.damping = h_params.damping;
    dp.dt = h_params.dt;
    dp.density = h_params.density;
    dp.implicit_fd_epsilon = fmaxf(h_params.implicit_fd_epsilon, 1e-6f);
    dp.gravity = h_params.gravity;
    dp.boundary_min = h_params.boundary_min;
    dp.boundary_max = h_params.boundary_max;
    dp.energyType = static_cast<int>(h_params.energyType);

    std::fill(h_mass.begin(), h_mass.end(), 0.0f);
    for (const auto& t : h_tet) {
        float tetMass = h_params.density * t.volume * 0.25f;
        h_mass[t.verticesIndex.x] += tetMass;
        h_mass[t.verticesIndex.y] += tetMass;
        h_mass[t.verticesIndex.z] += tetMass;
        h_mass[t.verticesIndex.w] += tetMass;
    }
    CUDA_CHECK(cudaMemcpy(d_mass, h_mass.data(), h_mass.size() * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpyToSymbol(c_params, &dp, sizeof(DevParams)));
}

void ElasticitySolver::ComputeTetInitVolume()
{
    int numTets = static_cast<int>(h_tet.size());

    k_ComputeTetInitVolume<<<grid1D(numTets), 256>>>(d_tet, d_vertex, numTets);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_tet.data(), d_tet,
        numTets * sizeof(Tetrahedron),
        cudaMemcpyDeviceToHost));

    std::fill(h_mass.begin(), h_mass.end(), 0.0f);
    float minVol = 1e30f;
    for (const auto& t : h_tet) {
        minVol = std::min(minVol, t.volume);
        float tetMass = h_params.density * t.volume * 0.25f;
        h_mass[t.verticesIndex.x] += tetMass;
        h_mass[t.verticesIndex.y] += tetMass;
        h_mass[t.verticesIndex.z] += tetMass;
        h_mass[t.verticesIndex.w] += tetMass;
    }

    CUDA_CHECK(cudaMemcpy(d_mass, h_mass.data(), h_mass.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "Min tet rest-volume: " << minVol << std::endl;
}

void ElasticitySolver::SetInitialOffset(const float3& offset)
{
    for (auto& v : h_vertex) {
        v.x += offset.x;
        v.y += offset.y;
        v.z += offset.z;
    }
    CUDA_CHECK(cudaMemcpy(d_vertex, h_vertex.data(), h_vertex.size() * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vertex_prev, h_vertex.data(), h_vertex.size() * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vertex_trial, h_vertex.data(), h_vertex.size() * sizeof(float3), cudaMemcpyHostToDevice));
}

bool ElasticitySolver::SolveImplicitCG()
{
    const int numVerts = static_cast<int>(h_vertex.size());
    const int numTets = static_cast<int>(h_tet.size());
    const float dt2_inv = 1.0f / (h_params.dt * h_params.dt);
    const float eps = fmaxf(h_params.implicit_fd_epsilon, 1e-6f);

    k_InitCG<<<grid1D(numVerts), 256>>>(d_cg_solution, d_cg_residual, d_cg_direction, d_cg_rhs, numVerts);
    CUDA_CHECK(cudaGetLastError());

    auto deviceDot = [&](const float3* a, const float3* b) -> float {
        CUDA_CHECK(cudaMemset(d_cg_scalar, 0, sizeof(float)));
        k_Dot<<<grid1D(numVerts), 256>>>(a, b, d_cg_scalar, numVerts);
        CUDA_CHECK(cudaGetLastError());
        float result = 0.0f;
        CUDA_CHECK(cudaMemcpy(&result, d_cg_scalar, sizeof(float), cudaMemcpyDeviceToHost));
        return result;
    };

    float rr = deviceDot(d_cg_residual, d_cg_residual);
    const float rhsNorm0 = std::sqrt(std::max(rr, 0.0f));
    if (rhsNorm0 < 1e-12f) {
        return true;
    }

    for (unsigned int iter = 0; iter < h_params.implicit_cg_max_iters; ++iter) {
        k_BuildTrialPositions<<<grid1D(numVerts), 256>>>(d_vertex_trial, d_vertex, d_cg_direction, eps, numVerts);
        CUDA_CHECK(cudaGetLastError());

        k_ZeroVec<<<grid1D(numVerts), 256>>>(d_force_trial, numVerts);
        CUDA_CHECK(cudaGetLastError());
        k_ComputeForces<<<grid1D(numTets), 256>>>(d_tet, d_vertex_trial, d_force_trial, numTets);
        CUDA_CHECK(cudaGetLastError());
        k_AddGravity<<<grid1D(numVerts), 256>>>(d_force_trial, d_mass, numVerts);
        CUDA_CHECK(cudaGetLastError());

        k_MassScaledCopy<<<grid1D(numVerts), 256>>>(d_cg_Ap, d_cg_direction, d_mass, dt2_inv, numVerts);
        CUDA_CHECK(cudaGetLastError());
        k_AddForceDifferenceToAp<<<grid1D(numVerts), 256>>>(d_cg_Ap, d_force_trial, d_force, 1.0f / eps, numVerts);
        CUDA_CHECK(cudaGetLastError());

        float pAp = deviceDot(d_cg_direction, d_cg_Ap);
        if (!std::isfinite(pAp) || fabsf(pAp) < 1e-20f) {
            return false;
        }

        float alpha = rr / pAp;
        if (!std::isfinite(alpha)) {
            return false;
        }

        k_CGStepSolutionResidual<<<grid1D(numVerts), 256>>>(d_cg_solution, d_cg_residual, d_cg_direction, d_cg_Ap, alpha, numVerts);
        CUDA_CHECK(cudaGetLastError());

        float rrNew = deviceDot(d_cg_residual, d_cg_residual);
        if (!std::isfinite(rrNew)) {
            return false;
        }

        if (std::sqrt(std::max(rrNew, 0.0f)) <= h_params.implicit_cg_tolerance * std::max(rhsNorm0, 1e-6f)) {
            return true;
        }

        float beta = rrNew / rr;
        rr = rrNew;
        k_CGStepDirection<<<grid1D(numVerts), 256>>>(d_cg_direction, d_cg_residual, beta, numVerts);
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}

void ElasticitySolver::StepExplicit()
{
    int numTets = static_cast<int>(h_tet.size());
    int numVerts = static_cast<int>(h_vertex.size());

    k_ZeroVec<<<grid1D(numVerts), 256>>>(d_force, numVerts);
    CUDA_CHECK(cudaGetLastError());

    k_ComputeForces<<<grid1D(numTets), 256>>>(d_tet, d_vertex, d_force, numTets);
    CUDA_CHECK(cudaGetLastError());

    k_AddGravity<<<grid1D(numVerts), 256>>>(d_force, d_mass, numVerts);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_force.data(), d_force, sizeof(float3) * h_force.size(), cudaMemcpyDeviceToHost));

    k_IntegrateExplicit<<<grid1D(numVerts), 256>>>(d_vertex, d_vertex_velocity, d_force, d_mass, numVerts);
    CUDA_CHECK(cudaGetLastError());

    k_BoundaryCheck<<<grid1D(numVerts), 256>>>(d_vertex, d_vertex_velocity, numVerts);
    CUDA_CHECK(cudaGetLastError());
}

void ElasticitySolver::StepImplicit()
{
    int numTets = static_cast<int>(h_tet.size());
    int numVerts = static_cast<int>(h_vertex.size());

    k_SetPredictor<<<grid1D(numVerts), 256>>>(d_vertex, d_vertex_prev, d_vertex_velocity, numVerts);
    CUDA_CHECK(cudaGetLastError());

    k_ZeroVec<<<grid1D(numVerts), 256>>>(d_force, numVerts);
    CUDA_CHECK(cudaGetLastError());
    k_ComputeForces<<<grid1D(numTets), 256>>>(d_tet, d_vertex, d_force, numTets);
    CUDA_CHECK(cudaGetLastError());
    k_AddGravity<<<grid1D(numVerts), 256>>>(d_force, d_mass, numVerts);
    CUDA_CHECK(cudaGetLastError());

    k_CopyVec<<<grid1D(numVerts), 256>>>(d_cg_rhs, d_force, numVerts);
    CUDA_CHECK(cudaGetLastError());

    bool cgOk = SolveImplicitCG();
    if (!cgOk) {
        k_CopyVec<<<grid1D(numVerts), 256>>>(d_vertex, d_vertex_prev, numVerts);
        CUDA_CHECK(cudaGetLastError());
        StepExplicit();
        return;
    }

    k_FinalizeImplicitUpdate<<<grid1D(numVerts), 256>>>(d_vertex, d_vertex_prev, d_vertex_velocity, d_cg_solution, numVerts);
    CUDA_CHECK(cudaGetLastError());

    k_BoundaryCheck<<<grid1D(numVerts), 256>>>(d_vertex, d_vertex_velocity, numVerts);
    CUDA_CHECK(cudaGetLastError());
}

void ElasticitySolver::Step()
{
    if (h_params.solverType == IMPLICIT) {
        if (h_params.energyType != NEOHOOKEAN) {
            if (!implicit_fallback_warned) {
                std::cerr << "Implicit FEM is only implemented for Neo-Hookean right now. Falling back to explicit integration." << std::endl;
                implicit_fallback_warned = true;
            }
            StepExplicit();
            return;
        }
        StepImplicit();
        return;
    }

    StepExplicit();
}
