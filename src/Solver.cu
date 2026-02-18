#include "../include/Solver.h"

__constant__ ElasticitySolver::Parameters g_params;

__global__ void ComputeTetInitVolumeKernel(Tetrahedron* d_tet, float3* d_vertex, int num_tets)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_tets)
    {
        Tetrahedron tet = d_tet[idx];
        float3 v0 = d_vertex[tet.verticesIndex.x];
        float3 v1 = d_vertex[tet.verticesIndex.y];
        float3 v2 = d_vertex[tet.verticesIndex.z];
        float3 v3 = d_vertex[tet.verticesIndex.w];

        // Compute the volume of the tetrahedron
        float volume = fabsf(dot(cross(v1 - v0, v2 - v0), v3 - v0)) / 6.0f;
        d_tet[idx].volume = volume;

        // Compute the rest state matrix Dm and its inverse
        float3 Dm_col0 = v1 - v0;
        float3 Dm_col1 = v2 - v0;
        float3 Dm_col2 = v3 - v0;
        mat3 Dm(Dm_col0, Dm_col1, Dm_col2);
        // Compute the inverse of Dm
        d_tet[idx].Dm_inv = mat3::inverse(Dm);
    }
}

__global__ void forward(float3* position, float3* velocity)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// add gravity
	velocity[idx] = velocity[idx] + g_params.gravity * g_params.dt;
    position[idx] = position[idx] + g_params.gravity * g_params.dt;

	// boundary condition
    if (position[idx].x < g_params.boundary_min.x) {
        position[idx].x = g_params.boundary_min.x;
        velocity[idx].x = 0;
	}
    else if(position[idx].x > g_params.boundary_max.x) {
        position[idx].x = g_params.boundary_max.x;
        velocity[idx].x = 0;
	}
    if (position[idx].y < g_params.boundary_min.y) {
        position[idx].y = g_params.boundary_min.y;
        velocity[idx].y = 0;
    }
    else if(position[idx].y > g_params.boundary_max.y) {
        position[idx].y = g_params.boundary_max.y;
        velocity[idx].y = 0;
    }
    if (position[idx].z < g_params.boundary_min.z) {
        position[idx].z = g_params.boundary_min.z;
        velocity[idx].z = 0;
    }
    else if(position[idx].z > g_params.boundary_max.z) {
        position[idx].z = g_params.boundary_max.z;
        velocity[idx].z = 0;
	}
}

__global__
void computeElasticForces(
    Tetrahedron* d_tet,
    float3* d_vertex,
    float3* d_force,
    int num_tets)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tets)
        return;

    Tetrahedron tet = d_tet[idx];

    // === current positions ===
    float3 x0 = d_vertex[tet.verticesIndex.x];
    float3 x1 = d_vertex[tet.verticesIndex.y];
    float3 x2 = d_vertex[tet.verticesIndex.z];
    float3 x3 = d_vertex[tet.verticesIndex.w];

    // === deformation gradient ===
    mat3 Ds(x1 - x0, x2 - x0, x3 - x0);
    mat3 F = Ds * tet.Dm_inv;

    mat3 P;

    float lambda = g_params.lambda;
    float mu     = g_params.mu;

    // ===============================
    // Material models
    // ===============================

    if (g_params.energyType == STVK)
    {
        mat3 E = 0.5f * (mat3::transpose(F) * F - mat3(1.0));
        mat3 S = lambda * mat3::trace(E) * mat3(1.0)
                 + 2.0f * mu * E;
        P = F * S;
    }
    // else if (g_params.energyType == COROTATED)
    // {
    //     // SVD
    //     mat3 U, V;
    //     float3 sigma;
    //     svd(F, U, sigma, V);

    //     mat3 R = U * mat3::transpose(V);

    //     float trace_term =
    //         mat3::trace(mat3::transpose(R) * F - mat3(1.0));

    //     P = 2.0f * mu * (F - R)
    //         + lambda * trace_term * R;
    // }
    else if (g_params.energyType == NEOHOOKEAN)
    {
        float J = mat3::determinant(F);
        mat3 FinvT = mat3::transpose(mat3::inverse(F));

        P = mu * (F - FinvT)
            + lambda * logf(J) * FinvT;
    }

    // ===============================
    // Compute gradients of shape functions
    // ===============================

    mat3 Dm_inv_T = mat3::transpose(tet.Dm_inv);

    float3 gradN1 = Dm_inv_T.column(0);
    float3 gradN2 = Dm_inv_T.column(1);
    float3 gradN3 = Dm_inv_T.column(2);
    float3 gradN0 = -(gradN1 + gradN2 + gradN3);

    float volume = tet.volume;

    float3 f0 = -volume * (P * gradN0);
    float3 f1 = -volume * (P * gradN1);
    float3 f2 = -volume * (P * gradN2);
    float3 f3 = -volume * (P * gradN3);

    // ===============================
    // Atomic add to global force array
    // ===============================

    atomicAddVec3(&d_force[tet.verticesIndex.x], f0);
    atomicAddVec3(&d_force[tet.verticesIndex.y], f1);
    atomicAddVec3(&d_force[tet.verticesIndex.z], f2);
    atomicAddVec3(&d_force[tet.verticesIndex.w], f3);
}

void ElasticitySolver::setParams()
{
    this->h_params.lambda = (this->h_params.youngs_modulus * this->h_params.poisson_ratio) / ((1 + this->h_params.poisson_ratio) * (1 - 2 * this->h_params.poisson_ratio));
    this->h_params.mu = this->h_params.youngs_modulus / (2 * (1 + this->h_params.poisson_ratio));
    this->h_params.dt = 1.f / 60.f / this->h_params.substeps;
    cudaMemcpyToSymbol((void*) &g_params, &(this->h_params), sizeof(Parameters));
}

void ElasticitySolver::ComputeTetInitVolume()
{
    // launch kernel to compute initial volume of each tetrahedron
    int num_tets = h_tet.size();
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_tets + threadsPerBlock - 1) / threadsPerBlock;
    ComputeTetInitVolumeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_tet, d_vertex, num_tets);
    cudaDeviceSynchronize();
}