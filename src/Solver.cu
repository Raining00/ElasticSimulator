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

__global__ void computeElasticForces(Tetrahedron* d_tet, float3* d_vertex, float3* d_vertex_rest, float3* d_vertex_velocity, int num_tets)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > num_tets)
        return;


}

void ElasticitySolver::setParams()
{
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