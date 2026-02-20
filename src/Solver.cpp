#include "Solver.h"
#include "iostream"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

bool ElasticitySolver::initialize(const std::vector<Tetrahedron>& tets, const std::vector<float3>& vertices)
{
    // Store data on host
    h_tet = tets;
    h_vertex = vertices;

    // Allocate and copy data to GPU
    CUDA_CHECK(cudaMalloc(&d_tet, tets.size() * sizeof(Tetrahedron)));
    CUDA_CHECK(cudaMemcpy(d_tet, tets.data(), tets.size() * sizeof(Tetrahedron), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_vertex, vertices.size() * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_vertex, vertices.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice));

    // For simplicity, we assume the rest positions are the same as the initial positions
    CUDA_CHECK(cudaMalloc(&d_vertex_rest, vertices.size() * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_vertex_rest, vertices.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice));

	// velocity initialization
	CUDA_CHECK(cudaMalloc(&d_vertex_velocity, vertices.size() * sizeof(float3)));
	// Initialize velocities to zero
	std::vector<float3> zero_velocity(vertices.size(), make_float3(0.0f, 0.0f, 0.0f));
	CUDA_CHECK(cudaMemcpy(d_vertex_velocity, zero_velocity.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice));

    // mass initialization
    CUDA_CHECK(cudaMalloc(&d_mass, vertices.size() * sizeof(float)));
    // Initialize masses to zero (this would typically be computed based on density and volume)
    std::vector<float> zero_mass(vertices.size(), 0.0f);
    CUDA_CHECK(cudaMemcpy(d_mass, zero_mass.data(), vertices.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    return true;
}


ElasticitySolver::~ElasticitySolver()
{
    // Free GPU memory
    cudaFree(d_tet);
    cudaFree(d_vertex);
    cudaFree(d_vertex_rest);
}