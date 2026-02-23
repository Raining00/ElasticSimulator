#include "Solver.h"
#include "iostream"
#include "../include/MeshToTet.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

bool ElasticitySolver::DataTransfer(const std::vector<Tetrahedron>& tets, const std::vector<float3>& vertices)
{
    // Allocate and copy data to GPU
    CUDA_CHECK(cudaMalloc(&d_tet, tets.size() * sizeof(Tetrahedron)));
    CUDA_CHECK(cudaMemcpy(d_tet, tets.data(), tets.size() * sizeof(Tetrahedron), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_vertex, vertices.size() * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_vertex, vertices.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice));

	// velocity initialization
	CUDA_CHECK(cudaMalloc(&d_vertex_velocity, vertices.size() * sizeof(float3)));
	// Initialize velocities to zero
	CUDA_CHECK(cudaMemset(d_vertex_velocity, 0, vertices.size() * sizeof(float3)));

    // force initialization
    CUDA_CHECK(cudaMalloc(&d_force, vertices.size() * sizeof(float3)));
    // Initialize forces to zero
	CUDA_CHECK(cudaMemset(d_force, 0, vertices.size() * sizeof(float3)));

    // mass initialization
    CUDA_CHECK(cudaMalloc(&d_mass, vertices.size() * sizeof(float)));
    // Initialize masses to zero (this would typically be computed based on density and volume)
    CUDA_CHECK(cudaMemset(d_mass, 0, vertices.size() * sizeof(float)));
    
    return true;
}

void ElasticitySolver::Initialize(const Mesh& mesh)
{
    // This function would convert the Mesh data into the format needed for the solver
    // For simplicity, we assume the Mesh class has methods
    tetrahedralizeMesh(mesh, h_tet, h_vertex);
	std::cout << "Tetrahedralization complete: " << h_tet.size() << " tetrahedra, " << h_vertex.size() << " vertices." << std::endl;
	h_force.resize(h_vertex.size(), make_float3(0.0f, 0.0f, 0.0f)); // Initialize forces to zero

    extractSurfaceTriangles(h_tet, h_vertex, suraceMesh);
    DataTransfer(h_tet, h_vertex);

    ComputeTetInitVolume();
}


void ElasticitySolver::ExportMesh(unsigned int frame)
{
    // Copy vertex data back to host and update the surface mesh vertices
    CUDA_CHECK(cudaMemcpy(this->suraceMesh.vertices.data(), d_vertex, h_vertex.size() * sizeof(float3), cudaMemcpyDeviceToHost));
    // The faces of the surface mesh remain unchanged, so we can directly save it
    std::string filename = "D:/Code/ElasticSimulator/output/frame_" + std::to_string(frame) + ".obj";
    saveOBJ(filename, suraceMesh);
}

void ElasticitySolver::PrintInfo() const
{
    std::cout << "Elasticity Solver Parameters:" << std::endl;
    std::cout << "Young's Modulus: " << h_params.youngs_modulus << std::endl;
    std::cout << "Poisson's Ratio: " << h_params.poisson_ratio << std::endl;
    std::cout << "Damping: " << h_params.damping << std::endl;
    std::cout << "Time Step: " << h_params.dt << std::endl;
    std::cout << "Density: " << h_params.density << std::endl;
    std::cout << "Substeps: " << h_params.substeps << std::endl;
    std::cout << "Lambda: " << h_params.lambda << std::endl;
    std::cout << "Mu: " << h_params.mu << std::endl;
    std::cout << "Energy Type: " << (h_params.energyType == STVK ? "STVK" : (h_params.energyType == COROTATED ? "Corotated" : "NeoHookean")) << std::endl;
    std::cout << "Gravity: (" << h_params.gravity.x << ", " << h_params.gravity.y << ", " << h_params.gravity.z << ")" << std::endl;
    std::cout << "Boundary Min: (" << h_params.boundary_min.x << ", " << h_params.boundary_min.y << ", " << h_params.boundary_min.z << ")" << std::endl;
    std::cout << "Boundary Max: (" << h_params.boundary_max.x << ", " << h_params.boundary_max.y << ", " << h_params.boundary_max.z << ")" << std::endl;
}

ElasticitySolver::~ElasticitySolver()
{
    // Free GPU memory
    cudaFree(d_tet);
    cudaFree(d_vertex);
    cudaFree(d_vertex_velocity);
    cudaFree(d_mass);
    cudaFree(d_force);
}