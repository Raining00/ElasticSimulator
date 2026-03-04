#include "Solver.h"
#include "iostream"
#include "MeshToTet.hpp"

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
    CUDA_CHECK(cudaMalloc(&d_tet, tets.size() * sizeof(Tetrahedron)));
    CUDA_CHECK(cudaMemcpy(d_tet, tets.data(), tets.size() * sizeof(Tetrahedron), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_vertex, vertices.size() * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_vertex, vertices.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_vertex_prev, vertices.size() * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_vertex_prev, vertices.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_vertex_trial, vertices.size() * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_vertex_trial, vertices.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_vertex_velocity, vertices.size() * sizeof(float3)));
    CUDA_CHECK(cudaMemset(d_vertex_velocity, 0, vertices.size() * sizeof(float3)));

    CUDA_CHECK(cudaMalloc(&d_force, vertices.size() * sizeof(float3)));
    CUDA_CHECK(cudaMemset(d_force, 0, vertices.size() * sizeof(float3)));

    CUDA_CHECK(cudaMalloc(&d_force_trial, vertices.size() * sizeof(float3)));
    CUDA_CHECK(cudaMemset(d_force_trial, 0, vertices.size() * sizeof(float3)));

    CUDA_CHECK(cudaMalloc(&d_mass, vertices.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_mass, 0, vertices.size() * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_cg_rhs, vertices.size() * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_cg_solution, vertices.size() * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_cg_residual, vertices.size() * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_cg_direction, vertices.size() * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_cg_Ap, vertices.size() * sizeof(float3)));
    CUDA_CHECK(cudaMemset(d_cg_rhs, 0, vertices.size() * sizeof(float3)));
    CUDA_CHECK(cudaMemset(d_cg_solution, 0, vertices.size() * sizeof(float3)));
    CUDA_CHECK(cudaMemset(d_cg_residual, 0, vertices.size() * sizeof(float3)));
    CUDA_CHECK(cudaMemset(d_cg_direction, 0, vertices.size() * sizeof(float3)));
    CUDA_CHECK(cudaMemset(d_cg_Ap, 0, vertices.size() * sizeof(float3)));

    CUDA_CHECK(cudaMalloc(&d_cg_scalar, sizeof(float)));

    return true;
}

void ElasticitySolver::Initialize(const Mesh& mesh)
{
    tetrahedralizeMesh(mesh, h_tet, h_vertex);
    std::cout << "Tetrahedralization complete: " << h_tet.size() << " tetrahedra, " << h_vertex.size() << " vertices." << std::endl;
    h_force.resize(h_vertex.size(), make_float3(0.0f, 0.0f, 0.0f));
    h_mass.resize(h_vertex.size(), 0.0f);

    extractSurfaceTriangles(h_tet, h_vertex, suraceMesh);
    DataTransfer(h_tet, h_vertex);
    params_ready = false;
    info_printed = false;
    implicit_fallback_warned = false;

    ComputeTetInitVolume();
}

void ElasticitySolver::ExportMesh(unsigned int frame)
{
    CUDA_CHECK(cudaMemcpy(this->suraceMesh.vertices.data(), d_vertex, h_vertex.size() * sizeof(float3), cudaMemcpyDeviceToHost));
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
    std::cout << "Solver Type: " << (h_params.solverType == EXPLICIT ? "Explicit" : "Implicit") << std::endl;
    std::cout << "Gravity: (" << h_params.gravity.x << ", " << h_params.gravity.y << ", " << h_params.gravity.z << ")" << std::endl;
    std::cout << "Boundary Min: (" << h_params.boundary_min.x << ", " << h_params.boundary_min.y << ", " << h_params.boundary_min.z << ")" << std::endl;
    std::cout << "Boundary Max: (" << h_params.boundary_max.x << ", " << h_params.boundary_max.y << ", " << h_params.boundary_max.z << ")" << std::endl;
    std::cout << "Implicit CG Max Iterations: " << h_params.implicit_cg_max_iters << std::endl;
    std::cout << "Implicit CG Tolerance: " << h_params.implicit_cg_tolerance << std::endl;
}

ElasticitySolver::~ElasticitySolver()
{
    cudaFree(d_tet);
    cudaFree(d_vertex);
    cudaFree(d_vertex_prev);
    cudaFree(d_vertex_trial);
    cudaFree(d_vertex_velocity);
    cudaFree(d_mass);
    cudaFree(d_force);
    cudaFree(d_force_trial);
    cudaFree(d_cg_rhs);
    cudaFree(d_cg_solution);
    cudaFree(d_cg_residual);
    cudaFree(d_cg_direction);
    cudaFree(d_cg_Ap);
    cudaFree(d_cg_scalar);
}

const Mesh& ElasticitySolver::GetSurfaceMesh() const
{
    return suraceMesh;
}

const float3* ElasticitySolver::GetDeviceVertices() const
{
    return d_vertex;
}

size_t ElasticitySolver::GetVertexCount() const
{
    return h_vertex.size();
}

ElasticitySolver::Parameters& ElasticitySolver::GetParameters()
{
    params_ready = false;
    return h_params;
}

const ElasticitySolver::Parameters& ElasticitySolver::GetParameters() const
{
    return h_params;
}

void ElasticitySolver::Simulate(unsigned int total_frame, bool export_results)
{
    if (!params_ready) {
        SetParams();
        params_ready = true;
    }
    if (!info_printed) {
        PrintInfo();
        info_printed = true;
    }

    std::cout << "Starting simulation: " << total_frame << " frames, "
        << h_params.substeps << " substeps/frame." << std::endl;

    for (unsigned int frame = 0; frame < total_frame; ++frame) {
        for (unsigned int sub = 0; sub < h_params.substeps; ++sub) {
            Step();
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::cout << "Frame " << frame << " done." << std::endl;

        if (export_results) {
            ExportMesh(frame);
        }
    }

    std::cout << "Simulation complete." << std::endl;
}

void ElasticitySolver::SimulateFrame(bool export_result)
{
    if (!params_ready) {
        SetParams();
        params_ready = true;
    }
    if (!info_printed) {
        PrintInfo();
        info_printed = true;
    }

    for (unsigned int sub = 0; sub < h_params.substeps; ++sub) {
        Step();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (export_result) {
        static unsigned int frame_id = 0;
        ExportMesh(frame_id++);
    }
}
