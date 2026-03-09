#include "Solver.h"
#include "iostream"
#include "MeshToTet.hpp"
#include "ProjectPaths.h"
#include <cuda_runtime_api.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

template <typename Real>
bool ElasticitySolverT<Real>::DataTransfer(const std::vector<Tetrahedron<Real>>& tets, const std::vector<Vec3>& vertices)
{
    // Allocate and copy data to GPU
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tet), tets.size() * sizeof(Tetrahedron<Real>)));
    CUDA_CHECK(cudaMemcpy(d_tet, tets.data(), tets.size() * sizeof(Tetrahedron<Real>), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_F), tets.size() * sizeof(mat3<Real>)));
    //CUDA_CHECK(cudaMemset(&d_F, 0, tets.size() * sizeof(mat3<Real>));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertex), vertices.size() * sizeof(Vec3)));
    CUDA_CHECK(cudaMemcpy(d_vertex, vertices.data(), vertices.size() * sizeof(Vec3), cudaMemcpyHostToDevice));

	// velocity initialization
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertex_velocity), vertices.size() * sizeof(Vec3)));
	// Initialize velocities to zero
	CUDA_CHECK(cudaMemset(d_vertex_velocity, 0, vertices.size() * sizeof(Vec3)));

    // force initialization
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_force), vertices.size() * sizeof(Vec3)));
    // Initialize forces to zero
	CUDA_CHECK(cudaMemset(d_force, 0, vertices.size() * sizeof(Vec3)));

    // mass initialization
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mass), vertices.size() * sizeof(Real)));
    // Initialize masses to zero (this would typically be computed based on density and volume)
    CUDA_CHECK(cudaMemset(d_mass, 0, vertices.size() * sizeof(Real)));

    
    return true;
}

template <typename Real>
void ElasticitySolverT<Real>::Initialize(const Mesh<Real>& mesh)
{
    std::vector<Vec3f> tet_vertices_f;
    // This function would convert the Mesh data into the format needed for the solver
    // For simplicity, we assume the Mesh class has methods
    tetrahedralizeMesh(mesh, h_tet, tet_vertices_f);
    h_vertex.resize(tet_vertices_f.size());
    for (size_t i = 0; i < tet_vertices_f.size(); ++i) {
        h_vertex[i] = Vec3{
            static_cast<Real>(tet_vertices_f[i].x),
            static_cast<Real>(tet_vertices_f[i].y),
            static_cast<Real>(tet_vertices_f[i].z)
        };
    }
	std::cout << "Tetrahedralization complete: " << h_tet.size() << " tetrahedra, " << h_vertex.size() << " vertices." << std::endl;

    extractSurfaceTriangles(h_tet, tet_vertices_f, suraceMesh);
    DataTransfer(h_tet, h_vertex);
    params_ready = false;
    info_printed = false;
}


template <typename Real>
void ElasticitySolverT<Real>::ExportMesh(unsigned int frame)
{
    // Copy vertex data back to host and update the surface mesh vertices.
    std::vector<Vec3> host_vertices(h_vertex.size());
    CUDA_CHECK(cudaMemcpy(host_vertices.data(), d_vertex, h_vertex.size() * sizeof(Vec3), cudaMemcpyDeviceToHost));
    this->suraceMesh.vertices.resize(host_vertices.size());
    for (size_t i = 0; i < host_vertices.size(); ++i) {
        suraceMesh.vertices[i] = Vec3 {
            static_cast<float>(host_vertices[i].x),
            static_cast<float>(host_vertices[i].y),
            static_cast<float>(host_vertices[i].z)
        };
    }
    // The faces of the surface mesh remain unchanged, so we can directly save it
    std::string filename = PROJECT_SOURCE_DIR "output/frame_" + std::to_string(frame) + ".obj";
    saveOBJ(filename, suraceMesh);
}

template <typename Real>
void ElasticitySolverT<Real>::PrintInfo() const
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

template <typename Real>
ElasticitySolverT<Real>::~ElasticitySolverT()
{
    // Free GPU memory
    cudaFree(d_tet);
    cudaFree(d_vertex);
    cudaFree(d_vertex_velocity);
    cudaFree(d_mass);
    cudaFree(d_force);
    cudaFree(d_F);
}

template <typename Real>
const Mesh<Real>& ElasticitySolverT<Real>::GetSurfaceMesh() const
{
    return suraceMesh;
}

template <typename Real>
const typename ElasticitySolverT<Real>::Vec3* ElasticitySolverT<Real>::GetDeviceVertices() const
{
    return d_vertex;
}

template <typename Real>
size_t ElasticitySolverT<Real>::GetVertexCount() const
{
    return h_vertex.size();
}


// ������ Main simulation loop ����������������������������������������������������������������������������������������������������

template <typename Real>
void ElasticitySolverT<Real>::Simulate(unsigned int total_frame, bool export_results)
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

    std::cout << "Precomputing volume and mass...." << std::endl;
    ComputeTetInitVolume();

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

template <typename Real>
void ElasticitySolverT<Real>::SimulateFrame(bool export_result)
{
    if (!params_ready) {
        SetParams();
        // precompute volumes and mass.
        ComputeTetInitVolume();
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

template bool ElasticitySolverT<float>::DataTransfer(const std::vector<Tetrahedron<float>>&, const std::vector<Vec3f>&);
template bool ElasticitySolverT<double>::DataTransfer(const std::vector<Tetrahedron<double>>&, const std::vector<Vec3d>&);

template void ElasticitySolverT<float>::Initialize(const Mesh<float>&);
template void ElasticitySolverT<double>::Initialize(const Mesh <double> &);

template void ElasticitySolverT<float>::Simulate(unsigned int, bool);
template void ElasticitySolverT<double>::Simulate(unsigned int, bool);

template void ElasticitySolverT<float>::SimulateFrame(bool);
template void ElasticitySolverT<double>::SimulateFrame(bool);

template void ElasticitySolverT<float>::ExportMesh(unsigned int);
template void ElasticitySolverT<double>::ExportMesh(unsigned int);

template const Mesh<float>& ElasticitySolverT<float>::GetSurfaceMesh() const;
template const Mesh<double>& ElasticitySolverT<double>::GetSurfaceMesh() const;

template const ElasticitySolverT<float>::Vec3* ElasticitySolverT<float>::GetDeviceVertices() const;
template const ElasticitySolverT<double>::Vec3* ElasticitySolverT<double>::GetDeviceVertices() const;

template size_t ElasticitySolverT<float>::GetVertexCount() const;
template size_t ElasticitySolverT<double>::GetVertexCount() const;

template ElasticitySolverT<float>::~ElasticitySolverT();
template ElasticitySolverT<double>::~ElasticitySolverT();


