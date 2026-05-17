#include "Solver.h"
#include "iostream"
#include "MeshToTet.hpp"
#include "ProjectPaths.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <cmath>
#include <cublas_v2.h>
#include <cusparse.h>


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

template <typename Real>
void CreateCSRMat(cusparseSpMatDescr_t& matA, int* d_A_row_offsets, int* d_A_col_indices, Real* d_A_values, int rows, int cols, int nnz);

template <>
void CreateCSRMat<float>(cusparseSpMatDescr_t& matA, int* d_A_row_offsets, int* d_A_col_indices, float* d_A_values, int rows, int cols, int nnz)
{
    cusparseCreateCsr(&matA,
        rows, cols, nnz,
        d_A_row_offsets, d_A_col_indices, d_A_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
}

template <>
void CreateCSRMat<double>(cusparseSpMatDescr_t& matA, int* d_A_row_offsets, int* d_A_col_indices, double* d_A_values, int rows, int cols, int nnz)
{
    cusparseCreateCsr(&matA,
        rows, cols, nnz,
        d_A_row_offsets, d_A_col_indices, d_A_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
}

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

    const size_t dof = vertices.size() * 3;
    h_constraint_dof_flags.assign(dof, 0);
    h_constraint_dof_targets.assign(dof, Real(0));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_constraint_dof_flags), dof * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_constraint_dof_flags, 0, dof * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_constraint_dof_targets), dof * sizeof(Real)));
    CUDA_CHECK(cudaMemset(d_constraint_dof_targets, 0, dof * sizeof(Real)));
    
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
    if(h_params.platformType == CPU)
    {
        h_velocity.assign(tet_vertices_f.size(), Vec3({Real(0), Real(0), Real(0)}));
        h_mass.assign(tet_vertices_f.size(), 0);
    }
    
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
    csr_ready = false;
    params_ready = false;
    info_printed = false;
    frame_counter = 0;
}

template <typename Real>
bool ElasticitySolverT<Real>::Initialize(const std::string& filename)
{
    // 0. .node file
    std::fstream nodeFS(filename + ".node");
    // Node count, 3 dim, no attribute, no boundary marker
    unsigned int nNodes, nDims, nNodeAttribs, nMarkers;
    nodeFS >> nNodes >> nDims >> nNodeAttribs >> nMarkers;
    bool nodeStartWithZero = false;

    std::vector<Vec3f> tet_vertices_f(nNodes);

    h_vertex.resize(nNodes);
    if(h_params.platformType == CPU)
    {
        h_velocity.assign(nNodes, Vec3({Real(0), Real(0), Real(0)}));
        h_mass.assign(nNodes, 0);
    }
    
    for (unsigned int i = 0; i < nNodes; ++i)
    {
        unsigned int _;
        nodeFS >> _ >> tet_vertices_f[i].x >> tet_vertices_f[i].y >> tet_vertices_f[i].z;
        if (i == 0 && _ == 0) nodeStartWithZero = true;
        for (unsigned int j = 0; j < nNodeAttribs + nMarkers; ++j)
            nodeFS >> _;
    }

    for (size_t i = 0; i < tet_vertices_f.size(); ++i) {
        h_vertex[i] = Vec3{
            static_cast<Real>(tet_vertices_f[i].x),
            static_cast<Real>(tet_vertices_f[i].y),
            static_cast<Real>(tet_vertices_f[i].z)
        };
    }

    // 1. .ele file
    std::fstream eleFS(filename + ".ele");
    // <# of tetrahedra> <nodes per tetrahedron> <# of attributes>
    unsigned int nTets, nNodesPerTet, nEleAttribs;
    eleFS >> nTets >> nNodesPerTet >> nEleAttribs;
    h_tet.resize(nTets);
    for (unsigned int i = 0; i < nTets; ++i)
    {
        Tetrahedron<Real>& tet = h_tet[i];
        unsigned int _;
        eleFS >> _ >> tet.verticesIndex.x >> tet.verticesIndex.y >> tet.verticesIndex.z >> tet.verticesIndex.w;
        if (nodeStartWithZero == false)
        {
            tet.verticesIndex.x -= 1;tet.verticesIndex.y -= 1; tet.verticesIndex.z -= 1; tet.verticesIndex.w -= 1;
        }
        for (unsigned int j = 0; j < nEleAttribs; ++j)
            eleFS >> _;
    }

    std::cout << "Tetrahedralization complete: " << h_tet.size() << " tetrahedra, " << h_vertex.size() << " vertices." << std::endl;
    extractSurfaceTriangles(h_tet, tet_vertices_f, suraceMesh);
    DataTransfer(h_tet, h_vertex);
    csr_ready = false;
    params_ready = false;
    info_printed = false;
    frame_counter = 0;

    return true;
}

template <typename Real>
void ElasticitySolverT<Real>::ExportMesh(unsigned int frame)
{
    std::vector<Vec3> host_vertices(h_vertex.size());
    if(h_params.platformType == GPU)
    {
        CUDA_CHECK(cudaMemcpy(host_vertices.data(), d_vertex, h_vertex.size() * sizeof(Vec3), cudaMemcpyDeviceToHost));
    }
    else
        host_vertices = h_vertex;
    // Copy vertex data back to host and update the surface mesh vertices.
    this->suraceMesh.vertices.resize(host_vertices.size());
    for (size_t i = 0; i < host_vertices.size(); ++i) {
        suraceMesh.vertices[i] = Vec3 {
            static_cast<float>(host_vertices[i].x),
            static_cast<float>(host_vertices[i].y),
            static_cast<float>(host_vertices[i].z)
        };
    }
    // The faces of the surface mesh remain unchanged, so we can directly save it
    std::string filename = PROJECT_SOURCE_DIR "/output/frame_" + std::to_string(frame) + ".obj";
    std::cout << "save mesh as: " << filename << std::endl;
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
    const char* energyName = "NeoHookean";
    if (h_params.energyType == STVK) energyName = "STVK";
    else if (h_params.energyType == COROTATED) energyName = "Corotated";
    else if (h_params.energyType == ARAP) energyName = "ARAP";
    std::cout << "Energy Type: " << energyName << std::endl;
    std::cout << "Gravity: (" << h_params.gravity.x << ", " << h_params.gravity.y << ", " << h_params.gravity.z << ")" << std::endl;
    std::cout << "Boundary Min: (" << h_params.boundary_min.x << ", " << h_params.boundary_min.y << ", " << h_params.boundary_min.z << ")" << std::endl;
    std::cout << "Boundary Max: (" << h_params.boundary_max.x << ", " << h_params.boundary_max.y << ", " << h_params.boundary_max.z << ")" << std::endl;
}

template <typename Real>
static typename ElasticitySolverT<Real>::Vec3 LocalToWorld(
    const typename ElasticitySolverT<Real>::KinematicCylinder& cylinder,
    const typename ElasticitySolverT<Real>::Vec3& local)
{
    return cylinder.rotation * local + cylinder.translation;
}

template <typename Real>
static typename ElasticitySolverT<Real>::Vec3 WorldToLocal(
    const typename ElasticitySolverT<Real>::KinematicCylinder& cylinder,
    const typename ElasticitySolverT<Real>::Vec3& world)
{
    return mat3<Real>::transpose(cylinder.rotation) * (world - cylinder.translation);
}

template <typename Real>
static bool CylinderContains(
    const typename ElasticitySolverT<Real>::KinematicCylinder& cylinder,
    const typename ElasticitySolverT<Real>::Vec3& world)
{
    const auto local = WorldToLocal<Real>(cylinder, world);
    if (local[1] > cylinder.height * Real(0.5) || local[1] < -cylinder.height * Real(0.5)) {
        return false;
    }
    return local[0] * local[0] + local[2] * local[2] <= cylinder.radius * cylinder.radius;
}

template <typename Real>
int ElasticitySolverT<Real>::AddKinematicCylinder(const Vec3& center, Real radius, Real height)
{
    KinematicCylinder cylinder;
    cylinder.translation = center;
    cylinder.rotation = mat3<Real>(Real(1));
    cylinder.radius = radius;
    cylinder.height = height;
    h_kinematicCylinders.push_back(cylinder);
    return static_cast<int>(h_kinematicCylinders.size() - 1);
}

template <typename Real>
void ElasticitySolverT<Real>::AttachKinematicConstraints(int shapeID)
{
    if (shapeID < 0 || shapeID >= static_cast<int>(h_kinematicCylinders.size())) {
        throw std::out_of_range("Invalid kinematic cylinder id");
    }

    const auto& cylinder = h_kinematicCylinders[static_cast<size_t>(shapeID)];
    int added = 0;
    for (int i = 0; i < static_cast<int>(h_vertex.size()); ++i) {
        if (!CylinderContains<Real>(cylinder, h_vertex[static_cast<size_t>(i)])) {
            continue;
        }

        bool alreadyConstrained = false;
        for (const auto& constraint : h_kinematicConstraints) {
            if (constraint.vertexID == i) {
                alreadyConstrained = true;
                break;
            }
        }
        if (alreadyConstrained) {
            continue;
        }

        KinematicConstraint constraint;
        constraint.vertexID = i;
        constraint.shapeID = shapeID;
        constraint.localPosition = WorldToLocal<Real>(cylinder, h_vertex[static_cast<size_t>(i)]);
        h_kinematicConstraints.push_back(constraint);
        ++added;
    }

    std::cout << "Attached " << added << " kinematic vertices to cylinder " << shapeID << std::endl;
    UpdateKinematicConstraints();
}

template <typename Real>
void ElasticitySolverT<Real>::RotateKinematicCylinderXKeepingLocalPoint(
    int shapeID,
    Real radians,
    const Vec3& localPoint,
    const Vec3& worldPin)
{
    if (shapeID < 0 || shapeID >= static_cast<int>(h_kinematicCylinders.size())) {
        throw std::out_of_range("Invalid kinematic cylinder id");
    }

    const Real c = std::cos(radians);
    const Real s = std::sin(radians);
    const mat3<Real> rotX(
        Real(1), Real(0), Real(0),
        Real(0), c, -s,
        Real(0), s, c);

    auto& cylinder = h_kinematicCylinders[static_cast<size_t>(shapeID)];
    cylinder.rotation = rotX * cylinder.rotation;
    const Vec3 world = LocalToWorld<Real>(cylinder, localPoint);
    cylinder.translation -= world - worldPin;
}

template <typename Real>
void ElasticitySolverT<Real>::UpdateKinematicConstraints()
{
    if (h_constraint_dof_flags.empty() || h_constraint_dof_targets.empty()) {
        return;
    }

    std::fill(h_constraint_dof_flags.begin(), h_constraint_dof_flags.end(), 0);
    std::fill(h_constraint_dof_targets.begin(), h_constraint_dof_targets.end(), Real(0));

    for (const auto& constraint : h_kinematicConstraints) {
        if (constraint.shapeID < 0 || constraint.shapeID >= static_cast<int>(h_kinematicCylinders.size())) {
            continue;
        }

        const auto& cylinder = h_kinematicCylinders[static_cast<size_t>(constraint.shapeID)];
        const Vec3 target = LocalToWorld<Real>(cylinder, constraint.localPosition);
        const int base = 3 * constraint.vertexID;
        for (int c = 0; c < 3; ++c) {
            h_constraint_dof_flags[static_cast<size_t>(base + c)] = 1;
            h_constraint_dof_targets[static_cast<size_t>(base + c)] = target[c];
        }
    }

    if (d_constraint_dof_flags && d_constraint_dof_targets) {
        CUDA_CHECK(cudaMemcpy(
            d_constraint_dof_flags,
            h_constraint_dof_flags.data(),
            h_constraint_dof_flags.size() * sizeof(int),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_constraint_dof_targets,
            h_constraint_dof_targets.data(),
            h_constraint_dof_targets.size() * sizeof(Real),
            cudaMemcpyHostToDevice));
    }
}

template <typename Real>
ElasticitySolverT<Real>::~ElasticitySolverT()
{
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_tet));
    CUDA_CHECK(cudaFree(d_vertex));
    CUDA_CHECK(cudaFree(d_vertex_velocity));
    CUDA_CHECK(cudaFree(d_mass));
    CUDA_CHECK(cudaFree(d_force));
    CUDA_CHECK(cudaFree(d_F));
    if (d_constraint_dof_flags) CUDA_CHECK(cudaFree(d_constraint_dof_flags));
    if (d_constraint_dof_targets) CUDA_CHECK(cudaFree(d_constraint_dof_targets));

    const bool isImplicitGPU =
        (h_params.platformType == GPU) &&
        (h_params.solverType == IMPLICIT || h_params.solverType == IMPLICIT_SPARSE);
    if (!isImplicitGPU)
        return;
    // CSR matrix
    if (d_A_row_offsets)  CUDA_CHECK(cudaFree(d_A_row_offsets));
    if (d_A_col_indices)  CUDA_CHECK(cudaFree(d_A_col_indices));
    if (d_A_diag_indices) CUDA_CHECK(cudaFree(d_A_diag_indices));
    if (d_A_values)       CUDA_CHECK(cudaFree(d_A_values));
    if (d_elem_to_A_csr)  CUDA_CHECK(cudaFree(d_elem_to_A_csr));

    // Linear-system buffers shared by dense CG and sparse PCG
    if (delta_x)       CUDA_CHECK(cudaFree(delta_x));
    if (d_b)           CUDA_CHECK(cudaFree(d_b));
    if (d_r)           CUDA_CHECK(cudaFree(d_r));
    if (d_p)           CUDA_CHECK(cudaFree(d_p));
    if (d_q)           CUDA_CHECK(cudaFree(d_q));
    if (d_z)           CUDA_CHECK(cudaFree(d_z));
    if (d_M_inv)       CUDA_CHECK(cudaFree(d_M_inv));
    if (d_x_tilde)     CUDA_CHECK(cudaFree(d_x_tilde));
    if (d_x0)          CUDA_CHECK(cudaFree(d_x0));
    if (d_energy)      CUDA_CHECK(cudaFree(d_energy));
    if (d_spmv_buffer) CUDA_CHECK(cudaFree(d_spmv_buffer));
    if (DnA)           CUDA_CHECK(cudaFree(DnA));
    // cublas handle
    if (vecP) cusparseDestroyDnVec(vecP);
    if (vecQ) cusparseDestroyDnVec(vecQ);
    if (A) cusparseDestroySpMat(A);
    if (cublasH) cublasDestroy(cublasH);
    if (cusparseH) cusparseDestroy(cusparseH);
}

template <typename Real>
void ElasticitySolverT<Real>::AdvanceFrame(bool export_result)
{
    if (h_params.platformType == CPU) {
        SimulateCPU(export_result, static_cast<int>(frame_counter));
        ++frame_counter;
        return;
    }

    SimulateFrame(export_result);
    ++frame_counter;
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


//  Main simulation loop
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
    PreCompute();
    if ((h_params.solverType == IMPLICIT || h_params.solverType == IMPLICIT_SPARSE) && !csr_ready) {
        std::cout << "Precomputing global CSR matrix topology..." << std::endl;
        BuildGlobalCsrFromTetMesh();
        UploadGlobalCsrToDevice();
        InitCUDALib();
        csr_ready = true;
    }

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
        PreCompute();
        if (h_params.solverType == IMPLICIT_SPARSE && !csr_ready) {
            BuildGlobalCsrFromTetMesh();
            UploadGlobalCsrToDevice();
            csr_ready = true;
        }
        if (h_params.solverType == IMPLICIT || h_params.solverType == IMPLICIT_SPARSE)
            InitCUDALib();
        params_ready = true;
    }
    if (!info_printed) {
        PrintInfo();
        info_printed = true;
    }

    UpdateKinematicConstraints();

    for (unsigned int sub = 0; sub < h_params.substeps; ++sub) {
        Step();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (export_result) {
        static unsigned int frame_id = 0;
        ExportMesh(frame_id++);
    }
}

template <typename Real>
void ElasticitySolverT<Real>::BuildGlobalCsrFromTetMesh()
{
    const int numVerts = static_cast<int>(h_vertex.size());
    const int numTets = static_cast<int>(h_tet.size());
    const int dofCount = numVerts * 3;

    h_A_row_offsets.assign(dofCount + 1, 0);
    h_A_col_indices.clear();
    h_A_diag_indices.assign(dofCount, -1);
    h_A_values.clear();
    h_elem_to_A_csr.assign(static_cast<size_t>(numTets) * 12 * 12, -1);

    std::vector<std::unordered_set<int>> rowConnectivity(static_cast<size_t>(dofCount));
    for (int row = 0; row < dofCount; ++row) {
        rowConnectivity[static_cast<size_t>(row)].insert(row);
    }

    for (const auto& tet : h_tet) {
        const int v[4] = {
            tet.verticesIndex[0], tet.verticesIndex[1],
            tet.verticesIndex[2], tet.verticesIndex[3]
        };
        for (int a = 0; a < 4; ++a) {
            for (int da = 0; da < 3; ++da) {
                const int row = v[a] * 3 + da;
                auto& cols = rowConnectivity[static_cast<size_t>(row)];
                for (int b = 0; b < 4; ++b) {
                    for (int db = 0; db < 3; ++db) {
                        cols.insert(v[b] * 3 + db);
                    }
                }
            }
        }
    }

    std::vector<std::unordered_map<int, int>> rowLookup(static_cast<size_t>(dofCount));
    int nnz = 0;
    for (int row = 0; row < dofCount; ++row) {
        h_A_row_offsets[static_cast<size_t>(row)] = nnz;
        auto cols = std::vector<int>(
            rowConnectivity[static_cast<size_t>(row)].begin(),
            rowConnectivity[static_cast<size_t>(row)].end());
        std::sort(cols.begin(), cols.end());
        for (const int col : cols) {
            h_A_col_indices.push_back(col);
            if (col == row) {
                h_A_diag_indices[static_cast<size_t>(row)] = nnz;
            }
            rowLookup[static_cast<size_t>(row)].insert({ col, nnz });
            ++nnz;
        }
    }
    h_A_row_offsets[static_cast<size_t>(dofCount)] = nnz;
    for (int row = 0; row < dofCount; ++row) {
        if (h_A_diag_indices[static_cast<size_t>(row)] < 0) {
            std::cerr << "CSR diagonal missing at row " << row << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    h_A_values.assign(static_cast<size_t>(nnz), static_cast<Real>(0));

    for (int e = 0; e < numTets; ++e) {
        const auto& tet = h_tet[static_cast<size_t>(e)];
        const int v[4] = {
            tet.verticesIndex[0], tet.verticesIndex[1],
            tet.verticesIndex[2], tet.verticesIndex[3]
        };
        for (int lr = 0; lr < 12; ++lr) {
            const int ra = lr / 3;
            const int rd = lr % 3;
            const int globalRow = v[ra] * 3 + rd;
            const auto& lookup = rowLookup[static_cast<size_t>(globalRow)];
            for (int lc = 0; lc < 12; ++lc) {
                const int ca = lc / 3;
                const int cd = lc % 3;
                const int globalCol = v[ca] * 3 + cd;
                const auto it = lookup.find(globalCol);
                if (it == lookup.end()) {
                    std::cerr << "CSR mapping error at element " << e
                              << ", local (" << lr << "," << lc << ")." << std::endl;
                    exit(EXIT_FAILURE);
                }
                const size_t mapIdx = static_cast<size_t>(e) * 12 * 12 + static_cast<size_t>(lr) * 12 + static_cast<size_t>(lc);
                h_elem_to_A_csr[mapIdx] = it->second;
            }
        }
    }

    std::cout << "CSR ready: dof=" << dofCount
              << ", nnz=" << nnz
              << ", diag-entries=" << h_A_diag_indices.size()
              << ", element-map entries=" << h_elem_to_A_csr.size()
              << std::endl;
}

template <typename Real>
void ElasticitySolverT<Real>::UploadGlobalCsrToDevice()
{
    cudaFree(d_A_row_offsets);
    cudaFree(d_A_col_indices);
    cudaFree(d_A_diag_indices);
    cudaFree(d_A_values);
    cudaFree(d_elem_to_A_csr);
    d_A_row_offsets = nullptr;
    d_A_col_indices = nullptr;
    d_A_diag_indices = nullptr;
    d_A_values = nullptr;
    d_elem_to_A_csr = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A_row_offsets), h_A_row_offsets.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_A_row_offsets, h_A_row_offsets.data(),
        h_A_row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A_col_indices), h_A_col_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_A_col_indices, h_A_col_indices.data(),
        h_A_col_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A_diag_indices), h_A_diag_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_A_diag_indices, h_A_diag_indices.data(),
        h_A_diag_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A_values), h_A_values.size() * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(d_A_values, h_A_values.data(),
        h_A_values.size() * sizeof(Real), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_elem_to_A_csr), h_elem_to_A_csr.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_elem_to_A_csr, h_elem_to_A_csr.data(),
        h_elem_to_A_csr.size() * sizeof(int), cudaMemcpyHostToDevice));
}

template <typename Real>
void ElasticitySolverT<Real>::InitCUDALib()
{
    if (cublasH != nullptr || cusparseH != nullptr) {
        return;
    }

    CUBLAS_CHECK(cublasCreate(&cublasH));
    cusparseCreate(&cusparseH);

    const size_t dof = h_vertex.size() * 3;

    // Buffers shared by both dense CG and sparse PCG paths.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&delta_x), dof * sizeof(Real)));
    CUDA_CHECK(cudaMemset(delta_x, 0, dof * sizeof(Real)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), dof * sizeof(Real)));
    CUDA_CHECK(cudaMemset(d_b, 0, dof * sizeof(Real)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_r), dof * sizeof(Real)));
    CUDA_CHECK(cudaMemset(d_r, 0, dof * sizeof(Real)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_p), dof * sizeof(Real)));
    CUDA_CHECK(cudaMemset(d_p, 0, dof * sizeof(Real)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_q), dof * sizeof(Real)));
    CUDA_CHECK(cudaMemset(d_q, 0, dof * sizeof(Real)));

    if (h_params.solverType == IMPLICIT) {
        // Dense pipeline: explicit dense matrix backing CG.
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&DnA), dof * dof * sizeof(Real)));
        CUDA_CHECK(cudaMemset(DnA, 0, dof * dof * sizeof(Real)));
    } else if (h_params.solverType == IMPLICIT_SPARSE) {
        // Sparse pipeline: cuSPARSE CSR matrix + Jacobi-preconditioned CG.
        CreateCSRMat<Real>(A, d_A_row_offsets, d_A_col_indices, d_A_values,
            static_cast<int>(dof), static_cast<int>(dof), static_cast<int>(h_A_values.size()));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_z), dof * sizeof(Real)));
        CUDA_CHECK(cudaMemset(d_z, 0, dof * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_M_inv), dof * sizeof(Real)));
        CUDA_CHECK(cudaMemset(d_M_inv, 0, dof * sizeof(Real)));

        // Line search buffers
        const size_t numVerts = h_vertex.size();
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_x_tilde), numVerts * sizeof(Vec3)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_x0), numVerts * sizeof(Vec3)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_energy), sizeof(Real)));

        cusparseCreateDnVec(&vecP, dof, d_p, std::is_same<Real, float>::value ? CUDA_R_32F : CUDA_R_64F);
        cusparseCreateDnVec(&vecQ, dof, d_q, std::is_same<Real, float>::value ? CUDA_R_32F : CUDA_R_64F);

        const Real one = static_cast<Real>(1);
        const Real zero = static_cast<Real>(0);
        cusparseSpMV_bufferSize(
            cusparseH,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one,
            A,
            vecP,
            &zero,
            vecQ,
            std::is_same<Real, float>::value ? CUDA_R_32F : CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            &spmv_buffer_size);
        CUDA_CHECK(cudaMalloc(&d_spmv_buffer, spmv_buffer_size));
    }

    cg_max_iters = static_cast<int>(dof);
}

template bool ElasticitySolverT<float>::DataTransfer(const std::vector<Tetrahedron<float>>&, const std::vector<Vec3f>&);
template bool ElasticitySolverT<double>::DataTransfer(const std::vector<Tetrahedron<double>>&, const std::vector<Vec3d>&);

template void ElasticitySolverT<float>::Initialize(const Mesh<float>&);
template void ElasticitySolverT<double>::Initialize(const Mesh <double> &);

template bool ElasticitySolverT<float>::Initialize(const std::string& name);
template bool ElasticitySolverT<double>::Initialize(const std::string& name);

template void ElasticitySolverT<float>::Simulate(unsigned int, bool);
template void ElasticitySolverT<double>::Simulate(unsigned int, bool);

template void ElasticitySolverT<float>::AdvanceFrame(bool);
template void ElasticitySolverT<double>::AdvanceFrame(bool);

template void ElasticitySolverT<float>::SimulateFrame(bool);
template void ElasticitySolverT<double>::SimulateFrame(bool);

template void ElasticitySolverT<float>::BuildGlobalCsrFromTetMesh();
template void ElasticitySolverT<double>::BuildGlobalCsrFromTetMesh();

template void ElasticitySolverT<float>::UploadGlobalCsrToDevice();
template void ElasticitySolverT<double>::UploadGlobalCsrToDevice();

template void ElasticitySolverT<float>::ExportMesh(unsigned int);
template void ElasticitySolverT<double>::ExportMesh(unsigned int);

template const Mesh<float>& ElasticitySolverT<float>::GetSurfaceMesh() const;
template const Mesh<double>& ElasticitySolverT<double>::GetSurfaceMesh() const;

template const ElasticitySolverT<float>::Vec3* ElasticitySolverT<float>::GetDeviceVertices() const;
template const ElasticitySolverT<double>::Vec3* ElasticitySolverT<double>::GetDeviceVertices() const;

template int ElasticitySolverT<float>::AddKinematicCylinder(const Vec3&, float, float);
template int ElasticitySolverT<double>::AddKinematicCylinder(const Vec3&, double, double);

template void ElasticitySolverT<float>::AttachKinematicConstraints(int);
template void ElasticitySolverT<double>::AttachKinematicConstraints(int);

template void ElasticitySolverT<float>::RotateKinematicCylinderXKeepingLocalPoint(int, float, const Vec3&, const Vec3&);
template void ElasticitySolverT<double>::RotateKinematicCylinderXKeepingLocalPoint(int, double, const Vec3&, const Vec3&);

template void ElasticitySolverT<float>::UpdateKinematicConstraints();
template void ElasticitySolverT<double>::UpdateKinematicConstraints();

template void ElasticitySolverT<float>::InitCUDALib();
template void ElasticitySolverT<double>::InitCUDALib();

template size_t ElasticitySolverT<float>::GetVertexCount() const;
template size_t ElasticitySolverT<double>::GetVertexCount() const;

template ElasticitySolverT<float>::~ElasticitySolverT();
template ElasticitySolverT<double>::~ElasticitySolverT();
