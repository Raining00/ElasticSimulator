#pragma once
#include "BaseStructure.hpp"
#include "PhysicsWorld.h"
#include <cstddef>
#include <type_traits>
#include <string>

enum EnergyType
{
    STVK = 0,
    COROTATED = 1,
    NEOHOOKEAN = 2
};

enum SolverType
{
    EXPLICIT = 0,
    IMPLICIT = 1,
};

enum Platform
{
    CPU = 0,
    GPU = 1,
};

//Froward declaration
typedef struct cublasContext* cublasHandle_t;
typedef struct cusparseContext* cusparseHandle_t;
typedef struct cusparseSpMatDescr* cusparseSpMatDescr_t;
typedef struct cusparseDnVecDescr* cusparseDnVecDescr_t;

template <typename Real>
class ElasticitySolverT : public PhysicsObjectT<Real>
{
public:
    using Scalar = Real;
    using Vec3 = Vector<Real, 3>;
    using AABB = PhysicsAABBT<Real>;
    using CollisionSettings = WorldCollisionSettingsT<Real>;

    struct Parameters
    {
        Real youngs_modulus = static_cast<Real>(1e6); // Young's modulus
        Real poisson_ratio = static_cast<Real>(0.45); // Poisson's ratio
        Real damping = static_cast<Real>(0.0); // Damping factor
		Real dt = static_cast<Real>(1e-3); // Time step
        Real density = static_cast<Real>(1000.0); // Material density
		unsigned int substeps = 10; // Number of substeps for the simulation
        // lame parameters
        Real lambda = static_cast<Real>(0.0); // First Lame parameter
        Real mu = static_cast<Real>(0.0); // Second Lame parameter (shear modulus)
        Real alpha = static_cast<Real>(0.0);
        EnergyType energyType = NEOHOOKEAN; // Type of elastic energy to use
        SolverType solverType = EXPLICIT; // Type of solver to use
        Platform   platformType = GPU;
        Real barrier_distance = static_cast<Real>(0.02);     // barrier active distance
        Real barrier_stiffness = static_cast<Real>(5e3);     // barrier strenth
		Vec3 gravity = { static_cast<Real>(0.0), static_cast<Real>(-9.81), static_cast<Real>(0.0) }; // Gravity vector
		Vec3 boundary_min = { static_cast<Real>(-10.0), static_cast<Real>(0.0), static_cast<Real>(-10.0) }; // Minimum boundary for collision
		Vec3 boundary_max = { static_cast<Real>(10.0), static_cast<Real>(10.0), static_cast<Real>(10.0) }; // Maximum boundary for collision
    };

    ElasticitySolverT() = default;
    ~ElasticitySolverT() override;

    void Initialize(const Mesh<Real>& mesh);

    bool Initialize(const std::string& filename);

    void AdvanceFrame(bool export_result = false) override;
    void Simulate(unsigned int total_frame = 30, bool export_results = true);
    void SimulateFrame(bool export_result = false);

    void SetInitialOffset(const Vec3& offset);
    void RotateVerticesByEulerAngles(const Vec3& euler_angles);
    void RotateVerticesAroundCentroidByEulerAngles(const Vec3& euler_angles);
    void ExportMesh(unsigned int frame);
    const Mesh<Real>& GetSurfaceMesh() const override;
    const Vec3* GetDeviceVertices() const override;
    size_t GetVertexCount() const override;
    PhysicsObjectType GetObjectType() const override;
    AABB GetWorldBounds() const override;
    void SetWorldCollisionSettings(const CollisionSettings& settings) override;
	Parameters& GetParameters() { return h_params; }
    const Parameters& GetParameters() const { return h_params; }

    void SimulateCPU(bool export_results = false, int frame = 0);

protected:
    bool DataTransfer(const std::vector<Tetrahedron<Real>>& tets, const std::vector<Vec3>& vertices);

    void ComputeTetInitVolume();
    void BuildGlobalCsrFromTetMesh();
    void UploadGlobalCsrToDevice();
    void InitCUDALib();

    void SetParams();
    void Step();
    void Step_Explicit();
    void Step_Implicit();

    void StepCPUExplicit();
    void StepCPUImplicit();

    void PrintInfo() const;

private:
    // host data
    std::vector<Tetrahedron<Real>> h_tet;
    std::vector<Vec3> h_vertex;
    std::vector<Vec3> h_velocity;
    std::vector<Real> h_mass;
	Parameters h_params;
    Mesh<Real> suraceMesh;

    //device data
    Tetrahedron<Real>* d_tet;
    Vec3* d_vertex;
	Vec3* d_vertex_velocity;
    Vec3* d_force;
    Real* d_mass;
    mat3<Real>* d_F;
    int* d_A_row_offsets = nullptr;
    int* d_A_col_indices = nullptr;
    int* d_A_diag_indices = nullptr;
    Real* d_A_values = nullptr;
    int* d_elem_to_A_csr = nullptr;

    // for implicit solver. A x = B
    Real* delta_x = nullptr;
    Real* d_b = nullptr;
    Real* d_r = nullptr;
    Real* d_p = nullptr;
    Real* d_q = nullptr;
    Real* DnA;
    cusparseSpMatDescr_t A = nullptr;
    cusparseDnVecDescr_t vecP = nullptr;
    cusparseDnVecDescr_t vecQ = nullptr;
    cublasHandle_t cublasH = nullptr;
    cusparseHandle_t cusparseH = nullptr;
    void* d_spmv_buffer = nullptr;
    size_t spmv_buffer_size = 0;
    int cg_max_iters = 0;
    Real cg_tolerance = static_cast<Real>(1e-6);

    std::vector<int> h_A_row_offsets;
    std::vector<int> h_A_col_indices;
    std::vector<int> h_A_diag_indices;
    std::vector<Real> h_A_values;
    std::vector<int> h_elem_to_A_csr;

    bool csr_ready = false;
    bool params_ready = false;
    bool info_printed = false;
    unsigned int frame_counter = 0;
};

using ElasticitySolverf = ElasticitySolverT<float>;
using ElasticitySolverd = ElasticitySolverT<double>;
using ElasticitySolver = ElasticitySolverf;
