#pragma once
#include "BaseStructure.hpp"
#include "SkeletonAnimation.hpp"
#include <array>
#include <cstddef>
#include <type_traits>
#include <string>
#include <vector>

enum EnergyType
{
    STVK = 0,
    COROTATED = 1,
    NEOHOOKEAN = 2,
    ARAP = 3
};

enum SolverType
{
    EXPLICIT = 0,
    IMPLICIT = 1,
    IMPLICIT_SPARSE = 2,
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
class ElasticitySolverT
{
public:
    using Scalar = Real;
    using Vec3 = Vector<Real, 3>;

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
        Real muscle_coupling_stiffness = static_cast<Real>(0.0);
        Real muscle_coupling_damping = static_cast<Real>(0.0);
    };

    ElasticitySolverT() = default;
    ~ElasticitySolverT();

    void Initialize(const Mesh<Real>& mesh);

    bool Initialize(const std::string& filename);

    void AdvanceFrame(bool export_result = false);
    void Simulate(unsigned int total_frame = 30, bool export_results = true);
    void SimulateFrame(bool export_result = false);

    void SetInitialOffset(const Vec3& offset);
    void RotateVerticesByEulerAngles(const Vec3& euler_angles);
    void RotateVerticesAroundCentroidByEulerAngles(const Vec3& euler_angles);
    int AddKinematicCylinder(const Vec3& center, Real radius, Real height);
    void AttachKinematicConstraints(int shapeID);
    void RotateKinematicCylinderXKeepingLocalPoint(int shapeID, Real radians, const Vec3& localPoint, const Vec3& worldPin);
    void UpdateKinematicConstraints();
    size_t GetKinematicConstraintCount() const { return h_kinematicConstraints.size(); }
    void BindSkeletonToTetMesh(const SkeletonFrameT<Real>& restFrame, Real maxDistance = static_cast<Real>(0));
    void UpdateSkeletonCouplingTargets(const SkeletonFrameT<Real>& frame);
    void ClearSkeletonCoupling();
    size_t GetSkeletonBindingCount() const { return h_skeletonBindingCount; }
    void ExportMesh(unsigned int frame);
    const Mesh<Real>& GetSurfaceMesh() const;
    const Vec3* GetDeviceVertices() const;
    size_t GetVertexCount() const;
	Parameters& GetParameters() { return h_params; }
    const Parameters& GetParameters() const { return h_params; }

    void SimulateCPU(bool export_results = false, int frame = 0);

protected:
    bool DataTransfer(const std::vector<Tetrahedron<Real>>& tets, const std::vector<Vec3>& vertices);

    void PreCompute();
    void BuildGlobalCsrFromTetMesh();
    void UploadGlobalCsrToDevice();
    void InitCUDALib();

    void SetParams();
    void Step();
    void Step_Explicit();
    void Step_Implicit();
    void Step_Implicit_Sparse();

    void StepCPUExplicit();
    void StepCPUImplicit();

    void PrintInfo() const;

public:
    struct KinematicCylinder
    {
        Vec3 translation = { Real(0), Real(0), Real(0) };
        mat3<Real> rotation = mat3<Real>(Real(1));
        Real radius = Real(0);
        Real height = Real(0);
    };

    struct KinematicConstraint
    {
        int vertexID = -1;
        int shapeID = -1;
        Vec3 localPosition = { Real(0), Real(0), Real(0) };
    };

    struct SkeletonVertexBinding
    {
        int boneID = -1;
        Vec3 restPosition = { Real(0), Real(0), Real(0) };
        Real weight = Real(0);
    };

private:
    // host data
    std::vector<Tetrahedron<Real>> h_tet;
    std::vector<Vec3> h_vertex;
    std::vector<Vec3> h_velocity;
    std::vector<Real> h_mass;
    std::vector<KinematicCylinder> h_kinematicCylinders;
    std::vector<KinematicConstraint> h_kinematicConstraints;
    std::vector<int> h_constraint_dof_flags;
    std::vector<Real> h_constraint_dof_targets;
    std::vector<SkeletonVertexBinding> h_skeletonBindings;
    std::vector<Vec3> h_skeletonTargets;
    std::vector<Vec3> h_skeletonTargetVelocities;
    std::vector<Real> h_skeletonWeights;
    std::vector<std::array<Real, 16>> h_skeletonRestInverseWorldMatrices;
    size_t h_skeletonBindingCount = 0;
    Real h_lastSkeletonTime = Real(0);
    bool h_hasLastSkeletonTime = false;
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
    int* d_constraint_dof_flags = nullptr;
    Real* d_constraint_dof_targets = nullptr;
    Vec3* d_skeleton_targets = nullptr;
    Vec3* d_skeleton_target_velocities = nullptr;
    Real* d_skeleton_weights = nullptr;

    // for implicit solver. A x = B
    Real* delta_x = nullptr;
    Real* d_b = nullptr;
    Real* d_r = nullptr;
    Real* d_p = nullptr;
    Real* d_q = nullptr;
    // Sparse PCG (IMPLICIT_SPARSE) extras: Jacobi-preconditioned residual + diag inverse
    Real* d_z = nullptr;
    Real* d_M_inv = nullptr;
    // Line search buffers (IMPLICIT_SPARSE only)
    Vec3* d_x_tilde = nullptr;   // inertial target: x_n + dt * v_n
    Vec3* d_x0 = nullptr;        // saved positions at start of Newton step
    Real* d_energy = nullptr;    // single-element device buffer for energy accumulation
    Real* DnA = nullptr;
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
