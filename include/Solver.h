#pragma once
#include "BaseStructure.hpp"
#include <cstddef>
#include <type_traits>

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
		Real dt = static_cast<Real>(1e-4); // Time step
        Real density = static_cast<Real>(1000.0); // Material density
		unsigned int substeps = 10; // Number of substeps for the simulation
        // lame parameters
        Real lambda = static_cast<Real>(0.0); // First Lame parameter
        Real mu = static_cast<Real>(0.0); // Second Lame parameter (shear modulus)
        EnergyType energyType = NEOHOOKEAN; // Type of elastic energy to use
        SolverType solverType = EXPLICIT; // Type of solver to use
		Vec3 gravity = { static_cast<Real>(0.0), static_cast<Real>(-9.81), static_cast<Real>(0.0) }; // Gravity vector
		Vec3 boundary_min = { static_cast<Real>(-10.0), static_cast<Real>(0.0), static_cast<Real>(-10.0) }; // Minimum boundary for collision
		Vec3 boundary_max = { static_cast<Real>(10.0), static_cast<Real>(10.0), static_cast<Real>(10.0) }; // Maximum boundary for collision
    };

    ElasticitySolverT() = default;
    ~ElasticitySolverT();

    void Initialize(const Mesh& mesh);

    void Simulate(unsigned int total_frame = 30, bool export_results = true);
    void SimulateFrame(bool export_result = false);

    void SetInitialOffset(const Vec3& offset);
    void ExportMesh(unsigned int frame);
    const Mesh& GetSurfaceMesh() const;
    const Vec3* GetDeviceVertices() const;
    size_t GetVertexCount() const;
	Parameters& GetParameters() { return h_params; }
    const Parameters& GetParameters() const { return h_params; }

protected:
    bool DataTransfer(const std::vector<Tetrahedron>& tets, const std::vector<Vec3>& vertices);

    void ComputeTetInitVolume();
    void SetParams();
    void Step();
    void Step_Explicit();
    void Step_Implicit();

    void PrintInfo() const;

private:
    // host data
    std::vector<Tetrahedron> h_tet;
    std::vector<Vec3> h_vertex;
	Parameters h_params;
    Mesh suraceMesh;

    //device data
    Tetrahedron* d_tet;
    Vec3* d_vertex;
	Vec3* d_vertex_velocity;
    Vec3* d_force;
    Real* d_mass;
    bool params_ready = false;
    bool info_printed = false;
};

using ElasticitySolverf = ElasticitySolverT<float>;
using ElasticitySolverd = ElasticitySolverT<double>;
using ElasticitySolver = ElasticitySolverf;
