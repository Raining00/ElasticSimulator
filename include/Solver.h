#pragma once
#include "BaseStructure.hpp"
#include <cstddef>

enum EnergyType
{
    STVK = 0,
    COROTATED = 1,
    NEOHOOKEAN = 2
};

class ElasticitySolver
{
public:
    struct Parameters
    {
        float youngs_modulus = 1e5; // Young's modulus
        float poisson_ratio = 0.45f; // Poisson's ratio
        float damping = 0.0f; // Damping factor
		float dt = 1e-4; // Time step
        float density = 1000.0f; // Material density
		unsigned int substeps = 10; // Number of substeps for the simulation
        // lame parameters
        float lambda = 0.0f; // First Lame parameter
        float mu = 0.0f; // Second Lame parameter (shear modulus)
        EnergyType energyType = NEOHOOKEAN; // Type of elastic energy to use
		float3 gravity = { 0.0f, -9.81f, 0.0f }; // Gravity vector
		float3 boundary_min = { -10.0f, 0.0f, -10.0f }; // Minimum boundary for collision
		float3 boundary_max = { 10.0f, 10.0f, 10.0f }; // Maximum boundary for collision
    };

    ElasticitySolver() = default;
    ~ElasticitySolver();

    void Initialize(const Mesh& mesh);

    void Simulate(unsigned int total_frame = 30, bool export_results = true);
    void SimulateFrame(bool export_result = false);

    void SetInitialOffset(const float3& offset);
    void ExportMesh(unsigned int frame);
    const Mesh& GetSurfaceMesh() const;
    const float3* GetDeviceVertices() const;
    size_t GetVertexCount() const;

protected:
    bool DataTransfer(const std::vector<Tetrahedron>& tets, const std::vector<float3>& vertices);

    void ComputeTetInitVolume();
    void SetParams();
    void Step();

    void PrintInfo() const;

private:
    // host data
    std::vector<Tetrahedron> h_tet;
    std::vector<float3> h_vertex;
    std::vector<float3> h_force; //debug
    std::vector<float> h_mass; //debug
	Parameters h_params;
    Mesh suraceMesh;

    //device data
    Tetrahedron* d_tet;
    float3* d_vertex;
	float3* d_vertex_velocity;
    float3* d_force;
    float* d_mass;
    bool params_ready = false;
    bool info_printed = false;
};
