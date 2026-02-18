#pragma once
#include "BaseStructure.hpp"

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
        float youngs_modulus = 1000.0f; // Young's modulus
        float poisson_ratio = 0.3f; // Poisson's ratio
        float damping = 0.01f; // Damping factor
		float dt = 1.f / 60.f; // Time step
		unsigned int substeps = 10; // Number of substeps for the simulation
        // lame parameters
        float lambda = 0.0f; // First Lame parameter
        float mu = 0.0f; // Second Lame parameter (shear modulus)
        EnergyType energyType = STVK; // Type of elastic energy to use
		float3 gravity = { 0.0f, -9.81f, 0.0f }; // Gravity vector
		float3 boundary_min = { -10.0f, 0.0f, -10.0f }; // Minimum boundary for collision
		float3 boundary_max = { 10.0f, 10.0f, 10.0f }; // Maximum boundary for collision
    };

    ElasticitySolver() = default;
    ~ElasticitySolver();

    bool initialize(const std::vector<Tetrahedron>& tets, const std::vector<float3>& vertices);

    void solve() {}

protected:
    void ComputeTetInitVolume();
    void setParams();

private:
    // host data
    std::vector<Tetrahedron> h_tet;
    std::vector<float3> h_vertex;
	Parameters h_params;

    //device data
    Tetrahedron* d_tet;
    float3* d_vertex;
    float3* d_vertex_rest;
	float3* d_vertex_velocity;
};