#pragma once

#include "helper_math.h"
#include "matrix.hpp"
#include <vector>

struct Triangle {
    int3 verticesIndex; // index of the three vertices
};

struct Mesh {
    std::vector<float3> vertices; // array of vertex positions
    std::vector<Triangle> faces; // array of triangles
};

struct Tetrahedron 
{
    int4 verticesIndex; // index of the four vertices
    mat3 Dm_inv; // inverse of the rest state matrix
    float volume; // rest volume
};

struct Particle {
    float3 position;
    float3 position_rest;
    float3 velocity;
    float3 force;
    float mass;
};