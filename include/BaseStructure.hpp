#pragma once

#include "math/helper_math.h"
#include "math/matrix.hpp"
#include <vector>

struct Triangle {
    Vec3i verticesIndex; // index of the three vertices
};

template <typename Real>
struct MeshT {
    using Vec3 = Vector<Real, 3>;
    std::vector<Vec3> vertices; // array of vertex positions
    std::vector<Triangle> faces; // array of triangles
};

template <typename Real>
struct TetrahedronT {
    Vec4i verticesIndex; // index of the four vertices
    mat3<Real> Dm_inv; // inverse of the rest state matrix
    Real volume; // rest volume
};

template <typename Real>
struct ParticleT {
    using Vec3 = Vector<Real, 3>;
    Vec3 position;
    Vec3 position_rest;
    Vec3 velocity;
    Vec3 force;
    Real mass;
};

using Mesh = MeshT<float>;
using Tetrahedron = TetrahedronT<float>;
using Particle = ParticleT<float>;
