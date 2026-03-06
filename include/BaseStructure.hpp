#pragma once

#include "math/helper_math.h"
#include "math/matrix.hpp"
#include <vector>

struct Triangle {
    Vec3i verticesIndex; // index of the three vertices
};

template <typename Real>
struct Mesh {
    using Vec3 = Vector<Real, 3>;
    std::vector<Vec3> vertices; // array of vertex positions
    std::vector<Triangle> faces; // array of triangles
};

template <typename Real>
struct Tetrahedron {
    Vec4i verticesIndex; // index of the four vertices
    mat3<Real> Dm_inv; // inverse of the rest state matrix
    Real volume; // rest volume
};

template <typename Real>
struct Particle {
    using Vec3 = Vector<Real, 3>;
    Vec3 position;
    Vec3 position_rest;
    Vec3 velocity;
    Vec3 force;
    Real mass;
};