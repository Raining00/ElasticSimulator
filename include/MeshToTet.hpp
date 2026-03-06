#pragma once
#include <string>
#include "BaseStructure.hpp"

class tetgenio;

template <typename Real>
bool loadOBJ(const std::string& filename, Mesh<Real>& mesh);

template <typename Real>
void saveOBJ(const std::string& filename, const Mesh<Real>& mesh);

template <typename Real>
void buildTetgenInput(const Mesh<Real>& mesh, tetgenio& in);

template <typename Real>
void tetrahedralizeMesh(const Mesh<Real>& mesh, 
    std::vector<Tetrahedron<Real>>& tets, 
    std::vector<Vec3f>& vertices);

template <typename Real>
void extractSurfaceTriangles(
    const std::vector<Tetrahedron<Real>>& tets,
    const std::vector<Vec3f>& vertices,
    Mesh<Real>& surfaceMesh
);

