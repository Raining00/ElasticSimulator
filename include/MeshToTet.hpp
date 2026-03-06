#pragma once
#include <string>
#include "BaseStructure.hpp"

class tetgenio;

bool loadOBJ(const std::string& filename, Mesh& mesh);
void saveOBJ(const std::string& filename, const Mesh& mesh);

void buildTetgenInput(const Mesh& mesh, tetgenio& in);

void tetrahedralizeMesh(const Mesh& mesh, 
    std::vector<Tetrahedron>& tets, 
    std::vector<Vec3f>& vertices);

void extractSurfaceTriangles(
    const std::vector<Tetrahedron>& tets,
    const std::vector<Vec3f>& vertices,
    Mesh& surfaceMesh
);

