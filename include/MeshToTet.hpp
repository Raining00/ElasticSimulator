#pragma once
#include <string>
#include "BaseStructure.hpp"

class tetgenio;
struct Tetrahedron;

bool loadOBJ(const std::string& filename, Mesh& mesh);

void buildTetgenInput(const Mesh& mesh, tetgenio& in);

void tetrahedralizeMesh(const Mesh& mesh, 
    std::vector<Tetrahedron>& tets, 
    std::vector<float3>& vertices);

void extractSurfaceTriangles(
    const std::vector<Tetrahedron>& tets,
    const std::vector<float3>& vertices);