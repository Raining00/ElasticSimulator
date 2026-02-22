#include <iostream>
#include "include/MeshToTet.hpp"
#include "include/Solver.h"

int main()
{
    float3 a = make_float3(1.0f, 2.0f, 3.0f);
    float3 b = make_float3(4.0f, 5.0f, 6.0f);
    float3 c = a + b;

	Mesh mesh, surfaceMesh;
    ElasticitySolver solver;

	std::vector<Tetrahedron> tets;
	std::vector<float3> vertices;
    if (loadOBJ("D:/Code/ElasticSimulator/bunny.obj", mesh)) {
        std::cout << "Loaded mesh with " << mesh.vertices.size() << " vertices and " << mesh.faces.size() << " faces." << std::endl;
        solver.Initialize(mesh);
		//extractSurfaceTriangles(tets, vertices, surfaceMesh);
        //saveOBJ("D:/Code/ElasticSimulator/surface_mesh.obj", surfaceMesh);
		solver.ExportMesh(0);
    } else {
        std::cerr << "Failed to load mesh." << std::endl;
	}

    std::cout << "Result of a + b: (" << c.x << ", " << c.y << ", " << c.z << ")" << std::endl;
}