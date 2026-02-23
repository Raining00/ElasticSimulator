#include <iostream>
#include "include/MeshToTet.hpp"
#include "include/Solver.h"

void scaleMesh(Mesh& mesh, float scale)
{
    for (auto& vertex : mesh.vertices) {
        vertex.x *= scale;
        vertex.y *= scale;
        vertex.z *= scale;
    }
}

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
		scaleMesh(mesh, 100);
        saveOBJ("D:/Code/ElasticSimulator/bunny_scaled.obj", mesh); 
        solver.Initialize(mesh);
		solver.SetInitialOffset(make_float3(0.0f, 0.2f, 0.0f)); // Set an initial offset for the simulation
        solver.Simulate(5);
    } else {
        std::cerr << "Failed to load mesh." << std::endl;
	}

    std::cout << "Result of a + b: (" << c.x << ", " << c.y << ", " << c.z << ")" << std::endl;
}