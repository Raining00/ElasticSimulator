#include <iostream>
#include "MeshToTet.hpp"
#include "Solver.h"
#include "render/RealtimeViewer.h"
#include "ProjectPaths.h"

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
	Mesh mesh;
    ElasticitySolver solver;
    RealtimeViewer viewer;

    if (!loadOBJ(PROJECT_SOURCE_DIR "/assets/sphere.obj", mesh)) {
        std::cerr << "Failed to load mesh." << std::endl;
        return 1;
	}
	scaleMesh(mesh, 3.0f);

    std::cout << "Loaded mesh with " << mesh.vertices.size() << " vertices and " << mesh.faces.size() << " faces." << std::endl;
    solver.Initialize(mesh);
    solver.SetInitialOffset(make_float3(0.0f, 0.3f, 0.0f));
    auto& p = solver.GetParameters();
    p.solverType = IMPLICIT;
    p.energyType = NEOHOOKEAN;
	p.dt = 1e-3f;

    if (!viewer.Initialize(solver.GetSurfaceMesh(), 1280, 720)) {
        std::cerr << "Failed to initialize realtime viewer." << std::endl;
        return 1;
    }

    while (!viewer.ShouldClose()) {
        solver.SimulateFrame(false);
        viewer.UpdateFromCuda(solver.GetDeviceVertices(), solver.GetVertexCount());
        viewer.RenderFrame();
        viewer.PollEvents();
    }

    return 0;
}
