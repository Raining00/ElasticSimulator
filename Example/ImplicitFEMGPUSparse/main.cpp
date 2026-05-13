#include <iostream>
#include "MeshToTet.hpp"
#include "PhysicsWorld.h"
#include "Solver.h"
#include "render/RealtimeViewer.h"
#include "ProjectPaths.h"
#include "glm/glm.hpp"

using Scalar = double;

int main()
{
    Mesh<Scalar> mesh;
    ElasticitySolverT<Scalar> solver;
    PhysicsWorldT<Scalar> world;
    RealtimeViewer<Scalar> viewer;

    auto& p = solver.GetParameters();
    p.energyType = NEOHOOKEAN;
    p.solverType = IMPLICIT_SPARSE;   // cuSPARSE CSR + Jacobi PCG pipeline
    p.dt = 1e-2;
    p.youngs_modulus = 1e6;
    p.poisson_ratio = 0.4;
    p.density = 1000;
    p.substeps = 1;

    auto& world_collision = world.GetCollisionSettings();
    world_collision.boundary_min   = p.boundary_min;
    world_collision.boundary_max   = p.boundary_max;
    world_collision.barrier_distance  = p.barrier_distance;
    world_collision.barrier_stiffness = p.barrier_stiffness;

    solver.Initialize(PROJECT_SOURCE_DIR "/assets/spot/spot.1");
    solver.RotateVerticesAroundCentroidByEulerAngles({ Scalar(glm::radians(90.0)), Scalar(0.0), Scalar(0.0) });
    solver.SetInitialOffset({ Scalar(0), Scalar(1), Scalar(0) });
    world.AddObject(solver);

    if (!viewer.Initialize(
        solver.GetSurfaceMesh(),
        world_collision.boundary_min,
        world_collision.boundary_max,
        1280,
        720)) {
        std::cerr << "Failed to initialize realtime viewer." << std::endl;
        return 1;
    }

    while (!viewer.ShouldClose()) {
        world.AdvanceFrame(false);
        viewer.UpdateFromCuda(solver.GetDeviceVertices(), solver.GetVertexCount());
        viewer.RenderFrame();
        viewer.PollEvents();
    }

    return 0;
}
