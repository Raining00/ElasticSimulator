#include <cmath>
#include <iostream>

#include "BaseStructure.hpp"
#include "Solver.h"
#include "render/RealtimeViewer.h"

using Scalar = double;

namespace {

Mesh<Scalar> BuildCylinderSurface(Scalar radius, Scalar yMin, Scalar yMax, int segments, int rings)
{
    Mesh<Scalar> mesh;
    const Scalar pi = static_cast<Scalar>(3.14159265358979323846);

    for (int r = 0; r <= rings; ++r) {
        const Scalar t = static_cast<Scalar>(r) / static_cast<Scalar>(rings);
        const Scalar y = yMin * (static_cast<Scalar>(1) - t) + yMax * t;
        for (int s = 0; s < segments; ++s) {
            const Scalar theta = static_cast<Scalar>(2) * pi * static_cast<Scalar>(s) / static_cast<Scalar>(segments);
            mesh.vertices.push_back({
                radius * std::cos(theta),
                y,
                radius * std::sin(theta)
            });
        }
    }

    const int bottomCenter = static_cast<int>(mesh.vertices.size());
    mesh.vertices.push_back({ Scalar(0), yMin, Scalar(0) });
    const int topCenter = static_cast<int>(mesh.vertices.size());
    mesh.vertices.push_back({ Scalar(0), yMax, Scalar(0) });

    for (int r = 0; r < rings; ++r) {
        for (int s = 0; s < segments; ++s) {
            const int next = (s + 1) % segments;
            const int v00 = r * segments + s;
            const int v01 = r * segments + next;
            const int v10 = (r + 1) * segments + s;
            const int v11 = (r + 1) * segments + next;
            mesh.faces.push_back({ { v00, v10, v11 } });
            mesh.faces.push_back({ { v00, v11, v01 } });
        }
    }

    for (int s = 0; s < segments; ++s) {
        const int next = (s + 1) % segments;
        mesh.faces.push_back({ { bottomCenter, next, s } });

        const int topBase = rings * segments;
        mesh.faces.push_back({ { topCenter, topBase + s, topBase + next } });
    }

    return mesh;
}

} // namespace

int main()
{
    ElasticitySolverT<Scalar> solver;
    RealtimeViewer<Scalar> viewer;

    auto& p = solver.GetParameters();
    p.energyType = NEOHOOKEAN;
    p.solverType = IMPLICIT_SPARSE;
    p.dt = static_cast<Scalar>(5e-3);
    p.youngs_modulus = static_cast<Scalar>(3e5);
    p.poisson_ratio = static_cast<Scalar>(0.4);
    p.density = static_cast<Scalar>(1000);
    p.substeps = 1;
    p.gravity = { Scalar(0), Scalar(0), Scalar(0) };
    p.boundary_min = { Scalar(-2), Scalar(-1), Scalar(-2) };
    p.boundary_max = { Scalar(2), Scalar(3), Scalar(2) };
    p.barrier_distance = static_cast<Scalar>(0.02);
    p.barrier_stiffness = static_cast<Scalar>(1e3);

    Mesh<Scalar> arm = BuildCylinderSurface(
        static_cast<Scalar>(0.22),
        static_cast<Scalar>(0.30),
        static_cast<Scalar>(1.75),
        28,
        14);
    solver.Initialize(arm);

    const Scalar boneRadius = static_cast<Scalar>(0.12);
    const Scalar boneLength = static_cast<Scalar>(0.48);
    const int shoulder = solver.AddKinematicCylinder({ Scalar(0), Scalar(0.62), Scalar(0) }, boneRadius, boneLength);
    const int forearm = solver.AddKinematicCylinder({ Scalar(0), Scalar(1.28), Scalar(0) }, boneRadius, boneLength);
    solver.AttachKinematicConstraints(shoulder);
    solver.AttachKinematicConstraints(forearm);

    std::cout << "Total kinematic constraints: " << solver.GetKinematicConstraintCount() << std::endl;

    if (!viewer.Initialize(
        solver.GetSurfaceMesh(),
        p.boundary_min,
        p.boundary_max,
        1280,
        720)) {
        std::cerr << "Failed to initialize realtime viewer." << std::endl;
        return 1;
    }

    int frame = 0;
    while (!viewer.ShouldClose()) {
        if (frame < 250) {
            solver.RotateKinematicCylinderXKeepingLocalPoint(
                forearm,
                static_cast<Scalar>(0.01),
                { Scalar(0), static_cast<Scalar>(-0.25), Scalar(0) },
                { Scalar(0), static_cast<Scalar>(1.03), Scalar(0) });
        }

        solver.AdvanceFrame(false);
        viewer.UpdateFromCuda(solver.GetDeviceVertices(), solver.GetVertexCount());
        viewer.RenderFrame();
        viewer.PollEvents();
        ++frame;
    }

    return 0;
}
