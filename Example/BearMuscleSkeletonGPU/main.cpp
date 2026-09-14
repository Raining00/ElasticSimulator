#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "BaseStructure.hpp"
#include "MeshToTet.hpp"
#include "ProjectPaths.h"
#include "SkeletonAnimation.hpp"
#include "Solver.h"
#include "render/RealtimeViewer.h"

using Scalar = double;

namespace {

struct Bounds
{
    Vector<Scalar, 3> min;
    Vector<Scalar, 3> max;
};

Bounds ComputeBounds(const Mesh<Scalar>& mesh)
{
    Bounds bounds{
        { Scalar(0), Scalar(0), Scalar(0) },
        { Scalar(0), Scalar(0), Scalar(0) }
    };
    if (mesh.vertices.empty()) {
        return bounds;
    }

    bounds.min = mesh.vertices.front();
    bounds.max = mesh.vertices.front();
    for (const auto& v : mesh.vertices) {
        for (int c = 0; c < 3; ++c) {
            bounds.min[c] = std::min(bounds.min[c], v[c]);
            bounds.max[c] = std::max(bounds.max[c], v[c]);
        }
    }
    return bounds;
}

Vector<Scalar, 3> ExpandMin(const Vector<Scalar, 3>& v, Scalar margin)
{
    return { v.x - margin, v.y - margin, v.z - margin };
}

Vector<Scalar, 3> ExpandMax(const Vector<Scalar, 3>& v, Scalar margin)
{
    return { v.x + margin, v.y + margin, v.z + margin };
}

} // namespace

int main(int argc, char** argv)
{
    bool headless = false;
    size_t headlessFrames = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--headless") == 0) {
            headless = true;
        } else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            headlessFrames = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        }
    }

    const std::string meshPath = PROJECT_SOURCE_DIR "/assets/bear/bear.obj";
    const std::string skeletonDir = PROJECT_SOURCE_DIR "/assets/bear/skeleton";

    Mesh<Scalar> bearMesh;
    if (!loadOBJ(meshPath, bearMesh)) {
        std::cerr << "Failed to load bear mesh: " << meshPath << std::endl;
        return 1;
    }
    std::cout << "Loaded bear surface: " << bearMesh.vertices.size()
              << " vertices, " << bearMesh.faces.size() << " triangles." << std::endl;

    SkeletonAnimationT<Scalar> skeletonAnimation;
    if (!skeletonAnimation.LoadFromDirectory(skeletonDir)) {
        std::cerr << "Failed to load skeleton sequence: " << skeletonDir << std::endl;
        return 1;
    }

    ElasticitySolverT<Scalar> solver;
    auto& p = solver.GetParameters();
    p.energyType = NEOHOOKEAN;
    p.solverType = IMPLICIT_SPARSE;
    p.platformType = GPU;
    p.substeps = 10;
    p.dt = static_cast<Scalar>(1) / (skeletonAnimation.Frame(0).fps * static_cast<Scalar>(p.substeps));
    p.youngs_modulus = static_cast<Scalar>(1e5);
    p.poisson_ratio = static_cast<Scalar>(0.42);
    p.density = static_cast<Scalar>(1000);
    p.damping = static_cast<Scalar>(0.01);
    p.gravity = { Scalar(0), Scalar(0), Scalar(0) };
    p.barrier_distance = static_cast<Scalar>(0.02);
    p.barrier_stiffness = static_cast<Scalar>(1e3);
    p.muscle_coupling_stiffness = static_cast<Scalar>(8e4);
    p.muscle_coupling_damping = static_cast<Scalar>(80);

    const Bounds bounds = ComputeBounds(bearMesh);
    const Scalar margin = static_cast<Scalar>(0.5);
    p.boundary_min = ExpandMin(bounds.min, margin);
    p.boundary_max = ExpandMax(bounds.max, margin);

    solver.Initialize(bearMesh);
    solver.BindSkeletonToTetMesh(skeletonAnimation.Frame(0), static_cast<Scalar>(0.35));

    std::cout << "Skeleton-coupled tet vertices: "
              << solver.GetSkeletonBindingCount() << std::endl;

    if (headless) {
        for (size_t frame = 0; frame < headlessFrames; ++frame) {
            const size_t skeletonFrame = std::min(frame, skeletonAnimation.FrameCount() - 1);
            solver.UpdateSkeletonCouplingTargets(skeletonAnimation.Frame(skeletonFrame));
            solver.AdvanceFrame(false);
        }
        std::cout << "Headless bear muscle-skeleton run complete: "
                  << headlessFrames << " frame(s)." << std::endl;
        return 0;
    }

    RealtimeViewer<Scalar> viewer;
    if (!viewer.Initialize(
        solver.GetSurfaceMesh(),
        p.boundary_min,
        p.boundary_max,
        1280,
        720)) {
        std::cerr << "Failed to initialize realtime viewer." << std::endl;
        return 1;
    }

    size_t frame = 0;
    while (!viewer.ShouldClose()) {
        if (!viewer.IsPaused()) {
            const size_t skeletonFrame = std::min(frame, skeletonAnimation.FrameCount() - 1);
            solver.UpdateSkeletonCouplingTargets(skeletonAnimation.Frame(skeletonFrame));
            solver.AdvanceFrame(false);
            ++frame;
        }

        viewer.UpdateFromCuda(solver.GetDeviceVertices(), solver.GetVertexCount());
        viewer.RenderFrame();
        viewer.PollEvents();
    }

    return 0;
}
