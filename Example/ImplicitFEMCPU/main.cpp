#include "MeshToTet.hpp"
#include "Solver.h"
#include "ProjectPaths.h"
#include "glm/glm.hpp"
#include <iostream>

using Scalar = double;

int main()
{
    ElasticitySolverT<Scalar> solver;
    auto& p = solver.GetParameters();
    p.energyType = NEOHOOKEAN;
    p.solverType = IMPLICIT;
    p.dt = 1.f / 120.0;
    p.youngs_modulus = 1e2;
    p.poisson_ratio = 0.4;
    p.density = 1;
    p.platformType = CPU;
    p.barrier_stiffness = 5e2;

    /*Mesh<Scalar> mesh;

    if (!loadOBJ(PROJECT_SOURCE_DIR "/assets/sphere.obj", mesh)) {
        std::cerr << "Failed to load mesh." << std::endl;
        return 1;
    }
    solver.Initialize(mesh);*/
    solver.Initialize(PROJECT_SOURCE_DIR "/assets/spot/spot.1");
    solver.RotateVerticesAroundCentroidByEulerAngles({ Scalar(glm::radians(90.0)), Scalar(0), Scalar(0) });
    solver.SetInitialOffset({ Scalar(0), Scalar(1), Scalar(0) });
    
    int current_frame = 0;
    int total_frame = 60;
    printf("Start simulation: \n");
    while (current_frame < total_frame) {
        solver.AdvanceFrame(true);
        current_frame++;
        printf("%i / %i \n", current_frame, total_frame);
    }

    return 0;
}
