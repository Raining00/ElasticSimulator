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
    p.dt = 1.f / 60.0;
    p.damping = 0.001;
    p.youngs_modulus = 1e2;
    p.poisson_ratio = 0.3;
    p.density = 1.0;
    p.platformType = CPU;

    solver.Initialize(PROJECT_SOURCE_DIR "/assets/ellell.1");
    solver.RotateVerticesAroundCentroidByEulerAngles({ Scalar(glm::radians(90.0)), Scalar(0.0), Scalar(0.0) });
    solver.SetInitialOffset({ Scalar(0), Scalar(1), Scalar(0) });
    
    int current_frame = 0;
    int total_frame = 200;
    printf("Start simulation: \n");
    while (current_frame < total_frame) {
        solver.SimulateCPU(true, current_frame);
        current_frame++;
        printf("%i / %i \n", current_frame, total_frame);
    }

    return 0;
}
