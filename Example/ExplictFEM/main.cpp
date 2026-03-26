#include "MeshToTet.hpp"
#include "Solver.h"
#include "ProjectPaths.h"
#include "glm/glm.hpp"

using Scalar = double;

int main()
{
    ElasticitySolverT<Scalar> solver;
    solver.Initialize(PROJECT_SOURCE_DIR "/assets/ellell.1");
    solver.RotateVerticesAroundCentroidByEulerAngles({ Scalar(0), Scalar(0.0), Scalar(0.0) });
    solver.SetInitialOffset({ Scalar(0), Scalar(1), Scalar(0) });
    auto& p = solver.GetParameters();
    p.energyType = NEOHOOKEAN;
    p.solverType = EXPLICIT;
    p.dt = 5e-4;
    p.damping = 0.001;
    p.youngs_modulus = 1e2;
    p.poisson_ratio = 0.3;
    p.density = 1.0;
    p.platformType = CPU;

    int current_frame = 0;
    int total_frame = 2000;
    printf("Start simulation: \n");
    while (current_frame < total_frame) {
        solver.SimulateCPU(true, current_frame);
        current_frame++;
        printf("%i / %i \n", current_frame, total_frame);
    }

    return 0;
}
