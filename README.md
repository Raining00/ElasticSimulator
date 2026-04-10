# ElasticSimulator

A GPU-accelerated finite-element soft-body simulator written in C++ and CUDA.
It implements explicit and implicit FEM pipelines for hyperelastic materials
(StVK, Corotated, Stable Neo-Hookean) and ships with a small real-time OpenGL
viewer for inspecting the results.

## Features

- Tetrahedral FEM elasticity with three energy models: StVK, Corotated, Stable Neo-Hookean.
- Multiple solver pipelines:
  - **Explicit FEM** — CPU and CUDA backends.
  - **Implicit FEM (dense)** — CUDA backend, dense matrix + cuBLAS CG.
  - **Implicit FEM (sparse)** — CUDA backend, cuSPARSE CSR matrix + Jacobi-preconditioned CG.
- Tetrahedralization of triangle meshes via TetGen, or direct loading of `.node`/`.ele` files.
- Real-time OpenGL viewer with CUDA interop for displaying the deforming surface mesh.
- Single- and double-precision support (`float` / `double`) through templated solvers.

## Repository layout

```
ElasticSimulator/
├── CMakeLists.txt
├── include/                    # Public headers (Solver, math, render, ...)
├── src/
│   ├── Solver/                 # Solver.cpp, Solvergpu.cu, Solvercpu.cpp
│   └── Render/                 # Real-time viewer + CUDA interop
├── Example/                    # Self-contained example apps
│   ├── FEMExplicitCPU/
│   ├── FEMExplicitGPU/
│   ├── ImplicitFEMCPU/
│   ├── ImplicitFEMGPU/         # dense CG
│   └── ImplicitFEMGPUSparse/   # cuSPARSE CSR + Jacobi PCG
├── extern/                     # git submodules: tetgen, glm, glfw
├── assets/                     # Example meshes (e.g. spot)
└── output/                     # Default OBJ export directory
```

## Requirements

- A CUDA-capable GPU (compute capability >= 7.5).
- CUDA Toolkit 11.x or later (cuBLAS and cuSPARSE are required).
- CMake >= 3.18.
- A C++17 compiler:
  - **Windows:** Visual Studio 2019 / 2022 (MSVC).
  - **Linux:** GCC >= 9 or Clang >= 10.
- OpenGL development headers (for the real-time viewer).
- Git (with submodule support).

## Cloning

The project uses git submodules for `tetgen`, `glm` and `glfw`. Clone with
submodules in one shot:

```bash
git clone --recursive git@github.com:Raining00/ElasticSimulator.git
cd ElasticSimulator
```

Or, if you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

## Building

The project is configured with CMake. From the repository root:

### Windows (Visual Studio)

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

The example executables will be placed under `build/bin/Release/`.

### Linux / WSL

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The example executables will be placed under `build/bin/`.

> **Note:** The default `CMAKE_CUDA_ARCHITECTURES` set in `CMakeLists.txt`
> targets a wide range of modern GPUs (sm_75 through sm_120). If your CUDA
> toolkit is older or you only need one architecture, override it on the
> command line, e.g. `-DCMAKE_CUDA_ARCHITECTURES=86`.

## Running the examples

Each subdirectory under `Example/` is built into its own executable. After
building, run them directly from the build output directory, for example:

```bash
# Windows
build\bin\Release\ImplicitFEM_GPU_Sparse.exe

# Linux
./build/bin/ImplicitFEM_GPU_Sparse
```

Available examples:

| Example                  | Solver pipeline                                       |
| ------------------------ | ----------------------------------------------------- |
| `FEMExplicitCPU`         | Explicit FEM, CPU                                     |
| `FEMExplicitGPU`         | Explicit FEM, CUDA                                    |
| `ImplicitFEMCPU`         | Implicit FEM, CPU                                     |
| `ImplicitFEM_GPU`        | Implicit FEM, CUDA, dense matrix + CG                 |
| `ImplicitFEM_GPU_Sparse` | Implicit FEM, CUDA, cuSPARSE CSR + Jacobi PCG         |

Simulated frames can be exported as `.obj` files into the `output/` directory
by passing `true` to `world.AdvanceFrame(...)` (see `Example/FEMExplicitGPU/main.cpp`).

## Using the solver in your own code

A minimal usage looks like this:

```cpp
#include "Solver.h"
#include "PhysicsWorld.h"
#include "MeshToTet.hpp"

using Scalar = double;

Mesh<Scalar> mesh;
loadOBJ("path/to/mesh.obj", mesh);

ElasticitySolverT<Scalar> solver;
auto& p = solver.GetParameters();
p.energyType    = NEOHOOKEAN;
p.solverType    = IMPLICIT_SPARSE;   // or IMPLICIT, EXPLICIT
p.dt            = 1e-3;
p.youngs_modulus = 1e2;
p.poisson_ratio = 0.4;
p.density       = 1.0;
p.substeps      = 1;

solver.Initialize(mesh);

PhysicsWorldT<Scalar> world;
world.AddObject(solver);

for (int frame = 0; frame < 200; ++frame) {
    world.AdvanceFrame(/*export=*/true);
}
```

The key knobs in `ElasticitySolverT::Parameters` are:

- `energyType` — `STVK`, `COROTATED`, or `NEOHOOKEAN`.
- `solverType` — `EXPLICIT`, `IMPLICIT` (dense GPU CG), or `IMPLICIT_SPARSE` (cuSPARSE CSR + Jacobi PCG).
- `platformType` — `CPU` or `GPU`.
- `dt`, `substeps`, `density`, `youngs_modulus`, `poisson_ratio`, `damping`.
- `gravity`, `boundary_min`, `boundary_max` for the world AABB.

## Limitations

This project is primarily a research / learning sandbox for FEM-based soft-body
simulation. A few rough edges to be aware of:

- **Collision handling is not fully stable or proper.** The current world
  collision is just an axis-aligned bounding-box projection with simple
  restitution and friction in `k_BoundaryCheck`. There is no continuous
  collision detection, no self-collision, and no robust contact resolution, so
  fast-moving or thin geometry can tunnel through the boundary or exhibit
  jitter at rest.
- **No object–object collisions.** The `PhysicsWorld` can hold multiple
  objects, but inter-object contact is not implemented — each elastic body
  only sees the world AABB.
- **Implicit solver supports Neo-Hookean only.** Both the dense GPU CG path
  (`IMPLICIT`) and the sparse cuSPARSE PCG path (`IMPLICIT_SPARSE`) currently
  assert on `NEOHOOKEAN` energy. StVK and Corotated are only wired up through
  the explicit pipeline.
- **Linear solver convergence is not adaptive.** The CG / PCG iteration count
  is capped at the system DoF and uses a fixed tolerance; there is no line
  search, no Newton outer loop, and no inertia/contact-aware preconditioning
  beyond simple Jacobi.
- **Dense GPU implicit path scales poorly.** It allocates a full `dof × dof`
  matrix in device memory, so it is only suitable for small meshes. For
  anything non-trivial, prefer `IMPLICIT_SPARSE`.
- **CPU paths are minimal.** They exist mainly as reference implementations
  and are not performance-tuned.
- **No checkpointing or scene description format.** Scenes are configured
  directly in C++ inside each `Example/.../main.cpp`.

Contributions and bug reports are welcome.

## License

See individual source files and `extern/` submodules for licensing details of
the third-party dependencies (TetGen, GLFW, GLM).
