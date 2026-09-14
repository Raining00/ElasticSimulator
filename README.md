# ElasticSimulator

A GPU-accelerated soft-body simulator built with **C++17 and CUDA**. ElasticSimulator uses tetrahedral finite elements to simulate elastic deformation, with CPU and GPU examples, an interactive OpenGL viewer, and skeleton-driven animation.

The project is intended for learning, experimentation, and prototyping in physically based animation. It includes explicit and implicit solvers, several elastic material models, and OBJ mesh export.

## Building

### Requirements

- CMake 3.18 or newer, with support for your chosen build generator.
- A C++17 compiler compatible with your CUDA Toolkit.
- An NVIDIA GPU and CUDA Toolkit with cuBLAS and cuSPARSE.
- OpenGL 3.3 and the platform development libraries required by GLFW.

TetGen, GLFW, and GLM are included as Git submodules. CUDA is currently required even for the CPU examples.

### Clone

```bash
git clone --recurse-submodules https://github.com/Raining00/ElasticSimulator.git
cd ElasticSimulator
```

For an existing clone, run `git submodule update --init --recursive`.

### Configure and compile

Before building, edit `CMAKE_CUDA_ARCHITECTURES` in [CMakeLists.txt](CMakeLists.txt) to match your GPU and installed Toolkit. For example, for a GPU with compute capability 8.6:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 86)
```

The current default lists multiple architectures, which older Toolkits may not recognize. Change the line in the file directly; the current configuration overrides a command-line `-D` setting.

**Windows — Visual Studio 2022**

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --parallel
```

Executables are written to `build/bin/Release/`. Use the generator matching your installed Visual Studio version.

**Linux**

With CUDA, OpenGL, and GLFW's platform development dependencies installed:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Executables are written to `build/bin/` for single-configuration generators. Linux builds are not currently covered by automated validation.

To build an individual example, append its target name, for example:

```bash
cmake --build build --config Release --target ArmBend_GPU
```

## Examples

Each example has its own entry point under [Example/](Example/). Scene settings, material parameters, and input paths can be adjusted in the corresponding `main.cpp`.

### Arm bending

[ArmBendGPU](Example/ArmBendGPU/main.cpp) · **`ArmBend_GPU`**

Generates a cylindrical soft body and bends it using two kinematic cylinder handles. The example uses the GPU sparse implicit solver and requires no external model files, making it a convenient starting point.

```powershell
# Windows
.\build\bin\Release\ArmBend_GPU.exe
```

```bash
# Linux
./build/bin/ArmBend_GPU
```

**The viewer starts paused. Press Space to begin the simulation.**

### Skeleton-driven soft body

[BearMuscleSkeletonGPU](Example/BearMuscleSkeletonGPU/main.cpp) · **`BearMuscleSkeleton_GPU`**

Loads the bear mesh from `assets/bear/bear.obj` and animation poses from `assets/bear/skeleton/`. The skeleton drives target positions coupled to the deformable body through stiffness and damping, demonstrating soft-tissue motion with the GPU sparse implicit solver.

The example also supports running without a viewer:

```powershell
.\build\bin\Release\BearMuscleSkeleton_GPU.exe --headless --frames 10
```

On Linux, use `./build/bin/BearMuscleSkeleton_GPU --headless --frames 10`. This mode still requires CUDA and does not export meshes by default.

For custom animation data, [export_fbx_animation_sequence.py](python/export_fbx_animation_sequence.py) can export per-frame OBJ meshes and skeleton JSON files using Blender:

```bash
blender --background --python python/export_fbx_animation_sequence.py -- --input path/to/character.fbx --output output/character
```

### Explicit FEM

| Example | Target | Description |
| --- | --- | --- |
| [FEMExplicitGPU](Example/FEMExplicitGPU/main.cpp) | `ExplicitFEM_GPU` | Simulates `assets/spot.obj` with CUDA explicit integration and an interactive viewer. |
| [FEMExplicitCPU](Example/FEMExplicitCPU/main.cpp) | `ExplicitFEM_CPU` | CPU reference example using the `assets/ellell.1.node` / `.ele` mesh and OBJ export. |

These examples provide a starting point for exploring elastic forces and the effect of time-step size on explicit integration.

### Implicit FEM

| Example | Target | Description |
| --- | --- | --- |
| [ImplicitFEMGPU](Example/ImplicitFEMGPU/main.cpp) | `ImplicitFEM_GPU` | Simulates `assets/spot.obj` using a dense GPU linear system and conjugate gradients, with interactive display. |
| [ImplicitFEMGPUSparse](Example/ImplicitFEMGPUSparse/main.cpp) | `ImplicitFEM_GPU_Sparse` | Uses a sparse CSR system and Jacobi-preconditioned conjugate gradients to reduce matrix storage. |
| [ImplicitFEMCPU](Example/ImplicitFEMCPU/main.cpp) | `ImplicitFEM_CPU` | CPU dense implicit reference example using `assets/spot/spot.1.node` / `.ele`, with OBJ export. |

Before running `ImplicitFEM_GPU_Sparse`, change its input path in `main.cpp` from `assets/bear.obj` to `assets/bear/bear.obj`, then rebuild the target. The dense examples are best suited to small meshes because matrix storage grows quadratically with the number of degrees of freedom.

### Sparse conjugate gradients

[sparseCG](Example/sparseCG/main.cpp) · **`sparseCG`**

A standalone CUDA sparse conjugate-gradient example for inspecting the linear solver independently of a deformable-body scene.

## Viewer Controls

| Input | Action |
| --- | --- |
| Space | Start or pause the simulation |
| 1 / 2 | Switch between orbit and free-flight cameras |
| Left mouse drag | Rotate the view |
| Middle mouse drag | Pan in orbit mode |
| Scroll wheel | Zoom in orbit mode |
| W / A / S / D | Move in free-flight mode |
| Q / E | Move down / up in free-flight mode |

To export simulation frames, create an `output/` directory in the repository root and call `solver.AdvanceFrame(true)` in an example. Files are saved as `output/frame_<index>.obj`; repeated runs may overwrite existing frames. The CPU explicit reference currently reuses frame indices across calls.

Asset and export paths are based on the source directory recorded at build time. Reconfigure and rebuild if you move the repository.

## Project Status

This is an experimental simulator. Collision handling is limited to simple boundary treatment; self-collision and general inter-object contact are not implemented. Contributions and reproducible bug reports are welcome through [GitHub Issues](https://github.com/Raining00/ElasticSimulator/issues).

## License

A project-level license has not yet been specified. Third-party components retain their own licenses; see [TetGen](extern/tetgen/LICENSE), [GLFW](extern/glfw/LICENSE.md), [GLM](extern/glm/copying.txt), and the notices in bundled source files.
