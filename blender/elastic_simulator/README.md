# Elastic Simulator Blender Add-on

This add-on exposes the ElasticSimulator FEM algorithm inside Blender without
using the repository's OpenGL viewer. It reads the selected Blender mesh,
tetrahedralizes it, runs an explicit elastic FEM simulation, and creates a new
animated mesh object with shape keys.

## Install

1. In Blender, open `Edit > Preferences > Add-ons > Install`.
2. Select `D:\Code\ElasticSimulator\blender\elastic_simulator\__init__.py`.
3. Enable `Elastic Simulator FEM`.
4. Open the 3D viewport sidebar and use the `Elastic FEM` tab.

The add-on imports only Blender modules at enable time. When you click
`Run Elastic FEM`, Blender's Python environment must have:

- `numpy`
- either `tetgen` or `ptetgen`

## Use

1. Select a closed surface mesh.
2. Adjust the FEM parameters in `View3D > Sidebar > Elastic FEM`.
3. Click `Run Elastic FEM`.

The generated object is named `<source>_ElasticFEM`. The animation is stored as
shape keys, so it can be rendered with Blender's normal viewport or render
pipeline.

## Editable Parameters

- Energy model: StVK, Corotated, or Neo-Hookean
- Young's modulus, Poisson ratio, density
- Time step, simulation steps, substeps, capture rate, Blender frame stride
- Damping
- Gravity vector
- Floor collision, floor height, restitution
- TetGen quality settings

This is intentionally algorithm-only: it does not link or use the C++ OpenGL
viewer target.
