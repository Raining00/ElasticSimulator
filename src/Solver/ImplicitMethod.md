# Implicit Method in This Solver

This note describes the implicit FEM path implemented in `src/Solver/Solver.cpp` and `src/Solver/Solver.cu`.

## 1. What the implicit path is trying to do

The implicit path is selected when:

```cpp
params.solverType = IMPLICIT;
```

At a high level, the solver tries to replace the explicit update with a linear solve for a displacement increment:

```text
A * Delta x = b
```

Then the state is updated by:

```text
x_{n+1} = x_n + Delta x
v_{n+1} = Delta x / dt
```

This is the role of `k_integrateImplicit(...)`.

## 2. Initialization unique to the implicit solver

Before the first implicit frame, the code performs extra setup:

1. `BuildGlobalCsrFromTetMesh()`
2. `UploadGlobalCsrToDevice()`
3. `InitCUDALib()`

### 2.1 CSR topology

`BuildGlobalCsrFromTetMesh()` builds the sparse matrix structure of the global system.

- Each vertex has 3 degrees of freedom.
- Total dof count is `3 * numVertices`.
- Every tetrahedron contributes a dense `12 x 12` local block because it couples 4 vertices x 3 dof.

The function:

- collects row connectivity from tetrahedron adjacency
- inserts diagonal entries
- stores CSR row offsets and column indices
- builds `elem_to_A_csr`, a lookup table from local element entry `(lr, lc)` to a global CSR index

That mapping is important because the GPU kernel later scatters each local stiffness entry into the correct sparse matrix slot.

### 2.2 GPU library objects

`InitCUDALib()` creates:

- a cuBLAS handle
- a cuSPARSE handle
- the CSR matrix descriptor `A`
- device buffers for `delta_x`, `b`, and `r`

The separate `Example/sparseCG/main.cpp` file shows a standalone GPU conjugate gradient example using cuBLAS and cuSPARSE. That example is the clearest reference for how the final solve is expected to work.

## 3. Material restriction in the current code

`Step_Implicit()` currently allows only:

```cpp
energyType == NEOHOOKEAN
```

If another energy model is selected, the solver exits.

So the implicit derivation below is specifically the Neo-Hookean one used in the code.

## 4. Current implicit substep pipeline

The current implementation does the following:

1. Compute elastic forces and current deformation gradients.
2. Compute each tetrahedron's tangent stiffness matrix.
3. Scatter those local matrices into the global CSR matrix.
4. Add mass to the diagonal and assemble the right-hand side.
5. Intended: solve `A * Delta x = b` with CG.
6. Integrate with `Delta x`.
7. Apply boundary handling.
8. Clear temporary buffers.

In code, that is roughly:

```cpp
k_ComputeForces(...);
k_computeK(...);
k_Assemble(...);
// CG solver: TODO
k_integrateImplicit(...);
k_BoundaryCheck(...);
```

One important implementation detail: unlike `Step_Explicit()`, the current implicit path does not call `k_AddGravity(...)`, so the assembled right-hand side currently uses elastic force only.

## 5. Deformation gradient and elastic force

Like the explicit solver, the implicit path first computes:

```text
F = Ds * Dm_inv
```

and evaluates the first Piola stress `P(F)`.

The same elemental nodal force formula is used:

```text
H  = -V0 * P * Dm_inv^T
f1, f2, f3 = columns of H
f0 = -(f1 + f2 + f3)
```

These forces are assembled into the vertex force array by `k_ComputeForces(...)`.

## 6. Local stiffness matrix

The main extra work in the implicit method is the tangent stiffness matrix.

The code comment summarizes the idea as:

```text
df/dx = vec(dF/dx)^T * vec(dP/dF) * vec(dF/dx)
```

For each tetrahedron, `k_computeK(...)` builds:

### 6.1 Shape-function gradient terms

Using `Dm_inv^T`, the code computes the four gradients `g[a]` of the tetrahedral basis functions.

From these, it forms `dF_dx[12]`, one `3 x 3` matrix per local degree of freedom.

These are stacked into the matrix `B`:

```text
B = BuildBMatrixFromdFdx(dF_dx)
```

where `B` has size `9 x 12`.

### 6.2 Material Hessian for Neo-Hookean energy

For Neo-Hookean elasticity, the code computes:

```text
J = det(F)
dJ/dF = [f1 x f2, f2 x f0, f0 x f1]
```

and then builds:

- `G`, an outer-product matrix from `dJ/dF`
- `hessianJ`, the second derivative structure of `J`
- `I9`, the `9 x 9` identity

The tangent `dP/dF` is assembled as:

```text
dP/dF =
    mu * I9
  + lambda * G
  + (lambda * (J - 1) - mu) * hessianJ
```

### 6.3 Element stiffness

The elemental tangent matrix is:

```text
Ke = -V0 * B^T * (dP/dF) * B
```

The code then scales it by `dt^2` while scattering into the global matrix:

```text
A_values += Ke * dt^2
```

using the precomputed `elem_to_A_csr` lookup.

## 7. Global system assembly

After the elastic contribution is added, `k_Assemble(...)` adds the mass term and builds the right-hand side.

For each vertex degree of freedom:

```text
A_diag += m
b = v_n * dt + (f / m) * dt^2
```

So the assembled linear system corresponds to the implementation comment:

```text
(M + dt^2 * K_like) * Delta x = v_n * dt + M^{-1} f * dt^2
```

More precisely, the code stores the mass directly on the diagonal of `A`, while the stiffness-like contribution was already scattered into the sparse matrix.

## 8. Solve step and current status

The intended next step is a conjugate gradient solve:

```text
A * Delta x = b
```

However, in the current `Step_Implicit()` implementation, this part is still marked:

```cpp
// CG solver.
```

No CG iterations are actually executed there yet.

That means:

- `delta_x` stays zero
- `k_integrateImplicit(...)` applies no position change
- the implicit path currently assembles the system but does not complete the physical update

This is why `Example/sparseCG/main.cpp` is important: it demonstrates the cuSPARSE + cuBLAS CG loop that still needs to be integrated into `Step_Implicit()`.

## 9. Boundary handling and cleanup

After the intended solve/integration stage, the code uses the same AABB collision kernel as the explicit method:

```text
k_BoundaryCheck(...)
```

Then it clears:

- global sparse values
- right-hand side `b`
- `delta_x`
- force buffer

so the next substep starts from a clean assembly state.

## 10. Why this is called implicit

The method is implicit because the new displacement is obtained by solving a system built from the current tangent response of the material.

Compared with the explicit method, the goal is:

- better stability for stiff materials
- larger usable time steps
- at the cost of assembling and solving a global sparse linear system each substep

## 11. Code-level summary

In this project, the implicit algorithm is intended to be:

```text
Precompute:
  V0, Dm_inv, lumped mass
  sparse CSR topology
  element-to-CSR scatter map

For each substep:
  compute F
  compute elastic force
  compute local tangent stiffness Ke
  assemble global sparse matrix A
  assemble rhs b
  solve A * Delta x = b
  update x and v from Delta x
  apply box collision
  clear assembly buffers
```

That is the basic algorithm represented by the current source code, with the important caveat that the actual CG solve is not yet wired into `Step_Implicit()`.
