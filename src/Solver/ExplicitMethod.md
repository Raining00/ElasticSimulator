# Explicit Method in This Solver

This note describes the explicit FEM update implemented in `src/Solver/Solver.cu`.

## 1. What the explicit step does

The explicit path is selected when:

```cpp
params.solverType = EXPLICIT;
```

At runtime, `ElasticitySolverT<Real>::Step()` dispatches to `Step_Explicit()`.

The per-substep pipeline in the current code is:

1. Add gravity to every vertex force.
2. Compute elastic force for each tetrahedron and scatter it to vertices.
3. Integrate velocity and position with symplectic Euler.
4. Project vertices back into the AABB boundary and modify velocity.

In code, the order is:

```cpp
k_AddGravity(...);
k_ComputeForces(...);
k_Integrate(...);
k_BoundaryCheck(...);
```

## 2. Precomputation

Before stepping, `ComputeTetInitVolume()` computes:

- Rest tetrahedron volume `V0`
- Inverse rest matrix `Dm_inv`
- Lumped vertex mass

For a tetrahedron with rest vertices `x0, x1, x2, x3`, the code builds:

```text
Dm = [x1 - x0, x2 - x0, x3 - x0]
V0 = |det(Dm)| / 6
Dm_inv = inverse(Dm)
```

Each tet contributes one quarter of its mass to each of its four vertices:

```text
tetMass = density * V0 / 4
```

This mass is computed once and then reused by later explicit substeps.

## 3. Deformation gradient

During each substep, the current deformation gradient is computed for each tetrahedron:

```text
Ds = [x1 - x0, x2 - x0, x3 - x0]
F  = Ds * Dm_inv
```

This is implemented by `computeF(...)`, and the result is stored in `d_F`.

## 4. Stress model

The solver supports three constitutive models:

- `STVK`
- `COROTATED`
- `NEOHOOKEAN`

The code computes the first Piola-Kirchhoff stress `P(F)` from the selected energy:

### StVK

```text
E = 0.5 * (F^T F - I)
P = F * (2 * mu * E + lambda * trace(E) * I)
```

### Corotated

Using the polar decomposition `F = R S`:

```text
P = 2 * mu * (F - R) + lambda * trace(R^T F - I) * R
```

### Neo-Hookean

With `J = det(F)`:

```text
P = mu * F + (lambda * (J - 1) - mu) * adj(F)^T
```

The implementation clamps `J` from below to avoid singular behavior.

## 5. Force assembly

Once `P` is known, the elemental force contribution is:

```text
H  = -V0 * P * Dm_inv^T
f1 = column 0 of H
f2 = column 1 of H
f3 = column 2 of H
f0 = -(f1 + f2 + f3)
```

The four nodal forces are atomically added into the global vertex force array.

This is the role of `k_ComputeForces(...)`.

## 6. Time integration

The explicit integrator is `k_Integrate(...)`. For each vertex:

```text
a = f / m
v_{n+1} = (1 - damping) * v_n + dt * a
x_{n+1} = x_n + dt * v_{n+1}
```

This is a symplectic Euler update:

- velocity is updated first
- the new velocity is used to update position

After integration, the force buffer is reset to zero for the next substep.

## 7. Boundary handling

`k_BoundaryCheck(...)` clamps positions into the axis-aligned box:

```text
[boundary_min, boundary_max]
```

If a vertex crosses a boundary:

- its position is projected back to the box face
- the normal velocity component is flipped and scaled by `restitution`
- tangential components are reduced by `friction`

This is a simple collision response model rather than a constraint solve.

## 8. Why this is called explicit

The method is explicit because it updates the state directly from already-known forces:

```text
force -> acceleration -> velocity -> position
```

There is no global linear solve in each substep.

The tradeoff is:

- simple and relatively cheap per step
- but it usually needs a smaller `dt` for stability, especially for stiff materials

## 9. Code-level summary

In this project, the explicit algorithm is:

```text
Precompute: V0, Dm_inv, lumped mass

For each substep:
  add gravity
  compute F
  compute P(F)
  assemble nodal forces
  symplectic Euler integrate
  apply box collision
```

That matches the current implementation in `Step_Explicit()`.
