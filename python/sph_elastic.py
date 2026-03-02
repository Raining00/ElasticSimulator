"""
SPH Elastic Simulation — Taichi
================================
Particle-based elastic solid simulation using the Smoothed Particle
Hydrodynamics (SPH) method with a Total-Lagrangian formulation.

Method Overview
---------------
Total-Lagrangian SPH (TLSPH) — all kernel summations are computed in the
*reference* (rest) configuration, so the neighbor list is built once and
reused throughout the simulation.  This avoids reconnecting neighbors as
the body deforms and naturally handles large deformations.

Deformation Gradient
--------------------
The corrected SPH deformation gradient for particle i is:

    F_i = [ Σ_j m_j/ρ0_j (x_j - x_i) ⊗ ∇W_ij0 ] · L_i⁻¹

where ∇W_ij0 is the kernel gradient evaluated at the *reference* positions
X_i, X_j, and L_i is the correction matrix (kernel gradient correction /
renormalization):

    L_i = Σ_j m_j/ρ0_j (X_j - X_i) ⊗ ∇W_ij0

This corrected formulation gives first-order consistency (reproduces linear
displacement fields exactly).

Constitutive Models
-------------------
Three analytic 1st Piola-Kirchhoff stress models (no autodiff):

  StVK:        P = F(2μE + λtr(E)I),  E = (FᵀF-I)/2
  Co-rotated:  P = 2μ(F-R) + λ(J-1)JF⁻ᵀ,  R from SVD polar decomp
  Neo-Hookean: P = μ(F-F⁻ᵀ) + λ(J-1)JF⁻ᵀ

Internal Force
--------------
The SPH discretisation of the divergence of stress gives the elastic
acceleration on particle i:

    a_i^elastic = (1/ρ0_i) Σ_j m_j [ P_i·L_i⁻ᵀ + P_j·L_j⁻ᵀ ] · ∇W_ij0

This is the symmetric, momentum-conserving form.

Artificial Viscosity
--------------------
A standard Monaghan-style artificial viscosity term damps spurious
oscillations without the need for a physical viscosity parameter.

Kernel
------
Cubic spline W(q, h),  q = |X_ij|/h  (compact support radius = 2h).

Integration
-----------
Explicit Velocity-Verlet (leap-frog) with a floor collision.

Default Scene
-------------
If no PLY file is given, a unit sphere is sampled with a Poisson-disk-like
stratified grid fill so particles are approximately uniformly spaced.
"""

import argparse
import os
import math
import numpy as np
import taichi as ti

# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SPH Elastic Simulation — Taichi")

    # Input
    p.add_argument("--ply",  type=str, default=None,
                   help="Path to input PLY point cloud (or surface mesh). "
                        "If omitted, a unit sphere particle cloud is generated.")
    p.add_argument("--sphere_radius", type=float, default=1.0,
                   help="Radius of the default unit sphere (default: 1.0)")
    p.add_argument("--sphere_particles", type=int, default=2000,
                   help="Target particle count for default sphere (default: 2000)")

    # Material
    p.add_argument("--model", type=str, default="corotated",
                   choices=["stvk", "corotated", "neohookean"],
                   help="Constitutive model (default: corotated)")
    p.add_argument("--E",       type=float, default=1e5,
                   help="Young's modulus (default: 1e5)")
    p.add_argument("--nu",      type=float, default=0.3,
                   help="Poisson's ratio (default: 0.3)")
    p.add_argument("--density", type=float, default=1000.0,
                   help="Reference density ρ₀ (default: 1000.0)")

    # SPH
    p.add_argument("--h_factor", type=float, default=1.2,
                   help="Smoothing length = h_factor × particle_spacing (default: 1.2)")
    p.add_argument("--art_visc_alpha", type=float, default=0.3,
                   help="Artificial viscosity coefficient α (default: 0.3)")
    p.add_argument("--max_neighbors", type=int, default=128,
                   help="Max neighbor storage per particle (default: 128)")

    # Simulation
    p.add_argument("--dt",      type=float, default=1e-4,
                   help="Time step (default: 1e-4)")
    p.add_argument("--steps",   type=int,   default=5000,
                   help="Total simulation steps (default: 5000)")
    p.add_argument("--gravity", type=float, default=-9.8,
                   help="Gravity in Y direction (default: -9.8)")
    p.add_argument("--damping", type=float, default=0.999,
                   help="Velocity damping factor per step (default: 0.999)")

    # Output
    p.add_argument("--export_every", type=int, default=10,
                   help="Export PLY every N steps (default: 10)")
    p.add_argument("--output_dir",   type=str, default="output_sph",
                   help="Output directory for PLY frames (default: output_sph)")
    p.add_argument("--gui",          action="store_true",
                   help="Real-time visualisation via ti.ui.Window")
    p.add_argument("--substeps",     type=int, default=5,
                   help="Sim substeps per GUI frame (default: 5)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Particle cloud helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_ply_points(path: str) -> np.ndarray:
    """
    Load a PLY file and return its vertex positions as (N,3) float32.
    Handles both point clouds and triangle meshes (vertices only).
    """
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError("pip install plyfile")
    ply  = PlyData.read(path)
    elem = ply["vertex"]
    x = np.array(elem["x"], dtype=np.float32)
    y = np.array(elem["y"], dtype=np.float32)
    z = np.array(elem["z"], dtype=np.float32)
    pts = np.stack([x, y, z], axis=1)
    print(f"[PLY] Loaded {len(pts)} points from '{path}'")
    return pts


def sample_sphere_particles(radius: float, n_target: int) -> np.ndarray:
    """
    Fill a sphere with ~n_target uniformly distributed particles using a
    stratified rejection-sampling approach on a 3-D grid.

    The grid spacing is chosen so the cube [ -r, r ]³ contains ≈ n_target
    particles after rejection; the actual count may differ slightly.
    """
    # Grid spacing that yields ≈ n_target inside the sphere
    vol_sphere = (4.0/3.0) * math.pi * radius**3
    vol_cube   = (2.0 * radius)**3
    fill_ratio = vol_sphere / vol_cube          # ≈ 0.5236
    n_grid     = int(round((n_target / fill_ratio) ** (1.0/3.0))) + 1
    spacing    = 2.0 * radius / n_grid

    coords = np.arange(n_grid) * spacing - radius + spacing * 0.5
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    # Reject outside sphere
    dist = np.linalg.norm(pts, axis=1)
    pts  = pts[dist <= radius].astype(np.float32)

    # Lift above floor (bottom of sphere at y=0.01)
    pts[:, 1] += radius + 0.01

    print(f"[Sphere] Sampled {len(pts)} particles "
          f"(target={n_target}, r={radius}, spacing≈{spacing:.4f})")
    return pts, spacing


def estimate_particle_spacing(pts: np.ndarray) -> float:
    """
    Estimate mean nearest-neighbor distance as a proxy for particle spacing.
    Uses a random subsample for speed.
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), min(500, len(pts)), replace=False)
    sub = pts[idx]
    # Vectorised pairwise distances on subsample
    diff = sub[:, None, :] - sub[None, :, :]       # (K,K,3)
    dist = np.linalg.norm(diff, axis=-1)            # (K,K)
    np.fill_diagonal(dist, np.inf)
    nn   = dist.min(axis=1)
    return float(nn.mean())


# ─────────────────────────────────────────────────────────────────────────────
# PLY exporter
# ─────────────────────────────────────────────────────────────────────────────
def export_ply(positions: np.ndarray, frame: int, output_dir: str):
    """Write particle positions to  <output_dir>/frame_<frame>.ply"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"frame_{frame:06d}.ply")
    n = len(positions)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for v in positions:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Taichi init
# ─────────────────────────────────────────────────────────────────────────────
ti.init(arch=ti.gpu, default_fp=ti.f32)


# ─────────────────────────────────────────────────────────────────────────────
# SPH Elastic Simulator
# ─────────────────────────────────────────────────────────────────────────────
@ti.data_oriented
class SPHElasticSimulator:
    """
    Total-Lagrangian SPH elastic simulator.

    Neighbor list
    -------------
    Built once from the reference (rest) positions X and stored as a fixed
    compact adjacency structure:
        neighbor_count[i]          – number of neighbors of particle i
        neighbor_list[i, 0..k-1]  – neighbor indices

    Kernel
    ------
    Cubic spline with compact support 2h:

        W(q) = σ * { 1 - 1.5q² + 0.75q³        0 ≤ q < 1
                   { 0.25(2-q)³                  1 ≤ q < 2
                   { 0                           q ≥ 2

        σ = 8/(π h³)  in 3-D

    Gradient:  ∇W_ij = dW/dq · (1/h) · (X_i - X_j)/|X_i - X_j|  (at ref pos)
    """

    def __init__(self, positions: np.ndarray, args,
                 particle_spacing: float):
        N = len(positions)
        self.N   = N
        self.dt  = args.dt
        self.gravity  = args.gravity
        self.model    = args.model
        self.damping  = args.damping
        self.art_visc = args.art_visc_alpha
        self.max_nb   = args.max_neighbors

        # Smoothing length
        self.h    = args.h_factor * particle_spacing
        self.h2   = self.h * self.h
        self.invh = 1.0 / self.h

        # Lamé parameters
        E, nu = args.E, args.nu
        self.mu  = E / (2.0*(1.0+nu))
        self.lam = E*nu / ((1.0+nu)*(1.0-2.0*nu))

        # Particle mass from reference density
        vol_particle  = particle_spacing**3          # approximate particle volume
        self.mass_p   = args.density * vol_particle
        self.rho0     = args.density

        print(f"[SPH] N={N}  h={self.h:.4f}  mass_p={self.mass_p:.4e}")
        print(f"      mu={self.mu:.1f}  lam={self.lam:.1f}")
        print(f"      model={self.model}  dt={self.dt}")

        # ── Taichi fields ─────────────────────────────────────────────────────
        # Reference (Lagrangian) positions — never change
        self.X   = ti.Vector.field(3, ti.f32, N)
        # Current positions, velocities, accelerations
        self.x   = ti.Vector.field(3, ti.f32, N)
        self.v   = ti.Vector.field(3, ti.f32, N)
        self.a   = ti.Vector.field(3, ti.f32, N)

        # Per-particle deformation gradient and Piola-Kirchhoff stress
        self.F   = ti.Matrix.field(3, 3, ti.f32, N)
        self.P   = ti.Matrix.field(3, 3, ti.f32, N)

        # Kernel correction matrix L_i and its inverse  (3×3 per particle)
        self.L    = ti.Matrix.field(3, 3, ti.f32, N)
        self.Linv = ti.Matrix.field(3, 3, ti.f32, N)

        # Reference density per particle (kernel sum)
        self.rho0_sph = ti.field(ti.f32, N)

        # Neighbor storage
        self.nb_count = ti.field(ti.i32, N)
        self.nb_list  = ti.field(ti.i32, (N, self.max_nb))

        # Material scalars
        self.mu_f  = ti.field(ti.f32, shape=())
        self.lam_f = ti.field(ti.f32, shape=())
        self.mu_f[None]  = float(self.mu)
        self.lam_f[None] = float(self.lam)

        # mass / rho0 scalar
        self.mass_f = ti.field(ti.f32, shape=())
        self.rho0_f = ti.field(ti.f32, shape=())
        self.mass_f[None] = float(self.mass_p)
        self.rho0_f[None] = float(self.rho0)

        # ── Initialise ────────────────────────────────────────────────────────
        self.X.from_numpy(positions.astype(np.float32))
        self.x.from_numpy(positions.astype(np.float32))
        self.v.fill(0.0)
        self.a.fill(0.0)

        # Identity deformation gradient
        I3 = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
        self.F.from_numpy(I3)

        # Build neighbor list from reference positions
        self._build_neighbor_list(positions)

        # Precompute correction matrices and reference SPH density
        self._precompute_correction()

    # ═══════════════════════════════════════════════════════════════════════
    # Kernel functions
    # ═══════════════════════════════════════════════════════════════════════
    @ti.func
    def _W(self, r: float) -> float:
        """Cubic spline kernel value. r = |X_ij|."""
        sigma = 8.0 / (math.pi * self.h**3)
        q = r * self.invh
        val = 0.0
        if q < 1.0:
            val = sigma * (1.0 - 1.5*q*q + 0.75*q*q*q)
        elif q < 2.0:
            t   = 2.0 - q
            val = sigma * 0.25 * t*t*t
        return val

    @ti.func
    def _dW_dr(self, r: float) -> float:
        """dW/dr (scalar). r = |X_ij|."""
        sigma = 8.0 / (math.pi * self.h**3)
        q  = r * self.invh
        dv = 0.0
        if q < 1.0:
            dv = sigma * (-3.0*q + 2.25*q*q) * self.invh
        elif q < 2.0:
            t  = 2.0 - q
            dv = sigma * (-0.75 * t*t) * self.invh
        return dv

    @ti.func
    def _gradW(self, Xi: ti.template(), Xj: ti.template()) -> ti.Vector:
        """
        Kernel gradient ∇_i W(Xi, Xj) evaluated at reference positions.
        ∇_i W = dW/dr · (Xi - Xj) / |Xi - Xj|
        """
        Xij = Xi - Xj
        r   = Xij.norm()
        res = ti.Vector([0.0, 0.0, 0.0])
    
        if r > 1e-12:
            res = self._dW_dr(r) * (Xij / r)
            
        return res

    # ═══════════════════════════════════════════════════════════════════════
    # Neighbor list (Python-side, builds once)
    # ═══════════════════════════════════════════════════════════════════════
    def _build_neighbor_list(self, positions: np.ndarray):
        """
        Build fixed neighbor list from reference positions using a spatial
        hash grid for O(N) performance.
        """
        N = self.N
        support = 2.0 * self.h       # kernel compact support radius

        # Cell grid
        bbox_min = positions.min(axis=0) - support
        bbox_max = positions.max(axis=0) + support
        cell_size = support

        def cell_idx(pt):
            return tuple(int((pt[k] - bbox_min[k]) / cell_size) for k in range(3))

        from collections import defaultdict
        grid = defaultdict(list)
        for i, pt in enumerate(positions):
            grid[cell_idx(pt)].append(i)

        nb_count_np = np.zeros(N, dtype=np.int32)
        nb_list_np  = np.full((N, self.max_nb), -1, dtype=np.int32)
        support2    = support * support

        for i, pi in enumerate(positions):
            ci = cell_idx(pi)
            count = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        nc = (ci[0]+dx, ci[1]+dy, ci[2]+dz)
                        for j in grid.get(nc, []):
                            if j == i:
                                continue
                            d2 = float(np.sum((pi - positions[j])**2))
                            if d2 < support2:
                                if count < self.max_nb:
                                    nb_list_np[i, count] = j
                                    count += 1
            nb_count_np[i] = count

        avg_nb = nb_count_np.mean()
        max_nb = nb_count_np.max()
        print(f"[SPH] Neighbor list built: avg={avg_nb:.1f}, max={max_nb}")
        if max_nb >= self.max_nb:
            print(f"  WARNING: some particles hit max_neighbors={self.max_nb}. "
                  f"Increase --max_neighbors.")

        self.nb_count.from_numpy(nb_count_np)
        self.nb_list.from_numpy(nb_list_np)

    # ═══════════════════════════════════════════════════════════════════════
    # Precompute correction matrices L_i and reference SPH density
    # ═══════════════════════════════════════════════════════════════════════
    @ti.kernel
    def _precompute_correction(self):
        """
        L_i = Σ_j (m/ρ₀) (X_j - X_i) ⊗ ∇W_ij0
        Linv_i = L_i⁻¹   (kernel gradient correction matrix)

        rho0_sph_i = Σ_j m · W(|X_ij|)  (SPH estimate of reference density)
        """
        m     = self.mass_f[None]
        rho0  = self.rho0_f[None]
        vol_j = m / rho0     # reference particle volume estimate

        for i in range(self.N):
            Xi  = self.X[i]
            L   = ti.Matrix.zero(ti.f32, 3, 3)
            rho = 0.0

            for k in range(self.nb_count[i]):
                j  = self.nb_list[i, k]
                Xj = self.X[j]
                Xij = Xj - Xi
                gW  = self._gradW(Xi, Xj)
                r   = Xij.norm()

                # L accumulation
                for a in ti.static(range(3)):
                    for b in ti.static(range(3)):
                        L[a, b] += vol_j * Xij[a] * gW[b]

                rho += m * self._W(r)

            self.rho0_sph[i] = rho

            # Invert L (with regularisation for degenerate cases)
            det = L.determinant()
            if ti.abs(det) > 1e-10:
                self.Linv[i] = L.inverse()
            else:
                self.Linv[i] = ti.Matrix.identity(ti.f32, 3)

            self.L[i] = L

    # ═══════════════════════════════════════════════════════════════════════
    # Deformation gradient computation
    # ═══════════════════════════════════════════════════════════════════════
    @ti.kernel
    def _compute_deformation_gradient(self):
        """
        F_i = [ Σ_j vol_j (x_j - x_i) ⊗ ∇W_ij0 ] · L_i⁻¹

        where ∇W_ij0 is evaluated at reference positions X.
        """
        m    = self.mass_f[None]
        rho0 = self.rho0_f[None]
        vol_j = m / rho0

        for i in range(self.N):
            xi    = self.x[i]
            Xi    = self.X[i]
            A     = ti.Matrix.zero(ti.f32, 3, 3)

            for k in range(self.nb_count[i]):
                j  = self.nb_list[i, k]
                xj = self.x[j]
                Xj = self.X[j]
                xij = xj - xi                        # current relative position
                gW0 = self._gradW(Xi, Xj)            # gradient at reference

                for a in ti.static(range(3)):
                    for b in ti.static(range(3)):
                        A[a, b] += vol_j * xij[a] * gW0[b]

            # Apply kernel gradient correction
            self.F[i] = A @ self.Linv[i]

    # ═══════════════════════════════════════════════════════════════════════
    # 1st Piola-Kirchhoff stress  P(F)
    # ═══════════════════════════════════════════════════════════════════════
    @ti.func
    def _pk1_stvk(self, F: ti.template()) -> ti.Matrix:
        mu  = self.mu_f[None]; lam = self.lam_f[None]
        I   = ti.Matrix.identity(ti.f32, 3)
        E   = (F.transpose() @ F - I) * 0.5
        return F @ (2.0*mu*E + lam*E.trace()*I)

    @ti.func
    def _pk1_corotated(self, F: ti.template()) -> ti.Matrix:
        mu  = self.mu_f[None]; lam = self.lam_f[None]
        U, sig, V = ti.svd(F, ti.f32)
        R   = U @ V.transpose()
        J   = F.determinant()
        Ft  = F.inverse().transpose()
        return 2.0*mu*(F - R) + lam*(J-1.0)*J*Ft

    @ti.func
    def _pk1_neohookean(self, F: ti.template()) -> ti.Matrix:
        mu  = self.mu_f[None]; lam = self.lam_f[None]
        J   = F.determinant()
        Ft  = F.inverse().transpose()
        return mu*(F - Ft) + lam*(J-1.0)*J*Ft

    # ═══════════════════════════════════════════════════════════════════════
    # Compute per-particle stress P
    # ═══════════════════════════════════════════════════════════════════════
    @ti.kernel
    def _compute_stress(self, model_id: int):
        for i in range(self.N):
            F_ = self.F[i]
            P  = ti.Matrix.zero(ti.f32, 3, 3)
            if   model_id == 0: P = self._pk1_stvk(F_)
            elif model_id == 1: P = self._pk1_corotated(F_)
            else:               P = self._pk1_neohookean(F_)
            self.P[i] = P

    # ═══════════════════════════════════════════════════════════════════════
    # Elastic + artificial viscosity forces → acceleration
    # ═══════════════════════════════════════════════════════════════════════
    @ti.kernel
    def _compute_acceleration(self):
        """
        SPH momentum equation (Total-Lagrangian, symmetric form):

            a_i = (1/rho0_i) Σ_j m_j (P_i·Linv_i + P_j·Linv_j) · ∇W_ij0
                + art_visc term
                + gravity

        The symmetric form exactly conserves linear momentum.

        Artificial viscosity (Monaghan 1992):
            Π_ij = -α h c_s μ_ij / (ρ_ij)
            μ_ij = v_ij·X_ij / (|X_ij|² + 0.01 h²)  (when v·X < 0)
            c_s  ≈ sqrt(E/ρ₀)   (SPH speed of sound estimate)
        """
        m      = self.mass_f[None]
        rho0   = self.rho0_f[None]
        alpha  = self.art_visc
        # Estimate wave speed from Young's modulus
        E_eff  = self.mu_f[None] * (3.0*self.lam_f[None] + 2.0*self.mu_f[None]) / \
                 (self.lam_f[None] + self.mu_f[None])   # Young's modulus from Lamé
        cs     = ti.sqrt(E_eff / rho0)

        for i in range(self.N):
            ai    = ti.Vector([0.0, self.gravity, 0.0])   # gravity
            Pi    = self.P[i]
            Li_inv = self.Linv[i]
            # Corrected stress: P_i · L_i^{-T}
            PLi   = Pi @ Li_inv.transpose()
            rho0i = rho0

            for k in range(self.nb_count[i]):
                j    = self.nb_list[i, k]
                Xi   = self.X[i]; Xj = self.X[j]
                xi   = self.x[i]; xj = self.x[j]
                vi   = self.v[i]; vj = self.v[j]

                gW0  = self._gradW(Xi, Xj)   # ∇_i W (at reference positions)
                Xij  = Xi - Xj               # reference relative position

                # Symmetric elastic stress contribution
                Pj   = self.P[j]
                Lj_inv = self.Linv[j]
                PLj  = Pj @ Lj_inv.transpose()

                # SPH force density: (P_i·Li_inv + P_j·Lj_inv) · ∇W0
                stress_term = (PLi + PLj) @ gW0
                ai += (m / rho0i) * stress_term

                # Artificial viscosity
                vij  = vi - vj
                Xij_norm2 = Xij.dot(Xij)
                v_dot_x   = vij.dot(Xij)

                if v_dot_x < 0.0:
                    mu_ij   = self.h * v_dot_x / (Xij_norm2 + 0.01*self.h2)
                    rho_avg = rho0      # reference density
                    Pi_visc = -alpha * cs * mu_ij / rho_avg
                    ai     += -m * Pi_visc * gW0

            self.a[i] = ai

    # ═══════════════════════════════════════════════════════════════════════
    # Velocity-Verlet integration + floor collision
    # ═══════════════════════════════════════════════════════════════════════
    @ti.kernel
    def _integrate(self, dt: float, damping: float):
        for i in range(self.N):
            self.v[i] = (self.v[i] + dt * self.a[i]) * damping
            self.x[i] = self.x[i] + dt * self.v[i]

            # Floor constraint  y >= 0
            if self.x[i][1] < 0.0:
                self.x[i][1] = 0.0
                if self.v[i][1] < 0.0:
                    self.v[i][1] *= -0.3

    # ═══════════════════════════════════════════════════════════════════════
    # Public step
    # ═══════════════════════════════════════════════════════════════════════
    def step(self, model_id: int):
        self._compute_deformation_gradient()
        self._compute_stress(model_id)
        self._compute_acceleration()
        self._integrate(self.dt, self.damping)

    def get_positions(self) -> np.ndarray:
        return self.x.to_numpy()

    # ═══════════════════════════════════════════════════════════════════════
    # GUI buffer (particle positions for ti.ui.Scene.particles)
    # ═══════════════════════════════════════════════════════════════════════
    def get_particle_radius(self) -> float:
        """Visual radius ≈ half the smoothing length."""
        return float(self.h) * 0.25


# ─────────────────────────────────────────────────────────────────────────────
# GUI runner
# ─────────────────────────────────────────────────────────────────────────────
def run_gui(sim: SPHElasticSimulator, model_id: int, args):
    window = ti.ui.Window(
        name=f"SPH Elastic — {args.model.upper()}",
        res=(1024, 768), vsync=True,
    )
    canvas = window.get_canvas()
    scene  = ti.ui.Scene()
    camera = ti.ui.Camera()

    # Auto-frame: estimate bounding box from initial positions
    pos0   = sim.get_positions()
    centre = pos0.mean(axis=0)
    radius = float(np.linalg.norm(pos0 - centre, axis=1).max())
    cam_dist = radius * 3.5

    camera.position(centre[0], centre[1] + radius, centre[2] + cam_dist)
    camera.lookat(centre[0], centre[1], centre[2])
    camera.up(0.0, 1.0, 0.0)

    p_radius = sim.get_particle_radius()

    import time
    step = 0; t0 = time.time()

    while window.running:
        for _ in range(args.substeps):
            sim.step(model_id)
            step += 1
            if step >= args.steps:
                break

        camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.LMB)
        scene.set_camera(camera)
        scene.ambient_light(color=(0.4, 0.4, 0.4))
        scene.point_light(pos=(centre[0]+3, centre[1]+5, centre[2]+3), color=(1.0, 1.0, 1.0))
        scene.point_light(pos=(centre[0]-3, centre[1]+2, centre[2]-2), color=(0.3, 0.4, 0.6))

        # Draw particles
        scene.particles(sim.x, radius=p_radius, color=(0.3, 0.65, 1.0))

        canvas.scene(scene)
        window.show()

        elapsed = time.time() - t0
        sps = step / elapsed if elapsed > 0 else 0.0
        window.GUI.begin("Info", 0.02, 0.02, 0.38, 0.18)
        window.GUI.text(f"Step    : {step} / {args.steps}")
        window.GUI.text(f"SPS     : {sps:.1f} sim-steps/s")
        window.GUI.text(f"Model   : {args.model.upper()}")
        window.GUI.text(f"Method  : Total-Lagrangian SPH")
        window.GUI.text(f"Particles: {sim.N}")
        window.GUI.end()

        if step >= args.steps:
            print(f"[GUI] Done after {step} steps.")
            while window.running:
                window.show()
            break


# ─────────────────────────────────────────────────────────────────────────────
# Headless export runner
# ─────────────────────────────────────────────────────────────────────────────
def run_export(sim: SPHElasticSimulator, model_id: int, args):
    import time
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[Export] PLY every {args.export_every} step(s) → '{args.output_dir}/'")
    t0 = time.time(); frame = 0

    for step in range(args.steps):
        sim.step(model_id)

        if step % args.export_every == 0:
            pos = sim.get_positions()
            export_ply(pos, frame, args.output_dir)
            elapsed = time.time() - t0
            cy = pos[:, 1].mean()
            print(f"  Step {step+1:6d}/{args.steps} | frame {frame:06d} | "
                  f"avg_y={cy:.4f} | elapsed={elapsed:.2f}s")
            frame += 1

    print(f"[Done] {frame} frames saved to '{args.output_dir}/'.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 1. Load or generate particle cloud
    if args.ply is not None:
        positions = load_ply_points(args.ply)
        spacing   = estimate_particle_spacing(positions)
        print(f"[PLY] Estimated particle spacing: {spacing:.4f}")
    else:
        print("[Particles] No PLY provided — generating default unit sphere.")
        positions, spacing = sample_sphere_particles(
            args.sphere_radius, args.sphere_particles
        )

    # 2. Build SPH simulator
    model_map = {"stvk": 0, "corotated": 1, "neohookean": 2}
    model_id  = model_map[args.model]

    sim = SPHElasticSimulator(positions, args, spacing)

    # 3. Run
    if args.gui:
        print(f"[GUI] SPH window — {args.substeps} substep(s)/frame  (ESC to quit)")
        run_gui(sim, model_id, args)
    else:
        run_export(sim, model_id, args)


if __name__ == "__main__":
    main()
