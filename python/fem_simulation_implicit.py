"""
FEM Simulation — Implicit Euler with Taichi
============================================
Supports StVK, Co-rotated, and Neo-Hookean energy models.

Integrator
----------
Backward (implicit) Euler:

    M (v_{n+1} - v_n) / dt = f_int(x_{n+1}) + f_ext
    x_{n+1} = x_n + dt * v_{n+1}

Substituting x_{n+1} = x_n + dt*v_{n+1} into the residual and linearising
around the predictor x_pred = x_n + dt*v_n yields the Newton system:

    (M/dt² - K(x_pred)) Δx = r(x_pred)

where K = ∂f_int/∂x is the tangent stiffness assembled analytically per element.

The linear system is solved with a matrix-free Projected Conjugate Gradient (PCG)
whose matrix-vector product Kx is computed by scatter-adding per-element
contributions, exactly matching a sparse global assembly but staying on-GPU.

Stiffness matrices (dP/dF per model)
-------------------------------------
  StVK        : dP/dF derived from C = FᵀF, E = (C-I)/2
  Co-rotated  : dP/dF via polar SVD, linearised rotation derivative
  Neo-Hookean : dP/dF = μI⊗I + (2μ-λ(J-1))F^{-T}⊗F^{-T}
                        + λJ(J-1) G  (G = ∂F^{-T}/∂F contraction)

Each 12×12 element stiffness Ke is scattered into the global residual via
the same H = -vol·P·Dm_invᵀ pattern used for forces, avoiding any global
sparse matrix storage.
"""

import argparse
import os
import numpy as np
import taichi as ti

# ─────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="FEM Simulation — Implicit Euler with Taichi"
    )
    parser.add_argument("--mesh", type=str, default=None,
                        help="Path to input mesh (.obj/.stl/.ply). "
                             "Omit to use default unit sphere.")
    parser.add_argument("--sphere_radius", type=float, default=1.0,
                        help="Radius of default sphere (default: 1.0)")
    parser.add_argument("--sphere_subdiv", type=int, default=3,
                        help="Subdivision level for default sphere (default: 3)")
    parser.add_argument("--model", type=str, default="corotated",
                        choices=["stvk", "corotated", "neohookean"],
                        help="Elastic energy model (default: corotated)")
    parser.add_argument("--E", type=float, default=1e5,
                        help="Young's modulus (default: 1e5)")
    parser.add_argument("--nu", type=float, default=0.45,
                        help="Poisson's ratio (default: 0.45)")
    parser.add_argument("--dt", type=float, default=1e-2,
                        help="Time step — implicit allows much larger dt (default: 1e-2)")
    parser.add_argument("--density", type=float, default=1000.0,
                        help="Material density (default: 1000.0)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Total simulation steps (default: 500)")
    parser.add_argument("--gravity", type=float, default=-9.8,
                        help="Gravity in Y direction (default: -9.8)")
    parser.add_argument("--newton_iter", type=int, default=5,
                        help="Newton iterations per time step (default: 5)")
    parser.add_argument("--cg_iter", type=int, default=100,
                        help="Max PCG iterations per Newton step (default: 100)")
    parser.add_argument("--cg_tol", type=float, default=1e-5,
                        help="PCG residual tolerance (default: 1e-5)")
    parser.add_argument("--rayleigh_alpha", type=float, default=0.0,
                        help="Rayleigh mass damping coefficient α (default: 0)")
    parser.add_argument("--rayleigh_beta", type=float, default=0.01,
                        help="Rayleigh stiffness damping coefficient β (default: 0.01)")
    parser.add_argument("--export_every", type=int, default=1,
                        help="Export OBJ every N steps (default: 1)")
    parser.add_argument("--output_dir", type=str, default="output_implicit",
                        help="Output directory for OBJ frames (default: output_implicit)")
    parser.add_argument("--gui", action="store_true",
                        help="Real-time visualisation via ti.ui.Window")
    parser.add_argument("--substeps", type=int, default=1,
                        help="Sim substeps per GUI frame (default: 1)")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Mesh helpers  (identical to explicit version)
# ─────────────────────────────────────────────
def load_mesh(path: str):
    try:
        import trimesh
    except ImportError:
        raise ImportError("pip install trimesh")
    mesh = trimesh.load(path, force="mesh")
    if mesh is None or len(mesh.vertices) == 0:
        raise ValueError(f"Failed to load mesh from: {path}")
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces    = np.array(mesh.faces,    dtype=np.int32)
    print(f"[Mesh] Loaded: {len(vertices)} vertices, {len(faces)} faces")
    return vertices, faces


def generate_sphere(radius: float = 1.0, subdivisions: int = 3):
    phi   = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  phi, 0],[ 1,  phi, 0],[-1, -phi, 0],[ 1, -phi, 0],
        [ 0, -1,  phi],[ 0,  1,  phi],[ 0, -1, -phi],[ 0,  1, -phi],
        [ phi, 0, -1],[ phi, 0,  1],[-phi, 0, -1],[-phi, 0,  1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts[0])
    tris = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=np.int32)
    for _ in range(subdivisions):
        new_tris = []; cache = {}
        def mid(a, b):
            k = (min(a,b), max(a,b))
            if k in cache: return cache[k]
            nonlocal verts
            m = (verts[a]+verts[b])*0.5; m /= np.linalg.norm(m)
            idx = len(verts); verts = np.vstack([verts, m]); cache[k] = idx; return idx
        for t in tris:
            a,b,c = int(t[0]),int(t[1]),int(t[2])
            ab,bc,ca = mid(a,b),mid(b,c),mid(c,a)
            new_tris.extend([[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]])
        tris = np.array(new_tris, dtype=np.int32)
    verts *= radius
    verts[:,1] += radius + 0.01
    print(f"[Sphere] {len(verts)} verts, {len(tris)} faces (r={radius}, sdiv={subdivisions})")
    return verts, tris


def tetrahedralize(vertices, faces):
    try:
        import tetgen
        tet = tetgen.TetGen(vertices, faces)
        nodes, elems = tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
        nodes = np.array(nodes, dtype=np.float64)
        elems = np.array(elems, dtype=np.int32)
        print(f"[TetGen] {len(nodes)} nodes, {len(elems)} tets")
        return nodes, elems
    except ImportError:
        pass
    try:
        import ptetgen
        r = ptetgen.tetrahedralize("pq1.414a0.1", vertices, faces)
        nodes = np.array(r.points, dtype=np.float64)
        elems = np.array(r.cells_dict.get("tetra",[]), dtype=np.int32)
        print(f"[ptetgen] {len(nodes)} nodes, {len(elems)} tets")
        return nodes, elems
    except ImportError:
        pass
    raise ImportError("Install tetgen: pip install tetgen")


def extract_surface(elems: np.ndarray):
    from collections import defaultdict
    local_faces = [(1,2,3),(0,2,3),(0,1,3),(0,1,2)]
    oriented    = [(1,3,2),(0,2,3),(0,3,1),(0,1,2)]
    face_count  = defaultdict(list)
    for tet in elems:
        for (li,lj,lk),(oi,oj,ok) in zip(local_faces, oriented):
            vi,vj,vk = int(tet[li]),int(tet[lj]),int(tet[lk])
            key = tuple(sorted((vi,vj,vk)))
            face_count[key].append((tet[oi],tet[oj],tet[ok]))
    surf_faces = [f[0] for f in face_count.values() if len(f)==1]
    surf_faces = np.array(surf_faces, dtype=np.int32)
    print(f"[Surface] {len(surf_faces)} triangles from {len(elems)} tets")
    return surf_faces


def build_surface_index_map(surf_faces, n_total):
    unique = np.unique(surf_faces)
    g2l    = np.full(n_total, -1, dtype=np.int32)
    g2l[unique] = np.arange(len(unique), dtype=np.int32)
    return unique, g2l[surf_faces]


def export_surface_obj(positions, surf_vert_indices, surf_faces_local, frame, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"frame_{frame:06d}.obj")
    sv = positions[surf_vert_indices]
    with open(path, "w") as f:
        f.write(f"# FEM implicit frame {frame}\n")
        for v in sv:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for tri in surf_faces_local:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


# ─────────────────────────────────────────────
# Taichi init
# ─────────────────────────────────────────────
ti.init(arch=ti.gpu, default_fp=ti.f32)


# ─────────────────────────────────────────────
# Implicit FEM Simulator
# ─────────────────────────────────────────────
@ti.data_oriented
class ImplicitFEMSimulator:
    """
    Backward Euler FEM simulator.

    Newton iteration per step
    -------------------------
    Given x_n, v_n, compute x_{n+1} via Newton iterations on the residual:

        g(x) = M(x - x_pred)/dt² - f_int(x) - f_ext = 0
        x_pred = x_n + dt*v_n   (inertia predictor)

    Linearise: g(x + Δx) ≈ g(x) + [M/dt² - K(x)] Δx = 0
    Solve with PCG:  A Δx = -g(x),   A = M/dt² - K

    The matrix-vector product A·p is computed on-GPU without assembling A.

    Rayleigh damping
    ----------------
    Augment system matrix:  A = M(1/dt² + α/dt) + K(1 + β/dt)
    Augment RHS with velocity-proportional damping term.
    """

    def __init__(self, nodes, elems, args):
        self.n_verts = len(nodes)
        self.n_tets  = len(elems)
        self.dt      = args.dt
        self.gravity = args.gravity
        self.model   = args.model

        # Lamé
        E, nu = args.E, args.nu
        self.mu  = E / (2.0*(1.0+nu))
        self.lam = E*nu / ((1.0+nu)*(1.0-2.0*nu))
        self.density       = args.density
        self.newton_iter   = args.newton_iter
        self.cg_iter       = args.cg_iter
        self.cg_tol        = args.cg_tol
        self.rayleigh_alpha = args.rayleigh_alpha
        self.rayleigh_beta  = args.rayleigh_beta

        print(f"[Implicit FEM] model={self.model}  mu={self.mu:.1f}  lam={self.lam:.1f}")
        print(f"  dt={self.dt}  newton={self.newton_iter}  cg_max={self.cg_iter}")

        # ── Primary fields ────────────────────────────────────────────────────
        self.x      = ti.Vector.field(3, ti.f32, self.n_verts)
        self.v      = ti.Vector.field(3, ti.f32, self.n_verts)
        self.mass   = ti.field(ti.f32, self.n_verts)

        # ── Tet geometry ──────────────────────────────────────────────────────
        self.tets   = ti.Vector.field(4, ti.i32, self.n_tets)
        self.Dm_inv = ti.Matrix.field(3, 3, ti.f32, self.n_tets)
        self.vol    = ti.field(ti.f32, self.n_tets)

        # ── Material scalars ──────────────────────────────────────────────────
        self.mu_f  = ti.field(ti.f32, shape=())
        self.lam_f = ti.field(ti.f32, shape=())
        self.mu_f[None]  = float(self.mu)
        self.lam_f[None] = float(self.lam)

        # ── Newton/PCG work fields ────────────────────────────────────────────
        # x_pred = x_n + dt*v_n
        self.x_pred = ti.Vector.field(3, ti.f32, self.n_verts)
        # elastic + gravity forces at current Newton iterate
        self.f_int  = ti.Vector.field(3, ti.f32, self.n_verts)
        # Newton residual  g = M(x - x_pred)/dt² - f_int - f_ext
        self.residual = ti.Vector.field(3, ti.f32, self.n_verts)
        # PCG vectors (all in "flat" 3N sense stored as VectorFields)
        self.p  = ti.Vector.field(3, ti.f32, self.n_verts)   # search direction
        self.Ap = ti.Vector.field(3, ti.f32, self.n_verts)   # A*p
        self.dx = ti.Vector.field(3, ti.f32, self.n_verts)   # solution Δx
        self.r  = ti.Vector.field(3, ti.f32, self.n_verts)   # CG residual
        self.z  = ti.Vector.field(3, ti.f32, self.n_verts)   # precond residual
        # Diagonal Jacobi preconditioner: diag(M/dt² - K)
        self.precond = ti.Vector.field(3, ti.f32, self.n_verts)

        # Scalar accumulators for dot products (atomics)
        self.dot_rz   = ti.field(ti.f32, shape=())
        self.dot_pAp  = ti.field(ti.f32, shape=())
        self.dot_rz_new = ti.field(ti.f32, shape=())

        # GUI buffers (lazily allocated)
        self.gui_verts   = None
        self.gui_indices = None

        # ── Initialise from numpy ─────────────────────────────────────────────
        self.x.from_numpy(nodes.astype(np.float32))
        self.v.fill(0.0)
        self.mass.fill(0.0)
        self.tets.from_numpy(elems.astype(np.int32))
        self._precompute()

    # ═══════════════════════════════════════════════════════════════════════════
    # Precompute
    # ═══════════════════════════════════════════════════════════════════════════
    @ti.kernel
    def _precompute(self):
        for i in range(self.n_tets):
            a,b,c,d = self.tets[i][0],self.tets[i][1],self.tets[i][2],self.tets[i][3]
            Dm  = ti.Matrix.cols([self.x[b]-self.x[a],
                                  self.x[c]-self.x[a],
                                  self.x[d]-self.x[a]])
            vol = ti.abs(Dm.determinant()) / 6.0
            self.vol[i]    = vol
            self.Dm_inv[i] = Dm.inverse()
            nm = self.density * vol / 4.0
            self.mass[a] += nm; self.mass[b] += nm
            self.mass[c] += nm; self.mass[d] += nm

    # ═══════════════════════════════════════════════════════════════════════════
    # Deformation gradient
    # ═══════════════════════════════════════════════════════════════════════════
    @ti.func
    def _F(self, i: int) -> ti.Matrix:
        a,b,c,d = self.tets[i][0],self.tets[i][1],self.tets[i][2],self.tets[i][3]
        Ds = ti.Matrix.cols([self.x[b]-self.x[a],
                              self.x[c]-self.x[a],
                              self.x[d]-self.x[a]])
        return Ds @ self.Dm_inv[i]

    # ═══════════════════════════════════════════════════════════════════════════
    # 1st PK stress  P(F)  — same as explicit version
    # ═══════════════════════════════════════════════════════════════════════════
    @ti.func
    def _pk1_stvk(self, F: ti.template()) -> ti.Matrix:
        mu  = self.mu_f[None]; lam = self.lam_f[None]
        I   = ti.Matrix.identity(ti.f32, 3)
        E   = (F.transpose()@F - I)*0.5
        return F @ (2.0*mu*E + lam*E.trace()*I)

    @ti.func
    def _pk1_corotated(self, F: ti.template()) -> ti.Matrix:
        mu  = self.mu_f[None]; lam = self.lam_f[None]
        U, sig, V = ti.svd(F, ti.f32)
        R  = U @ V.transpose()
        J  = F.determinant()
        Ft = F.inverse().transpose()
        return 2.0*mu*(F-R) + lam*(J-1.0)*J*Ft

    @ti.func
    def _pk1_neohookean(self, F: ti.template()) -> ti.Matrix:
        mu  = self.mu_f[None]; lam = self.lam_f[None]
        J   = F.determinant()
        Ft  = F.inverse().transpose()
        return mu*(F - Ft) + lam*(J-1.0)*J*Ft

    # ═══════════════════════════════════════════════════════════════════════════
    # dP/dF·dF  (analytic, column-wise, needed for stiffness matvec)
    # ═══════════════════════════════════════════════════════════════════════════
    @ti.func
    def _dpk1_stvk(self, F: ti.template(), dF: ti.template()) -> ti.Matrix:
        """Directional derivative of P_StVK in direction dF."""
        mu  = self.mu_f[None]; lam = self.lam_f[None]
        I   = ti.Matrix.identity(ti.f32, 3)
        E   = (F.transpose()@F - I)*0.5
        dE  = (F.transpose()@dF + dF.transpose()@F)*0.5
        dS  = 2.0*mu*dE + lam*dE.trace()*I
        return dF @ (2.0*mu*E + lam*E.trace()*I) + F @ dS

    @ti.func
    def _dpk1_corotated(self, F: ti.template(), dF: ti.template()) -> ti.Matrix:
        """
        Linearised co-rotated:  dP = 2μ(dF - dR) + λ[(J F^{-T}:dF)(2J-1)F^{-T}
                                                     + J(J-1)dF^{-T}]
        dR is linearised via SVD (Müller et al. approach):
            dR = R * skew( R^T dF S^{-1} + S^{-1} (R^T dF)^T )
        where S = V diag(sig) V^T  (symmetric part of polar decomp)
        """
        mu  = self.mu_f[None]; lam = self.lam_f[None]
        U, sig, V = ti.svd(F, ti.f32)
        R  = U @ V.transpose()
        S  = V @ ti.Matrix([
            [sig[0,0], 0, 0],
            [0, sig[1,1], 0],
            [0, 0, sig[2,2]]], dt=ti.f32) @ V.transpose()

        # Linearised rotation: solve Ax = b  (3x3 skew system)
        # dR = R * skew(w),  w solved from the Givens system
        # Use the closed-form from Barbic 2012 (symmetric case)
        RtdF = R.transpose() @ dF
        # Skew part  W = (RtdF @ S_inv - S_inv @ RtdF^T) / 2  — antisymmetric
        S_inv = V @ ti.Matrix([
            [1.0/sig[0,0], 0, 0],
            [0, 1.0/sig[1,1], 0],
            [0, 0, 1.0/sig[2,2]]], dt=ti.f32) @ V.transpose()
        W = (RtdF @ S_inv - S_inv @ RtdF.transpose()) * 0.5
        # dR = R * W  (antisymmetric W acting as infinitesimal rotation)
        dR = R @ W

        J   = F.determinant()
        Ft  = F.inverse().transpose()
        # d(F^{-T})/dF · dF  = -F^{-T} dF^T F^{-T}
        dFt = -Ft @ dF.transpose() @ Ft

        # Jacobian derivative: dJ = J * tr(F^{-1} dF) = J*(Ft:dF)
        dJ  = J * (Ft * dF).sum()

        dP  = (2.0*mu*(dF - dR)
               + lam*((2.0*J-1.0)*dJ*Ft + J*(J-1.0)*dFt))
        return dP

    @ti.func
    def _dpk1_neohookean(self, F: ti.template(), dF: ti.template()) -> ti.Matrix:
        """dP/dF · dF for stable Neo-Hookean."""
        mu  = self.mu_f[None]; lam = self.lam_f[None]
        J   = F.determinant()
        Ft  = F.inverse().transpose()
        dJ  = J * (Ft * dF).sum()
        dFt = -Ft @ dF.transpose() @ Ft
        dP  = (mu*dF
               + (-mu + lam*(J-1.0))*dFt*J + lam*(J-1.0)*dJ*Ft
               + lam*J*dJ*Ft + lam*J*(J-1.0)*dFt)
        # Simplified:
        dP  = (mu*dF
               + lam*(2.0*J-1.0)*dJ*Ft
               + (lam*(J-1.0)*J - mu)*dFt)
        return dP

    # ═══════════════════════════════════════════════════════════════════════════
    # Assemble internal forces f_int  (scatter from per-tet H)
    # ═══════════════════════════════════════════════════════════════════════════
    @ti.kernel
    def _compute_f_int(self, model_id: int):
        for i in range(self.n_verts):
            self.f_int[i] = ti.Vector([0.0, self.gravity*self.mass[i], 0.0])

        for i in range(self.n_tets):
            a,b,c,d = self.tets[i][0],self.tets[i][1],self.tets[i][2],self.tets[i][3]
            F_ = self._F(i)

            P = ti.Matrix.zero(ti.f32, 3, 3)
            if   model_id == 0: P = self._pk1_stvk(F_)
            elif model_id == 1: P = self._pk1_corotated(F_)
            else:               P = self._pk1_neohookean(F_)

            H  = -self.vol[i] * P @ self.Dm_inv[i].transpose()
            fb = ti.Vector([H[0,0],H[1,0],H[2,0]])
            fc = ti.Vector([H[0,1],H[1,1],H[2,1]])
            fd = ti.Vector([H[0,2],H[1,2],H[2,2]])
            fa = -(fb+fc+fd)
            self.f_int[a] += fa; self.f_int[b] += fb
            self.f_int[c] += fc; self.f_int[d] += fd

    # ═══════════════════════════════════════════════════════════════════════════
    # Newton residual:  residual = M(x-x_pred)/dt² - f_int
    # (gravity already included in f_int)
    # ═══════════════════════════════════════════════════════════════════════════
    @ti.kernel
    def _compute_residual(self, dt: float, ra: float):
        """
        g = M(x - x_pred)/dt² - f_int
        Rayleigh mass damping adds M·α·v/dt to the inertia term.
        """
        idt2 = 1.0 / (dt*dt)
        for i in range(self.n_verts):
            m = self.mass[i]
            inertia = m * (self.x[i] - self.x_pred[i]) * idt2
            # Rayleigh mass damping contribution to residual
            # approximate v_{n+1} ≈ (x - x_pred)/dt  (first-order)
            v_new = (self.x[i] - self.x_pred[i]) / dt
            damp  = m * ra * v_new
            self.residual[i] = inertia + damp - self.f_int[i]

    # ═══════════════════════════════════════════════════════════════════════════
    # Matrix-free  A·p  where  A = M(1/dt² + α/dt) - K(1 + β/dt)
    #
    # The stiffness contribution K·p is assembled exactly as forces but using
    # the directional derivative dP/dF in place of P, with dF driven by dp.
    # ═══════════════════════════════════════════════════════════════════════════
    @ti.kernel
    def _matvec(self, model_id: int, dt: float, ra: float, rb: float):
        """
        Ap = A · p  =  [M(1/dt²+α/dt) - (1+β/dt)·K] · p

        Sign convention: A = M_eff/dt² + stiffness  but K appears with -1
        because  A Δx = -g  and  g = M(x-x_pred)/dt² - f_int
        so the stiffness term is   -∂f_int/∂x = K  (positive semi-definite).
        """
        idt2   = 1.0/(dt*dt)
        idt    = 1.0/dt
        k_scale = 1.0 + rb*idt    # stiffness scaled by Rayleigh β

        # Inertia + Rayleigh-mass part
        for i in range(self.n_verts):
            m = self.mass[i]
            self.Ap[i] = m*(idt2 + ra*idt) * self.p[i]

        # Stiffness part: scatter dP·dF contributions
        for i in range(self.n_tets):
            a,b,c,d = self.tets[i][0],self.tets[i][1],self.tets[i][2],self.tets[i][3]
            F_ = self._F(i)
            Di = self.Dm_inv[i]

            # Build dDs from search direction p (same shape as Ds)
            dDs = ti.Matrix.cols([self.p[b]-self.p[a],
                                   self.p[c]-self.p[a],
                                   self.p[d]-self.p[a]])
            dF_ = dDs @ Di

            dP = ti.Matrix.zero(ti.f32, 3, 3)
            if   model_id == 0: dP = self._dpk1_stvk(F_, dF_)
            elif model_id == 1: dP = self._dpk1_corotated(F_, dF_)
            else:               dP = self._dpk1_neohookean(F_, dF_)

            # dH = -vol * dP * Dm_inv^T   (same scatter as forces)
            dH  = -self.vol[i] * k_scale * dP @ Di.transpose()
            dfb = ti.Vector([dH[0,0],dH[1,0],dH[2,0]])
            dfc = ti.Vector([dH[0,1],dH[1,1],dH[2,1]])
            dfd = ti.Vector([dH[0,2],dH[1,2],dH[2,2]])
            dfa = -(dfb+dfc+dfd)

            # Subtract  K·p  from Ap  (A = M_eff - K, so Ap += M·p - K·p)
            self.Ap[a] -= dfa; self.Ap[b] -= dfb
            self.Ap[c] -= dfc; self.Ap[d] -= dfd

    # ═══════════════════════════════════════════════════════════════════════════
    # Diagonal Jacobi preconditioner:  diag(M/dt² - diag(K))
    # We approximate diag(K) from the per-tet contributions using a
    # "diagonal extraction" pass (inexpensive).
    # ═══════════════════════════════════════════════════════════════════════════
    @ti.kernel
    def _build_precond(self, model_id: int, dt: float, ra: float, rb: float):
        idt2   = 1.0/(dt*dt); idt = 1.0/dt
        k_scale = 1.0 + rb*idt
        for i in range(self.n_verts):
            m = self.mass[i]
            self.precond[i] = ti.Vector([m*(idt2+ra*idt), m*(idt2+ra*idt), m*(idt2+ra*idt)])

        for i in range(self.n_tets):
            a,b,c,d = self.tets[i][0],self.tets[i][1],self.tets[i][2],self.tets[i][3]
            F_ = self._F(i)
            Di = self.Dm_inv[i]
            # Extract diagonal via 9 unit dF probes would be expensive;
            # use a scalar stiffness bound: k_diag ≈ vol * (2μ+λ) * |Dm_inv|²
            mu  = self.mu_f[None]; lam = self.lam_f[None]
            stiff = self.vol[i] * k_scale * (2.0*mu + lam)
            Di_norm2 = (Di*Di).sum()
            contrib = stiff * Di_norm2 / 4.0   # divided among 4 nodes
            for ci in ti.static(range(3)):
                self.precond[a][ci] += contrib
                self.precond[b][ci] += contrib
                self.precond[c][ci] += contrib
                self.precond[d][ci] += contrib

    # ═══════════════════════════════════════════════════════════════════════════
    # PCG helper kernels
    # ═══════════════════════════════════════════════════════════════════════════
    @ti.kernel
    def _dot_rz(self):
        s = 0.0
        for i in range(self.n_verts):
            s += self.r[i].dot(self.z[i])
        self.dot_rz[None] = s

    @ti.kernel
    def _dot_pAp(self):
        s = 0.0
        for i in range(self.n_verts):
            s += self.p[i].dot(self.Ap[i])
        self.dot_pAp[None] = s

    @ti.kernel
    def _dot_rz_new_kernel(self):
        s = 0.0
        for i in range(self.n_verts):
            s += self.r[i].dot(self.z[i])
        self.dot_rz_new[None] = s

    @ti.kernel
    def _apply_precond(self):
        for i in range(self.n_verts):
            for ci in ti.static(range(3)):
                p = self.precond[i][ci]
                self.z[i][ci] = self.r[i][ci] / (p if p > 1e-12 else 1.0)

    @ti.kernel
    def _init_cg(self):
        """dx=0, r=-residual, z=precond(r), p=z"""
        for i in range(self.n_verts):
            self.dx[i] = ti.Vector([0.0,0.0,0.0])
            self.r[i]  = -self.residual[i]

    @ti.kernel
    def _update_dx_r(self, alpha: float):
        for i in range(self.n_verts):
            self.dx[i] += alpha * self.p[i]
            self.r[i]  -= alpha * self.Ap[i]

    @ti.kernel
    def _update_p(self, beta: float):
        for i in range(self.n_verts):
            self.p[i] = self.z[i] + beta * self.p[i]

    @ti.kernel
    def _r_norm2(self) -> float:
        s = 0.0
        for i in range(self.n_verts):
            s += self.r[i].dot(self.r[i])
        return s

    # ═══════════════════════════════════════════════════════════════════════════
    # Apply solution and floor collision
    # ═══════════════════════════════════════════════════════════════════════════
    @ti.kernel
    def _apply_dx(self, dt: float):
        for i in range(self.n_verts):
            self.x[i] += self.dx[i]
            # Update velocity
            self.v[i] = (self.x[i] - self.x_pred[i]) / dt

    @ti.kernel
    def _floor_collision(self):
        for i in range(self.n_verts):
            if self.x[i][1] < 0.0:
                self.x[i][1] = 0.0
                if self.v[i][1] < 0.0:
                    self.v[i][1] *= -0.3

    @ti.kernel
    def _set_predictor(self, dt: float):
        for i in range(self.n_verts):
            self.x_pred[i] = self.x[i] + dt * self.v[i]
            self.x[i]      = self.x_pred[i]   # warm-start x at predictor

    # ═══════════════════════════════════════════════════════════════════════════
    # Full PCG solve:  A Δx = -g
    # ═══════════════════════════════════════════════════════════════════════════
    def _pcg_solve(self, model_id: int, dt: float):
        ra = self.rayleigh_alpha
        rb = self.rayleigh_beta

        self._build_precond(model_id, dt, ra, rb)
        self._init_cg()
        self._apply_precond()          # z = M^{-1} r

        # p = z
        self.p.copy_from(self.z)      # ti field copy

        self._dot_rz()
        rz = self.dot_rz[None]
        if abs(rz) < 1e-30:
            return 0                  # already converged

        for it in range(self.cg_iter):
            self._matvec(model_id, dt, ra, rb)   # Ap = A*p
            self._dot_pAp()
            pAp = self.dot_pAp[None]
            if abs(pAp) < 1e-30:
                break
            alpha = rz / pAp
            self._update_dx_r(alpha)             # dx += α p,  r -= α Ap

            r2 = self._r_norm2()
            if r2 < self.cg_tol**2 * self.n_verts:
                break

            self._apply_precond()                # z = M^{-1} r
            self._dot_rz_new_kernel()
            rz_new = self.dot_rz_new[None]
            beta = rz_new / rz
            rz   = rz_new
            self._update_p(beta)                 # p = z + β p

        return it + 1

    # ═══════════════════════════════════════════════════════════════════════════
    # Public step
    # ═══════════════════════════════════════════════════════════════════════════
    def step(self, model_id: int):
        dt = self.dt

        # 1. Compute inertia predictor  x_pred = x_n + dt*v_n
        self._set_predictor(dt)

        # 2. Newton iterations
        for n_it in range(self.newton_iter):
            # Compute f_int at current x
            self._compute_f_int(model_id)
            # Compute Newton residual g
            self._compute_residual(dt, self.rayleigh_alpha)
            # Solve  A Δx = -g  with PCG
            self._pcg_solve(model_id, dt)
            # Update x
            self._apply_dx(dt)

        # 3. Floor collision
        self._floor_collision()

    def get_positions(self):
        return self.x.to_numpy()

    # ═══════════════════════════════════════════════════════════════════════════
    # GUI buffer helpers  (identical to explicit version)
    # ═══════════════════════════════════════════════════════════════════════════
    def setup_gui_buffers(self, surf_vert_indices, surf_faces_local):
        self._surf_vert_indices = surf_vert_indices
        K = len(surf_vert_indices); M = len(surf_faces_local)
        self.gui_verts   = ti.Vector.field(3, ti.f32, K)
        self.gui_indices = ti.field(ti.i32, 3*M)
        self.gui_indices.from_numpy(surf_faces_local.flatten().astype("int32"))

    @ti.kernel
    def _update_gui_kernel(self, surf_idx: ti.types.ndarray()):
        for i in range(surf_idx.shape[0]):
            self.gui_verts[i] = self.x[surf_idx[i]]

    def update_gui_buffers(self):
        self._update_gui_kernel(self._surf_vert_indices)


# ─────────────────────────────────────────────
# GUI runner
# ─────────────────────────────────────────────
def run_gui(sim, model_id, args, surf_vert_indices, surf_faces_local):
    sim.setup_gui_buffers(surf_vert_indices, surf_faces_local)

    window = ti.ui.Window(
        name=f"FEM Implicit — {args.model.upper()}",
        res=(1024, 768), vsync=True,
    )
    canvas = window.get_canvas()
    scene  = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 2.0, 5.0)
    camera.lookat(0.0, 0.5, 0.0)
    camera.up(0.0, 1.0, 0.0)

    import time
    step = 0; t0 = time.time()

    while window.running:
        for _ in range(args.substeps):
            sim.step(model_id)
            step += 1
            if step >= args.steps: break

        sim.update_gui_buffers()
        camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.LMB)
        scene.set_camera(camera)
        scene.ambient_light(color=(0.35, 0.35, 0.35))
        scene.point_light(pos=(3.0, 5.0, 3.0),   color=(1.0, 1.0, 1.0))
        scene.point_light(pos=(-3.0, 3.0, -2.0), color=(0.4, 0.4, 0.5))
        scene.mesh(sim.gui_verts, indices=sim.gui_indices,
                   color=(0.3, 0.65, 1.0), two_sided=True)
        scene.particles(sim.gui_verts, radius=0.003, color=(1.0, 1.0, 1.0))
        canvas.scene(scene)
        window.show()

        elapsed = time.time() - t0
        sps = step / elapsed if elapsed > 0 else 0.0
        window.GUI.begin("Info", 0.02, 0.02, 0.35, 0.15)
        window.GUI.text(f"Step   : {step} / {args.steps}")
        window.GUI.text(f"SPS    : {sps:.2f}")
        window.GUI.text(f"Model  : {args.model.upper()}")
        window.GUI.text(f"Solver : Implicit PCG")
        window.GUI.end()

        if step >= args.steps:
            print(f"[GUI] Done after {step} steps.")
            while window.running:
                window.show()
            break


# ─────────────────────────────────────────────
# Headless export runner
# ─────────────────────────────────────────────
def run_export(sim, model_id, args, surf_vert_indices, surf_faces_local):
    import time
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[Export] OBJ every {args.export_every} step(s) → '{args.output_dir}/'")
    t0 = time.time(); frame = 0

    for step in range(args.steps):
        sim.step(model_id)
        if step % args.export_every == 0:
            pos = sim.get_positions()
            export_surface_obj(pos, surf_vert_indices, surf_faces_local,
                               frame, args.output_dir)
            elapsed = time.time() - t0
            cy = pos[:,1].mean()
            print(f"  Step {step+1:6d}/{args.steps} | frame {frame:06d} | "
                  f"avg_y={cy:.4f} | elapsed={elapsed:.2f}s")
            frame += 1

    print(f"[Done] {frame} frames saved to '{args.output_dir}/'.")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    # 1. Mesh
    if args.mesh is not None:
        verts_surf, faces_surf = load_mesh(args.mesh)
    else:
        print("[Mesh] No mesh provided — generating default unit sphere.")
        verts_surf, faces_surf = generate_sphere(args.sphere_radius, args.sphere_subdiv)

    # 2. Tetrahedralize
    nodes, elems = tetrahedralize(verts_surf, faces_surf)

    # 3. Surface extraction
    surf_faces = extract_surface(elems)
    surf_vert_indices, surf_faces_local = build_surface_index_map(surf_faces, len(nodes))

    # 4. Build simulator
    sim = ImplicitFEMSimulator(nodes, elems, args)

    model_map = {"stvk": 0, "corotated": 1, "neohookean": 2}
    model_id  = model_map[args.model]

    # 5. Run
    if args.gui:
        print(f"[GUI] Implicit solver — {args.substeps} substep(s)/frame")
        run_gui(sim, model_id, args, surf_vert_indices, surf_faces_local)
    else:
        run_export(sim, model_id, args, surf_vert_indices, surf_faces_local)


if __name__ == "__main__":
    main()


# python fem_simulation_implicit.py --model neohookean --gui