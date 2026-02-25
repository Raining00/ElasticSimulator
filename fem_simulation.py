"""
FEM Simulation using Taichi
Supports StVK, Co-rotated, and Neo-Hookean energy models
Explicit time integration
Exports surface mesh as OBJ per frame
"""

import argparse
import os
import numpy as np
import taichi as ti

# ─────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="FEM Simulation with Taichi")
    parser.add_argument("--mesh", type=str, default=None,
                        help="Path to input mesh file (.obj, .stl, .ply, etc.). "
                             "If omitted, a unit sphere is used as the default mesh.")
    parser.add_argument("--sphere_radius", type=float, default=1.0,
                        help="Radius of the default sphere mesh (default: 1.0)")
    parser.add_argument("--sphere_subdiv", type=int, default=3,
                        help="Subdivision level for default sphere (default: 3)")
    parser.add_argument("--model", type=str, default="stvk",
                        choices=["stvk", "corotated", "neohookean"],
                        help="Elastic energy model (default: stvk)")
    parser.add_argument("--E", type=float, default=1e5,
                        help="Young's modulus (default: 1e5)")
    parser.add_argument("--nu", type=float, default=0.45,
                        help="Poisson's ratio (default: 0.45)")
    parser.add_argument("--dt", type=float, default=1e-4,
                        help="Time step (default: 1e-4)")
    parser.add_argument("--density", type=float, default=1000.0,
                        help="Material density (default: 1000.0)")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Total simulation steps (default: 10000)")
    parser.add_argument("--gravity", type=float, default=-9.8,
                        help="Gravity in Y direction (default: -9.8)")
    parser.add_argument("--export_every", type=int, default=1,
                        help="Export surface OBJ every N steps (default: 1, ignored in GUI mode)")
    parser.add_argument("--output_dir", type=str, default="output_frames",
                        help="Directory to save frame OBJ files (default: output_frames)")
    # ── GUI options ────────────────────────────────────────────────────────────
    parser.add_argument("--gui", action="store_true",
                        help="Show real-time surface mesh in a ti.ui.Window instead of exporting OBJs")
    parser.add_argument("--substeps", type=int, default=10,
                        help="Simulation sub-steps per rendered frame in GUI mode (default: 5)")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Mesh Loading
# ─────────────────────────────────────────────
def load_mesh(path: str):
    """Load surface mesh using trimesh."""
    try:
        import trimesh
    except ImportError:
        raise ImportError("Please install trimesh: pip install trimesh")

    mesh = trimesh.load(path, force="mesh")
    if mesh is None or len(mesh.vertices) == 0:
        raise ValueError(f"Failed to load mesh from: {path}")

    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int32)
    print(f"[Mesh] Loaded: {len(vertices)} vertices, {len(faces)} faces")
    return vertices, faces


def generate_sphere(radius: float = 1.0, subdivisions: int = 3):
    """
    Generate a UV/icosphere surface mesh (vertices + triangular faces) without
    any external dependencies. Uses iterative icosphere subdivision.

    Parameters
    ----------
    radius      : float  sphere radius
    subdivisions: int    number of subdivision iterations (3 → ~1280 faces)

    Returns
    -------
    vertices : np.ndarray  (N, 3) float64
    faces    : np.ndarray  (M, 3) int32
    """
    # ── seed icosahedron ──────────────────────────────────────────────────────
    phi = (1.0 + np.sqrt(5.0)) / 2.0          # golden ratio
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ], dtype=np.float64)
    # Normalise onto unit sphere
    verts /= np.linalg.norm(verts[0])

    tris = np.array([
        [0,11, 5],[0, 5, 1],[0, 1, 7],[0, 7,10],[0,10,11],
        [1, 5, 9],[5,11, 4],[11,10, 2],[10, 7, 6],[7, 1, 8],
        [3, 9, 4],[3, 4, 2],[3, 2, 6],[3, 6, 8],[3, 8, 9],
        [4, 9, 5],[2, 4,11],[6, 2,10],[8, 6, 7],[9, 8, 1],
    ], dtype=np.int32)

    # ── subdivide ─────────────────────────────────────────────────────────────
    for _ in range(subdivisions):
        new_tris = []
        midpoint_cache = {}

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key in midpoint_cache:
                return midpoint_cache[key]
            nonlocal verts
            mid = (verts[a] + verts[b]) * 0.5
            mid /= np.linalg.norm(mid)           # project onto sphere
            idx = len(verts)
            verts = np.vstack([verts, mid])
            midpoint_cache[key] = idx
            return idx

        for t in tris:
            a, b, c = int(t[0]), int(t[1]), int(t[2])
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_tris.extend([
                [a, ab, ca],
                [b, bc, ab],
                [c, ca, bc],
                [ab, bc, ca],
            ])
        tris = np.array(new_tris, dtype=np.int32)

    verts = verts * radius
    # Lift sphere so it sits just above y=0
    verts[:, 1] += radius + 0.01

    print(f"[Sphere] Generated: {len(verts)} vertices, {len(tris)} faces "
          f"(radius={radius}, subdiv={subdivisions})")
    return verts, tris


# ─────────────────────────────────────────────
# Surface Extraction
# ─────────────────────────────────────────────
def extract_surface(elems: np.ndarray):
    """
    Extract the surface triangles from a tetrahedral mesh.

    A triangle face is on the surface if and only if it appears in exactly
    one tetrahedron (interior faces are shared by two tets).

    Each tet [a,b,c,d] has four faces:
        (b,c,d), (a,c,d), (a,b,d), (a,b,c)
    We store each face as a *sorted* tuple for counting, but keep the
    original winding so we can recover an outward-pointing normal later.

    Returns
    -------
    surf_faces : np.ndarray  shape (M, 3) int32
        Triangle indices into the tet-node array.
    """
    from collections import defaultdict

    # Map sorted-face-key -> list of (original winding) faces
    face_count = defaultdict(list)

    # The four faces of a tet, with consistent outward winding
    # for a positively-oriented tet [a,b,c,d]:
    local_faces = [
        (1, 2, 3),  # face opposite a  → b,c,d
        (0, 2, 3),  # face opposite b  → a,c,d  (reversed for outward)
        (0, 1, 3),  # face opposite c  → a,b,d
        (0, 1, 2),  # face opposite d  → a,b,c  (reversed for outward)
    ]
    # Outward-pointing winding (alternating orientation)
    local_faces_oriented = [
        (1, 3, 2),
        (0, 2, 3),
        (0, 3, 1),
        (0, 1, 2),
    ]

    for tet in elems:
        for (li, lj, lk), (oi, oj, ok) in zip(local_faces, local_faces_oriented):
            vi, vj, vk = int(tet[li]), int(tet[lj]), int(tet[lk])
            key = tuple(sorted((vi, vj, vk)))
            face_count[key].append((tet[oi], tet[oj], tet[ok]))

    # Keep only faces that belong to exactly one tet (boundary faces)
    surf_faces = []
    for key, faces in face_count.items():
        if len(faces) == 1:
            surf_faces.append(faces[0])

    surf_faces = np.array(surf_faces, dtype=np.int32)
    print(f"[Surface] Extracted {len(surf_faces)} surface triangles "
          f"from {len(elems)} tetrahedra.")
    return surf_faces


def build_surface_index_map(surf_faces: np.ndarray, n_total_verts: int):
    """
    Build a compact vertex array for the surface so the exported OBJ
    only references the vertices that actually appear on the surface.

    Returns
    -------
    surf_vert_indices : np.ndarray  shape (K,) int32
        Global vertex indices that are on the surface (sorted).
    surf_faces_local : np.ndarray  shape (M, 3) int32
        Remapped face indices into surf_vert_indices.
    """
    unique_verts = np.unique(surf_faces)          # sorted global indices
    global_to_local = np.full(n_total_verts, -1, dtype=np.int32)
    global_to_local[unique_verts] = np.arange(len(unique_verts), dtype=np.int32)
    surf_faces_local = global_to_local[surf_faces]
    return unique_verts, surf_faces_local


# ─────────────────────────────────────────────
# OBJ Exporter
# ─────────────────────────────────────────────
def export_surface_obj(positions: np.ndarray,
                       surf_vert_indices: np.ndarray,
                       surf_faces_local: np.ndarray,
                       frame: int,
                       output_dir: str):
    """
    Write the current surface mesh to  <output_dir>/frame_<frame>.obj

    Parameters
    ----------
    positions        : (N, 3) float  – all tet-node positions at current step
    surf_vert_indices: (K,)   int    – global indices of surface vertices
    surf_faces_local : (M, 3) int    – face indices into the compact surface array
    frame            : int           – current frame number
    output_dir       : str           – destination directory
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"frame_{frame:06d}.obj")

    surf_verts = positions[surf_vert_indices]   # (K, 3) compact vertex positions

    with open(path, "w") as f:
        f.write(f"# FEM surface export — frame {frame}\n")
        f.write(f"# Vertices: {len(surf_verts)}  Faces: {len(surf_faces_local)}\n")

        for v in surf_verts:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")

        # OBJ face indices are 1-based
        for tri in surf_faces_local:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


# ─────────────────────────────────────────────
# Tetrahedralization via TetGen
# ─────────────────────────────────────────────
def tetrahedralize(vertices, faces):
    """Use tetgen (via tetgenpy or ptetgen) to generate tetrahedra."""
    try:
        import tetgen
        tet = tetgen.TetGen(vertices, faces)
        tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
        nodes, elems = tet.node, tet.elem
        nodes = np.array(nodes, dtype=np.float64)
        elems = np.array(elems, dtype=np.int32)
        print(f"[TetGen] Generated: {len(nodes)} nodes, {len(elems)} tetrahedra")
        return nodes, elems
    except ImportError:
        pass

    # Fallback: try ptetgen
    try:
        import ptetgen
        result = ptetgen.tetrahedralize("pq1.414a0.1", vertices, faces)
        nodes = np.array(result.points, dtype=np.float64)
        elems = np.array(result.cells_dict.get("tetra", []), dtype=np.int32)
        print(f"[ptetgen] Generated: {len(nodes)} nodes, {len(elems)} tetrahedra")
        return nodes, elems
    except ImportError:
        pass

    raise ImportError(
        "TetGen binding not found. Install via:\n"
        "  pip install tetgen        (PyTetGen)\n"
        "  pip install ptetgen       (alternative)\n"
    )


# ─────────────────────────────────────────────
# Taichi Initialization
# ─────────────────────────────────────────────
ti.init(arch=ti.gpu, default_fp=ti.f32)

# ─────────────────────────────────────────────
# Simulation Class
# ─────────────────────────────────────────────
@ti.data_oriented
class FEMSimulator:
    def __init__(self, nodes, elems, args):
        self.n_verts = len(nodes)
        self.n_tets  = len(elems)
        self.dt      = args.dt
        self.gravity = args.gravity
        self.model   = args.model

        # Lamé parameters
        E, nu = args.E, args.nu
        self.mu     = E / (2.0 * (1.0 + nu))
        self.lam    = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.density = args.density

        print(f"[FEM] Model={self.model}, mu={self.mu:.2f}, lam={self.lam:.2f}")
        print(f"[FEM] Verts={self.n_verts}, Tets={self.n_tets}")

        # ── Taichi fields ──────────────────────────────
        self.x      = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        self.v      = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        self.f      = ti.Vector.field(3, dtype=ti.f32, shape=self.n_verts)
        self.mass   = ti.field(dtype=ti.f32, shape=self.n_verts)

        self.tets   = ti.Vector.field(4, dtype=ti.i32, shape=self.n_tets)
        # Dm_inv: 3x3 reference shape matrix inverse per tet
        self.Dm_inv = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_tets)
        self.vol    = ti.field(dtype=ti.f32, shape=self.n_tets)

        # Scalar material params
        self.mu_field  = ti.field(dtype=ti.f32, shape=())
        self.lam_field = ti.field(dtype=ti.f32, shape=())
        self.mu_field[None]  = float(self.mu)
        self.lam_field[None] = float(self.lam)

        # ── Initialize from numpy ──────────────────────
        self.x.from_numpy(nodes.astype(np.float32))
        self.v.fill(0.0)
        self.f.fill(0.0)
        self.mass.fill(0.0)
        self.tets.from_numpy(elems.astype(np.int32))

        self._precompute()

        # ── GUI surface buffers (populated lazily by setup_gui_buffers) ────────
        self.gui_verts  = None   # ti.Vector.field for surface vertex positions
        self.gui_indices = None  # ti.field for flat triangle index list

    def setup_gui_buffers(self,
                          surf_vert_indices: "np.ndarray",
                          surf_faces_local:  "np.ndarray"):
        """
        Allocate Taichi fields consumed by ti.ui.Scene.mesh().

        ti.ui.Scene.mesh() expects:
          vertices  – ti.Vector.field(3, f32, shape=K)   (surface verts only)
          indices   – ti.field(i32, shape=3*M)            (flat triangle list)

        We store surf_vert_indices so _update_gui_buffers can gather positions.
        """
        self._surf_vert_indices = surf_vert_indices          # (K,) global→local
        K = len(surf_vert_indices)
        M = len(surf_faces_local)

        self.gui_verts   = ti.Vector.field(3, dtype=ti.f32, shape=K)
        self.gui_indices = ti.field(dtype=ti.i32, shape=3 * M)

        # Upload static index buffer (topology never changes)
        flat_idx = surf_faces_local.flatten().astype("int32")
        self.gui_indices.from_numpy(flat_idx)

    @ti.kernel
    def _update_gui_buffers_kernel(self, surf_idx: ti.types.ndarray()):
        """Gather surface vertex positions from the full position field."""
        for i in range(surf_idx.shape[0]):
            self.gui_verts[i] = self.x[surf_idx[i]]

    def update_gui_buffers(self):
        """Call every frame to sync surface positions into the GUI field."""
        self._update_gui_buffers_kernel(self._surf_vert_indices)

    # ──────────────────────────────────────────────────
    @ti.kernel
    def _precompute(self):
        """Compute Dm_inv and lumped mass for each tet."""
        for i in range(self.n_tets):
            a = self.tets[i][0]
            b = self.tets[i][1]
            c = self.tets[i][2]
            d = self.tets[i][3]

            xa = self.x[a]; xb = self.x[b]
            xc = self.x[c]; xd = self.x[d]

            Dm = ti.Matrix.cols([xb - xa, xc - xa, xd - xa])
            det = Dm.determinant()
            vol = ti.abs(det) / 6.0
            self.vol[i] = vol
            self.Dm_inv[i] = Dm.inverse()

            node_mass = self.density * vol / 4.0
            self.mass[a] += node_mass
            self.mass[b] += node_mass
            self.mass[c] += node_mass
            self.mass[d] += node_mass

    # ──────────────────────────────────────────────────
    @ti.func
    def _deformation_gradient(self, i: int) -> ti.Matrix:
        a = self.tets[i][0]; b = self.tets[i][1]
        c = self.tets[i][2]; d = self.tets[i][3]
        Ds = ti.Matrix.cols([self.x[b] - self.x[a],
                              self.x[c] - self.x[a],
                              self.x[d] - self.x[a]])
        return Ds @ self.Dm_inv[i]

    # ──────────────────────────────────────────────────
    @ti.func
    def _pk1_stvk(self, F: ti.template()) -> ti.Matrix:
        """St. Venant-Kirchhoff: P = F(2μE + λtr(E)I), E = (FᵀF - I)/2"""
        mu  = self.mu_field[None]
        lam = self.lam_field[None]
        I   = ti.Matrix.identity(ti.f32, 3)
        E   = (F.transpose() @ F - I) * 0.5
        trE = E.trace()
        S   = 2.0 * mu * E + lam * trE * I   # 2nd PK stress
        return F @ S

    @ti.func
    def _pk1_corotated(self, F: ti.template()) -> ti.Matrix:
        """Co-rotated linear elasticity via polar decomposition R,S=polar(F)
        P = 2μ(F-R) + λ(J-1)JF^{-T}   (simplified co-rotated)
        """
        mu  = self.mu_field[None]
        lam = self.lam_field[None]
        # Polar decomposition: F = R * S  (R orthogonal, S sym. pos. def.)
        # Approximate via one SVD-free iteration using the Higham method
        # We use SVD via QR proxy — Taichi supports ti.svd
        U, sig, V = ti.svd(F, ti.f32)
        R = U @ V.transpose()
        J = F.determinant()
        Finv_T = F.inverse().transpose()
        P = 2.0 * mu * (F - R) + lam * (J - 1.0) * J * Finv_T
        return P

    @ti.func
    def _pk1_neohookean(self, F: ti.template()) -> ti.Matrix:
        """Stable Neo-Hookean: P = μ(F - F^{-T}) + λ(J-1)JF^{-T}"""
        mu  = self.mu_field[None]
        lam = self.lam_field[None]
        J = F.determinant()
        Finv_T = F.inverse().transpose()
        P = mu * (F - Finv_T) + lam * (J - 1.0) * J * Finv_T
        return P

    # ──────────────────────────────────────────────────
    @ti.kernel
    def _compute_forces(self, model_id: int):
        """Accumulate elastic forces. model_id: 0=stvk,1=corot,2=neohook"""
        # Reset forces (gravity applied separately)
        for i in range(self.n_verts):
            self.f[i] = ti.Vector([0.0, self.gravity * self.mass[i], 0.0])

        for i in range(self.n_tets):
            a = self.tets[i][0]; b = self.tets[i][1]
            c = self.tets[i][2]; d = self.tets[i][3]

            F = self._deformation_gradient(i)

            P = ti.Matrix.zero(ti.f32, 3, 3)
            if model_id == 0:
                P = self._pk1_stvk(F)
            elif model_id == 1:
                P = self._pk1_corotated(F)
            else:
                P = self._pk1_neohookean(F)

            # Force on each node: f = -vol * P * Dm_inv^T
            H = -self.vol[i] * P @ self.Dm_inv[i].transpose()

            # Columns of H are forces on nodes b, c, d
            fb = ti.Vector([H[0, 0], H[1, 0], H[2, 0]])
            fc = ti.Vector([H[0, 1], H[1, 1], H[2, 1]])
            fd = ti.Vector([H[0, 2], H[1, 2], H[2, 2]])
            fa = -(fb + fc + fd)

            self.f[a] += fa
            self.f[b] += fb
            self.f[c] += fc
            self.f[d] += fd

    # ──────────────────────────────────────────────────
    @ti.kernel
    def _integrate(self, dt: float):
        """Explicit Euler integration + simple floor collision."""
        for i in range(self.n_verts):
            self.v[i] += dt * self.f[i] / self.mass[i]
            self.x[i] += dt * self.v[i]

            # Floor constraint y >= 0
            if self.x[i][1] < 0.0:
                self.x[i][1] = 0.0
                if self.v[i][1] < 0.0:
                    self.v[i][1] *= -0.3   # restitution

    # ──────────────────────────────────────────────────
    def step(self, model_id: int):
        self._compute_forces(model_id)
        self._integrate(self.dt)

    def get_positions(self):
        return self.x.to_numpy()


# ─────────────────────────────────────────────
# GUI runner
# ─────────────────────────────────────────────
def run_gui(sim, model_id: int, args,
            surf_vert_indices: "np.ndarray",
            surf_faces_local:  "np.ndarray"):
    """
    Real-time visualisation using ti.ui.Window + ti.ui.Scene.mesh().

    Controls
    --------
    Mouse drag   – orbit camera
    Scroll wheel – zoom
    ESC / close  – quit
    """
    sim.setup_gui_buffers(surf_vert_indices, surf_faces_local)

    window = ti.ui.Window(
        name=f"FEM — {args.model.upper()}",
        res=(1024, 768),
        vsync=True,
    )
    canvas = window.get_canvas()
    scene  = ti.ui.Scene()
    camera = ti.ui.Camera()

    # Position camera to frame the mesh sensibly
    camera.position(0.0, 2.0, 5.0)
    camera.lookat(0.0, 0.5, 0.0)
    camera.up(0.0, 1.0, 0.0)

    step = 0
    import time
    t0 = time.time()

    while window.running:
        # ── Advance physics ──────────────────────────────────────────────────
        for _ in range(args.substeps):
            sim.step(model_id)
            step += 1
            if step >= args.steps:
                break

        # ── Sync surface positions to GPU render buffer ───────────────────────
        sim.update_gui_buffers()

        # ── Render ────────────────────────────────────────────────────────────
        camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.LMB)
        scene.set_camera(camera)

        scene.ambient_light(color=(0.35, 0.35, 0.35))
        scene.point_light(pos=(3.0, 5.0, 3.0),  color=(1.0, 1.0, 1.0))
        scene.point_light(pos=(-3.0, 3.0, -2.0), color=(0.4, 0.4, 0.5))

        # Draw the deforming surface mesh
        scene.mesh(
            sim.gui_verts,
            indices=sim.gui_indices,
            color=(0.3, 0.65, 1.0),
            two_sided=True,
        )

        # Optionally overlay vertex dots for clarity at low resolutions
        scene.particles(sim.gui_verts, radius=0.003, color=(1.0, 1.0, 1.0))

        canvas.scene(scene)
        window.show()

        elapsed = time.time() - t0
        fps = step / elapsed if elapsed > 0 else 0.0
        window.GUI.begin("Info", 0.02, 0.02, 0.3, 0.12)
        window.GUI.text(f"Step : {step} / {args.steps}")
        window.GUI.text(f"FPS  : {fps:.1f} sim-steps/s")
        window.GUI.text(f"Model: {args.model.upper()}")
        window.GUI.end()

        if step >= args.steps:
            print(f"[GUI] Simulation complete after {step} steps.")
            # Keep window open until user closes it
            while window.running:
                window.show()
            break


# ─────────────────────────────────────────────
# Headless export runner
# ─────────────────────────────────────────────
def run_export(sim, model_id: int, args,
               surf_vert_indices: "np.ndarray",
               surf_faces_local:  "np.ndarray"):
    """Step the simulation and write one OBJ per export cadence."""
    import time
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[Export] Saving surface OBJ every {args.export_every} step(s) "
          f"→ '{args.output_dir}/'")

    t0 = time.time()
    frame = 0

    for step in range(args.steps):
        sim.step(model_id)

        if (step % args.export_every) == 0:
            pos = sim.get_positions()
            export_surface_obj(
                pos, surf_vert_indices, surf_faces_local,
                frame, args.output_dir
            )
            elapsed = time.time() - t0
            cy = pos[:, 1].mean()
            print(f"  Step {step+1:6d}/{args.steps} | frame {frame:06d} | "
                  f"avg_y={cy:.4f} | elapsed={elapsed:.2f}s")
            frame += 1

    print(f"[Done] Simulation finished. {frame} frames saved to '{args.output_dir}/'.")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    # 1. Load or generate surface mesh
    if args.mesh is not None:
        verts_surf, faces_surf = load_mesh(args.mesh)
    else:
        print("[Mesh] No mesh provided — generating default unit sphere.")
        verts_surf, faces_surf = generate_sphere(
            radius=args.sphere_radius,
            subdivisions=args.sphere_subdiv,
        )

    # 2. Tetrahedralize
    nodes, elems = tetrahedralize(verts_surf, faces_surf)

    # 3. Extract surface of the tet mesh + build compact index map
    surf_faces = extract_surface(elems)
    surf_vert_indices, surf_faces_local = build_surface_index_map(
        surf_faces, len(nodes)
    )

    # 4. Build simulator
    sim = FEMSimulator(nodes, elems, args)

    model_map = {"stvk": 0, "corotated": 1, "neohookean": 2}
    model_id  = model_map[args.model]

    # 5. Run: GUI or headless export
    if args.gui:
        print(f"[GUI] Opening window — {args.substeps} substep(s)/frame, "
              f"press ESC or close to quit.")
        run_gui(sim, model_id, args, surf_vert_indices, surf_faces_local)
    else:
        run_export(sim, model_id, args, surf_vert_indices, surf_faces_local)


if __name__ == "__main__":
    main()



# python fem_simulation.py --model corotated --gui