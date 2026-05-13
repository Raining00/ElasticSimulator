bl_info = {
    "name": "Elastic Simulator FEM",
    "author": "ElasticSimulator",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Elastic FEM",
    "description": "Run the ElasticSimulator FEM algorithm on a Blender mesh without the OpenGL viewer.",
    "category": "Physics",
}

import bpy
import bmesh
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
)
from bpy.types import Operator, Panel, PropertyGroup
from mathutils import Vector


ENERGY_MODEL_ITEMS = (
    ("stvk", "StVK", "St. Venant-Kirchhoff explicit FEM"),
    ("corotated", "Corotated", "Corotated linear elasticity"),
    ("neohookean", "Neo-Hookean", "Stable Neo-Hookean elasticity"),
)


def _import_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "NumPy is required by Elastic Simulator FEM. Install it into Blender's Python first."
        ) from exc
    return np


def _triangulated_mesh_from_object(obj):
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()
    bm.faces.index_update()

    world = obj.matrix_world
    verts = [world @ v.co for v in bm.verts]
    faces = [[loop.vert.index for loop in face.loops] for face in bm.faces]
    bm.free()

    np = _import_numpy()
    vertices = np.array([(v.x, v.y, v.z) for v in verts], dtype=np.float64)
    triangles = np.array(faces, dtype=np.int32)
    if len(vertices) < 4 or len(triangles) == 0:
        raise RuntimeError("Select a closed triangle surface mesh with at least four vertices.")
    return vertices, triangles


def _tetrahedralize(vertices, faces, min_dihedral, min_ratio):
    np = _import_numpy()
    try:
        import tetgen

        tet = tetgen.TetGen(vertices, faces)
        tet.tetrahedralize(order=1, mindihedral=min_dihedral, minratio=min_ratio)
        return np.array(tet.node, dtype=np.float64), np.array(tet.elem, dtype=np.int32)
    except ImportError:
        pass

    try:
        import ptetgen

        result = ptetgen.tetrahedralize(
            f"pq{min_ratio:g}", vertices.astype(np.float64), faces.astype(np.int32)
        )
        return (
            np.array(result.points, dtype=np.float64),
            np.array(result.cells_dict.get("tetra", []), dtype=np.int32),
        )
    except ImportError as exc:
        raise RuntimeError(
            "A TetGen Python binding is required. Install either 'tetgen' or 'ptetgen' "
            "into Blender's Python environment."
        ) from exc


def _extract_surface(elems):
    from collections import defaultdict

    np = _import_numpy()
    oriented = ((1, 3, 2), (0, 2, 3), (0, 3, 1), (0, 1, 2))
    face_count = defaultdict(list)
    for tet in elems:
        for a, b, c in oriented:
            face = (int(tet[a]), int(tet[b]), int(tet[c]))
            face_count[tuple(sorted(face))].append(face)

    surface = [faces[0] for faces in face_count.values() if len(faces) == 1]
    if not surface:
        raise RuntimeError("TetGen produced no boundary faces.")
    return np.array(surface, dtype=np.int32)


def _build_surface_map(surface_faces, vertex_count):
    np = _import_numpy()
    surface_vertex_indices = np.unique(surface_faces)
    global_to_local = np.full(vertex_count, -1, dtype=np.int32)
    global_to_local[surface_vertex_indices] = np.arange(len(surface_vertex_indices), dtype=np.int32)
    return surface_vertex_indices, global_to_local[surface_faces]


class NumpyFEMSimulator:
    def __init__(self, nodes, elems, settings):
        np = _import_numpy()
        self.np = np
        self.x = nodes.astype(np.float64).copy()
        self.v = np.zeros_like(self.x)
        self.f = np.zeros_like(self.x)
        self.elems = elems.astype(np.int32)
        self.dt = float(settings.dt)
        self.gravity = np.array(
            [settings.gravity_x, settings.gravity_y, settings.gravity_z],
            dtype=np.float64,
        )
        self.damping = float(settings.damping)
        self.floor_collision = bool(settings.floor_collision)
        self.floor_y = float(settings.floor_y)
        self.restitution = float(settings.restitution)
        self.energy_model = settings.energy_model
        self.density = float(settings.density)
        self.mu = settings.youngs_modulus / (2.0 * (1.0 + settings.poisson_ratio))
        self.lam = (
            settings.youngs_modulus
            * settings.poisson_ratio
            / ((1.0 + settings.poisson_ratio) * (1.0 - 2.0 * settings.poisson_ratio))
        )
        self.dm_inv = np.zeros((len(elems), 3, 3), dtype=np.float64)
        self.vol = np.zeros(len(elems), dtype=np.float64)
        self.mass = np.zeros(len(nodes), dtype=np.float64)
        self._precompute()

    def _precompute(self):
        np = self.np
        for i, tet in enumerate(self.elems):
            a, b, c, d = tet
            dm = np.column_stack((self.x[b] - self.x[a], self.x[c] - self.x[a], self.x[d] - self.x[a]))
            det = np.linalg.det(dm)
            vol = abs(det) / 6.0
            if vol <= 1e-14:
                continue
            self.dm_inv[i] = np.linalg.inv(dm)
            self.vol[i] = vol
            node_mass = self.density * vol / 4.0
            self.mass[tet] += node_mass
        self.mass[self.mass <= 1e-12] = 1.0

    def _pk1_stvk(self, f):
        np = self.np
        eye = np.eye(3)
        strain = 0.5 * (f.T @ f - eye)
        stress = 2.0 * self.mu * strain + self.lam * np.trace(strain) * eye
        return f @ stress

    def _pk1_corotated(self, f):
        np = self.np
        u, _, vh = np.linalg.svd(f)
        r = u @ vh
        if np.linalg.det(r) < 0.0:
            u[:, -1] *= -1.0
            r = u @ vh
        j = max(np.linalg.det(f), 1e-8)
        finv_t = np.linalg.inv(f).T
        return 2.0 * self.mu * (f - r) + self.lam * (j - 1.0) * j * finv_t

    def _pk1_neohookean(self, f):
        np = self.np
        j = max(np.linalg.det(f), 1e-8)
        finv_t = np.linalg.inv(f).T
        return self.mu * (f - finv_t) + self.lam * (j - 1.0) * j * finv_t

    def step(self):
        np = self.np
        self.f[:, :] = self.gravity[None, :] * self.mass[:, None]

        for i, tet in enumerate(self.elems):
            if self.vol[i] <= 0.0:
                continue
            a, b, c, d = tet
            ds = np.column_stack((self.x[b] - self.x[a], self.x[c] - self.x[a], self.x[d] - self.x[a]))
            f = ds @ self.dm_inv[i]
            if self.energy_model == "stvk":
                p = self._pk1_stvk(f)
            elif self.energy_model == "corotated":
                p = self._pk1_corotated(f)
            else:
                p = self._pk1_neohookean(f)

            h = -self.vol[i] * p @ self.dm_inv[i].T
            fb = h[:, 0]
            fc = h[:, 1]
            fd = h[:, 2]
            fa = -(fb + fc + fd)
            self.f[a] += fa
            self.f[b] += fb
            self.f[c] += fc
            self.f[d] += fd

        self.v += self.dt * self.f / self.mass[:, None]
        if self.damping > 0.0:
            self.v *= max(0.0, 1.0 - self.damping)
        self.x += self.dt * self.v

        if self.floor_collision:
            below = self.x[:, 1] < self.floor_y
            self.x[below, 1] = self.floor_y
            falling = below & (self.v[:, 1] < 0.0)
            self.v[falling, 1] *= -self.restitution


def _simulate(settings, vertices, faces):
    np = _import_numpy()
    nodes, elems = _tetrahedralize(vertices, faces, settings.min_dihedral, settings.min_ratio)
    if len(elems) == 0:
        raise RuntimeError("Tetrahedralization produced no tetrahedra.")

    surface_faces = _extract_surface(elems)
    surface_indices, local_faces = _build_surface_map(surface_faces, len(nodes))
    sim = NumpyFEMSimulator(nodes, elems, settings)

    frames = [sim.x[surface_indices].copy()]
    step_count = max(1, int(settings.steps))
    export_every = max(1, int(settings.export_every))
    for step in range(1, step_count + 1):
        for _ in range(max(1, int(settings.substeps))):
            sim.step()
        if step % export_every == 0:
            frames.append(sim.x[surface_indices].copy())

    return np.array(frames), local_faces


def _create_shape_key_animation(context, source_obj, frames, faces, settings):
    mesh = bpy.data.meshes.new(f"{source_obj.name}_ElasticFEMMesh")
    mesh.from_pydata(frames[0].tolist(), [], faces.tolist())
    mesh.update()

    obj = bpy.data.objects.new(f"{source_obj.name}_ElasticFEM", mesh)
    context.collection.objects.link(obj)
    context.view_layer.objects.active = obj
    obj.select_set(True)

    obj.shape_key_add(name="Basis")
    frame_start = context.scene.frame_start
    frame_stride = max(1, int(settings.blender_frame_stride))
    keys = []

    for i, positions in enumerate(frames):
        key = obj.shape_key_add(name=f"FEM_{i:04d}")
        for vertex_index, co in enumerate(positions):
            key.data[vertex_index].co = Vector(co)
        keys.append(key)

    for i, key in enumerate(keys):
        current_frame = frame_start + i * frame_stride
        previous_frame = max(frame_start, current_frame - frame_stride)
        next_frame = current_frame + frame_stride

        key.value = 0.0
        key.keyframe_insert("value", frame=previous_frame)
        key.value = 1.0
        key.keyframe_insert("value", frame=current_frame)
        key.value = 0.0
        key.keyframe_insert("value", frame=next_frame)

    context.scene.frame_end = frame_start + max(1, len(frames) - 1) * frame_stride
    return obj


class ELASTIC_FEM_Settings(PropertyGroup):
    energy_model: EnumProperty(
        name="Energy Model",
        items=ENERGY_MODEL_ITEMS,
        default="neohookean",
    )
    youngs_modulus: FloatProperty(name="Young's Modulus", default=1.0e6, min=1.0)
    poisson_ratio: FloatProperty(name="Poisson Ratio", default=0.45, min=0.0, max=0.49)
    density: FloatProperty(name="Density", default=1000.0, min=1e-6)
    dt: FloatProperty(name="Time Step", default=1.0e-4, min=1e-8, precision=6)
    steps: IntProperty(name="Steps", default=120, min=1)
    substeps: IntProperty(name="Substeps", default=1, min=1)
    export_every: IntProperty(name="Capture Every", default=2, min=1)
    blender_frame_stride: IntProperty(name="Frame Stride", default=1, min=1)
    damping: FloatProperty(name="Damping", default=0.0, min=0.0, max=1.0)
    gravity_x: FloatProperty(name="Gravity X", default=0.0)
    gravity_y: FloatProperty(name="Gravity Y", default=-9.81)
    gravity_z: FloatProperty(name="Gravity Z", default=0.0)
    floor_collision: BoolProperty(name="Floor Collision", default=True)
    floor_y: FloatProperty(name="Floor Y", default=0.0)
    restitution: FloatProperty(name="Restitution", default=0.3, min=0.0, max=1.0)
    min_dihedral: FloatProperty(name="Tet Min Dihedral", default=20.0, min=0.0, max=60.0)
    min_ratio: FloatProperty(name="Tet Min Ratio", default=1.5, min=1.0, max=5.0)


class ELASTIC_FEM_OT_Run(Operator):
    bl_idname = "elastic_fem.run"
    bl_label = "Run Elastic FEM"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        source_obj = context.object
        if source_obj is None or source_obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh object before running Elastic FEM.")
            return {"CANCELLED"}

        settings = context.scene.elastic_fem_settings
        try:
            vertices, faces = _triangulated_mesh_from_object(source_obj)
            frames, surface_faces = _simulate(settings, vertices, faces)
            result = _create_shape_key_animation(context, source_obj, frames, surface_faces, settings)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        self.report({"INFO"}, f"Created {result.name} with {len(frames)} simulated FEM frames.")
        return {"FINISHED"}


class ELASTIC_FEM_PT_Panel(Panel):
    bl_label = "Elastic FEM"
    bl_idname = "ELASTIC_FEM_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Elastic FEM"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.elastic_fem_settings

        layout.prop(settings, "energy_model")
        layout.prop(settings, "youngs_modulus")
        layout.prop(settings, "poisson_ratio")
        layout.prop(settings, "density")

        layout.separator()
        layout.prop(settings, "dt")
        layout.prop(settings, "steps")
        layout.prop(settings, "substeps")
        layout.prop(settings, "export_every")
        layout.prop(settings, "blender_frame_stride")
        layout.prop(settings, "damping")

        layout.separator()
        layout.prop(settings, "gravity_x")
        layout.prop(settings, "gravity_y")
        layout.prop(settings, "gravity_z")
        layout.prop(settings, "floor_collision")
        if settings.floor_collision:
            layout.prop(settings, "floor_y")
            layout.prop(settings, "restitution")

        layout.separator()
        layout.prop(settings, "min_dihedral")
        layout.prop(settings, "min_ratio")

        layout.separator()
        layout.operator("elastic_fem.run", icon="MOD_PHYSICS")


classes = (
    ELASTIC_FEM_Settings,
    ELASTIC_FEM_OT_Run,
    ELASTIC_FEM_PT_Panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.elastic_fem_settings = PointerProperty(type=ELASTIC_FEM_Settings)


def unregister():
    del bpy.types.Scene.elastic_fem_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
