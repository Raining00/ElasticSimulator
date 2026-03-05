import argparse
from pathlib import Path

import numpy as np
import taichi as ti


def parse_args():
    parser = argparse.ArgumentParser(description="Taichi SPH elastic body simulation")
    parser.add_argument("--ply", type=str, default=None, help="Input particle cloud PLY file")
    parser.add_argument("--gui", action="store_true", help="Run interactive Taichi GUI (no export)")
    parser.add_argument("--steps", type=int, default=800, help="Number of frames to run")
    parser.add_argument("--substeps", type=int, default=4, help="Simulation substeps per frame")
    parser.add_argument("--dt", type=float, default=5e-4, help="Simulation time step")
    parser.add_argument("--num_particles", type=int, default=1800,
                        help="Particle count for generated sphere")
    parser.add_argument("--h", type=float, default=0.12, help="SPH smoothing radius")
    parser.add_argument("--rho0", type=float, default=1000.0, help="Rest density")
    parser.add_argument("--pressure_k", type=float, default=30.0, help="Pressure stiffness")
    parser.add_argument("--viscosity", type=float, default=0.08, help="Viscosity coefficient")
    parser.add_argument("--elastic_k", type=float, default=1200.0, help="Elastic spring stiffness")
    parser.add_argument("--elastic_damp", type=float, default=1.8, help="Elastic spring damping")
    parser.add_argument("--gravity", type=float, default=-9.8, help="Gravity along Y")
    parser.add_argument("--global_damping", type=float, default=0.999, help="Velocity damping")
    parser.add_argument("--max_neighbors", type=int, default=96, help="Neighbor cap per particle")
    parser.add_argument("--box_half_extent", type=float, default=1.6, help="X/Z boundary half extent")
    parser.add_argument("--output_dir", type=str, default="output_sph_elastic",
                        help="Export folder when not using GUI")
    parser.add_argument("--export_every", type=int, default=1, help="Export every N frames")
    parser.add_argument("--arch", type=str, default="gpu", choices=["gpu", "cpu", "cuda", "vulkan"],
                        help="Taichi backend")
    parser.add_argument("--render_radius", type=float, default=0.01, help="GUI particle radius")
    return parser.parse_args()


def sample_unit_sphere_particles(n: int, lift: float = 1.25) -> np.ndarray:
    points = []
    while len(points) < n:
        samples = np.random.uniform(-1.0, 1.0, size=(n * 2, 3))
        keep = np.sum(samples * samples, axis=1) <= 1.0
        valid = samples[keep]
        points.extend(valid.tolist())
    pts = np.array(points[:n], dtype=np.float32)
    pts[:, 1] += lift
    return pts


def load_ply_particles(path: str) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("Please install open3d to load PLY: pip install open3d") from exc

    pcd = o3d.io.read_point_cloud(path)

    pts = np.asarray(pcd.points, dtype=np.float32)
    pts = pts - np.mean(pts, axis=0, keepdims=True)
    min_y = np.min(pts[:, 1])
    pts[:, 1] += (0.2 - min_y)
    return pts


def write_ply(path: str, positions: np.ndarray):
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {positions.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in positions:
            f.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f}\n")


def build_rest_neighbors(positions: np.ndarray, h: float, max_neighbors: int):
    n = positions.shape[0]
    ids = np.full((n, max_neighbors), -1, dtype=np.int32)
    rest_len = np.zeros((n, max_neighbors), dtype=np.float32)
    counts = np.zeros(n, dtype=np.int32)
    radius2 = h * h

    for i in range(n):
        d = positions - positions[i]
        dist2 = np.sum(d * d, axis=1)
        valid = np.where((dist2 > 1e-12) & (dist2 < radius2))[0]
        if valid.size > 0:
            order = valid[np.argsort(dist2[valid])]
            m = min(max_neighbors, order.shape[0])
            nbr = order[:m]
            ids[i, :m] = nbr
            rest_len[i, :m] = np.sqrt(dist2[nbr]).astype(np.float32)
            counts[i] = m
    return counts, ids, rest_len


@ti.data_oriented
class SPHElasticBody:
    def __init__(self, positions: np.ndarray, rest_counts, rest_ids, rest_len, args):
        self.n = positions.shape[0]
        self.max_neighbors = args.max_neighbors
        self.h = args.h
        self.h2 = args.h * args.h
        self.dt = args.dt
        self.rho0 = args.rho0
        self.pressure_k = args.pressure_k
        self.viscosity = args.viscosity
        self.elastic_k = args.elastic_k
        self.elastic_damp = args.elastic_damp
        self.gravity = args.gravity
        self.global_damping = args.global_damping
        self.box_half_extent = args.box_half_extent
        self.mass = 1.0

        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.a = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.rho = ti.field(dtype=ti.f32, shape=self.n)
        self.pressure = ti.field(dtype=ti.f32, shape=self.n)

        self.nbr_count = ti.field(dtype=ti.i32, shape=self.n)
        self.nbr_ids = ti.field(dtype=ti.i32, shape=(self.n, self.max_neighbors))

        self.rest_count = ti.field(dtype=ti.i32, shape=self.n)
        self.rest_ids = ti.field(dtype=ti.i32, shape=(self.n, self.max_neighbors))
        self.rest_len = ti.field(dtype=ti.f32, shape=(self.n, self.max_neighbors))

        self.x.from_numpy(positions.astype(np.float32))
        self.v.fill(0.0)
        self.a.fill(0.0)
        self.rest_count.from_numpy(rest_counts)
        self.rest_ids.from_numpy(rest_ids)
        self.rest_len.from_numpy(rest_len)

    @ti.func
    def poly6(self, r: ti.f32) -> ti.f32:
        result = 0.0
        if 0.0 <= r and r < self.h:
            x = self.h2 - r * r
            result = 315.0 / (64.0 * np.pi * self.h ** 9) * x * x * x
        return result

    @ti.func
    def spiky_grad(self, r_vec):
        r = r_vec.norm()
        grad = ti.Vector([0.0, 0.0, 0.0])
        if 1e-6 < r and r < self.h:
            coeff = -45.0 / (np.pi * self.h ** 6) * (self.h - r) * (self.h - r)
            grad = coeff * r_vec / r
        return grad

    @ti.func
    def visc_lap(self, r: ti.f32) -> ti.f32:
        result = 0.0
        if 0.0 <= r and r < self.h:
            result = 45.0 / (np.pi * self.h ** 6) * (self.h - r)
        return result

    @ti.kernel
    def find_neighbors(self):
        for i in range(self.n):
            count = 0
            xi = self.x[i]
            for j in range(self.n):
                if i != j:
                    r2 = (xi - self.x[j]).norm_sqr()
                    if r2 < self.h2 and count < self.max_neighbors:
                        self.nbr_ids[i, count] = j
                        count += 1
            self.nbr_count[i] = count

    @ti.kernel
    def compute_density_pressure(self):
        for i in range(self.n):
            rho_i = self.mass * self.poly6(0.0)
            for k in range(self.nbr_count[i]):
                j = self.nbr_ids[i, k]
                rij = self.x[i] - self.x[j]
                rho_i += self.mass * self.poly6(rij.norm())
            self.rho[i] = ti.max(rho_i, 1e-6)
            self.pressure[i] = self.pressure_k * (self.rho[i] - self.rho0)

    @ti.kernel
    def compute_acceleration(self):
        for i in range(self.n):
            self.a[i] = ti.Vector([0.0, self.gravity, 0.0])

        for i in range(self.n):
            xi = self.x[i]
            vi = self.v[i]
            rhoi = self.rho[i]
            for k in range(self.nbr_count[i]):
                j = self.nbr_ids[i, k]
                xj = self.x[j]
                vj = self.v[j]
                rhoj = self.rho[j]
                rij = xi - xj
                r = rij.norm()
                grad = self.spiky_grad(rij)
                self.a[i] += -self.mass * (
                    self.pressure[i] / (rhoi * rhoi) + self.pressure[j] / (rhoj * rhoj)
                ) * grad
                self.a[i] += self.viscosity * self.mass * (vj - vi) / rhoj * self.visc_lap(r)

            for k in range(self.rest_count[i]):
                j = self.rest_ids[i, k]
                r0 = self.rest_len[i, k]
                rij = xi - self.x[j]
                r = rij.norm()
                if r > 1e-6:
                    d = rij / r
                    rel_v = (self.v[i] - self.v[j]).dot(d)
                    self.a[i] += -self.elastic_k * (r - r0) * d
                    self.a[i] += -self.elastic_damp * rel_v * d

    @ti.kernel
    def integrate(self):
        for i in range(self.n):
            self.v[i] += self.dt * self.a[i]
            self.v[i] *= self.global_damping
            self.x[i] += self.dt * self.v[i]

            if self.x[i][1] < 0.0:
                self.x[i][1] = 0.0
                if self.v[i][1] < 0.0:
                    self.v[i][1] *= -0.25
                self.v[i][0] *= 0.96
                self.v[i][2] *= 0.96

            if self.x[i][0] < -self.box_half_extent:
                self.x[i][0] = -self.box_half_extent
                if self.v[i][0] < 0.0:
                    self.v[i][0] *= -0.25
            if self.x[i][0] > self.box_half_extent:
                self.x[i][0] = self.box_half_extent
                if self.v[i][0] > 0.0:
                    self.v[i][0] *= -0.25
            if self.x[i][2] < -self.box_half_extent:
                self.x[i][2] = -self.box_half_extent
                if self.v[i][2] < 0.0:
                    self.v[i][2] *= -0.25
            if self.x[i][2] > self.box_half_extent:
                self.x[i][2] = self.box_half_extent
                if self.v[i][2] > 0.0:
                    self.v[i][2] *= -0.25

    def step(self):
        self.find_neighbors()
        self.compute_density_pressure()
        self.compute_acceleration()
        self.integrate()


def init_taichi(arch_name: str):
    arch_map = {
        "cpu": ti.cpu,
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
        "gpu": ti.gpu,
    }
    ti.init(arch=arch_map[arch_name], default_fp=ti.f32)


def main():
    args = parse_args()
    init_taichi(args.arch)

    if args.ply is None:
        print("[Init] No --ply provided. Generating unit sphere particle cloud.")
        pos = sample_unit_sphere_particles(args.num_particles, lift=1.25)
    else:
        print(f"[Init] Loading particle cloud from: {args.ply}")
        pos = load_ply_particles(args.ply)

    print("[Init] Building rest neighbor data...")
    rest_counts, rest_ids, rest_len = build_rest_neighbors(pos, args.h, args.max_neighbors)
    sim = SPHElasticBody(pos, rest_counts, rest_ids, rest_len, args)
    print(f"[Init] Particle count: {sim.n}")

    if args.gui:
        window = ti.ui.Window("SPH Elastic Body (Taichi)", (1024, 768), vsync=True)
        canvas = window.get_canvas()
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()
        camera.position(2.6, 1.7, 2.6)
        camera.lookat(0.0, 0.6, 0.0)
        camera.up(0.0, 1.0, 0.0)
        frame = 0
        while window.running and frame < args.steps:
            for _ in range(args.substeps):
                sim.step()
            camera.track_user_inputs(window, movement_speed=0.02, hold_key=ti.ui.RMB)
            scene.set_camera(camera)
            scene.point_light((1.8, 2.4, 1.2), color=(1.0, 1.0, 1.0))
            scene.point_light((-1.8, 2.2, -1.4), color=(0.5, 0.5, 0.5))
            scene.ambient_light((0.35, 0.35, 0.35))
            scene.particles(sim.x, radius=args.render_radius, color=(0.2, 0.65, 1.0))
            canvas.set_background_color((0.02, 0.04, 0.08))
            canvas.scene(scene)
            window.show()
            frame += 1
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame in range(args.steps):
        for _ in range(args.substeps):
            sim.step()
        if frame % args.export_every == 0:
            p = sim.x.to_numpy()
            out = out_dir / f"frame_{frame:05d}.ply"
            write_ply(str(out), p)
            print(f"[Export] {out}")


if __name__ == "__main__":
    main()
