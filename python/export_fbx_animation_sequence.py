#!/usr/bin/env python3
"""Export an animated FBX as per-frame OBJ meshes and skeleton JSON files.

Run with Blender, for example:

    blender --background --python python/export_fbx_animation_sequence.py -- \
        --input assets/cat/Cat-Walk.fbx \
        --output output/cat_walk

The script writes:

    <output>/mesh/000.obj
    <output>/skeleton/000.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import bpy
    from mathutils import Matrix
except ImportError as exc:
    raise SystemExit(
        "This script must be run with Blender's Python interpreter, e.g.\n"
        "  blender --background --python python/export_fbx_animation_sequence.py -- "
        "--input model.fbx --output output_dir"
    ) from exc


def blender_argv() -> list[str]:
    """Return arguments after Blender's '--' separator."""
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export FBX animation frames to OBJ meshes and skeleton JSON."
    )
    parser.add_argument("--input", "-i", required=True, help="Input FBX file.")
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory. mesh/ and skeleton/ are created under it.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Export source frames in reverse order while keeping output numbering increasing.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="First source frame to export. Defaults to the imported scene start frame.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last source frame to export, inclusive. Defaults to the imported scene end frame.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Source frame stride. Defaults to 1.",
    )
    parser.add_argument(
        "--keep-materials",
        action="store_true",
        help="Write OBJ material references. Disabled by default for simpler simulation input.",
    )
    return parser.parse_args(blender_argv())


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_fbx(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input FBX does not exist: {path}")
    try:
        bpy.ops.preferences.addon_enable(module="io_scene_fbx")
    except Exception:
        pass
    bpy.ops.import_scene.fbx(filepath=str(path))


def detected_frame_bounds() -> tuple[int, int]:
    ranges = [action.frame_range for action in bpy.data.actions]
    if not ranges:
        scene = bpy.context.scene
        return int(scene.frame_start), int(scene.frame_end)

    start = math.floor(min(frame_range[0] for frame_range in ranges))
    end = math.ceil(max(frame_range[1] for frame_range in ranges))
    return int(start), int(end)


def frame_range_from_scene(args: argparse.Namespace) -> list[int]:
    detected_start, detected_end = detected_frame_bounds()
    start = detected_start if args.start_frame is None else args.start_frame
    end = detected_end if args.end_frame is None else args.end_frame
    if args.step <= 0:
        raise ValueError("--step must be positive")
    if end < start:
        raise ValueError(f"Invalid frame range: start={start}, end={end}")

    frames = list(range(start, end + 1, args.step))
    if args.reverse:
        frames.reverse()
    return frames


def selected_mesh_objects() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def armature_objects() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]


def matrix_to_rows(matrix: Matrix) -> list[list[float]]:
    return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]


def vector_to_list(vector) -> list[float]:
    return [float(vector.x), float(vector.y), float(vector.z)]


def export_obj(path: Path, meshes: list[bpy.types.Object], keep_materials: bool) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    if meshes:
        bpy.context.view_layer.objects.active = meshes[0]

    if hasattr(bpy.ops, "wm") and hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(
            filepath=str(path),
            export_selected_objects=True,
            apply_modifiers=True,
            export_materials=keep_materials,
            forward_axis="NEGATIVE_Z",
            up_axis="Y",
        )
        return

    try:
        bpy.ops.preferences.addon_enable(module="io_scene_obj")
    except Exception:
        pass

    bpy.ops.export_scene.obj(
        filepath=str(path),
        use_selection=True,
        use_animation=False,
        use_mesh_modifiers=True,
        use_materials=keep_materials,
        axis_forward="-Z",
        axis_up="Y",
        path_mode="AUTO" if keep_materials else "STRIP",
    )


def export_skeleton_json(
    path: Path,
    armatures: list[bpy.types.Object],
    source_frame: int,
    output_index: int,
) -> None:
    scene = bpy.context.scene
    payload = {
        "source_frame": int(source_frame),
        "output_index": int(output_index),
        "time_seconds": float(source_frame / scene.render.fps),
        "fps": float(scene.render.fps),
        "armatures": [],
    }

    for armature in armatures:
        bone_names = [bone.name for bone in armature.data.bones]
        bone_index = {name: index for index, name in enumerate(bone_names)}
        armature_payload = {
            "name": armature.name,
            "world_matrix": matrix_to_rows(armature.matrix_world),
            "bones": [],
        }

        for bone in armature.data.bones:
            pose_bone = armature.pose.bones.get(bone.name)
            if pose_bone is None:
                continue

            parent_name = bone.parent.name if bone.parent else None
            parent_index = bone_index[parent_name] if parent_name else -1
            pose_armature_matrix = pose_bone.matrix.copy()
            world_matrix = armature.matrix_world @ pose_armature_matrix

            if pose_bone.parent is not None:
                local_pose_matrix = pose_bone.parent.matrix.inverted() @ pose_armature_matrix
            else:
                local_pose_matrix = pose_armature_matrix

            head_world = armature.matrix_world @ pose_bone.head
            tail_world = armature.matrix_world @ pose_bone.tail

            armature_payload["bones"].append(
                {
                    "name": bone.name,
                    "parent": parent_index,
                    "parent_name": parent_name,
                    "head": vector_to_list(head_world),
                    "tail": vector_to_list(tail_world),
                    "rest_matrix": matrix_to_rows(bone.matrix_local),
                    "local_pose_matrix": matrix_to_rows(local_pose_matrix),
                    "pose_matrix": matrix_to_rows(pose_armature_matrix),
                    "world_matrix": matrix_to_rows(world_matrix),
                }
            )

        payload["armatures"].append(armature_payload)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    mesh_dir = output_dir / "mesh"
    skeleton_dir = output_dir / "skeleton"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    skeleton_dir.mkdir(parents=True, exist_ok=True)

    clear_scene()
    import_fbx(input_path)

    scene = bpy.context.scene
    meshes = selected_mesh_objects()
    armatures = armature_objects()
    frames = frame_range_from_scene(args)

    if not meshes:
        raise RuntimeError(f"No mesh objects were imported from {input_path}")
    if not armatures:
        print(f"[Warning] No armature objects were imported from {input_path}")

    print(f"[FBX] {input_path}")
    print(f"[Output] {output_dir}")
    print(f"[Meshes] {len(meshes)}")
    print(f"[Armatures] {len(armatures)}")
    print(f"[Frames] {len(frames)} ({frames[0]} -> {frames[-1]})")

    for output_index, source_frame in enumerate(frames):
        scene.frame_set(source_frame)
        bpy.context.view_layer.update()

        stem = f"{output_index:03d}"
        export_obj(mesh_dir / f"{stem}.obj", meshes, args.keep_materials)
        export_skeleton_json(
            skeleton_dir / f"{stem}.json",
            armatures,
            source_frame,
            output_index,
        )
        print(f"[Export] {stem}: source frame {source_frame}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
