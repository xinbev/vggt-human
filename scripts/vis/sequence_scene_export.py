"""Portable, baked per-frame assets and index for a cached sequence viewer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def export_single_frame_scene(
    output_root: Path,
    frame: dict[str, Any],
    clouds: dict[str, tuple[np.ndarray, np.ndarray]],
    meshes: list[dict[str, Any]],
    cameras: dict[str, dict[str, Any]],
    settings: dict[str, Any],
    viewport: dict[str, Any] | None = None,
    environment_meshes: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    folder_name: str | None = None,
) -> Path:
    """Create a fresh folder; all positions are baked into the viewer world frame."""
    output_root.mkdir(parents=True, exist_ok=True)
    frame_index = int(frame["frame_index"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    target = output_root / (folder_name or f"frame_{frame_index:04d}_{stamp}")
    target.mkdir(exist_ok=False)
    incomplete = target / ".incomplete"
    incomplete.write_text("Single-frame export is in progress.\n", encoding="utf-8")
    files: dict[str, Any] = {"environment": {}, "environment_meshes": {}, "people": []}
    for source, (points, colors) in clouds.items():
        filename = f"environment_{source}.ply"
        write_point_ply(target / filename, points, colors)
        files["environment"][source] = {"file": filename, "point_count": int(len(points))}
    for source, (vertices, colors, faces) in (environment_meshes or {}).items():
        filename = f"environment_{source}_mesh.ply"
        write_triangle_ply(target / filename, vertices, colors, faces)
        files["environment_meshes"][source] = {
            "file": filename, "vertex_count": int(len(vertices)), "face_count": int(len(faces))
        }
    for mesh in meshes:
        filename = str(mesh["name"]) + ".obj"
        write_mesh_obj(target / filename, mesh["vertices"], mesh["faces"], mesh["color"],
                       mesh.get("opacity", 1.0))
        files["people"].append({
            "file": filename,
            "material_file": Path(filename).with_suffix(".mtl").name,
            "kind": str(mesh["kind"]),
            "track_id": mesh.get("track_id"),
            "query_index": mesh.get("query_index"),
            "opacity": float(mesh.get("opacity", 1.0)),
            "vertex_count": int(len(mesh["vertices"])),
            "face_count": int(len(mesh["faces"])),
        })
    camera_payload = {
        "coordinate_convention": "Same world XYZ as exported PLY/OBJ. Camera local +Z is forward, +X right, +Y down. w2c maps world to camera. Matrices use column vectors.",
        "image_hw": [int(v) for v in np.asarray(frame.get("rgb_chw")).shape[-2:]] if frame.get("rgb_chw") is not None else None,
        "intrinsic_3x3": _jsonable(frame["intrinsic"]),
        "cameras": {name: _camera_record(camera) for name, camera in cameras.items()},
        "viewport": _jsonable(viewport),
    }
    (target / "cameras.json").write_text(json.dumps(camera_payload, indent=2), encoding="utf-8")
    manifest = {
        "format": "vggt_omega_single_frame_scene_v1",
        "frame_index": frame_index,
        "source_frame_index": int(frame.get("source_frame_index", -1)),
        "frame_id": str(frame.get("frame_id", frame_index)),
        "source_image": str(frame.get("image", "")),
        "world_units": "VGGT viewer world units; metric scale is not guaranteed",
        "settings": _jsonable(settings),
        "files": {**files, "cameras": "cameras.json"},
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    incomplete.unlink()
    return target


def finalize_sequence_scene_export(
    batch_dir: Path, records: list[dict[str, Any]], viewport: dict[str, Any] | None
) -> Path:
    """Publish the index only after every per-frame manifest is complete."""
    if not records:
        raise ValueError("Cannot finalize an empty sequence export")
    for record in records:
        manifest = batch_dir / str(record["manifest"])
        if not manifest.is_file() or (manifest.parent / ".incomplete").exists():
            raise RuntimeError(f"Incomplete exported frame: {manifest}")
    payload = {
        "format": "vggt_omega_sequence_scene_export_v1",
        "num_frames": len(records),
        "viewport_at_export": _jsonable(viewport),
        "frames": records,
    }
    manifest_path = batch_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (batch_dir / ".incomplete").unlink(missing_ok=True)
    return manifest_path


def write_point_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(colors, dtype=np.uint8)
    if rgb.shape == (3,):
        rgb = np.broadcast_to(rgb, xyz.shape)
    if rgb.shape != xyz.shape or not np.isfinite(xyz).all():
        raise ValueError(f"Invalid point cloud for {path.name}: {xyz.shape}, {rgb.shape}")
    packed = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for index, name in enumerate(("x", "y", "z")):
        packed[name] = xyz[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        packed[name] = rgb[:, index]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("wb") as file:
        file.write(header.encode("ascii"))
        packed.tofile(file)


def write_mesh_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, color: Any,
                   opacity: float = 1.0) -> None:
    xyz = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not np.isfinite(xyz).all() or (triangles.size and (triangles.min() < 0 or triangles.max() >= len(xyz))):
        raise ValueError(f"Invalid mesh for {path.name}")
    rgb = np.asarray(color, dtype=np.uint8).reshape(3)
    with path.open("w", encoding="ascii", newline="\n") as file:
        file.write(f"mtllib {path.stem}.mtl\no {path.stem}\nusemtl smpl_color\n")
        for x, y, z in xyz:
            file.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for a, b, c in triangles + 1:
            file.write(f"f {a} {b} {c}\n")
    (path.with_suffix(".mtl")).write_text(
        "newmtl smpl_color\nKd " + " ".join(f"{value / 255:.6f}" for value in rgb)
        + f"\nd {float(opacity):.6f}\n",
        encoding="ascii",
    )


def write_triangle_ply(path: Path, vertices: np.ndarray, colors: np.ndarray, faces: np.ndarray) -> None:
    xyz = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    triangles = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if rgb.shape != xyz.shape or not np.isfinite(xyz).all():
        raise ValueError(f"Invalid environment mesh vertices for {path.name}")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(xyz)):
        raise ValueError(f"Invalid environment mesh faces for {path.name}")
    packed = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for index, name in enumerate(("x", "y", "z")):
        packed[name] = xyz[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        packed[name] = rgb[:, index]
    face_records = np.empty(len(triangles), dtype=[("count", "u1"), ("indices", "<i4", (3,))])
    face_records["count"] = 3
    face_records["indices"] = triangles
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(triangles)}\nproperty list uchar int vertex_indices\nend_header\n")
    with path.open("wb") as file:
        file.write(header.encode("ascii"))
        packed.tofile(file)
        face_records.tofile(file)


def _camera_record(camera: dict[str, Any]) -> dict[str, Any]:
    rotation = np.asarray(camera["rotation_c2w"], dtype=np.float64).reshape(3, 3)
    position = np.asarray(camera["position"], dtype=np.float64).reshape(3)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = position
    w2c = np.linalg.inv(c2w)
    return {
        "position_xyz": position.tolist(),
        "rotation_c2w_3x3": rotation.tolist(),
        "camera_to_world_4x4": c2w.tolist(),
        "world_to_camera_4x4": w2c.tolist(),
        "fov_y_radians": float(camera["fov"]),
        "aspect": float(camera["aspect"]),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
