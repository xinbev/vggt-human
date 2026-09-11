"""Disk format for replaying sequence point clouds and SMPL meshes without inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

CACHE_FORMAT = "vggt_omega_sequence_viewer_cache_v1"
MANIFEST_NAME = "manifest.json"
FACES_NAME = "smpl_faces.npy"
RAW_CAMERA_TRAJECTORY_NAME = "camera_trajectory_raw.npy"
HSI_CAMERA_TRAJECTORY_NAME = "camera_trajectory_hsi.npy"


def export_sequence_viewer_cache(scene: dict[str, Any], cache_dir: str | Path) -> Path:
    """Export final filtered HSI points and final SMPL meshes frame by frame."""
    root = Path(cache_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    frames = list(scene.get("frames", []))
    if not frames:
        raise ValueError("Cannot export an empty viewer scene")

    faces, vertex_count = _find_smpl_topology(frames)
    np.save(root / FACES_NAME, faces.astype(np.int32, copy=False), allow_pickle=False)
    raw_trajectory = np.asarray(
        scene.get("camera_trajectory_raw", np.empty((0, 3))),
        dtype=np.float32,
    ).reshape(-1, 3)
    hsi_trajectory = np.asarray(
        scene.get("camera_trajectory_hsi", np.empty((0, 3))),
        dtype=np.float32,
    ).reshape(-1, 3)
    np.save(root / RAW_CAMERA_TRAJECTORY_NAME, raw_trajectory, allow_pickle=False)
    np.save(root / HSI_CAMERA_TRAJECTORY_NAME, hsi_trajectory, allow_pickle=False)

    frame_records: list[dict[str, Any]] = []
    total_points = 0
    total_people = 0
    for position, frame in enumerate(frames):
        points = np.asarray(frame.get("hsi_points", np.empty((0, 3))), dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(frame.get("hsi_colors", np.empty((0, 3))), dtype=np.uint8).reshape(-1, 3)
        if colors.shape[0] != points.shape[0]:
            raise ValueError(f"Point/color count mismatch at cache frame {position}: {points.shape} vs {colors.shape}")

        people = _collect_frame_people(frame, vertex_count)
        camera = _collect_frame_camera(frame)
        frame_file = f"frame_{position:04d}.npz"
        np.savez(
            root / frame_file,
            points=points,
            colors=colors,
            smpl_vertices=people["vertices"],
            smpl_track_ids=people["track_ids"],
            smpl_query_indices=people["query_indices"],
            smpl_confidences=people["confidences"],
            smpl_track_qualities=people["track_qualities"],
            smpl_colors=people["colors"],
            intrinsic=camera["intrinsic"],
            raw_extrinsic=camera["raw_extrinsic"],
            hsi_extrinsic=camera["hsi_extrinsic"],
            raw_camera_rotation_c2w=camera["raw_camera_rotation_c2w"],
            raw_camera_position=camera["raw_camera_position"],
            raw_camera_fov=camera["raw_camera_fov"],
            raw_camera_aspect=camera["raw_camera_aspect"],
            hsi_camera_rotation_c2w=camera["hsi_camera_rotation_c2w"],
            hsi_camera_position=camera["hsi_camera_position"],
            hsi_camera_fov=camera["hsi_camera_fov"],
            hsi_camera_aspect=camera["hsi_camera_aspect"],
        )
        point_count = int(points.shape[0])
        people_count = int(people["vertices"].shape[0])
        total_points += point_count
        total_people += people_count
        frame_records.append(
            {
                "position": int(position),
                "frame_index": int(frame.get("frame_index", position)),
                "frame_id": str(frame.get("frame_id", position)),
                "source_image": str(frame.get("image", "")),
                "file": frame_file,
                "point_count": point_count,
                "people_count": people_count,
            }
        )
        if (position + 1) % 25 == 0 or position + 1 == len(frames):
            print(f"[viewer-cache] exported {position + 1}/{len(frames)} frames", flush=True)

    manifest = {
        "format": CACHE_FORMAT,
        "num_frames": len(frame_records),
        "point_source": "hsi_depth_filtered_human_points",
        "mesh_source": "hsi_smpl_with_base_fallback",
        "total_points": int(total_points),
        "total_people": int(total_people),
        "smpl_vertex_count": int(vertex_count),
        "smpl_face_count": int(faces.shape[0]),
        "smpl_faces_file": FACES_NAME,
        "camera_parameters": {
            "per_frame": [
                "intrinsic",
                "raw_extrinsic",
                "hsi_extrinsic",
                "raw_camera_rotation_c2w",
                "raw_camera_position",
                "raw_camera_fov",
                "raw_camera_aspect",
                "hsi_camera_rotation_c2w",
                "hsi_camera_position",
                "hsi_camera_fov",
                "hsi_camera_aspect",
            ],
            "raw_trajectory_file": RAW_CAMERA_TRAJECTORY_NAME,
            "hsi_trajectory_file": HSI_CAMERA_TRAJECTORY_NAME,
        },
        "scene_metadata": {
            "image_hw": scene.get("image_hw", []),
            "hsi_scene_affine_mode": scene.get("hsi_scene_affine_mode", "unknown"),
            "scene_scale_prealign": scene.get("scene_scale_prealign", "unknown"),
            "trstr_active": bool(scene.get("trstr_active", False)),
        },
        "frames": frame_records,
    }
    manifest_path = root / MANIFEST_NAME
    temporary_manifest = root / f"{MANIFEST_NAME}.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    print(
        f"[viewer-cache] ready: {manifest_path} | frames={len(frame_records)} "
        f"points={total_points} people={total_people}",
        flush=True,
    )
    return manifest_path


def load_sequence_viewer_manifest(cache_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(cache_dir).expanduser().resolve()
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing viewer cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != CACHE_FORMAT:
        raise ValueError(f"Unsupported viewer cache format: {manifest.get('format')!r}")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Viewer cache has no frame records: {manifest_path}")
    faces_path = root / str(manifest.get("smpl_faces_file", FACES_NAME))
    if not faces_path.is_file():
        raise FileNotFoundError(f"Missing viewer cache SMPL topology: {faces_path}")
    return root, manifest


def _find_smpl_topology(frames: list[dict[str, Any]]) -> tuple[np.ndarray, int]:
    for frame in frames:
        for person in frame.get("people", []):
            vertices = person.get("hsi_vertices")
            if vertices is None:
                vertices = person.get("base_vertices")
            faces = person.get("faces")
            if vertices is None or faces is None:
                continue
            vertices_np = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
            faces_np = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
            if vertices_np.shape[0] > 0 and faces_np.shape[0] > 0:
                return faces_np, int(vertices_np.shape[0])
    raise ValueError("Viewer scene contains no exportable SMPL topology")


def _collect_frame_people(frame: dict[str, Any], vertex_count: int) -> dict[str, np.ndarray]:
    vertices: list[np.ndarray] = []
    track_ids: list[int] = []
    query_indices: list[int] = []
    confidences: list[float] = []
    track_qualities: list[float] = []
    colors: list[tuple[int, int, int]] = []
    for person in frame.get("people", []):
        mesh = person.get("hsi_vertices")
        if mesh is None:
            mesh = person.get("base_vertices")
        if mesh is None:
            continue
        mesh_np = np.asarray(mesh, dtype=np.float32).reshape(-1, 3)
        if mesh_np.shape[0] != vertex_count or not np.isfinite(mesh_np).all():
            continue
        vertices.append(mesh_np)
        track_ids.append(int(person.get("track_id", -1)))
        query_indices.append(int(person.get("query_index", -1)))
        confidences.append(float(person.get("confidence", float("nan"))))
        track_qualities.append(float(person.get("track_quality", float("nan"))))
        colors.append(tuple(int(value) for value in person.get("color", (204, 51, 51))))
    return {
        "vertices": np.stack(vertices).astype(np.float32, copy=False) if vertices else np.empty((0, vertex_count, 3), dtype=np.float32),
        "track_ids": np.asarray(track_ids, dtype=np.int32),
        "query_indices": np.asarray(query_indices, dtype=np.int32),
        "confidences": np.asarray(confidences, dtype=np.float32),
        "track_qualities": np.asarray(track_qualities, dtype=np.float32),
        "colors": np.asarray(colors, dtype=np.uint8).reshape(-1, 3),
    }


def _collect_frame_camera(frame: dict[str, Any]) -> dict[str, np.ndarray]:
    intrinsic = _matrix(frame, "intrinsic", (3, 3))
    raw_extrinsic = _matrix(frame, "raw_extrinsic", (3, 4))
    hsi_extrinsic = _matrix(frame, "hsi_extrinsic", (3, 4))
    raw_camera = _camera_pose(frame.get("raw_camera"))
    hsi_camera = _camera_pose(frame.get("hsi_camera", frame.get("camera")))
    return {
        "intrinsic": intrinsic,
        "raw_extrinsic": raw_extrinsic,
        "hsi_extrinsic": hsi_extrinsic,
        "raw_camera_rotation_c2w": raw_camera["rotation_c2w"],
        "raw_camera_position": raw_camera["position"],
        "raw_camera_fov": raw_camera["fov"],
        "raw_camera_aspect": raw_camera["aspect"],
        "hsi_camera_rotation_c2w": hsi_camera["rotation_c2w"],
        "hsi_camera_position": hsi_camera["position"],
        "hsi_camera_fov": hsi_camera["fov"],
        "hsi_camera_aspect": hsi_camera["aspect"],
    }


def _matrix(frame: dict[str, Any], key: str, shape: tuple[int, int]) -> np.ndarray:
    value = frame.get(key)
    if value is None:
        raise ValueError(f"Viewer cache frame is missing required camera matrix: {key}")
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"Viewer cache camera matrix {key} must have shape {shape}, got {array.shape}")
    return array


def _camera_pose(value: Any) -> dict[str, np.ndarray]:
    if not isinstance(value, dict):
        raise ValueError("Viewer cache frame is missing required camera pose")
    rotation = np.asarray(value.get("rotation_c2w"), dtype=np.float32)
    position = np.asarray(value.get("position"), dtype=np.float32)
    if rotation.shape != (3, 3) or position.shape != (3,):
        raise ValueError(
            f"Viewer cache camera pose has invalid shapes: rotation={rotation.shape}, position={position.shape}"
        )
    return {
        "rotation_c2w": rotation,
        "position": position,
        "fov": np.asarray(float(value.get("fov", 0.0)), dtype=np.float32),
        "aspect": np.asarray(float(value.get("aspect", 0.0)), dtype=np.float32),
    }
