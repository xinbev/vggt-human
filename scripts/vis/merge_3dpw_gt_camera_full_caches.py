#!/usr/bin/env python
"""Merge chunked full-viewer caches using native 3DPW camera extrinsics.

The model is still responsible for depth and SMPL reconstruction.  This tool
only changes the display/world gauge: each cached camera-space reconstruction
is placed into the native 3DPW camera world with ``cam_poses``.  Consequently
the result is an oracle-style visualisation, not an RGB-only evaluation.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vis.full_viewer_cache_io import (  # noqa: E402
    FULL_CACHE_FORMAT,
    INCOMPLETE_NAME,
    MANIFEST_NAME,
    SCENE_METADATA_NAME,
    SMPL_FACES_NAME,
)


WORLD_POINT_KEYS = (
    "raw_points",
    "raw_points_full",
    "hsi_points",
    "hsi_points_full",
    "raw_mesh_vertices",
    "hsi_mesh_vertices",
)
WORLD_PERSON_KEYS = ("base_vertices", "hsi_vertices")
CAM_PERSON_KEYS = ("base_vertices_cam", "hsi_vertices_cam")


def main() -> None:
    args = parse_args()
    input_root = resolve_path(args.input_root)
    gt_path = resolve_path(args.gt_pkl)
    output_dir = resolve_path(args.output_dir)
    gt = load_3dpw_camera(gt_path)
    cache_dirs = discover_cache_dirs(input_root)
    if not cache_dirs:
        raise FileNotFoundError(f"No full viewer caches found below {input_root}")

    selected: dict[int, tuple[Path, dict[str, Any]]] = {}
    first_manifest: dict[str, Any] | None = None
    first_cache_dir: Path | None = None
    for cache_dir in cache_dirs:
        manifest_path = cache_dir / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_chunk_cache(cache_dir, manifest)
        if first_manifest is None:
            first_manifest = manifest
            first_cache_dir = cache_dir
        for record in manifest["frames"]:
            source_index = resolve_gt_frame_index(
                frame_id=str(record.get("frame_id", "")),
                source_index=int(record.get("source_frame_index", -1)),
                gt=gt,
                frame_index_offset=int(args.frame_index_offset),
            )
            # A rerun can leave older chunk directories under the same output
            # root.  When the caller gives an expected length, do not let
            # those stale frames extend the merged sequence.
            if int(args.expected_frames) > 0 and source_index >= int(args.expected_frames):
                continue
            if source_index < 0 or source_index >= gt["cam_poses"].shape[0]:
                raise IndexError(
                    f"Frame {source_index} from {cache_dir} is outside 3DPW camera range "
                    f"[0,{gt['cam_poses'].shape[0] - 1}]"
                )
            if source_index in selected:
                # Overlap is useful for chunk-boundary diagnostics.  The first
                # copy is retained, so the merged sequence still has one
                # canonical frame per native 3DPW camera index.
                continue
            selected[source_index] = (cache_dir, record)

    if first_manifest is None or first_cache_dir is None:
        raise RuntimeError("No valid chunk manifest was loaded")
    expected = int(args.expected_frames)
    if expected > 0:
        missing = sorted(set(range(expected)) - set(selected))
        if missing:
            raise ValueError(f"Merged caches do not cover all expected frames; missing {missing[:20]}")
    manifest = write_merged_cache(
        selected=selected,
        gt=gt,
        gt_path=gt_path,
        output_dir=output_dir,
        first_cache_dir=first_cache_dir,
        first_manifest=first_manifest,
    )
    ordered_indices = sorted(selected)
    print(
        json.dumps(
            {
                "output_manifest": str(manifest),
                "frames": len(ordered_indices),
                "first_source_frame": int(ordered_indices[0]),
                "last_source_frame": int(ordered_indices[-1]),
                "camera_source": "native_3dpw_cam_poses_T_w2c",
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Directory containing chunk_* full_viewer_cache directories")
    parser.add_argument("--gt-pkl", required=True, help="Native 3DPW sequence pickle")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-frames", type=int, default=0, help="Require coverage of [0, expected_frames)")
    parser.add_argument("--frame-index-offset", type=int, default=0)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def discover_cache_dirs(root: Path) -> list[Path]:
    # Only consume the explicitly generated chunk caches.  Restricting the
    # search prevents a previous merged output under the same output root from
    # being fed back into the merge on a rerun.
    manifests = sorted(root.glob("chunk_*/full_viewer_cache/manifest.json"))
    candidates = []
    for manifest in manifests:
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(payload.get("format", "")).startswith("vggt_omega_full_sequence_viewer_cache"):
            candidates.append(manifest.parent)
    return sorted(candidates, key=lambda path: natural_key(path.name))


def validate_chunk_cache(cache_dir: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("format") != FULL_CACHE_FORMAT:
        raise ValueError(f"Unsupported full cache format in {cache_dir}: {manifest.get('format')!r}")
    if (cache_dir / INCOMPLETE_NAME).exists():
        raise RuntimeError(f"Chunk cache is incomplete: {cache_dir}")
    records = manifest.get("frames")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Chunk cache has no frames: {cache_dir}")
    for required in (manifest.get("scene_metadata_file", SCENE_METADATA_NAME), manifest.get("smpl_faces_file", SMPL_FACES_NAME)):
        if not (cache_dir / str(required)).is_file():
            raise FileNotFoundError(f"Chunk cache is missing {required}: {cache_dir}")


def write_merged_cache(
    selected: dict[int, tuple[Path, dict[str, Any]]],
    gt: dict[str, np.ndarray],
    gt_path: Path,
    output_dir: Path,
    first_cache_dir: Path,
    first_manifest: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    incomplete = output_dir / INCOMPLETE_NAME
    incomplete.write_text("GT-camera full-cache merge is in progress.\n", encoding="utf-8")
    faces = np.load(
        first_cache_dir / str(first_manifest.get("smpl_faces_file", SMPL_FACES_NAME)),
        allow_pickle=False,
    ).astype(np.int32, copy=False)
    np.save(output_dir / SMPL_FACES_NAME, faces, allow_pickle=False)
    with (first_cache_dir / str(first_manifest.get("scene_metadata_file", SCENE_METADATA_NAME))).open("rb") as file:
        scene_metadata = pickle.load(file)  # noqa: S301 - trusted project-local cache.
    if not isinstance(scene_metadata, dict):
        raise TypeError("Invalid scene metadata in first chunk")

    output_records: list[dict[str, Any]] = []
    trajectory: list[np.ndarray] = []
    for output_position, source_index in enumerate(sorted(selected)):
        cache_dir, source_record = selected[source_index]
        source_file = cache_dir / str(source_record["file"])
        if not source_file.is_file():
            raise FileNotFoundError(f"Missing cached frame: {source_file}")
        with source_file.open("rb") as file:
            frame = pickle.load(file)  # noqa: S301 - trusted project-local cache.
        if not isinstance(frame, dict):
            raise TypeError(f"Invalid cached frame: {source_file}")
        transformed = place_frame_in_gt_camera_world(frame, gt, source_index)
        transformed["frame_index"] = int(output_position)
        frame_file = f"frame_{output_position:04d}.pkl"
        with (output_dir / frame_file).open("wb") as file:
            pickle.dump(transformed, file, protocol=pickle.HIGHEST_PROTOCOL)
        trajectory.append(np.asarray(transformed["hsi_camera"]["position"], dtype=np.float32))
        output_records.append(
            {
                "position": int(output_position),
                "frame_index": int(output_position),
                "source_frame_index": int(source_index),
                "frame_id": str(transformed.get("frame_id", source_index)),
                "source_image": str(transformed.get("image", source_record.get("source_image", ""))),
                "file": frame_file,
            }
        )
        if (output_position + 1) % 25 == 0 or output_position + 1 == len(selected):
            print(f"[3dpw-gt-camera-merge] wrote {output_position + 1}/{len(selected)} frames", flush=True)

    trajectory_np = np.stack(trajectory).astype(np.float32, copy=False)
    scene_metadata.update(
        {
            "camera_trajectory": trajectory_np,
            "camera_trajectory_raw": trajectory_np.copy(),
            "camera_trajectory_hsi": trajectory_np.copy(),
            "camera_source": "native_3dpw_cam_poses_T_w2c",
            "gt_camera_oracle": True,
            "gt_camera_pkl": str(gt_path),
            "gt_camera_intrinsics_native": gt["intrinsics"].astype(np.float32),
        }
    )
    with (output_dir / SCENE_METADATA_NAME).open("wb") as file:
        pickle.dump(scene_metadata, file, protocol=pickle.HIGHEST_PROTOCOL)
    viewer_args = dict(first_manifest.get("viewer_args", {}))
    viewer_args.update(
        {
            "gt_camera_oracle": True,
            "gt_camera_pkl": str(gt_path),
            "selected_source_indices": sorted(selected),
        }
    )
    output_manifest = {
        "format": FULL_CACHE_FORMAT,
        "num_frames": len(output_records),
        "scene_metadata_file": SCENE_METADATA_NAME,
        "smpl_faces_file": SMPL_FACES_NAME,
        "camera_parameters": {
            "per_frame": ["intrinsic", "raw_extrinsic", "hsi_extrinsic", "gt_extrinsic", "raw_camera", "hsi_camera", "gt_camera"],
            "scene": ["camera_trajectory", "camera_trajectory_raw", "camera_trajectory_hsi"],
            "source": "native 3DPW cam_poses; predicted intrinsics retained for processed image plane",
            "storage": "frame pickle and scene_metadata.pkl",
        },
        "viewer_args": viewer_args,
        "frames": output_records,
        "note": "GT-camera oracle visualisation; model-predicted depth and SMPL are retained.",
    }
    manifest_path = output_dir / MANIFEST_NAME
    temporary = output_dir / f"{MANIFEST_NAME}.tmp"
    temporary.write_text(json.dumps(output_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(manifest_path)
    incomplete.unlink(missing_ok=True)
    return manifest_path


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(piece) if piece.isdigit() else piece for piece in re.split(r"(\d+)", value))


def load_3dpw_camera(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing 3DPW GT pickle: {path}")
    with path.open("rb") as file:
        payload = pickle.load(file, encoding="latin1")
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid 3DPW payload: {path}")
    cam_poses = np.asarray(payload.get("cam_poses"), dtype=np.float32)
    if cam_poses.ndim != 3 or cam_poses.shape[1:] != (4, 4):
        raise ValueError(f"3DPW cam_poses must have shape [T,4,4], got {cam_poses.shape}")
    intrinsics = np.asarray(payload.get("cam_intrinsics"), dtype=np.float32).reshape(3, 3)
    frame_ids = np.asarray(payload.get("img_frame_ids", np.arange(len(cam_poses))), dtype=np.int64).reshape(-1)
    if frame_ids.shape != (len(cam_poses),):
        raise ValueError(f"3DPW img_frame_ids must have shape [{len(cam_poses)}], got {frame_ids.shape}")
    return {"cam_poses": cam_poses, "intrinsics": intrinsics, "img_frame_ids": frame_ids}


def resolve_gt_frame_index(
    frame_id: str,
    source_index: int,
    gt: dict[str, np.ndarray],
    frame_index_offset: int,
) -> int:
    """Map a cached image filename to the native 3DPW camera-array index."""
    native_ids = np.asarray(gt["img_frame_ids"], dtype=np.int64)
    match = re.search(r"(\d+)$", Path(frame_id).stem)
    if match is not None:
        image_id = int(match.group(1))
        locations = np.flatnonzero(native_ids == image_id)
        if len(locations) == 1:
            return int(locations[0]) + int(frame_index_offset)
    return int(source_index) + int(frame_index_offset)


def place_frame_in_gt_camera_world(frame: dict[str, Any], gt: dict[str, np.ndarray], source_index: int) -> dict[str, Any]:
    result = dict(frame)
    gt_w2c = np.asarray(gt["cam_poses"][source_index], dtype=np.float32)
    gt_r = gt_w2c[:3, :3]
    gt_t = gt_w2c[:3, 3]
    raw_extrinsic = np.asarray(frame["raw_extrinsic"], dtype=np.float32)
    hsi_extrinsic = np.asarray(frame["hsi_extrinsic"], dtype=np.float32)
    for key in WORLD_POINT_KEYS:
        if key in result and result[key] is not None:
            extrinsic = raw_extrinsic if key.startswith("raw_") else hsi_extrinsic
            result[key] = predicted_world_to_gt_world(np.asarray(result[key]), extrinsic, gt_r, gt_t)
    updated_people = []
    for person in result.get("people", []):
        person = dict(person)
        for key in WORLD_PERSON_KEYS:
            if key in person and person[key] is not None:
                extrinsic = raw_extrinsic if key.startswith("base_") else hsi_extrinsic
                person[key] = predicted_world_to_gt_world(np.asarray(person[key]), extrinsic, gt_r, gt_t)
        for key in CAM_PERSON_KEYS:
            if key in person and person[key] is not None:
                person[key] = np.asarray(person[key], dtype=np.float32).copy()
        updated_people.append(person)
    result["people"] = updated_people
    # Keep camera-space fields unchanged, but make the world/extrinsic contract
    # explicit: viewer interactions now use the native 3DPW camera world.
    result["source_frame_index"] = int(source_index)
    result["predicted_raw_extrinsic"] = raw_extrinsic.copy()
    result["predicted_hsi_extrinsic"] = hsi_extrinsic.copy()
    result["predicted_raw_camera"] = dict(result.get("raw_camera", {}))
    result["predicted_hsi_camera"] = dict(result.get("hsi_camera", {}))
    result["raw_extrinsic"] = gt_w2c[:3, :4].copy()
    result["hsi_extrinsic"] = gt_w2c[:3, :4].copy()
    result["gt_extrinsic"] = gt_w2c.copy()
    result["gt_intrinsic_native"] = gt["intrinsics"].copy()
    # The processed/model intrinsic stays in ``intrinsic`` because it is
    # required to interpret the cached depth maps.  Viewer frusta, however,
    # should use native 3DPW intrinsics so their FOV/aspect match the GT camera.
    result["raw_camera"] = camera_pose_from_extrinsic(result["raw_extrinsic"], gt["intrinsics"])
    result["hsi_camera"] = camera_pose_from_extrinsic(result["hsi_extrinsic"], gt["intrinsics"])
    result["camera"] = dict(result["hsi_camera"])
    result["gt_camera"] = dict(result["hsi_camera"])
    return result


def predicted_world_to_gt_world(points: np.ndarray, predicted_w2c: np.ndarray, gt_r: np.ndarray, gt_t: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    shape = values.shape
    flat = values.reshape(-1, 3)
    pred_r = np.asarray(predicted_w2c[:3, :3], dtype=np.float32)
    pred_t = np.asarray(predicted_w2c[:3, 3], dtype=np.float32)
    camera = flat @ pred_r.T + pred_t[None, :]
    world = (camera - gt_t[None, :]) @ gt_r
    return world.reshape(shape).astype(np.float32, copy=False)


def camera_pose_from_extrinsic(extrinsic: np.ndarray, intrinsic: np.ndarray) -> dict[str, Any]:
    rotation = np.asarray(extrinsic[:3, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:3, 3], dtype=np.float32)
    rotation_c2w = rotation.T
    position = -rotation_c2w @ translation
    fx = max(float(intrinsic[0, 0]), 1e-6)
    fy = max(float(intrinsic[1, 1]), 1e-6)
    width = max(float(intrinsic[0, 2]) * 2.0, 1.0)
    height = max(float(intrinsic[1, 2]) * 2.0, 1.0)
    return {
        "rotation_c2w": rotation_c2w,
        "position": position,
        "fov": np.float32(2.0 * np.arctan((height * 0.5) / fy)),
        "aspect": np.float32(width / height),
    }


if __name__ == "__main__":
    main()
