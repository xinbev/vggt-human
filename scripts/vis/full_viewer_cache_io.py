"""Lossless disk cache for replaying the full sequence Viser scene."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

FULL_CACHE_FORMAT = "vggt_omega_full_sequence_viewer_cache_v1"
MANIFEST_NAME = "manifest.json"
SCENE_METADATA_NAME = "scene_metadata.pkl"
SMPL_FACES_NAME = "smpl_faces.npy"
INCOMPLETE_NAME = ".incomplete"


def export_full_sequence_viewer_cache(
    scene: dict[str, Any],
    args: Any,
    cache_dir: str | Path,
) -> Path:
    """Persist every field used by ``SequenceViewer`` without duplicating SMPL faces."""
    root = Path(cache_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    incomplete_path = root / INCOMPLETE_NAME
    incomplete_path.write_text("Cache export is in progress.\n", encoding="utf-8")
    frames = list(scene.get("frames", []))
    if not frames:
        raise ValueError("Cannot export an empty full viewer scene")

    faces = _find_smpl_faces(frames)
    np.save(root / SMPL_FACES_NAME, faces.astype(np.int32, copy=False), allow_pickle=False)

    scene_metadata = {key: value for key, value in scene.items() if key != "frames"}
    with (root / SCENE_METADATA_NAME).open("wb") as file:
        pickle.dump(scene_metadata, file, protocol=pickle.HIGHEST_PROTOCOL)

    records: list[dict[str, Any]] = []
    for position, frame in enumerate(frames):
        frame_file = f"frame_{position:04d}.pkl"
        cached_frame = dict(frame)
        cached_people = []
        for person in frame.get("people", []):
            cached_person = dict(person)
            cached_person.pop("faces", None)
            cached_people.append(cached_person)
        cached_frame["people"] = cached_people
        with (root / frame_file).open("wb") as file:
            pickle.dump(cached_frame, file, protocol=pickle.HIGHEST_PROTOCOL)
        records.append(
            {
                "position": int(position),
                "frame_index": int(frame.get("frame_index", position)),
                "source_frame_index": int(frame.get("source_frame_index", -1)),
                "frame_id": str(frame.get("frame_id", position)),
                "source_image": str(frame.get("image", "")),
                "file": frame_file,
            }
        )
        if (position + 1) % 25 == 0 or position + 1 == len(frames):
            print(f"[full-viewer-cache] exported {position + 1}/{len(frames)} frames", flush=True)

    manifest = {
        "format": FULL_CACHE_FORMAT,
        "num_frames": len(records),
        "scene_metadata_file": SCENE_METADATA_NAME,
        "smpl_faces_file": SMPL_FACES_NAME,
        "camera_parameters": {
            "per_frame": [
                "intrinsic",
                "raw_extrinsic",
                "hsi_extrinsic",
                "raw_camera",
                "hsi_camera",
                "camera",
            ],
            "scene": [
                "camera_trajectory",
                "camera_trajectory_raw",
                "camera_trajectory_hsi",
            ],
            "storage": "frame pickle and scene_metadata.pkl",
        },
        "viewer_args": _json_safe(vars(args)),
        "frames": records,
        "note": "Trusted local pickle cache containing the complete SequenceViewer scene.",
    }
    manifest_path = root / MANIFEST_NAME
    temporary_manifest = root / f"{MANIFEST_NAME}.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    incomplete_path.unlink(missing_ok=True)
    print(f"[full-viewer-cache] ready: {manifest_path}", flush=True)
    return manifest_path


def load_full_sequence_viewer_cache(cache_dir: str | Path) -> tuple[dict[str, Any], SimpleNamespace, Path]:
    """Load a trusted cache and restore the shared SMPL topology on every person."""
    root = Path(cache_dir).expanduser().resolve()
    manifest_path = root / MANIFEST_NAME
    incomplete_path = root / INCOMPLETE_NAME
    if incomplete_path.exists():
        raise RuntimeError(f"Full viewer cache export is incomplete: {incomplete_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing full viewer cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FULL_CACHE_FORMAT:
        raise ValueError(f"Unsupported full viewer cache format: {manifest.get('format')!r}")

    metadata_path = root / str(manifest.get("scene_metadata_file", SCENE_METADATA_NAME))
    faces_path = root / str(manifest.get("smpl_faces_file", SMPL_FACES_NAME))
    if not metadata_path.is_file() or not faces_path.is_file():
        raise FileNotFoundError(f"Incomplete full viewer cache under {root}")
    with metadata_path.open("rb") as file:
        scene = pickle.load(file)  # noqa: S301 - cache is a trusted, project-local artifact.
    if not isinstance(scene, dict):
        raise TypeError(f"Invalid cached scene metadata: {type(scene).__name__}")
    faces = np.load(faces_path, allow_pickle=False).astype(np.int64, copy=False)

    frames = []
    records = manifest.get("frames", [])
    if not isinstance(records, list) or not records:
        raise ValueError(f"Full viewer cache has no frames: {manifest_path}")
    for position, record in enumerate(records):
        frame_path = root / str(record["file"])
        if not frame_path.is_file():
            raise FileNotFoundError(f"Missing cached frame: {frame_path}")
        with frame_path.open("rb") as file:
            frame = pickle.load(file)  # noqa: S301 - cache is a trusted, project-local artifact.
        if not isinstance(frame, dict):
            raise TypeError(f"Invalid cached frame {position}: {type(frame).__name__}")
        for person in frame.get("people", []):
            person["faces"] = faces
        frames.append(frame)
        if (position + 1) % 25 == 0 or position + 1 == len(records):
            print(f"[full-viewer-cache] loaded {position + 1}/{len(records)} frames", flush=True)
    scene["frames"] = frames
    viewer_args = SimpleNamespace(**dict(manifest.get("viewer_args", {})))
    return scene, viewer_args, manifest_path


def apply_full_viewer_startup_overrides(
    viewer_args: SimpleNamespace,
    point_size: float | None = None,
    human_mask_dilation_px: int | None = None,
    filter_human_points: bool | None = None,
) -> SimpleNamespace:
    """Apply validated display/filter overrides to cached SequenceViewer args."""
    if point_size is not None:
        if not 0.0005 <= float(point_size) <= 0.08:
            raise ValueError(f"point_size must be within [0.0005, 0.08], got {point_size}")
        viewer_args.point_size = float(point_size)
    if human_mask_dilation_px is not None:
        if not 0 <= int(human_mask_dilation_px) <= 32:
            raise ValueError(
                f"human_mask_dilation_px must be within [0, 32], got {human_mask_dilation_px}"
            )
        viewer_args.human_mask_dilation_px = int(human_mask_dilation_px)
        viewer_args.rebuild_human_filter_on_start = True
    else:
        viewer_args.rebuild_human_filter_on_start = False
    if filter_human_points is not None:
        viewer_args.filter_human_points = bool(filter_human_points)
    return viewer_args


def _find_smpl_faces(frames: list[dict[str, Any]]) -> np.ndarray:
    for frame in frames:
        for person in frame.get("people", []):
            faces = person.get("faces")
            if faces is not None:
                faces_np = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
                if faces_np.shape[0] > 0:
                    return faces_np
    raise ValueError("Full viewer scene contains no SMPL faces")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)
