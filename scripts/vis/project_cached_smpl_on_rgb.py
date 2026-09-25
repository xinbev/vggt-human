#!/usr/bin/env python3
"""Project SMPL meshes from a viewer cache onto their original RGB frames.

The cache stores camera-space vertices, world-space vertices, the processed
image intrinsic, and the original image path.  This script keeps that camera
contract intact and only maps projected pixels back through the same
crop/resize/pad geometry used by the inference loader.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.data.geometry import compute_resize_geometry


PALETTE = (
    (126, 78, 226),
    (32, 169, 232),
    (226, 118, 48),
    (48, 188, 110),
    (220, 70, 128),
    (210, 170, 46),
)


def main() -> None:
    args = parse_args()
    cache_dir = resolve_path(args.cache_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_dir)
    selections = select_frames(cache["records"], args)
    if not selections:
        raise RuntimeError("No cache frames selected")

    results = []
    for record in selections:
        result = render_record(record, cache, args, output_dir)
        results.append(result)
        print(
            f"[cached-smpl-overlay] {result['frame_id']}: "
            f"people={result['people_rendered']} image={result['output_image']}",
            flush=True,
        )

    summary = {
        "cache_dir": str(cache_dir),
        "cache_format": cache["format"],
        "mesh_source": args.mesh_source,
        "camera_source": args.camera_source,
        "resize_mode": args.resize_mode,
        "image_resolution": int(args.image_resolution),
        "patch_size": int(args.patch_size),
        "results": results,
    }
    summary_path = output_dir / "projection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[cached-smpl-overlay] summary={summary_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, help="full_viewer_cache or viewer_cache directory")
    parser.add_argument("--output-dir", default="outputs/vis/cached_smpl_projection")
    parser.add_argument("--frame-index", type=int, default=None, help="Cache position; default is the first frame")
    parser.add_argument("--frame-id", default="", help="Exact cached frame_id, e.g. image_00042")
    parser.add_argument("--all", action="store_true", help="Render every cached frame")
    parser.add_argument("--image-root", default="", help="Use this directory instead of each record's source image parent")
    parser.add_argument("--mesh-source", choices=("base", "hsi"), default="hsi")
    parser.add_argument("--camera-source", choices=("raw", "hsi"), default="hsi")
    parser.add_argument("--resize-mode", choices=("balanced", "max_size", "square_legacy"), default="balanced")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--face-stride", type=int, default=1, help="Draw every Nth face; 1 keeps the full mesh")
    parser.add_argument("--line-width", type=int, default=1)
    parser.add_argument("--fill-alpha", type=int, default=92)
    parser.add_argument("--edge-alpha", type=int, default=235)
    parser.add_argument("--no-fill", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_cache(cache_dir: Path) -> dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_format = str(manifest.get("format", ""))
    records = manifest.get("frames")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Cache has no frame records: {manifest_path}")
    faces_name = str(manifest.get("smpl_faces_file", manifest.get("faces_file", "smpl_faces.npy")))
    faces_path = cache_dir / faces_name
    if not faces_path.is_file():
        raise FileNotFoundError(f"Missing SMPL topology: {faces_path}")
    faces = np.asarray(np.load(faces_path, allow_pickle=False), dtype=np.int64).reshape(-1, 3)
    if faces.size == 0:
        raise ValueError(f"SMPL topology is empty: {faces_path}")
    if cache_format == "vggt_omega_full_sequence_viewer_cache_v1":
        kind = "full"
    elif cache_format == "vggt_omega_sequence_viewer_cache_v1":
        kind = "light"
    else:
        raise ValueError(f"Unsupported viewer cache format: {cache_format!r}")
    return {
        "format": cache_format,
        "kind": kind,
        "root": cache_dir,
        "records": records,
        "faces": faces,
        "scene_metadata": manifest.get("scene_metadata", {}),
    }


def select_frames(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.all and (args.frame_index is not None or args.frame_id):
        raise ValueError("--all cannot be combined with --frame-index or --frame-id")
    if args.all:
        return records
    if args.frame_id:
        matches = [record for record in records if str(record.get("frame_id")) == args.frame_id]
        if not matches:
            raise KeyError(f"No cached frame_id={args.frame_id!r}")
        return matches
    index = 0 if args.frame_index is None else int(args.frame_index)
    if index < 0 or index >= len(records):
        raise IndexError(f"frame-index {index} outside [0, {len(records) - 1}]")
    return [records[index]]


def render_record(record: dict[str, Any], cache: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    frame = load_frame(cache, record)
    image_path = resolve_source_image(str(record.get("source_image", frame.get("image", ""))), args.image_root)
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Original image not found: {image_path}. Pass --image-root when cache paths point to another machine."
        )
    image = Image.open(image_path).convert("RGB")
    orig_hw = (image.height, image.width)
    processed_hw = infer_processed_hw(frame, cache)
    intrinsic = require_matrix(frame, "intrinsic", (3, 3))
    extrinsic = require_matrix(frame, f"{args.camera_source}_extrinsic", (3, 4))
    geometry = compute_resize_geometry(
        orig_hw,
        image_resolution=int(args.image_resolution),
        patch_size=int(args.patch_size),
        mode=args.resize_mode,
    )
    projected_intrinsic, mapping = intrinsic_to_original(intrinsic, orig_hw, processed_hw, geometry)
    people = load_people(frame, cache, args)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    render_reports = []
    for person_index, person in enumerate(people):
        vertices, coordinate_source = choose_vertices(person, args, extrinsic)
        uv, valid = project_vertices(vertices, projected_intrinsic)
        report = draw_mesh(
            overlay,
            uv,
            vertices[:, 2],
            valid,
            cache["faces"],
            color=PALETTE[person_index % len(PALETTE)],
            line_width=max(1, int(args.line_width)),
            fill_alpha=int(args.fill_alpha),
            edge_alpha=int(args.edge_alpha),
            face_stride=max(1, int(args.face_stride)),
            draw_fill=not args.no_fill,
        )
        report.update(
            {
                "person_index": person_index,
                "track_id": int(person.get("track_id", -1)),
                "query_index": int(person.get("query_index", -1)),
                "confidence": finite_float(person.get("confidence")),
                "coordinate_source": coordinate_source,
            }
        )
        render_reports.append(report)

    composited = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    frame_id = str(record.get("frame_id", record.get("position", "frame")))
    output_image = output_dir / f"{frame_id}_smpl_overlay.png"
    composited.save(output_image)
    metadata = {
        "frame_id": frame_id,
        "position": int(record.get("position", 0)),
        "source_frame_index": int(record.get("source_frame_index", -1)),
        "source_image": str(image_path),
        "image_hw": [int(orig_hw[0]), int(orig_hw[1])],
        "processed_hw": [int(processed_hw[0]), int(processed_hw[1])],
        "intrinsic_processed": intrinsic.tolist(),
        "intrinsic_original": projected_intrinsic.tolist(),
        "extrinsic": extrinsic.tolist(),
        "pixel_mapping": mapping,
        "people": render_reports,
        "output_image": str(output_image),
    }
    (output_dir / f"{frame_id}_smpl_overlay.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "frame_id": frame_id,
        "output_image": str(output_image),
        "people_rendered": len(render_reports),
        "people": render_reports,
    }


def load_frame(cache: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    path = cache["root"] / str(record["file"])
    if cache["kind"] == "full":
        with path.open("rb") as file:
            frame = pickle.load(file)  # trusted project-local inference cache
        if not isinstance(frame, dict):
            raise TypeError(f"Invalid cached frame: {path}")
        return frame
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files} | {"_record": record}


def load_people(frame: dict[str, Any], cache: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    if cache["kind"] == "full":
        people = frame.get("people", [])
        if not isinstance(people, list):
            raise TypeError("Full cache frame 'people' must be a list")
        return people
    vertices = np.asarray(frame.get("smpl_vertices", np.empty((0, 0, 3))), dtype=np.float32)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"Light cache smpl_vertices must be [P,V,3], got {vertices.shape}")
    track_ids = np.asarray(frame.get("smpl_track_ids", np.full((vertices.shape[0],), -1))).reshape(-1)
    query_indices = np.asarray(frame.get("smpl_query_indices", np.full((vertices.shape[0],), -1))).reshape(-1)
    confidences = np.asarray(frame.get("smpl_confidences", np.full((vertices.shape[0],), np.nan))).reshape(-1)
    people = []
    for index, mesh in enumerate(vertices):
        people.append(
            {
                "hsi_vertices" if args.mesh_source == "hsi" else "base_vertices": mesh,
                "track_id": int(track_ids[index]) if index < len(track_ids) else -1,
                "query_index": int(query_indices[index]) if index < len(query_indices) else -1,
                "confidence": float(confidences[index]) if index < len(confidences) else float("nan"),
            }
        )
    return people


def choose_vertices(person: dict[str, Any], args: argparse.Namespace, extrinsic: np.ndarray) -> tuple[np.ndarray, str]:
    cam_key = f"{args.mesh_source}_vertices_cam"
    world_key = f"{args.mesh_source}_vertices"
    if args.mesh_source == args.camera_source and person.get(cam_key) is not None:
        vertices = np.asarray(person[cam_key], dtype=np.float32).reshape(-1, 3)
        return vertices, cam_key
    if person.get(world_key) is None:
        raise ValueError(f"Person has neither {cam_key} nor {world_key}")
    world = np.asarray(person[world_key], dtype=np.float32).reshape(-1, 3)
    return (world @ extrinsic[:, :3].T + extrinsic[:, 3]), world_key + "->camera"


def require_matrix(frame: dict[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    value = frame.get(key)
    if value is None:
        raise ValueError(f"Cache frame is missing {key}")
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != shape:
        raise ValueError(f"{key} must have shape {shape}, got {matrix.shape}")
    return matrix


def resolve_source_image(value: str, image_root: str) -> Path:
    source = Path(value).expanduser()
    if image_root:
        root = Path(image_root).expanduser().resolve()
        return root / source.name
    return source.resolve() if source.is_absolute() else (ROOT / source).resolve()


def infer_processed_hw(frame: dict[str, Any], cache: dict[str, Any]) -> tuple[int, int]:
    rgb = frame.get("rgb_chw")
    if rgb is not None:
        array = np.asarray(rgb)
        if array.ndim == 3:
            return int(array.shape[-2]), int(array.shape[-1])
    metadata_hw = frame.get("image_hw")
    if metadata_hw is None:
        metadata = cache.get("scene_metadata", {})
        if isinstance(metadata, dict):
            metadata_hw = metadata.get("image_hw")
    if metadata_hw is None:
        metadata_path = cache["root"] / "scene_metadata.pkl"
        if metadata_path.is_file():
            with metadata_path.open("rb") as file:
                metadata = pickle.load(file)
            metadata_hw = metadata.get("image_hw") if isinstance(metadata, dict) else None
    if metadata_hw is not None:
        values = np.asarray(metadata_hw).reshape(-1)
        if values.size >= 2:
            return int(values[0]), int(values[1])
    intrinsic = np.asarray(frame["intrinsic"])
    return max(1, int(round(float(intrinsic[1, 2]) * 2))), max(1, int(round(float(intrinsic[0, 2]) * 2)))


def intrinsic_to_original(
    intrinsic: np.ndarray,
    orig_hw: tuple[int, int],
    processed_hw: tuple[int, int],
    geometry: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    resized_h, resized_w = [int(value) for value in geometry.resized_hw]
    processed_h, processed_w = [int(value) for value in processed_hw]
    if processed_h < resized_h or processed_w < resized_w:
        raise ValueError(
            f"Cached processed image {processed_hw} is smaller than reconstructed resize {geometry.resized_hw}; "
            "pass the same --image-resolution/--resize-mode used during inference."
        )
    pad_h = processed_h - resized_h
    pad_w = processed_w - resized_w
    pad_top = (pad_h // (2 * int(geometry.patch_size))) * int(geometry.patch_size)
    pad_left = (pad_w // (2 * int(geometry.patch_size))) * int(geometry.patch_size)
    sx = float(geometry.scale_xy[0])
    sy = float(geometry.scale_xy[1])
    x1, y1, _, _ = [int(value) for value in geometry.crop_xyxy]
    # uv_processed = A * uv_original + b, so K_original follows the same map.
    mapped = np.asarray(intrinsic, dtype=np.float32).copy()
    mapped[0, 0] = mapped[0, 0] / max(sx, 1e-8)
    mapped[0, 2] = (mapped[0, 2] - float(pad_left)) / max(sx, 1e-8) + float(x1)
    mapped[1, 1] = mapped[1, 1] / max(sy, 1e-8)
    mapped[1, 2] = (mapped[1, 2] - float(pad_top)) / max(sy, 1e-8) + float(y1)
    return mapped, {
        "crop_xyxy": [x1, y1, int(geometry.crop_xyxy[2]), int(geometry.crop_xyxy[3])],
        "resized_hw": [resized_h, resized_w],
        "processed_hw": [processed_h, processed_w],
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "scale_xy": [sx, sy],
    }


def project_vertices(vertices: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 1e-5)
    uv = np.full((vertices.shape[0], 2), np.nan, dtype=np.float32)
    z = vertices[valid, 2]
    uv[valid, 0] = vertices[valid, 0] / z * float(intrinsic[0, 0]) + float(intrinsic[0, 2])
    uv[valid, 1] = vertices[valid, 1] / z * float(intrinsic[1, 1]) + float(intrinsic[1, 2])
    return uv, valid


def draw_mesh(
    overlay: Image.Image,
    uv: np.ndarray,
    depth: np.ndarray,
    valid_vertices: np.ndarray,
    faces: np.ndarray,
    color: tuple[int, int, int],
    line_width: int,
    fill_alpha: int,
    edge_alpha: int,
    face_stride: int,
    draw_fill: bool,
) -> dict[str, int]:
    width, height = overlay.size
    candidates = faces[::face_stride]
    valid_faces = []
    for face in candidates:
        if not valid_vertices[face].all():
            continue
        points = uv[face]
        if not np.isfinite(points).all():
            continue
        if max(points[:, 0]) < 0 or min(points[:, 0]) >= width or max(points[:, 1]) < 0 or min(points[:, 1]) >= height:
            continue
        valid_faces.append((float(np.mean(depth[face])), face))
    valid_faces.sort(key=lambda item: item[0], reverse=True)
    draw = ImageDraw.Draw(overlay, "RGBA")
    fill = (*color, max(0, min(255, int(fill_alpha))))
    # A darker outline keeps the individual SMPL triangles visible over the fill.
    edge_color = tuple(max(0, int(channel * 0.42)) for channel in color)
    edge = (*edge_color, max(0, min(255, int(edge_alpha))))
    for _, face in valid_faces:
        points = [(int(round(uv[index, 0])), int(round(uv[index, 1]))) for index in face]
        if draw_fill:
            draw.polygon(points, fill=fill)
        draw.line(points + [points[0]], fill=edge, width=max(1, int(line_width)), joint="curve")
    return {"faces_drawn": len(valid_faces), "vertices_valid": int(valid_vertices.sum())}


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


if __name__ == "__main__":
    main()
