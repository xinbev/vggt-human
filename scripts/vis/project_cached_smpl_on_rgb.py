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
        "render_scale": int(args.render_scale),
        "fill_alpha": int(args.fill_alpha),
        "wireframe": bool(args.wireframe),
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
    parser.add_argument("--fill-alpha", type=int, default=225, help="Mesh opacity, 255 is opaque")
    parser.add_argument("--edge-alpha", type=int, default=235)
    parser.add_argument("--render-scale", type=int, default=2, help="Supersampling factor for smooth boundaries")
    parser.add_argument("--wireframe", action="store_true", help="Draw dark triangle edges over the shaded mesh")
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
    render_scale = max(1, int(args.render_scale))
    render_hw = (int(image.height) * render_scale, int(image.width) * render_scale)
    mesh_rgb = np.zeros((render_hw[0], render_hw[1], 3), dtype=np.uint8)
    mesh_alpha = np.zeros((render_hw[0], render_hw[1]), dtype=np.uint8)
    mesh_depth = np.full((render_hw[0], render_hw[1]), np.inf, dtype=np.float32)
    render_reports = []
    for person_index, person in enumerate(people):
        vertices, coordinate_source = choose_vertices(person, args, extrinsic)
        uv, valid = project_vertices(vertices, projected_intrinsic)
        report = rasterize_smooth_mesh(
            mesh_rgb,
            mesh_alpha,
            mesh_depth,
            uv,
            vertices,
            vertices[:, 2],
            valid,
            cache["faces"],
            color=PALETTE[person_index % len(PALETTE)],
            fill_alpha=int(args.fill_alpha),
            face_stride=max(1, int(args.face_stride)),
            draw_fill=not args.no_fill,
            render_scale=render_scale,
        )
        if args.wireframe:
            report["wireframe_edges"] = draw_wireframe(
                mesh_rgb,
                mesh_alpha,
                uv,
                valid,
                cache["faces"],
                color=PALETTE[person_index % len(PALETTE)],
                line_width=max(1, int(args.line_width)),
                edge_alpha=int(args.edge_alpha),
                render_scale=render_scale,
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

    mesh_rgba = np.concatenate((mesh_rgb, mesh_alpha[..., None]), axis=-1)
    mesh_layer = Image.fromarray(mesh_rgba, mode="RGBA")
    if render_scale != 1:
        mesh_layer = mesh_layer.resize(image.size, Image.Resampling.LANCZOS)
    composited = Image.alpha_composite(image.convert("RGBA"), mesh_layer).convert("RGB")
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
        "render": {
            "render_scale": int(args.render_scale),
            "fill_alpha": int(args.fill_alpha),
            "wireframe": bool(args.wireframe),
        },
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


def rasterize_smooth_mesh(
    mesh_rgb: np.ndarray,
    mesh_alpha: np.ndarray,
    mesh_depth: np.ndarray,
    uv: np.ndarray,
    vertices: np.ndarray,
    depth: np.ndarray,
    valid_vertices: np.ndarray,
    faces: np.ndarray,
    color: tuple[int, int, int],
    fill_alpha: int,
    face_stride: int,
    draw_fill: bool,
    render_scale: int,
) -> dict[str, int]:
    """Software rasterizer with a z-buffer and interpolated vertex normals."""
    height, width = mesh_depth.shape
    scale = max(1, int(render_scale))
    scaled_uv = uv * float(scale)
    normals = vertex_normals(vertices, faces, valid_vertices)
    valid_faces = 0
    for face in faces[::face_stride]:
        if not valid_vertices[face].all():
            continue
        points = scaled_uv[face]
        if not np.isfinite(points).all():
            continue
        if max(points[:, 0]) < 0 or min(points[:, 0]) >= width or max(points[:, 1]) < 0 or min(points[:, 1]) >= height:
            continue
        x0 = max(0, int(np.floor(np.min(points[:, 0]))))
        x1 = min(width - 1, int(np.ceil(np.max(points[:, 0]))))
        y0 = max(0, int(np.floor(np.min(points[:, 1]))))
        y1 = min(height - 1, int(np.ceil(np.max(points[:, 1]))))
        if x1 < x0 or y1 < y0:
            continue
        p0, p1, p2 = points.astype(np.float32)
        denominator = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(float(denominator)) < 1e-6:
            continue
        grid_x, grid_y = np.meshgrid(
            np.arange(x0, x1 + 1, dtype=np.float32) + 0.5,
            np.arange(y0, y1 + 1, dtype=np.float32) + 0.5,
        )
        w0 = ((p1[1] - p2[1]) * (grid_x - p2[0]) + (p2[0] - p1[0]) * (grid_y - p2[1])) / denominator
        w1 = ((p2[1] - p0[1]) * (grid_x - p2[0]) + (p0[0] - p2[0]) * (grid_y - p2[1])) / denominator
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not bool(inside.any()):
            continue
        z = w0 * depth[face[0]] + w1 * depth[face[1]] + w2 * depth[face[2]]
        closer = inside & (z < mesh_depth[y0 : y1 + 1, x0 : x1 + 1])
        if not bool(closer.any()):
            continue
        normal = (
            w0[..., None] * normals[face[0]]
            + w1[..., None] * normals[face[1]]
            + w2[..., None] * normals[face[2]]
        )
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
        light = np.asarray([0.25, -0.35, 0.90], dtype=np.float32)
        light /= np.linalg.norm(light)
        diffuse = np.abs(np.sum(normal * light, axis=-1))
        # Ambient + diffuse + a restrained camera-facing highlight gives the
        # dense SMPL surface a continuous purple material instead of flat blocks.
        brightness = 0.38 + 0.62 * diffuse
        shaded = np.clip(np.asarray(color, dtype=np.float32)[None, None, :] * brightness[..., None], 0.0, 255.0)
        target_rgb = mesh_rgb[y0 : y1 + 1, x0 : x1 + 1]
        target_alpha = mesh_alpha[y0 : y1 + 1, x0 : x1 + 1]
        target_depth = mesh_depth[y0 : y1 + 1, x0 : x1 + 1]
        target_depth[closer] = z[closer]
        target_rgb[closer] = shaded[closer].astype(np.uint8)
        target_alpha[closer] = max(0, min(255, int(fill_alpha))) if draw_fill else 0
        valid_faces += 1
    return {"faces_drawn": int(valid_faces), "vertices_valid": int(valid_vertices.sum())}


def vertex_normals(vertices: np.ndarray, faces: np.ndarray, valid_vertices: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float32)
    for face in faces:
        if not valid_vertices[face].all():
            continue
        p0, p1, p2 = vertices[face]
        normal = np.cross(p1 - p0, p2 - p0).astype(np.float32)
        length = float(np.linalg.norm(normal))
        if length > 1e-8:
            normal /= length
            normals[face] += normal
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1e-6)
    normals[~valid_vertices] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return normals


def draw_wireframe(
    mesh_rgb: np.ndarray,
    mesh_alpha: np.ndarray,
    uv: np.ndarray,
    valid_vertices: np.ndarray,
    faces: np.ndarray,
    color: tuple[int, int, int],
    line_width: int,
    edge_alpha: int,
    render_scale: int,
) -> int:
    """Optional diagnostic wireframe; smooth shading remains the default."""
    image = Image.fromarray(np.concatenate((mesh_rgb, mesh_alpha[..., None]), axis=-1), mode="RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    edge_color = tuple(max(0, int(channel * 0.35)) for channel in color) + (max(0, min(255, int(edge_alpha))),)
    drawn = 0
    for face in faces:
        if not valid_vertices[face].all() or not np.isfinite(uv[face]).all():
            continue
        points = [(int(round(float(uv[index, 0]) * render_scale)), int(round(float(uv[index, 1]) * render_scale))) for index in face]
        draw.line(points + [points[0]], fill=edge_color, width=max(1, int(line_width) * int(render_scale)), joint="curve")
        drawn += 1
    updated = np.asarray(image, dtype=np.uint8)
    mesh_rgb[...] = updated[..., :3]
    mesh_alpha[...] = updated[..., 3]
    return drawn


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


if __name__ == "__main__":
    main()
