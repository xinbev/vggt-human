#!/usr/bin/env python3
"""Generate RGB mask visualizations from a cached VGGT-Omega sequence.

The cache remains the only inference input.  Cached SMPL vertices are projected
to the source image to obtain one box prompt per person, and SAM2 supplies the
pixel masks used for the visualization.  No VGGT, NLF, or SMPL inference is
performed here.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vis.project_cached_smpl_on_rgb import (  # noqa: E402
    choose_vertices,
    infer_processed_hw,
    intrinsic_to_original,
    load_cache,
    load_frame,
    load_people,
    project_vertices,
    require_matrix,
    resolve_source_image,
    select_frames,
)
from vggt_omega.tracking.sam2_masks import SAM2BoxMaskPredictor  # noqa: E402
from vggt_omega.tracking.schema import Detection  # noqa: E402
from vggt_omega.data.geometry import compute_resize_geometry  # noqa: E402


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

    predictor = SAM2BoxMaskPredictor(
        sam2_root=resolve_path(args.sam2_root),
        checkpoint=resolve_path(args.sam2_checkpoint),
        model_cfg=args.sam2_model_cfg,
        device=args.device,
        multimask_output=not args.single_mask,
    )
    results = []
    for record in selections:
        result = render_record(record, cache, args, output_dir, predictor)
        results.append(result)
        print(
            f"[cached-sam2-mask] {result['frame_id']}: "
            f"people={result['people_detected']} mask_pixels={result['union_mask_pixels']} "
            f"image={result['output_image']}",
            flush=True,
        )

    exported_cache = None
    if args.export_cache_dir:
        if not args.all:
            raise ValueError("--export-cache-dir requires --all so the exported cache keeps every frame")
        if cache["kind"] != "full":
            raise ValueError("--export-cache-dir is supported only for full_viewer_cache")
        exported_cache = export_sam2_full_cache(cache, args, results)

    summary = {
        "cache_dir": str(cache_dir),
        "cache_format": cache["format"],
        "sam2_root": str(resolve_path(args.sam2_root)),
        "sam2_checkpoint": str(resolve_path(args.sam2_checkpoint)),
        "sam2_model_cfg": args.sam2_model_cfg,
        "device": args.device,
        "prompt_source": "projected_cached_smpl_vertices",
        "mesh_source": args.mesh_source,
        "camera_source": args.camera_source,
        "box_expand_ratio": float(args.box_expand_ratio),
        "exported_cache": None if exported_cache is None else str(exported_cache),
        "results": results,
    }
    summary_path = output_dir / "sam2_mask_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[cached-sam2-mask] summary={summary_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, help="full_viewer_cache or viewer_cache directory")
    parser.add_argument("--output-dir", default="outputs/vis/cached_sam2_masks")
    parser.add_argument(
        "--export-cache-dir",
        default="",
        help="Optional full-viewer cache rewritten with SAM2-filtered raw/HSI points; requires --all",
    )
    parser.add_argument("--frame-index", type=int, default=None, help="Cache position; default is the first frame")
    parser.add_argument("--frame-id", default="", help="Exact cached frame_id, e.g. image_00042")
    parser.add_argument("--all", action="store_true", help="Process every cached frame")
    parser.add_argument("--image-root", default="", help="Use this directory for source image basenames")
    parser.add_argument("--sam2-root", default="third_party/sam2")
    parser.add_argument("--sam2-checkpoint", default="third_party/weights/sam/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="cuda", help="SAM2 device, usually cuda or cpu")
    parser.add_argument("--single-mask", action="store_true", help="Ask SAM2 for one mask per box")
    parser.add_argument("--mesh-source", choices=("base", "hsi"), default="hsi")
    parser.add_argument("--camera-source", choices=("raw", "hsi"), default="hsi")
    parser.add_argument("--resize-mode", choices=("balanced", "max_size", "square_legacy"), default="balanced")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--box-expand-ratio", type=float, default=0.05)
    parser.add_argument("--mask-alpha", type=int, default=105, help="Mask overlay alpha in [0, 255]")
    parser.add_argument("--draw-boxes", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def render_record(
    record: dict[str, Any],
    cache: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    predictor: SAM2BoxMaskPredictor,
) -> dict[str, Any]:
    mask_alpha = int(args.mask_alpha)
    if not 0 <= mask_alpha <= 255:
        raise ValueError(f"--mask-alpha must be within [0, 255], got {mask_alpha}")
    if float(args.box_expand_ratio) < 0.0 or float(args.box_expand_ratio) >= 1.0:
        raise ValueError("--box-expand-ratio must be within [0, 1)")

    frame = load_frame(cache, record)
    image_path = resolve_source_image(str(record.get("source_image", frame.get("image", ""))), args.image_root)
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Original image not found: {image_path}. Pass --image-root when cache paths point to another machine."
        )
    image = Image.open(image_path).convert("RGB")
    image_rgb = np.asarray(image, dtype=np.uint8)
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

    detections: list[Detection] = []
    prompts: list[dict[str, Any]] = []
    for person_index, person in enumerate(people):
        vertices, coordinate_source = choose_vertices(person, args, extrinsic)
        uv, valid = project_vertices(vertices, projected_intrinsic)
        box = projected_box(uv, valid, image.width, image.height, float(args.box_expand_ratio))
        if box is None:
            continue
        detections.append(
            Detection(
                bbox_xyxy=[float(value) for value in box],
                score=finite_float(person.get("confidence")) or 1.0,
                det_id=person_index,
                source="cached_smpl_projection",
            )
        )
        prompts.append(
            {
                "det_id": int(person_index),
                "person_index": int(person_index),
                "track_id": int(person.get("track_id", -1)),
                "query_index": int(person.get("query_index", -1)),
                "confidence": finite_float(person.get("confidence")),
                "bbox_xyxy": [float(value) for value in box],
                "coordinate_source": coordinate_source,
            }
        )

    # OpenCV/SAM2 expects a contiguous image.  A channel-reversed NumPy view
    # has negative strides and can crash some native OpenCV builds.
    image_bgr = np.ascontiguousarray(image_rgb[..., ::-1])
    masks, mask_meta = predictor.predict_for_detections(image_bgr, detections)
    union = np.zeros((image.height, image.width), dtype=np.bool_)
    overlay = image.convert("RGBA")
    overlay_array = np.asarray(overlay, dtype=np.uint8).copy()
    draw = ImageDraw.Draw(overlay)
    saved_masks: dict[str, np.ndarray] = {}
    reports = []
    for prompt in prompts:
        det_id = int(prompt["det_id"])
        mask = np.asarray(masks.get(det_id, np.zeros_like(union)), dtype=np.bool_)
        if mask.shape != union.shape:
            raise ValueError(f"SAM2 mask shape {mask.shape} does not match image {union.shape}")
        union |= mask
        color = PALETTE[int(prompt["person_index"]) % len(PALETTE)]
        overlay_array[mask, :3] = np.asarray(color, dtype=np.uint8)
        overlay_array[mask, 3] = np.uint8(mask_alpha)
        key = f"person_{int(prompt['person_index']):06d}"
        saved_masks[key] = mask.astype(np.uint8)
        report = dict(prompt)
        report.update(mask_meta.get(det_id, {}))
        reports.append(report)
        if args.draw_boxes:
            x1, y1, x2, y2 = [int(round(value)) for value in prompt["bbox_xyxy"]]
            draw.rectangle((x1, y1, x2, y2), outline=tuple(color) + (255,), width=2)

    overlay = Image.alpha_composite(image.convert("RGBA"), Image.fromarray(overlay_array, mode="RGBA"))
    if args.draw_boxes:
        draw = ImageDraw.Draw(overlay)
        for report in reports:
            x1, y1, x2, y2 = [int(round(value)) for value in report["bbox_xyxy"]]
            color = PALETTE[int(report["person_index"]) % len(PALETTE)]
            label = f"id={report['track_id']} score={report.get('sam2_score', 0.0):.2f}"
            draw.rectangle((x1, y1, x2, y2), outline=color + (255,), width=2)
            draw.text((x1 + 2, max(0, y1 - 14)), label, fill=color + (255,))

    frame_id = str(record.get("frame_id", record.get("position", "frame")))
    output_image = output_dir / f"{frame_id}_sam2_overlay.png"
    union_path = output_dir / f"{frame_id}_sam2_union_mask.png"
    masks_path = output_dir / f"{frame_id}_sam2_masks.npz"
    metadata_path = output_dir / f"{frame_id}_sam2.json"
    overlay.convert("RGB").save(output_image)
    Image.fromarray((union.astype(np.uint8) * 255), mode="L").save(union_path)
    np.savez_compressed(masks_path, **saved_masks)
    metadata = {
        "frame_id": frame_id,
        "position": int(record.get("position", 0)),
        "source_frame_index": int(record.get("source_frame_index", -1)),
        "source_image": str(image_path),
        "image_hw": [int(orig_hw[0]), int(orig_hw[1])],
        "processed_hw": [int(processed_hw[0]), int(processed_hw[1])],
        "intrinsic_processed": intrinsic.tolist(),
        "intrinsic_original": projected_intrinsic.tolist(),
        "pixel_mapping": mapping,
        "prompt_source": "projected_cached_smpl_vertices",
        "mask_source": "sam2_box_prompt",
        "union_mask_pixels": int(union.sum()),
        "people": reports,
        "output_image": str(output_image),
        "union_mask": str(union_path),
        "masks_npz": str(masks_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "frame_id": frame_id,
        "position": int(record.get("position", 0)),
        "output_image": str(output_image),
        "union_mask": str(union_path),
        "masks_npz": str(masks_path),
        "metadata": str(metadata_path),
        "people_detected": len(reports),
        "union_mask_pixels": int(union.sum()),
    }


def export_sam2_full_cache(cache: dict[str, Any], args: argparse.Namespace, results: list[dict[str, Any]]) -> Path:
    """Write a full cache whose filtered point clouds use the SAM2 union mask."""
    source_root = Path(cache["root"])
    output_root = resolve_path(args.export_cache_dir)
    if output_root == source_root:
        raise ValueError("--export-cache-dir must differ from --cache-dir")
    output_root.mkdir(parents=True, exist_ok=True)
    result_by_position = {int(item["position"]): item for item in results}
    manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    metadata_name = str(manifest.get("scene_metadata_file", "scene_metadata.pkl"))
    faces_name = str(manifest.get("smpl_faces_file", "smpl_faces.npy"))
    shutil.copy2(source_root / metadata_name, output_root / metadata_name)
    shutil.copy2(source_root / faces_name, output_root / faces_name)
    records = manifest.get("frames", [])
    expected_positions = set(range(len(records)))
    if set(result_by_position) != expected_positions:
        raise ValueError("SAM2 export result count does not match the full cache frame count")
    for position, record in enumerate(records):
        source_frame_path = source_root / str(record["file"])
        with source_frame_path.open("rb") as file:
            frame = pickle.load(file)
        if not isinstance(frame, dict):
            raise TypeError(f"Invalid cached frame: {source_frame_path}")
        union_path = Path(result_by_position[position]["union_mask"])
        union_original = np.asarray(Image.open(union_path).convert("L"), dtype=np.uint8) > 0
        processed_hw = _cached_frame_hw(frame)
        union_processed = np.asarray(
            Image.fromarray(union_original.astype(np.uint8) * 255, mode="L").resize(
                (processed_hw[1], processed_hw[0]), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        ) > 0
        frame["raw_human_exclusion_mask"] = union_processed
        frame["hsi_human_exclusion_mask"] = union_processed.copy()
        rgb = np.asarray(frame.get("rgb_chw"), dtype=np.float32)
        intrinsic = np.asarray(frame.get("intrinsic"), dtype=np.float32)
        if rgb.shape != (3, *processed_hw) or intrinsic.shape != (3, 3):
            raise ValueError(f"Frame {position} is missing full-cache RGB/camera fields needed for point export")
        for point_prefix in ("raw", "hsi"):
            depth = np.asarray(frame.get(f"{point_prefix}_depth_map"), dtype=np.float32)
            extrinsic = np.asarray(frame.get(f"{point_prefix}_extrinsic"), dtype=np.float32)
            if depth.shape != processed_hw or extrinsic.shape != (3, 4):
                raise ValueError(f"Frame {position} is missing {point_prefix} depth/camera fields needed for point export")
            points, colors = depth_to_world_points_numpy(
                depth,
                rgb,
                intrinsic,
                extrinsic,
                step=int(frame.get("depth_point_stride", 1)),
                max_scene_depth=float(frame.get("max_scene_depth", 0.0)),
                exclude_mask=union_processed,
            )
            frame[f"{point_prefix}_points"] = points
            frame[f"{point_prefix}_colors"] = colors
        destination = output_root / str(record["file"])
        with destination.open("wb") as file:
            pickle.dump(frame, file, protocol=pickle.HIGHEST_PROTOCOL)
    manifest["sam2_mask_source"] = {
        "mask_source": "sam2_box_prompt",
        "prompt_source": "projected_cached_smpl_vertices",
        "source_cache": str(source_root),
    }
    manifest["note"] = "Full cache rewritten with SAM2-filtered raw/HSI point clouds."
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[cached-sam2-mask] exported full cache={output_root}", flush=True)
    return output_root


def _cached_frame_hw(frame: dict[str, Any]) -> tuple[int, int]:
    depth = frame.get("hsi_depth_map", frame.get("raw_depth_map"))
    array = np.asarray(depth)
    if array.ndim != 2:
        raise ValueError(f"Cached depth must be [H,W], got {array.shape}")
    return int(array.shape[0]), int(array.shape[1])


def depth_to_world_points_numpy(
    depth: np.ndarray,
    rgb_chw: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    step: int,
    max_scene_depth: float,
    exclude_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    stride = max(1, int(step))
    ys, xs = np.meshgrid(
        np.arange(0, height, stride, dtype=np.int32),
        np.arange(0, width, stride, dtype=np.int32),
        indexing="ij",
    )
    z = depth[ys, xs]
    fx = max(float(intrinsic[0, 0]), 1e-6)
    fy = max(float(intrinsic[1, 1]), 1e-6)
    x = (xs.astype(np.float32) - float(intrinsic[0, 2])) / fx * z
    y = (ys.astype(np.float32) - float(intrinsic[1, 2])) / fy * z
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > 1e-6)
    if float(max_scene_depth) > 0.0:
        valid &= z <= float(max_scene_depth)
    valid &= ~np.asarray(exclude_mask, dtype=np.bool_)[ys, xs]
    camera = np.stack((x, y, z), axis=-1)[valid].astype(np.float32, copy=False)
    colors_image = np.asarray(rgb_chw, dtype=np.float32)
    if colors_image.shape[-2:] != (height, width):
        resized = Image.fromarray(np.clip(colors_image.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8))
        colors_image = np.asarray(resized.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32).transpose(2, 0, 1) / 255.0
    colors = (colors_image[:, ys, xs].transpose(1, 2, 0)[valid].clip(0.0, 1.0) * 255.0).astype(np.uint8)
    rotation = np.asarray(extrinsic[:3, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:3, 3], dtype=np.float32)
    world = ((camera - translation[None, :]) @ rotation).astype(np.float32, copy=False)
    return world, colors


def projected_box(
    uv: np.ndarray,
    valid: np.ndarray,
    image_width: int,
    image_height: int,
    expand_ratio: float,
) -> np.ndarray | None:
    finite = valid & np.isfinite(uv).all(axis=1)
    if int(finite.sum()) < 3:
        return None
    points = uv[finite]
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    width = max(float(x2 - x1), 1.0)
    height = max(float(y2 - y1), 1.0)
    x1 -= width * expand_ratio
    x2 += width * expand_ratio
    y1 -= height * expand_ratio
    y2 += height * expand_ratio
    box = np.asarray(
        [np.clip(x1, 0.0, image_width - 1.0), np.clip(y1, 0.0, image_height - 1.0),
         np.clip(x2, 0.0, image_width - 1.0), np.clip(y2, 0.0, image_height - 1.0)],
        dtype=np.float32,
    )
    if float(box[2] - box[0]) < 2.0 or float(box[3] - box[1]) < 2.0:
        return None
    return box


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


if __name__ == "__main__":
    main()
