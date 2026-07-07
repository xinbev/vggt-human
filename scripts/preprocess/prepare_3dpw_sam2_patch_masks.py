from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.tracking.query_builder import pixel_mask_to_patch_mask
from vggt_omega.tracking.sam2_masks import SAM2BoxMaskPredictor
from vggt_omega.tracking.schema import Detection
from vggt_omega.training.config import load_yaml_config, require_path
from vggt_omega.data.geometry import compute_resize_geometry


def main() -> None:
    args = parse_args()
    path_config = load_yaml_config(args.path_config)
    root = resolve_path(args.root or require_path(path_config, "datasets.threedpw_root"))
    annotation_root = resolve_path(args.annotation_root or require_path(path_config, "datasets.threedpw_smpl_base_root"))
    output_root = resolve_path(args.output_root or require_path(path_config, "datasets.threedpw_sam2_patch_masks_root"))
    output_root.mkdir(parents=True, exist_ok=True)

    predictor = SAM2BoxMaskPredictor(
        sam2_root=resolve_path(args.sam2_root or require_path(path_config, "third_party.sam2_root")),
        checkpoint=resolve_path(args.sam2_checkpoint or require_path(path_config, "third_party.sam2_checkpoint")),
        model_cfg=args.sam2_model_cfg,
        device=args.device,
        multimask_output=not args.sam2_single_mask,
    )

    summary: dict[str, Any] = {
        "root": str(root),
        "annotation_root": str(annotation_root),
        "output_root": str(output_root),
        "image_size": int(args.image_size),
        "image_resolution": int(args.image_resolution or args.image_size),
        "resize_mode": str(args.resize_mode),
        "patch_size": int(args.patch_size),
        "mask_patch_threshold": float(args.mask_patch_threshold),
        "min_mask_patches": int(args.min_mask_patches),
        "splits": {},
    }
    for split in args.splits:
        split_summary = prepare_split(root, annotation_root, output_root, split, predictor, args)
        summary["splits"][split] = split_summary
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SAM2 patch-mask cache for 3DPW SMPL-base training/eval")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--root", default="")
    parser.add_argument("--annotation-root", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam2-root", default="")
    parser.add_argument("--sam2-checkpoint", default="")
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-single-mask", action="store_true")
    parser.add_argument("--image-size", type=int, default=512, help="Legacy square size fallback; prefer --image-resolution")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", default="balanced", choices=["balanced", "max_size", "square_legacy"])
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--mask-patch-threshold", type=float, default=0.10)
    parser.add_argument("--min-mask-patches", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def prepare_split(
    root: Path,
    annotation_root: Path,
    output_root: Path,
    split: str,
    predictor: SAM2BoxMaskPredictor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    annotation_path = annotation_root / f"{split}.pkl"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"3DPW annotation cache not found: {annotation_path}")
    output_path = output_root / f"{split}.pkl"
    if output_path.is_file() and not args.overwrite:
        print(f"[3dpw-sam2] split={split} exists, skip: {output_path}", flush=True)
        return {"cache": str(output_path), "skipped_existing": True}

    with annotation_path.open("rb") as file:
        annotation = pickle.load(file)
    frames = annotation.get("frames")
    if not isinstance(frames, dict):
        raise TypeError(f"Invalid 3DPW annotation cache: {annotation_path}")

    frame_items = list(sorted(frames.items()))
    if int(args.max_frames) > 0:
        frame_items = frame_items[: int(args.max_frames)]

    max_num_patches = 0
    cache_frames: dict[str, dict[int, dict[str, Any]]] = {}
    stats = {
        "frames": 0,
        "persons": 0,
        "valid_masks": 0,
        "too_small_masks": 0,
        "missing_boxes": 0,
        "missing_images": 0,
    }
    for frame_idx, (frame_key, frame) in enumerate(frame_items):
        image_relpath = str(frame["image_relpath"])
        image_path = root / "imageFiles" / image_relpath
        frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            stats["missing_images"] += 1
            continue
        image_h, image_w = frame_bgr.shape[:2]
        geometry = compute_resize_geometry(
            (int(image_h), int(image_w)),
            image_resolution=int(args.image_resolution or args.image_size),
            patch_size=int(args.patch_size),
            mode=str(args.resize_mode),
        )
        target_hw = geometry.input_hw
        grid_hw = (target_hw[0] // int(args.patch_size), target_hw[1] // int(args.patch_size))
        num_patches = int(grid_hw[0] * grid_hw[1])
        max_num_patches = max(max_num_patches, num_patches)
        detections = build_detections(frame.get("persons", []))
        if not detections:
            stats["missing_boxes"] += 1
            continue
        masks, metadata = predictor.predict_for_detections(frame_bgr, detections)
        packed_by_person: dict[int, dict[str, Any]] = {}
        for det in detections:
            person_id = int(det.det_id)
            stats["persons"] += 1
            pixel_mask = masks.get(person_id)
            if pixel_mask is None:
                continue
            patch_mask = pixel_mask_to_patch_mask(
                pixel_mask,
                image_hw=target_hw,
                patch_size=int(args.patch_size),
                threshold=float(args.mask_patch_threshold),
            ).reshape(-1)
            patch_count = int(patch_mask.sum())
            if patch_count < int(args.min_mask_patches):
                stats["too_small_masks"] += 1
                continue
            packed_by_person[person_id] = {
                "bits": np.packbits(patch_mask.astype(np.uint8)),
                "patch_count": patch_count,
                "target_hw": [int(target_hw[0]), int(target_hw[1])],
                "grid_hw": [int(grid_hw[0]), int(grid_hw[1])],
                "num_patches": int(num_patches),
                "resize_mode": str(args.resize_mode),
                "image_resolution": int(args.image_resolution or args.image_size),
                **metadata.get(person_id, {}),
            }
            stats["valid_masks"] += 1
        if packed_by_person:
            cache_frames[str(frame_key)] = packed_by_person
        stats["frames"] += 1
        if int(args.log_interval) > 0 and (frame_idx + 1) % int(args.log_interval) == 0:
            print(
                "[3dpw-sam2] "
                f"split={split} processed={frame_idx + 1}/{len(frame_items)} "
                f"frames={stats['frames']} valid_masks={stats['valid_masks']}",
                flush=True,
            )

    cache = {
        "version": 1,
        "split": split,
        "image_size": int(args.image_size),
        "image_resolution": int(args.image_resolution or args.image_size),
        "resize_mode": str(args.resize_mode),
        "patch_size": int(args.patch_size),
        "num_patches": int(max_num_patches),
        "mask_patch_threshold": float(args.mask_patch_threshold),
        "min_mask_patches": int(args.min_mask_patches),
        "frames": cache_frames,
    }
    with output_path.open("wb") as file:
        pickle.dump(cache, file, protocol=pickle.HIGHEST_PROTOCOL)
    out_summary = {**stats, "cache": str(output_path), "cache_frames": len(cache_frames), "num_patches": int(max_num_patches)}
    print(f"[3dpw-sam2] split={split} -> {output_path} {out_summary}", flush=True)
    return out_summary


def build_detections(persons: list[dict[str, Any]]) -> list[Detection]:
    detections: list[Detection] = []
    for person in persons:
        if not bool(person.get("bbox_valid", False)):
            continue
        box = person.get("bbox_xyxy_pixels")
        if box is None:
            continue
        person_id = int(person.get("person_id", len(detections)))
        detections.append(
            Detection(
                bbox_xyxy=[float(v) for v in np.asarray(box, dtype=np.float32).reshape(4)],
                score=1.0,
                class_id=0,
                class_name="person",
                source="3dpw_gt_box",
                det_id=person_id,
            )
        )
    return detections


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
