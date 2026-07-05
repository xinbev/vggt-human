from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.data.hf_bedlam import HFBedlamDataset, _load_person
from vggt_omega.tracking.query_builder import pixel_mask_to_patch_mask
from vggt_omega.tracking.sam2_masks import SAM2BoxMaskPredictor
from vggt_omega.tracking.schema import Detection
from vggt_omega.training.config import load_yaml_config, require_path


def main() -> None:
    args = parse_args()
    path_config = load_yaml_config(args.path_config)
    images_root = resolve_path(args.images_root or require_path(path_config, "datasets.hf_bedlam_images_root"))
    npz_root = resolve_path(args.npz_root or require_path(path_config, "datasets.hf_bedlam_npz_root"))
    output_root = resolve_path(args.output_root or require_path(path_config, "datasets.hf_bedlam_sam2_patch_masks_root"))
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = HFBedlamDataset(
        images_root=images_root,
        npz_root=npz_root,
        sequence_length=1,
        stride=1,
        image_size=int(args.image_size),
        max_humans=int(args.max_humans),
        require_smpl=True,
        require_boxes=True,
        bbox_expand=float(args.bbox_expand),
        transl_add_cam_ext=bool(args.transl_add_cam_ext),
        skip_missing_images=True,
        max_npz_files=int(args.max_npz_files),
        max_frames=int(args.max_frames),
    )
    predictor = SAM2BoxMaskPredictor(
        sam2_root=resolve_path(args.sam2_root or require_path(path_config, "third_party.sam2_root")),
        checkpoint=resolve_path(args.sam2_checkpoint or require_path(path_config, "third_party.sam2_checkpoint")),
        model_cfg=args.sam2_model_cfg,
        device=args.device,
        multimask_output=not args.sam2_single_mask,
    )

    output_path = output_root / f"{args.output_split}.pkl"
    if output_path.is_file() and not args.overwrite:
        print(f"[hf-bedlam-sam2] exists, skip: {output_path}", flush=True)
        summary = {"cache": str(output_path), "skipped_existing": True}
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return

    frame_items = sorted(dataset._frames.items(), key=lambda item: item[0])  # noqa: SLF001
    if int(args.max_output_frames) > 0:
        frame_items = frame_items[: int(args.max_output_frames)]

    num_patches = (int(args.image_size) // int(args.patch_size)) ** 2
    cache_frames: dict[str, dict[int, dict[str, Any]]] = {}
    stats = {
        "frames": 0,
        "persons": 0,
        "valid_masks": 0,
        "too_small_masks": 0,
        "missing_boxes": 0,
        "missing_images": 0,
        "person_load_errors": 0,
    }
    for frame_idx, (frame_key, frame) in enumerate(frame_items):
        frame_bgr = cv2.imread(str(frame["image_path"]), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            stats["missing_images"] += 1
            continue
        image_h, image_w = frame_bgr.shape[:2]
        data = dataset._load_npz(frame["npz_path"])  # noqa: SLF001
        persons: list[dict[str, Any]] = []
        for person_idx in frame["person_indices"]:
            try:
                persons.append(
                    _load_person(
                        data,
                        int(person_idx),
                        (int(image_h), int(image_w)),
                        float(args.bbox_expand),
                        bool(args.transl_add_cam_ext),
                    )
                )
            except Exception:
                stats["person_load_errors"] += 1
        detections = build_detections(persons, image_w=image_w, image_h=image_h)
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
                image_size=int(args.image_size),
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
                **metadata.get(person_id, {}),
            }
            stats["valid_masks"] += 1
        if packed_by_person:
            cache_frames[str(frame_key)] = packed_by_person
        stats["frames"] += 1
        if int(args.log_interval) > 0 and (frame_idx + 1) % int(args.log_interval) == 0:
            print(
                "[hf-bedlam-sam2] "
                f"processed={frame_idx + 1}/{len(frame_items)} "
                f"frames={stats['frames']} valid_masks={stats['valid_masks']}",
                flush=True,
            )

    cache = {
        "version": 1,
        "split": str(args.output_split),
        "image_size": int(args.image_size),
        "patch_size": int(args.patch_size),
        "num_patches": int(num_patches),
        "mask_patch_threshold": float(args.mask_patch_threshold),
        "min_mask_patches": int(args.min_mask_patches),
        "frames": cache_frames,
    }
    with output_path.open("wb") as file:
        pickle.dump(cache, file, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "images_root": str(images_root),
        "npz_root": str(npz_root),
        "cache": str(output_path),
        "cache_frames": len(cache_frames),
        "num_patches": int(num_patches),
        **stats,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SAM2 patch-mask cache for HF BEDLAM SMPL-base training")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--images-root", default="")
    parser.add_argument("--npz-root", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--output-split", default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam2-root", default="")
    parser.add_argument("--sam2-checkpoint", default="")
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-single-mask", action="store_true")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--max-humans", type=int, default=20)
    parser.add_argument("--bbox-expand", type=float, default=0.15)
    parser.add_argument("--transl-add-cam-ext", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-patch-threshold", type=float, default=0.10)
    parser.add_argument("--min-mask-patches", type=int, default=4)
    parser.add_argument("--max-npz-files", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="Limit source frame index construction in HFBedlamDataset")
    parser.add_argument("--max-output-frames", type=int, default=0, help="Limit frames actually processed by SAM2")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def build_detections(persons: list[dict[str, Any]], image_w: int, image_h: int) -> list[Detection]:
    detections: list[Detection] = []
    for person in persons:
        if not bool(person.get("bbox_valid", False)):
            continue
        box = person.get("bbox_cxcywh_norm")
        if box is None:
            continue
        person_id = int(person.get("person_id", len(detections)))
        xyxy = cxcywh_norm_to_xyxy(np.asarray(box, dtype=np.float32).reshape(4), image_w=image_w, image_h=image_h)
        detections.append(
            Detection(
                bbox_xyxy=[float(v) for v in xyxy],
                score=1.0,
                class_id=0,
                class_name="person",
                source="hf_bedlam_gt_box",
                det_id=person_id,
            )
        )
    return detections


def cxcywh_norm_to_xyxy(box: np.ndarray, image_w: int, image_h: int) -> np.ndarray:
    cx, cy, w, h = [float(v) for v in box.reshape(4)]
    bw = w * float(max(image_w, 1))
    bh = h * float(max(image_h, 1))
    x1 = cx * float(max(image_w, 1)) - 0.5 * bw
    y1 = cy * float(max(image_h, 1)) - 0.5 * bh
    x2 = x1 + bw
    y2 = y1 + bh
    return np.asarray(
        [
            np.clip(x1, 0.0, float(max(image_w - 1, 1))),
            np.clip(y1, 0.0, float(max(image_h - 1, 1))),
            np.clip(x2, 0.0, float(max(image_w - 1, 1))),
            np.clip(y2, 0.0, float(max(image_h - 1, 1))),
        ],
        dtype=np.float32,
    )


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
