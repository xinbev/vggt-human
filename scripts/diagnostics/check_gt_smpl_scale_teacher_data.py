#!/usr/bin/env python
"""Pre-training validation for GT-SMPL scale-teacher data.

This script checks the exact geometry used by the Stage1 GT-visible-SMPL
scale teacher before training starts:

1. BEDLAM processed RGB/depth/K are loaded through the training dataset.
2. VGGT baseline predicts raw depth on the same processed image plane.
3. GT SMPL vertices are projected with dataset K into RGB/depth.
4. Local GT-depth visibility filtering is applied exactly like the loss.
5. Per-person accepted/skipped status, teacher scale, and visual overlays are
   exported for manual inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

# Compatibility patch for old chumpy on Python 3.11+.
import inspect
from collections import namedtuple

if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = getargspec

if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "complex"):
    np.complex = complex

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.evaluate_hsi_refine_metrics import (  # noqa: E402
    canonical_depth,
    decode_smpl_batch,
    project_points,
    scale_points_to_depth,
)
from scripts.train.train_smpl import apply_overrides, build_model, load_initial_checkpoint, load_yaml_config  # noqa: E402
from scripts.vis.visualize_hsi_depth_smpl_diagnostics import find_bedlam_window_index, tensor_image_to_pil  # noqa: E402
from vggt_omega.data import BedlamDataset  # noqa: E402
from vggt_omega.data.bedlam_boxes import cxcywh_norm_to_xyxy  # noqa: E402
from vggt_omega.data.geometry import resolve_image_size_config  # noqa: E402
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.training.config import deep_update, require_path  # noqa: E402
from vggt_omega.training.hungarian_losses import (  # noqa: E402
    _robust_median_filter,
    _sample_depth_pair_nearest_to_point_z,
    _subsample_points_per_person,
)
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402
from vggt_omega.utils.rotation import rot6d_to_axis_angle  # noqa: E402


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args)
    dataset = build_dataset(config, args)
    indices = select_indices(dataset, args, config)

    model = build_model(config).to(device)
    load_initial_checkpoint(model, config, device)
    model.eval()
    smpl = SMPLLayer(require_path(config, "assets.smpl_model_dir", allow_empty=False)).to(device).eval()

    all_summaries: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample_idx, dataset_idx in enumerate(indices):
            sample = dataset[dataset_idx]
            batch = {
                key: value.unsqueeze(0).to(device, non_blocking=True)
                for key, value in sample.items()
                if isinstance(value, torch.Tensor)
            }
            predictions = model(batch["images"])
            sample_dir = output_dir / f"sample_{sample_idx:03d}_dataset_{dataset_idx:06d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            all_summaries.append(validate_sample(sample_dir, dataset, dataset_idx, sample, batch, predictions, smpl, args))

    summary = {
        "num_samples": len(all_summaries),
        "samples": all_summaries,
        "global": aggregate_summary(all_summaries),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary)
    print(json.dumps({"summary": str(summary_path), "output_dir": str(output_dir)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GT-SMPL visible scale-teacher data before Stage1 training")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_hsi_nlf_provider.yaml")
    parser.add_argument("--output-dir", default="outputs/debug/gt_smpl_scale_teacher_data")
    parser.add_argument("--split", default="Training")
    parser.add_argument("--image", default="", help="Optional BEDLAM image path used as the first frame of a window")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--source", choices=("vertices", "joints"), default="vertices")
    parser.add_argument("--visibility-tolerance-m", type=float, default=0.20)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--max-points-per-person", type=int, default=512)
    parser.add_argument("--min-points-per-person", type=int, default=32)
    parser.add_argument("--min-visible-points", type=int, default=128)
    parser.add_argument("--mad-multiplier", type=float, default=2.5)
    parser.add_argument("--overlay-max-points-per-person", type=int, default=260)
    parser.add_argument("--filtered-overlay-max-points", type=int, default=160)
    parser.add_argument("--scale-warn-low", type=float, default=1.0)
    parser.add_argument("--scale-warn-high", type=float, default=200.0)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.train_config))
    config = apply_overrides(config, args.override)
    config.setdefault("model", {})["enable_camera"] = True
    config.setdefault("model", {})["enable_depth"] = True
    config.setdefault("model", {})["enable_smpl"] = False
    config.setdefault("model", {})["enable_hsi_refine"] = False
    config.setdefault("data", {})["require_depth"] = True
    config.setdefault("data", {})["require_boxes"] = True
    config.setdefault("checkpoint", {})["load_vggt_baseline"] = True
    return config


def build_dataset(config: dict[str, Any], args: argparse.Namespace) -> BedlamDataset:
    data_cfg = config["data"]
    image_size, image_resolution = resolve_image_size_config(data_cfg)
    return BedlamDataset(
        root=require_path(config, data_cfg.get("root_key", "datasets.bedlam_root"), allow_empty=False),
        split=str(args.split),
        sequence_length=int(data_cfg["sequence_length"]),
        stride=int(data_cfg["stride"]),
        image_size=image_size,
        image_resolution=image_resolution,
        resize_mode=str(data_cfg.get("resize_mode", "balanced")),
        max_humans=int(data_cfg.get("max_humans", config.get("model", {}).get("num_smpl_queries", 20))),
        require_smpl=True,
        require_depth=True,
        boxes_root=require_path(config, data_cfg["boxes_root_key"], allow_empty=False),
        require_boxes=True,
        query_source=str(data_cfg.get("query_source", "persons")),
        patch_size=int(config.get("model", {}).get("patch_size", 16)),
        mask_patch_threshold=float(data_cfg.get("mask_patch_threshold", 0.10)),
        min_mask_patches=int(data_cfg.get("min_mask_patches", 4)),
    )


def select_indices(dataset: BedlamDataset, args: argparse.Namespace, config: dict[str, Any]) -> list[int]:
    if args.image:
        data_cfg = config["data"]
        root = Path(require_path(config, data_cfg.get("root_key", "datasets.bedlam_root"), allow_empty=False)).expanduser()
        first = find_bedlam_window_index(dataset, Path(args.image).expanduser(), root, str(args.split))
        return [first + offset for offset in range(max(int(args.num_samples), 1)) if first + offset < len(dataset)]
    start = max(int(args.start_index), 0)
    end = min(len(dataset), start + max(int(args.num_samples), 1))
    return list(range(start, end))


def validate_sample(
    sample_dir: Path,
    dataset: BedlamDataset,
    dataset_idx: int,
    sample: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    predictions: dict[str, torch.Tensor],
    smpl: SMPLLayer,
    args: argparse.Namespace,
) -> dict[str, Any]:
    raw_depth = canonical_depth(predictions["depth"]).float()
    gt_depth = canonical_depth(batch["gt_depth"]).to(device=raw_depth.device, dtype=raw_depth.dtype)
    if gt_depth.shape[-2:] != raw_depth.shape[-2:]:
        gt_depth = F.interpolate(
            gt_depth.reshape(-1, 1, *gt_depth.shape[-2:]),
            size=raw_depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape(*gt_depth.shape[:2], *raw_depth.shape[-2:])

    image_hw = (int(batch["images"].shape[-2]), int(batch["images"].shape[-1]))
    depth_hw = (int(raw_depth.shape[-2]), int(raw_depth.shape[-1]))
    k_dataset = batch["K_scal3r"].to(device=raw_depth.device, dtype=raw_depth.dtype)
    k_vggt = encoding_to_camera(predictions["pose_enc"], image_size_hw=image_hw, build_intrinsics=True)[1]

    gt_poses = rot6d_to_axis_angle(batch["gt_pose_6d"].reshape(-1, 24, 6)).reshape(*batch["gt_pose_6d"].shape[:3], 72)
    gt = decode_smpl_batch(gt_poses, batch["gt_betas"], batch["gt_transl_cam"], smpl)

    frame_infos = frame_source_infos(dataset, dataset_idx)
    frame_summaries = []
    all_rows: list[dict[str, Any]] = []
    for frame_idx in range(int(batch["images"].shape[1])):
        frame_dir = sample_dir / f"frame_{frame_idx:02d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        rgb = tensor_image_to_pil(sample["images"][frame_idx].detach().cpu())
        source_info = frame_infos[frame_idx]
        frame_summary, rows = validate_frame(
            frame_dir=frame_dir,
            rgb=rgb,
            image_hw=image_hw,
            depth_hw=depth_hw,
            raw_depth=raw_depth[0, frame_idx],
            gt_depth=gt_depth[0, frame_idx],
            k_dataset=k_dataset[0, frame_idx],
            k_vggt=k_vggt[0, frame_idx],
            vertices=gt["vertices"][0, frame_idx],
            joints=gt["joints"][0, frame_idx],
            smpl_mask=batch["smpl_mask"][0, frame_idx].bool(),
            gt_boxes=batch["gt_boxes"][0, frame_idx],
            boxes_mask=batch["boxes_mask"][0, frame_idx].bool(),
            person_ids=batch["person_ids"][0, frame_idx],
            source_info=source_info,
            smpl=smpl,
            args=args,
        )
        frame_summaries.append(frame_summary)
        all_rows.extend(rows)

    write_csv(sample_dir / "teacher_points.csv", all_rows)
    sample_summary = {
        "dataset_index": int(dataset_idx),
        "sample_dir": str(sample_dir),
        "image_hw": list(image_hw),
        "depth_hw": list(depth_hw),
        "image_depth_same_hw": list(image_hw) == list(depth_hw),
        "frames": frame_summaries,
        "alerts": collect_alerts(frame_summaries),
    }
    (sample_dir / "sample_summary.json").write_text(json.dumps(sample_summary, indent=2), encoding="utf-8")
    return sample_summary


def validate_frame(
    *,
    frame_dir: Path,
    rgb: Image.Image,
    image_hw: tuple[int, int],
    depth_hw: tuple[int, int],
    raw_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    k_dataset: torch.Tensor,
    k_vggt: torch.Tensor,
    vertices: torch.Tensor,
    joints: torch.Tensor,
    smpl_mask: torch.Tensor,
    gt_boxes: torch.Tensor,
    boxes_mask: torch.Tensor,
    person_ids: torch.Tensor,
    source_info: dict[str, Any],
    smpl: SMPLLayer,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rgb.save(frame_dir / "processed_rgb.png")
    valid_slots = torch.where(smpl_mask)[0]
    points_cam_all = vertices if args.source == "vertices" else joints
    points_cam = points_cam_all[valid_slots]
    points_cam = _subsample_points_per_person(points_cam, int(args.max_points_per_person))

    if points_cam.numel() == 0:
        visibility = empty_visibility(points_cam, raw_depth)
    else:
        projected_image = project_points(points_cam, k_dataset.to(dtype=points_cam.dtype))
        projected_depth = scale_points_to_depth(projected_image, image_hw, depth_hw[0], depth_hw[1])
        frame_idx = torch.zeros(points_cam.shape[0], dtype=torch.long, device=points_cam.device)
        sampled_raw, sampled_gt, visible = _sample_depth_pair_nearest_to_point_z(
            raw_depth=raw_depth.reshape(1, *raw_depth.shape[-2:]),
            gt_depth=gt_depth.reshape(1, *gt_depth.shape[-2:]),
            points_2d=projected_depth,
            points_z=points_cam[..., 2],
            frame_idx=frame_idx,
            window=int(args.window),
            tolerance_m=float(args.visibility_tolerance_m),
        )
        visible = visible & torch.isfinite(sampled_raw) & torch.isfinite(sampled_gt) & (sampled_raw > 1e-6) & (points_cam[..., 2] > 1e-6)
        robust = torch.zeros_like(visible)
        if visible.any():
            candidates = points_cam[..., 2].to(dtype=raw_depth.dtype) / sampled_raw.clamp(min=1e-6)
            robust_flat = _robust_median_filter(candidates[visible], float(args.mad_multiplier))
            flat_positions = visible.reshape(-1).nonzero(as_tuple=False).reshape(-1)
            robust.reshape(-1)[flat_positions[robust_flat]] = True
        visibility = {
            "points_cam": points_cam,
            "projected_image": projected_image,
            "projected_depth": projected_depth,
            "sampled_raw": sampled_raw,
            "sampled_gt": sampled_gt,
            "visible": visible,
            "robust": robust,
            "scale_candidates": points_cam[..., 2].to(dtype=raw_depth.dtype) / sampled_raw.clamp(min=1e-6),
        }

    filtered_people = decode_filtered_people(source_info, smpl, k_dataset, k_dataset.device, k_dataset.dtype)
    frame_rows, person_summaries = build_person_reports(
        valid_slots=valid_slots,
        visibility=visibility,
        gt_boxes=gt_boxes,
        boxes_mask=boxes_mask,
        person_ids=person_ids,
        image_hw=image_hw,
        args=args,
    )
    frame_rows = [{**row, "frame": source_info["frame_id"], "sample_frame_index": source_info["frame_idx"]} for row in frame_rows]
    sidecar = source_info["sidecar"]
    raw_person_count = int(source_info["raw_person_count"])
    sidecar_persons = sidecar.get("persons", []) if isinstance(sidecar, dict) else []
    sidecar_train_valid = sum(1 for person in sidecar_persons if bool(person.get("train_valid", person.get("valid", person.get("bbox_valid", False)))))
    sidecar_filter_reasons = Counter(
        str(person.get("filtered_reason", "ok"))
        for person in sidecar_persons
        if not bool(person.get("train_valid", person.get("valid", person.get("bbox_valid", False))))
    )

    teacher_valid_points = int(visibility["robust"].sum().detach().cpu()) if visibility["robust"].numel() else 0
    visible_points = int(visibility["visible"].sum().detach().cpu()) if visibility["visible"].numel() else 0
    teacher_ok = teacher_valid_points >= int(args.min_visible_points)
    teacher_scale = scale_stats(visibility["scale_candidates"][visibility["robust"]]) if teacher_ok else {"count": teacher_valid_points}
    alerts = frame_alerts(
        image_hw=image_hw,
        depth_hw=depth_hw,
        raw_person_count=raw_person_count,
        sidecar_train_valid=sidecar_train_valid,
        loader_person_count=int(smpl_mask.sum().detach().cpu()),
        teacher_valid_points=teacher_valid_points,
        teacher_scale=teacher_scale,
        args=args,
    )

    visuals = save_visuals(
        frame_dir=frame_dir,
        rgb=rgb,
        raw_depth=raw_depth,
        gt_depth=gt_depth,
        image_hw=image_hw,
        depth_hw=depth_hw,
        k_dataset=k_dataset,
        k_vggt=k_vggt,
        gt_boxes=gt_boxes,
        boxes_mask=boxes_mask,
        visibility=visibility,
        valid_slots=valid_slots,
        person_summaries=person_summaries,
        filtered_people=filtered_people,
        sidecar_persons=sidecar_persons,
        args=args,
    )
    frame_summary = {
        "frame_id": source_info["frame_id"],
        "rgb_path": str(source_info["rgb_path"]),
        "smpl_path": str(source_info["smpl_path"]),
        "sidecar_path": str(source_info["sidecar_path"]),
        "raw_gt_person_count": raw_person_count,
        "sidecar_person_count": len(sidecar_persons),
        "sidecar_train_valid_count": sidecar_train_valid,
        "sidecar_filtered_count": len(sidecar_persons) - sidecar_train_valid,
        "sidecar_filter_reasons": dict(sidecar_filter_reasons),
        "loader_smpl_count": int(smpl_mask.sum().detach().cpu()),
        "loader_box_count": int(boxes_mask.sum().detach().cpu()),
        "visible_points_before_robust": visible_points,
        "teacher_valid_points_after_robust": teacher_valid_points,
        "teacher_frame_usable": bool(teacher_ok),
        "teacher_scale": teacher_scale,
        "persons": person_summaries,
        "visuals": visuals,
        "alerts": alerts,
    }
    (frame_dir / "frame_summary.json").write_text(json.dumps(frame_summary, indent=2), encoding="utf-8")
    return frame_summary, frame_rows


def empty_visibility(points_cam: torch.Tensor, raw_depth: torch.Tensor) -> dict[str, torch.Tensor]:
    empty_points = points_cam.new_zeros((*points_cam.shape[:-1], 2))
    empty_scalar = raw_depth.new_zeros(points_cam.shape[:-1])
    empty_bool = torch.zeros(points_cam.shape[:-1], dtype=torch.bool, device=points_cam.device)
    return {
        "points_cam": points_cam,
        "projected_image": empty_points,
        "projected_depth": empty_points,
        "sampled_raw": empty_scalar,
        "sampled_gt": empty_scalar,
        "visible": empty_bool,
        "robust": empty_bool,
        "scale_candidates": empty_scalar,
    }


def build_person_reports(
    *,
    valid_slots: torch.Tensor,
    visibility: dict[str, torch.Tensor],
    gt_boxes: torch.Tensor,
    boxes_mask: torch.Tensor,
    person_ids: torch.Tensor,
    image_hw: tuple[int, int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    people: list[dict[str, Any]] = []
    for local_idx, slot_tensor in enumerate(valid_slots.detach().cpu()):
        slot = int(slot_tensor.item())
        visible = visibility["visible"][local_idx]
        robust = visibility["robust"][local_idx]
        scales = visibility["scale_candidates"][local_idx]
        person_visible_points = int(visible.sum().detach().cpu())
        person_robust_points = int(robust.sum().detach().cpu())
        enough_person_points = person_robust_points >= int(args.min_points_per_person)
        person_scale = scale_stats(scales[robust]) if person_robust_points else {"count": 0}
        projected_bbox = projected_bbox_xyxy(visibility["projected_image"][local_idx], image_hw)
        box_iou = None
        if bool(boxes_mask[slot]) and projected_bbox is not None:
            box_iou = box_iou_xyxy(projected_bbox, cxcywh_to_xyxy_tuple(gt_boxes[slot], image_hw))
        people.append(
            {
                "slot": slot,
                "track_id": int(person_ids[slot].detach().cpu()),
                "box_valid": bool(boxes_mask[slot]),
                "visible_points_before_robust": person_visible_points,
                "teacher_valid_points_after_robust": person_robust_points,
                "used_by_teacher_person_gate": bool(enough_person_points),
                "scale": person_scale,
                "projected_bbox_xyxy": list(projected_bbox) if projected_bbox is not None else None,
                "projected_bbox_iou_with_sidecar_box": box_iou,
            }
        )
        flat_limit = min(int(args.overlay_max_points_per_person), int(visibility["points_cam"].shape[1]))
        if flat_limit <= 0:
            continue
        sample_indices = torch.linspace(0, int(visibility["points_cam"].shape[1]) - 1, flat_limit).round().long()
        for point_idx in sample_indices.tolist():
            point = visibility["projected_image"][local_idx, point_idx]
            row = {
                "slot": slot,
                "point_index": int(point_idx),
                "x": scalar(point[0]),
                "y": scalar(point[1]),
                "z_smpl": scalar(visibility["points_cam"][local_idx, point_idx, 2]),
                "raw_depth": scalar(visibility["sampled_raw"][local_idx, point_idx]),
                "gt_depth": scalar(visibility["sampled_gt"][local_idx, point_idx]),
                "visible_by_gt_depth": bool(visible[point_idx].detach().cpu()),
                "robust_teacher_point": bool(robust[point_idx].detach().cpu()),
                "scale_candidate": scalar(scales[point_idx]),
            }
            rows.append(row)
    return rows, people


def save_visuals(
    *,
    frame_dir: Path,
    rgb: Image.Image,
    raw_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    image_hw: tuple[int, int],
    depth_hw: tuple[int, int],
    k_dataset: torch.Tensor,
    k_vggt: torch.Tensor,
    gt_boxes: torch.Tensor,
    boxes_mask: torch.Tensor,
    visibility: dict[str, torch.Tensor],
    valid_slots: torch.Tensor,
    person_summaries: list[dict[str, Any]],
    filtered_people: list[dict[str, Any]],
    sidecar_persons: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, str]:
    depth_valid = torch.isfinite(gt_depth) & (gt_depth > 1e-6)
    raw_valid = torch.isfinite(raw_depth) & (raw_depth > 1e-6)
    gt_depth_img = colorize_depth(gt_depth.detach().cpu().numpy(), depth_valid.detach().cpu().numpy()).resize(
        (image_hw[1], image_hw[0]), Image.BILINEAR
    )
    raw_depth_img = colorize_depth(raw_depth.detach().cpu().numpy(), raw_valid.detach().cpu().numpy()).resize(
        (image_hw[1], image_hw[0]), Image.BILINEAR
    )
    paths = {
        "rgb_teacher_points": frame_dir / "rgb_gt_smpl_teacher_points.png",
        "gt_depth_teacher_points": frame_dir / "gt_depth_teacher_points.png",
        "vggt_raw_depth_teacher_points": frame_dir / "vggt_raw_depth_teacher_points.png",
        "rgb_datasetK_vs_vggtK": frame_dir / "rgb_datasetK_vs_vggtK_projection_check.png",
    }
    for key, background in (
        ("rgb_teacher_points", rgb.copy()),
        ("gt_depth_teacher_points", gt_depth_img.convert("RGB")),
        ("vggt_raw_depth_teacher_points", raw_depth_img.convert("RGB")),
    ):
        draw_teacher_overlay(
            background,
            image_hw,
            gt_boxes,
            boxes_mask,
            visibility,
            valid_slots,
            person_summaries,
            filtered_people,
            sidecar_persons,
            args,
        ).save(paths[key])
    draw_k_comparison_overlay(
        rgb.copy(),
        image_hw,
        k_dataset,
        k_vggt,
        visibility["points_cam"],
        valid_slots,
        gt_boxes,
        boxes_mask,
        args,
    ).save(paths["rgb_datasetK_vs_vggtK"])
    return {key: str(path) for key, path in paths.items()}


def draw_teacher_overlay(
    image: Image.Image,
    image_hw: tuple[int, int],
    gt_boxes: torch.Tensor,
    boxes_mask: torch.Tensor,
    visibility: dict[str, torch.Tensor],
    valid_slots: torch.Tensor,
    person_summaries: list[dict[str, Any]],
    filtered_people: list[dict[str, Any]],
    sidecar_persons: list[dict[str, Any]],
    args: argparse.Namespace,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), "green=teacher robust, yellow=visible outlier, red=not depth-visible, magenta=filtered sidecar GT", fill=(255, 255, 255))
    for slot in range(int(gt_boxes.shape[0])):
        if bool(boxes_mask[slot]):
            draw_box(draw, gt_boxes[slot], image_hw, (20, 220, 255), f"box slot{slot}")
    for person in sidecar_persons:
        if bool(person.get("train_valid", person.get("valid", person.get("bbox_valid", False)))):
            continue
        raw_box = person.get("raw_bbox_cxcywh_norm", person.get("bbox_cxcywh_norm"))
        if raw_box is None:
            continue
        try:
            box = torch.as_tensor(raw_box, dtype=torch.float32)
            if float(box[2]) > 0.0 and float(box[3]) > 0.0:
                draw_box(draw, box, image_hw, (255, 80, 220), f"filtered p{person.get('person_index', '?')} {person.get('filtered_reason', '')}")
        except (TypeError, ValueError):
            pass
    for local_idx, person in enumerate(person_summaries):
        projected = visibility["projected_image"][local_idx]
        visible = visibility["visible"][local_idx]
        robust = visibility["robust"][local_idx]
        draw_points(draw, projected[~visible], (255, 80, 60), image_hw, args.overlay_max_points_per_person, radius=1)
        draw_points(draw, projected[visible & ~robust], (255, 210, 40), image_hw, args.overlay_max_points_per_person, radius=2)
        draw_points(draw, projected[robust], (60, 255, 90), image_hw, args.overlay_max_points_per_person, radius=2)
        label_anchor = first_valid_point(projected, image_hw)
        if label_anchor is not None:
            scale = person.get("scale", {}).get("median")
            label = (
                f"slot{person['slot']} pts={person['teacher_valid_points_after_robust']}"
                f" scale={scale:.2f}" if scale is not None else f"slot{person['slot']} pts=0"
            )
            draw.text((label_anchor[0] + 4, label_anchor[1] + 4), label, fill=(255, 255, 255))
    for filtered in filtered_people:
        draw_points(draw, filtered["projected"], (255, 80, 220), image_hw, int(args.filtered_overlay_max_points), radius=1)
    return image


def draw_k_comparison_overlay(
    image: Image.Image,
    image_hw: tuple[int, int],
    k_dataset: torch.Tensor,
    k_vggt: torch.Tensor,
    points_cam: torch.Tensor,
    valid_slots: torch.Tensor,
    gt_boxes: torch.Tensor,
    boxes_mask: torch.Tensor,
    args: argparse.Namespace,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), "GT SMPL projection: cyan=dataset K used by teacher, orange=VGGT predicted K", fill=(255, 255, 255))
    for slot in range(int(gt_boxes.shape[0])):
        if bool(boxes_mask[slot]):
            draw_box(draw, gt_boxes[slot], image_hw, (20, 220, 255), f"box slot{slot}")
    for local_idx, slot_tensor in enumerate(valid_slots.detach().cpu()):
        points = points_cam[local_idx]
        idx = subsample_indices(points.shape[0], int(args.overlay_max_points_per_person), points.device)
        projected_dataset = project_points(points[idx], k_dataset.to(dtype=points.dtype))
        projected_vggt = project_points(points[idx], k_vggt.to(dtype=points.dtype))
        draw_points(draw, projected_dataset, (20, 220, 255), image_hw, int(args.overlay_max_points_per_person), radius=1)
        draw_points(draw, projected_vggt, (255, 150, 30), image_hw, int(args.overlay_max_points_per_person), radius=1)
        anchor = first_valid_point(projected_dataset, image_hw)
        if anchor is not None:
            draw.text((anchor[0] + 4, anchor[1] + 4), f"slot{int(slot_tensor)}", fill=(255, 255, 255))
    return image


def decode_filtered_people(
    source_info: dict[str, Any],
    smpl: SMPLLayer,
    k_dataset: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    raw_persons = source_info.get("raw_persons", [])
    sidecar = source_info.get("sidecar", {})
    sidecar_persons = sidecar.get("persons", []) if isinstance(sidecar, dict) else []
    filtered = []
    for person in sidecar_persons:
        if bool(person.get("train_valid", person.get("valid", person.get("bbox_valid", False)))):
            continue
        try:
            person_idx = int(person.get("person_index", -1))
        except (TypeError, ValueError):
            person_idx = -1
        if person_idx < 0 or person_idx >= len(raw_persons):
            continue
        raw = raw_persons[person_idx]
        if not isinstance(raw, dict):
            continue
        try:
            root = torch.as_tensor(raw["smplx_root_pose"], dtype=torch.float32, device=device).reshape(1, 3)
            body = torch.as_tensor(raw["smplx_body_pose"], dtype=torch.float32, device=device).reshape(21, 3)
            pose = torch.cat([root, body, body.new_zeros(2, 3)], dim=0).reshape(1, 72)
            betas = torch.as_tensor(raw["smplx_shape"], dtype=torch.float32, device=device).reshape(1, -1)[:, :10]
            transl = torch.as_tensor(raw["smplx_transl"], dtype=dtype, device=device).reshape(1, 3)
            verts, _ = smpl(pose, betas)
            verts = verts[0].to(dtype=dtype) + transl[0, None, :]
            idx = subsample_indices(verts.shape[0], 160, device)
            projected = project_points(verts[idx], k_dataset.to(device=device, dtype=dtype))
            filtered.append({"person_index": person_idx, "reason": person.get("filtered_reason", ""), "projected": projected})
        except (KeyError, RuntimeError, ValueError):
            continue
    return filtered


def frame_source_infos(dataset: BedlamDataset, dataset_idx: int) -> list[dict[str, Any]]:
    seq_idx, start_idx = dataset._index[dataset_idx]
    seq_dir, frame_ids = dataset._sequences[seq_idx]
    selected = [frame_ids[start_idx + step * dataset.stride] for step in range(dataset.sequence_length)]
    infos = []
    for frame_idx, frame_id in enumerate(selected):
        rgb_path = seq_dir / "rgb" / f"{frame_id}.png"
        smpl_path = seq_dir / "smpl" / f"{frame_id}.pkl"
        sidecar_path = dataset._box_path(seq_dir, frame_id)
        raw_persons = load_pickle_list(smpl_path)
        sidecar = load_pickle_dict(sidecar_path)
        infos.append(
            {
                "frame_idx": frame_idx,
                "frame_id": frame_id,
                "rgb_path": rgb_path,
                "smpl_path": smpl_path,
                "sidecar_path": sidecar_path,
                "raw_person_count": len(raw_persons),
                "raw_persons": raw_persons,
                "sidecar": sidecar,
            }
        )
    return infos


def load_pickle_list(path: Path) -> list[Any]:
    with path.open("rb") as file:
        value = pickle.load(file)
    return value if isinstance(value, list) else []


def load_pickle_dict(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        value = pickle.load(file)
    return value if isinstance(value, dict) else {}


def frame_alerts(
    *,
    image_hw: tuple[int, int],
    depth_hw: tuple[int, int],
    raw_person_count: int,
    sidecar_train_valid: int,
    loader_person_count: int,
    teacher_valid_points: int,
    teacher_scale: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    alerts = []
    if image_hw != depth_hw:
        alerts.append("image_hw_differs_from_depth_hw_projection_scaling_used")
    if raw_person_count > sidecar_train_valid:
        alerts.append("raw_gt_has_more_people_than_train_valid_sidecar_filtered_some_people")
    if sidecar_train_valid != loader_person_count:
        alerts.append("loader_person_count_differs_from_sidecar_train_valid_count")
    if teacher_valid_points < int(args.min_visible_points):
        alerts.append("not_enough_teacher_points_frame_will_not_train_scale")
    scale = teacher_scale.get("median")
    if scale is not None and (float(scale) < float(args.scale_warn_low) or float(scale) > float(args.scale_warn_high)):
        alerts.append("teacher_scale_outside_expected_range")
    return alerts


def aggregate_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    frames = [frame for sample in samples for frame in sample.get("frames", [])]
    if not frames:
        return {"num_frames": 0}
    raw_total = sum(int(frame.get("raw_gt_person_count", 0)) for frame in frames)
    sidecar_total = sum(int(frame.get("sidecar_train_valid_count", 0)) for frame in frames)
    loader_total = sum(int(frame.get("loader_smpl_count", 0)) for frame in frames)
    teacher_frames = sum(1 for frame in frames if bool(frame.get("teacher_frame_usable", False)))
    scales = [frame["teacher_scale"]["median"] for frame in frames if frame.get("teacher_scale", {}).get("median") is not None]
    return {
        "num_frames": len(frames),
        "raw_gt_person_total": raw_total,
        "sidecar_train_valid_total": sidecar_total,
        "loader_smpl_total": loader_total,
        "teacher_usable_frames": teacher_frames,
        "teacher_usable_frame_ratio": float(teacher_frames / max(len(frames), 1)),
        "teacher_scale_median": float(np.median(scales)) if scales else None,
        "alerts": collect_alerts(frames),
    }


def collect_alerts(frames: list[dict[str, Any]]) -> list[str]:
    alerts = []
    for frame in frames:
        alerts.extend(frame.get("alerts", []))
    return sorted(set(alerts))


def print_summary(summary: dict[str, Any]) -> None:
    global_stats = summary.get("global", {})
    print("========== GT-SMPL scale teacher data check ==========")
    print(f"samples: {summary.get('num_samples')} frames: {global_stats.get('num_frames')}")
    print(
        "persons raw/train_valid/loader: "
        f"{global_stats.get('raw_gt_person_total')}/"
        f"{global_stats.get('sidecar_train_valid_total')}/"
        f"{global_stats.get('loader_smpl_total')}"
    )
    print(
        "teacher usable frames: "
        f"{global_stats.get('teacher_usable_frames')}/"
        f"{global_stats.get('num_frames')} "
        f"scale_median={global_stats.get('teacher_scale_median')}"
    )
    alerts = global_stats.get("alerts", [])
    print(f"alerts: {alerts if alerts else '<none>'}")


def scale_stats(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach().float()
    values = values[torch.isfinite(values) & (values > 1e-6)]
    if values.numel() == 0:
        return {"count": 0}
    return {
        "count": int(values.numel()),
        "min": scalar(values.min()),
        "p05": scalar(torch.quantile(values, 0.05)),
        "median": scalar(values.median()),
        "p95": scalar(torch.quantile(values, 0.95)),
        "max": scalar(values.max()),
        "mean": scalar(values.mean()),
        "std": scalar(values.std(unbiased=False)) if values.numel() > 1 else 0.0,
    }


def projected_bbox_xyxy(points: torch.Tensor, image_hw: tuple[int, int]) -> tuple[float, float, float, float] | None:
    h, w = image_hw
    pts = points.detach().float().cpu().numpy()
    ok = np.isfinite(pts).all(axis=1) & (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    if not ok.any():
        return None
    x0, y0 = pts[ok].min(axis=0)
    x1, y1 = pts[ok].max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def cxcywh_to_xyxy_tuple(box: torch.Tensor, image_hw: tuple[int, int]) -> tuple[float, float, float, float]:
    xyxy = cxcywh_norm_to_xyxy(box.detach().cpu().float(), image_hw)
    return tuple(float(v) for v in xyxy.reshape(4).tolist())


def box_iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    area_a = max(ax1 - ax0, 0.0) * max(ay1 - ay0, 0.0)
    area_b = max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 1e-8 else 0.0


def draw_box(draw: ImageDraw.ImageDraw, box_cxcywh: torch.Tensor, image_hw: tuple[int, int], fill: tuple[int, int, int], text: str = "") -> None:
    x0, y0, x1, y1 = cxcywh_to_xyxy_tuple(box_cxcywh, image_hw)
    draw.rectangle([x0, y0, x1, y1], outline=fill, width=2)
    if text:
        draw.text((x0 + 2, y0 + 2), text, fill=fill)


def draw_points(
    draw: ImageDraw.ImageDraw,
    points: torch.Tensor,
    color: tuple[int, int, int],
    image_hw: tuple[int, int],
    max_points: int,
    radius: int = 2,
) -> None:
    if points.numel() == 0:
        return
    idx = subsample_indices(points.shape[0], max_points, points.device)
    pts = points[idx].detach().float().cpu().numpy()
    h, w = image_hw
    for x, y in pts:
        if not np.isfinite(x) or not np.isfinite(y) or x < 0 or y < 0 or x >= w or y >= h:
            continue
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)


def first_valid_point(points: torch.Tensor, image_hw: tuple[int, int]) -> tuple[float, float] | None:
    h, w = image_hw
    for x, y in points.detach().float().cpu().numpy():
        if np.isfinite(x) and np.isfinite(y) and 0 <= x < w and 0 <= y < h:
            return float(x), float(y)
    return None


def subsample_indices(total: int, max_count: int, device: torch.device) -> torch.Tensor:
    total = int(total)
    max_count = int(max_count)
    if total <= 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    if max_count <= 0 or total <= max_count:
        return torch.arange(total, dtype=torch.long, device=device)
    return torch.linspace(0, total - 1, steps=max_count, device=device).round().long()


def colorize_depth(array: np.ndarray, valid: np.ndarray | None = None, robust: bool = True) -> Image.Image:
    arr = np.asarray(array, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(arr)
    else:
        valid = valid & np.isfinite(arr)
    vals = arr[valid]
    if vals.size == 0:
        lo, hi = 0.0, 1.0
    elif robust and vals.size > 10:
        lo, hi = float(np.percentile(vals, 2.0)), float(np.percentile(vals, 98.0))
    else:
        lo, hi = float(vals.min()), float(vals.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    norm = np.zeros_like(arr, dtype=np.float32)
    norm[valid] = np.clip((arr[valid] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    r = (255.0 * norm).astype(np.uint8)
    g = (255.0 * (1.0 - np.abs(norm - 0.5) * 2.0)).astype(np.uint8)
    b = (255.0 * (1.0 - norm)).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[~valid] = np.array([0, 0, 0], dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def scalar(value: torch.Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        value = float(value.detach().float().cpu().reshape(-1)[0])
    value = float(value)
    return value if math.isfinite(value) else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
