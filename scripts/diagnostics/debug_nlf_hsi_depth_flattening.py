#!/usr/bin/env python
"""Detailed NLF-HSI depth flattening diagnostics.

This script intentionally inspects every geometry handoff that can turn a
depth map into an almost-planar scene:

1. processed RGB / GT depth / VGGT raw depth / HSI affine depth
2. dataset intrinsics vs VGGT-predicted intrinsics
3. GT SMPL projection into GT depth
4. NLF base and HSI-refined SMPL projection into raw/HSI/GT depth
5. human ROI depth statistics and robust affine fit
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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
    greedy_match,
    human_roi_depth_mask,
    load_training_checkpoint,
    load_vggt_baseline,
    project_points,
    sample_depth_at_points,
    scale_points_to_depth,
)
from scripts.train.train_smpl import apply_overrides, build_model, load_yaml_config  # noqa: E402
from scripts.vis.visualize_hsi_depth_smpl_diagnostics import (  # noqa: E402
    find_bedlam_window_index,
    tensor_image_to_pil,
)
from vggt_omega.data import BedlamDataset  # noqa: E402
from vggt_omega.data.geometry import resolve_image_size_config  # noqa: E402
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.training.config import deep_update, require_path  # noqa: E402
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
    load_vggt_baseline(model, config, device)
    load_training_checkpoint(model, Path(args.checkpoint).expanduser(), device)
    model.eval()
    smpl = SMPLLayer(require_path(config, "assets.smpl_model_dir", allow_empty=False)).to(device).eval()

    all_summaries = []
    with torch.no_grad():
        for local_idx, dataset_idx in enumerate(indices):
            sample = dataset[dataset_idx]
            batch = {
                key: value.unsqueeze(0).to(device, non_blocking=True)
                for key, value in sample.items()
                if isinstance(value, torch.Tensor)
            }
            predictions = model(
                batch["images"],
                smpl_query_boxes=batch["gt_boxes"] if args.use_gt_box_prior else None,
                smpl_query_boxes_mask=batch["boxes_mask"] if args.use_gt_box_prior else None,
                smpl_track_ids=batch.get("gt_track_ids", batch.get("person_ids")),
                smpl_track_mask=batch.get("gt_track_mask", batch.get("person_id_mask")),
            )
            sample_dir = output_dir / f"sample_{local_idx:03d}_dataset_{dataset_idx:06d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            all_summaries.append(debug_sample(sample_dir, sample, batch, predictions, smpl, args))

    summary = {
        "checkpoint": str(args.checkpoint),
        "num_samples": len(all_summaries),
        "samples": all_summaries,
        "global_alerts": collect_global_alerts(all_summaries),
    }
    out_json = output_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_human_summary(summary)
    print(json.dumps({"summary": str(out_json), "output_dir": str(output_dir)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug NLF-HSI depth flattening and SMPL/depth projection")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_hsi_nlf_provider.yaml")
    parser.add_argument("--output-dir", default="outputs/debug/nlf_hsi_depth_flattening")
    parser.add_argument("--image", default="", help="Optional BEDLAM image path used as the first frame of a window")
    parser.add_argument("--split", default="Training")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--device", default="")
    parser.add_argument("--use-gt-box-prior", action="store_true")
    parser.add_argument("--conf-threshold", type=float, default=0.10)
    parser.add_argument("--roi-expand", type=float, default=0.65)
    parser.add_argument("--depth-max-m", type=float, default=30.0)
    parser.add_argument("--flat-std-ratio-threshold", type=float, default=0.12)
    parser.add_argument("--flat-gradient-ratio-threshold", type=float, default=0.12)
    parser.add_argument("--small-scale-threshold", type=float, default=0.12)
    parser.add_argument("--vertex-sample-stride", type=int, default=25)
    parser.add_argument("--overlay-max-vertices", type=int, default=600)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.train_config))
    config = apply_overrides(config, args.override)
    config.setdefault("model", {})["enable_camera"] = True
    config.setdefault("model", {})["enable_depth"] = True
    config.setdefault("model", {})["enable_hsi_refine"] = True
    config.setdefault("model", {})["smpl_provider"] = "nlf"
    config.setdefault("model", {})["nlf_use_detector"] = False
    config.setdefault("model", {})["nlf_require_boxes"] = True
    config.setdefault("data", {})["require_depth"] = True
    config.setdefault("data", {})["require_boxes"] = True
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


def debug_sample(
    sample_dir: Path,
    sample: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    predictions: dict[str, torch.Tensor],
    smpl: SMPLLayer,
    args: argparse.Namespace,
) -> dict[str, Any]:
    raw_depth, hsi_depth, gt_depth = depth_triplet(predictions, batch)
    image_hw = (int(batch["images"].shape[-2]), int(batch["images"].shape[-1]))
    depth_hw = (int(raw_depth.shape[-2]), int(raw_depth.shape[-1]))
    k_vggt = encoding_to_camera(predictions["pose_enc"], image_size_hw=image_hw, build_intrinsics=True)[1]
    k_dataset = batch["K_scal3r"].to(device=k_vggt.device, dtype=k_vggt.dtype)

    base = decode_smpl_batch(predictions["pred_poses"], predictions["pred_betas"], predictions["pred_transl_cam"], smpl)
    hsi = decode_smpl_batch(
        predictions["hsi_refined_pred_poses"],
        predictions["hsi_refined_pred_betas"],
        predictions["hsi_refined_pred_transl_cam"],
        smpl,
    )
    gt_poses = rot6d_to_axis_angle(batch["gt_pose_6d"].reshape(-1, 24, 6)).reshape(*batch["gt_pose_6d"].shape[:3], 72)
    gt = decode_smpl_batch(gt_poses, batch["gt_betas"], batch["gt_transl_cam"], smpl)

    frame_summaries = []
    for frame_idx in range(int(batch["images"].shape[1])):
        rgb = tensor_image_to_pil(sample["images"][frame_idx].detach().cpu())
        frame_dir = sample_dir / f"frame_{frame_idx:02d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_summary = debug_frame(
            frame_dir=frame_dir,
            rgb=rgb,
            image_hw=image_hw,
            depth_hw=depth_hw,
            raw_depth=raw_depth[0, frame_idx],
            hsi_depth=hsi_depth[0, frame_idx],
            gt_depth=gt_depth[0, frame_idx],
            gt_boxes=batch["gt_boxes"][0, frame_idx],
            boxes_mask=batch["boxes_mask"][0, frame_idx].bool(),
            smpl_mask=batch["smpl_mask"][0, frame_idx].bool(),
            k_dataset=k_dataset[0, frame_idx],
            k_vggt=k_vggt[0, frame_idx],
            base={key: value[0, frame_idx] for key, value in base.items()},
            hsi={key: value[0, frame_idx] for key, value in hsi.items()},
            gt={key: value[0, frame_idx] for key, value in gt.items()},
            confs=squeeze_confs(predictions["pred_confs"])[0, frame_idx],
            predictions=predictions,
            frame_idx=frame_idx,
            args=args,
        )
        frame_summaries.append(frame_summary)
    sample_summary = {
        "sample_dir": str(sample_dir),
        "image_hw": list(image_hw),
        "depth_hw": list(depth_hw),
        "num_frames": len(frame_summaries),
        "frames": frame_summaries,
        "alerts": collect_global_alerts([{"frames": frame_summaries}]),
    }
    (sample_dir / "sample_summary.json").write_text(json.dumps(sample_summary, indent=2), encoding="utf-8")
    return sample_summary


def depth_triplet(predictions: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_depth = canonical_depth(predictions["depth"]).float()
    gt_depth = canonical_depth(batch["gt_depth"]).to(device=raw_depth.device, dtype=raw_depth.dtype)
    if gt_depth.shape[-2:] != raw_depth.shape[-2:]:
        gt_depth = F.interpolate(
            gt_depth.reshape(-1, 1, *gt_depth.shape[-2:]),
            size=raw_depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape(*gt_depth.shape[:2], *raw_depth.shape[-2:])
    scale = predictions.get("hsi_scene_scale")
    bias = predictions.get("hsi_scene_depth_bias")
    if scale is None or bias is None:
        return raw_depth, raw_depth, gt_depth
    scale = scale.to(device=raw_depth.device, dtype=raw_depth.dtype).reshape(*raw_depth.shape[:2], 1, 1)
    bias = bias.to(device=raw_depth.device, dtype=raw_depth.dtype).reshape(*raw_depth.shape[:2], 1, 1)
    return raw_depth, raw_depth * scale + bias, gt_depth


def debug_frame(
    *,
    frame_dir: Path,
    rgb: Image.Image,
    image_hw: tuple[int, int],
    depth_hw: tuple[int, int],
    raw_depth: torch.Tensor,
    hsi_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    gt_boxes: torch.Tensor,
    boxes_mask: torch.Tensor,
    smpl_mask: torch.Tensor,
    k_dataset: torch.Tensor,
    k_vggt: torch.Tensor,
    base: dict[str, torch.Tensor],
    hsi: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    confs: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    frame_idx: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rgb.save(frame_dir / "processed_rgb.png")
    valid = depth_valid_mask(gt_depth, raw_depth, hsi_depth, float(args.depth_max_m))
    roi_mask = human_roi_depth_mask(
        gt_boxes.reshape(1, 1, *gt_boxes.shape).to(device=raw_depth.device, dtype=raw_depth.dtype),
        boxes_mask.reshape(1, 1, *boxes_mask.shape).to(device=raw_depth.device).bool(),
        depth_hw[0],
        depth_hw[1],
        expand=float(args.roi_expand),
    )[0, 0]
    roi_valid = valid & roi_mask

    depth_stats = {
        "full": depth_region_stats(raw_depth, hsi_depth, gt_depth, valid),
        "human_roi": depth_region_stats(raw_depth, hsi_depth, gt_depth, roi_valid),
        "near_valid_pixels": int(valid.sum().detach().cpu()),
        "roi_valid_pixels": int(roi_valid.sum().detach().cpu()),
        "raw": array_stats(raw_depth[valid]),
        "hsi": array_stats(hsi_depth[valid]),
        "gt": array_stats(gt_depth[valid]),
        "raw_gradient": gradient_stats(raw_depth, valid),
        "hsi_gradient": gradient_stats(hsi_depth, valid),
        "gt_gradient": gradient_stats(gt_depth, valid),
        "hsi_vs_raw_std_ratio": safe_ratio(std_value(hsi_depth[valid]), std_value(raw_depth[valid])),
        "hsi_vs_raw_gradient_ratio": safe_ratio(
            gradient_stats(hsi_depth, valid).get("median"),
            gradient_stats(raw_depth, valid).get("median"),
        ),
        "robust_affine_roi_raw_to_gt": robust_affine_fit(raw_depth[roi_valid], gt_depth[roi_valid]),
    }

    scale_bias = frame_scale_bias(predictions, frame_idx)
    projection_stats, rows = projection_debug_rows(
        image_hw=image_hw,
        depth_hw=depth_hw,
        raw_depth=raw_depth,
        hsi_depth=hsi_depth,
        gt_depth=gt_depth,
        gt_boxes=gt_boxes,
        boxes_mask=boxes_mask,
        smpl_mask=smpl_mask,
        k_dataset=k_dataset,
        k_vggt=k_vggt,
        base=base,
        hsi=hsi,
        gt=gt,
        confs=confs,
        args=args,
    )

    visuals = save_frame_visuals(
        frame_dir=frame_dir,
        rgb=rgb,
        raw_depth=raw_depth,
        hsi_depth=hsi_depth,
        gt_depth=gt_depth,
        valid=valid,
        roi_mask=roi_mask,
        image_hw=image_hw,
        depth_hw=depth_hw,
        gt_boxes=gt_boxes,
        boxes_mask=boxes_mask,
        k_dataset=k_dataset,
        k_vggt=k_vggt,
        base=base,
        hsi=hsi,
        gt=gt,
        confs=confs,
        args=args,
    )
    csv_path = frame_dir / "projection_depth_samples.csv"
    write_csv(csv_path, rows)

    alerts = frame_alerts(depth_stats, scale_bias, projection_stats, args)
    summary = {
        "frame_dir": str(frame_dir),
        "image_hw": list(image_hw),
        "depth_hw": list(depth_hw),
        "intrinsics": {
            "dataset": tensor_to_list(k_dataset),
            "vggt": tensor_to_list(k_vggt),
            "diff": intrinsics_diff(k_dataset, k_vggt),
        },
        "hsi_affine": scale_bias,
        "depth_stats": depth_stats,
        "projection_stats": projection_stats,
        "alerts": alerts,
        "visualizations": visuals | {"projection_depth_samples_csv": str(csv_path)},
    }
    (frame_dir / "frame_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def projection_debug_rows(
    *,
    image_hw: tuple[int, int],
    depth_hw: tuple[int, int],
    raw_depth: torch.Tensor,
    hsi_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    gt_boxes: torch.Tensor,
    boxes_mask: torch.Tensor,
    smpl_mask: torch.Tensor,
    k_dataset: torch.Tensor,
    k_vggt: torch.Tensor,
    base: dict[str, torch.Tensor],
    hsi: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    confs: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gt_idx = torch.nonzero(smpl_mask, as_tuple=False).flatten()
    pred_idx = torch.nonzero(confs >= float(args.conf_threshold), as_tuple=False).flatten()
    matches = greedy_match(base["joints"][pred_idx, :24], gt["joints"][gt_idx, :24]) if gt_idx.numel() and pred_idx.numel() else []
    rows: list[dict[str, Any]] = []
    person_summaries = []
    for pred_local, gt_local in matches:
        q = int(pred_idx[pred_local].item())
        g = int(gt_idx[gt_local].item())
        person_summary = {
            "query": q,
            "gt_index": g,
            "confidence": scalar(confs[q]),
            "gt_box": tensor_to_list(gt_boxes[g]) if g < int(gt_boxes.shape[0]) and bool(boxes_mask[g]) else None,
        }
        for label, points, intrinsics, depth_name, depth in [
            ("gt_joints_datasetK_gtDepth", gt["joints"][g, :24], k_dataset, "gt", gt_depth),
            ("gt_joints_vggtK_gtDepth", gt["joints"][g, :24], k_vggt, "gt", gt_depth),
            ("base_joints_vggtK_rawDepth", base["joints"][q, :24], k_vggt, "raw", raw_depth),
            ("base_joints_vggtK_hsiDepth", base["joints"][q, :24], k_vggt, "hsi", hsi_depth),
            ("hsi_joints_vggtK_hsiDepth", hsi["joints"][q, :24], k_vggt, "hsi", hsi_depth),
            ("hsi_joints_vggtK_gtDepth", hsi["joints"][q, :24], k_vggt, "gt", gt_depth),
        ]:
            stats, point_rows = point_depth_stats(label, points, intrinsics, depth, image_hw, depth_hw, person=q, gt_index=g, depth_name=depth_name)
            person_summary[label] = stats
            rows.extend(point_rows)

        vertex_stride = max(int(args.vertex_sample_stride), 1)
        for label, vertices, intrinsics, depth_name, depth in [
            ("gt_vertices_datasetK_gtDepth", gt["vertices"][g, ::vertex_stride], k_dataset, "gt", gt_depth),
            ("gt_vertices_vggtK_gtDepth", gt["vertices"][g, ::vertex_stride], k_vggt, "gt", gt_depth),
            ("base_vertices_vggtK_hsiDepth", base["vertices"][q, ::vertex_stride], k_vggt, "hsi", hsi_depth),
            ("hsi_vertices_vggtK_hsiDepth", hsi["vertices"][q, ::vertex_stride], k_vggt, "hsi", hsi_depth),
        ]:
            stats, point_rows = point_depth_stats(label, vertices, intrinsics, depth, image_hw, depth_hw, person=q, gt_index=g, depth_name=depth_name)
            person_summary[label] = stats
            rows.extend(point_rows)

        person_summary["gt_projected_bbox_datasetK_iou_sidecar"] = projected_bbox_iou(
            gt["vertices"][g, :: max(vertex_stride, 1)], k_dataset, gt_boxes[g], boxes_mask[g], image_hw
        )
        person_summary["gt_projected_bbox_vggtK_iou_sidecar"] = projected_bbox_iou(
            gt["vertices"][g, :: max(vertex_stride, 1)], k_vggt, gt_boxes[g], boxes_mask[g], image_hw
        )
        person_summaries.append(person_summary)

    return {
        "num_gt": int(gt_idx.numel()),
        "num_pred": int(pred_idx.numel()),
        "num_matched": int(len(matches)),
        "persons": person_summaries,
    }, rows


def point_depth_stats(
    label: str,
    points: torch.Tensor,
    intrinsics: torch.Tensor,
    depth: torch.Tensor,
    image_hw: tuple[int, int],
    depth_hw: tuple[int, int],
    *,
    person: int,
    gt_index: int,
    depth_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    projected_img = project_points(points, intrinsics)
    projected_depth = scale_points_to_depth(projected_img, image_hw, depth_hw[0], depth_hw[1])
    sampled, valid = sample_depth_at_points(depth, projected_depth)
    z = points[:, 2].to(dtype=sampled.dtype)
    use = valid & torch.isfinite(sampled) & torch.isfinite(z) & (z > 1e-6)
    delta = sampled - z
    rows = []
    for idx in range(int(points.shape[0])):
        rows.append(
            {
                "label": label,
                "person_query": person,
                "gt_index": gt_index,
                "point_index": idx,
                "depth_name": depth_name,
                "x_img": scalar(projected_img[idx, 0]),
                "y_img": scalar(projected_img[idx, 1]),
                "x_depth": scalar(projected_depth[idx, 0]),
                "y_depth": scalar(projected_depth[idx, 1]),
                "point_z": scalar(z[idx]),
                "sampled_depth": scalar(sampled[idx]),
                "sample_minus_z": scalar(delta[idx]),
                "valid": bool(use[idx].detach().cpu()),
            }
        )
    return {
        "valid_points": int(use.sum().detach().cpu()),
        "total_points": int(points.shape[0]),
        "median_sample_minus_z_m": scalar(delta[use].median()) if use.any() else None,
        "mean_abs_sample_minus_z_m": scalar(torch.abs(delta[use]).mean()) if use.any() else None,
        "p90_abs_sample_minus_z_m": quantile_abs(delta[use], 0.90) if use.any() else None,
        "projected_in_depth_ratio": float(use.sum().detach().cpu()) / max(int(points.shape[0]), 1),
    }, rows


def save_frame_visuals(
    *,
    frame_dir: Path,
    rgb: Image.Image,
    raw_depth: torch.Tensor,
    hsi_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid: torch.Tensor,
    roi_mask: torch.Tensor,
    image_hw: tuple[int, int],
    depth_hw: tuple[int, int],
    gt_boxes: torch.Tensor,
    boxes_mask: torch.Tensor,
    k_dataset: torch.Tensor,
    k_vggt: torch.Tensor,
    base: dict[str, torch.Tensor],
    hsi: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    confs: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    paths["depth_triptych"] = str(save_depth_triptych(frame_dir / "depth_gt_raw_hsi_errors.png", gt_depth, raw_depth, hsi_depth, valid))
    paths["roi_mask"] = str(save_mask_overlay(frame_dir / "human_roi_mask.png", rgb, roi_mask, depth_hw, image_hw))
    paths["flatten_profiles"] = str(save_depth_profiles(frame_dir / "depth_centerline_profiles.png", gt_depth, raw_depth, hsi_depth, valid))
    paths["projection_rgb"] = str(
        save_projection_overlay(
            frame_dir / "projection_overlay_rgb.png",
            rgb.copy(),
            image_hw,
            gt_boxes,
            boxes_mask,
            k_dataset,
            k_vggt,
            base,
            hsi,
            gt,
            confs,
            args,
        )
    )
    for mode, filename in [
        ("gt_dataset", "projection_gt_datasetK_only.png"),
        ("gt_vggt", "projection_gt_vggtK_only.png"),
        ("base", "projection_base_vggtK_only.png"),
        ("hsi", "projection_hsi_vggtK_only.png"),
    ]:
        paths[f"projection_{mode}"] = str(
            save_projection_overlay(
                frame_dir / filename,
                rgb.copy(),
                image_hw,
                gt_boxes,
                boxes_mask,
                k_dataset,
                k_vggt,
                base,
                hsi,
                gt,
                confs,
                args,
                mode=mode,
            )
        )
    depth_bg = colorize_depth(gt_depth.detach().cpu().numpy(), valid.detach().cpu().numpy()).resize((image_hw[1], image_hw[0]), Image.BILINEAR).convert("RGB")
    paths["projection_gt_depth"] = str(
        save_projection_overlay(
            frame_dir / "projection_overlay_gt_depth.png",
            depth_bg,
            image_hw,
            gt_boxes,
            boxes_mask,
            k_dataset,
            k_vggt,
            base,
            hsi,
            gt,
            confs,
            args,
        )
    )
    for mode, filename in [
        ("gt_dataset", "projection_gt_depth_gt_datasetK_only.png"),
        ("gt_vggt", "projection_gt_depth_gt_vggtK_only.png"),
        ("base", "projection_gt_depth_base_vggtK_only.png"),
        ("hsi", "projection_gt_depth_hsi_vggtK_only.png"),
    ]:
        paths[f"projection_gt_depth_{mode}"] = str(
            save_projection_overlay(
                frame_dir / filename,
                depth_bg.copy(),
                image_hw,
                gt_boxes,
                boxes_mask,
                k_dataset,
                k_vggt,
                base,
                hsi,
                gt,
                confs,
                args,
                mode=mode,
            )
        )
    return paths


def save_depth_triptych(path: Path, gt_depth: torch.Tensor, raw_depth: torch.Tensor, hsi_depth: torch.Tensor, valid: torch.Tensor) -> Path:
    valid_np = valid.detach().cpu().numpy()
    gt_np = gt_depth.detach().cpu().numpy()
    raw_np = raw_depth.detach().cpu().numpy()
    hsi_np = hsi_depth.detach().cpu().numpy()
    err_raw = np.abs(raw_np - gt_np)
    err_hsi = np.abs(hsi_np - gt_np)
    images = [
        ("GT depth", colorize_depth(gt_np, valid_np)),
        ("VGGT raw", colorize_depth(raw_np, valid_np)),
        ("HSI depth", colorize_depth(hsi_np, valid_np)),
        ("raw abs err", colorize_depth(err_raw, valid_np, robust=True)),
        ("HSI abs err", colorize_depth(err_hsi, valid_np, robust=True)),
    ]
    w, h = images[0][1].size
    out = Image.new("RGB", (w * len(images), h + 22), "white")
    draw = ImageDraw.Draw(out)
    for idx, (title, img) in enumerate(images):
        out.paste(img.convert("RGB"), (idx * w, 22))
        draw.text((idx * w + 4, 4), title, fill=(0, 0, 0))
    out.save(path)
    return path


def save_mask_overlay(path: Path, rgb: Image.Image, mask: torch.Tensor, depth_hw: tuple[int, int], image_hw: tuple[int, int]) -> Path:
    mask_img = Image.fromarray((mask.detach().cpu().numpy().astype(np.uint8) * 180), mode="L")
    mask_img = mask_img.resize((image_hw[1], image_hw[0]), Image.NEAREST)
    overlay = Image.new("RGBA", rgb.size, (255, 0, 0, 0))
    overlay.putalpha(mask_img)
    out = Image.alpha_composite(rgb.convert("RGBA"), overlay).convert("RGB")
    del depth_hw
    out.save(path)
    return path


def save_depth_profiles(path: Path, gt_depth: torch.Tensor, raw_depth: torch.Tensor, hsi_depth: torch.Tensor, valid: torch.Tensor) -> Path:
    arrays = {
        "gt": gt_depth.detach().cpu().numpy(),
        "raw": raw_depth.detach().cpu().numpy(),
        "hsi": hsi_depth.detach().cpu().numpy(),
    }
    valid_np = valid.detach().cpu().numpy()
    row = valid_np.shape[0] // 2
    values = []
    for arr in arrays.values():
        values.extend(arr[row][valid_np[row]].tolist())
    lo, hi = robust_range(np.asarray(values, dtype=np.float32), robust=True)
    width = valid_np.shape[1]
    height = 220
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    colors = {"gt": (40, 160, 40), "raw": (60, 120, 230), "hsi": (220, 40, 160)}
    for name, arr in arrays.items():
        profile = arr[row]
        pts = []
        for x, val in enumerate(profile):
            if not np.isfinite(val):
                continue
            y = int((1.0 - np.clip((float(val) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)) * (height - 28)) + 12
            pts.append((x, y))
        if len(pts) > 1:
            draw.line(pts, fill=colors[name], width=2)
        draw.text((8, 8 + 18 * list(arrays).index(name)), f"{name} center row", fill=colors[name])
    out.save(path)
    return path


def save_projection_overlay(
    path: Path,
    image: Image.Image,
    image_hw: tuple[int, int],
    gt_boxes: torch.Tensor,
    boxes_mask: torch.Tensor,
    k_dataset: torch.Tensor,
    k_vggt: torch.Tensor,
    base: dict[str, torch.Tensor],
    hsi: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    confs: torch.Tensor,
    args: argparse.Namespace,
    mode: str = "all",
) -> Path:
    draw = ImageDraw.Draw(image)
    if mode in {"all", "gt_dataset", "gt_vggt"}:
        for idx in range(int(gt_boxes.shape[0])):
            if bool(boxes_mask[idx]):
                draw_box(draw, gt_boxes[idx], image_hw, fill=(255, 180, 0), text=f"box{idx}")

    gt_idx = torch.nonzero(torch.isfinite(gt["transl"][:, 2]) & (gt["transl"][:, 2] > 1e-6), as_tuple=False).flatten()
    pred_idx = torch.nonzero(confs >= float(args.conf_threshold), as_tuple=False).flatten()
    matches = greedy_match(base["joints"][pred_idx, :24], gt["joints"][gt_idx, :24]) if gt_idx.numel() and pred_idx.numel() else []
    for pred_local, gt_local in matches:
        q = int(pred_idx[pred_local].item())
        g = int(gt_idx[gt_local].item())
        if mode in {"all", "gt_dataset"}:
            draw_projected_points(draw, gt["joints"][g, :24], k_dataset, (50, 220, 50), image_hw, label=f"GT{g} datasetK")
            vertices = gt["vertices"][g]
            stride = max(int(math.ceil(vertices.shape[0] / max(int(args.overlay_max_vertices), 1))), 1)
            draw_projected_points(draw, vertices[::stride], k_dataset, (20, 120, 20), image_hw, radius=1)
        if mode in {"all", "gt_vggt"}:
            draw_projected_points(draw, gt["joints"][g, :24], k_vggt, (255, 220, 0), image_hw, label=f"GT{g} vggtK")
        if mode in {"all", "base"}:
            draw_projected_points(draw, base["joints"][q, :24], k_vggt, (0, 210, 255), image_hw, label=f"base{q}")
        if mode in {"all", "hsi"}:
            draw_projected_points(draw, hsi["joints"][q, :24], k_vggt, (255, 50, 200), image_hw, label=f"hsi{q}")
    label = {
        "all": "green=GT datasetK yellow=GT VGGTK cyan=NLF/base magenta=HSI",
        "gt_dataset": "GT SMPL projected with dataset K",
        "gt_vggt": "GT SMPL projected with VGGT predicted K",
        "base": "NLF/base SMPL projected with VGGT predicted K",
        "hsi": "HSI refined SMPL projected with VGGT predicted K",
    }.get(mode, mode)
    draw.text((8, 8), label, fill=(255, 255, 255))
    image.save(path)
    return path


def draw_box(draw: ImageDraw.ImageDraw, box_cxcywh: torch.Tensor, image_hw: tuple[int, int], fill: tuple[int, int, int], text: str = "") -> None:
    h, w = image_hw
    cx, cy, bw, bh = [float(v) for v in box_cxcywh.detach().cpu().reshape(-1).tolist()]
    x0 = (cx - bw * 0.5) * w
    y0 = (cy - bh * 0.5) * h
    x1 = (cx + bw * 0.5) * w
    y1 = (cy + bh * 0.5) * h
    draw.rectangle([x0, y0, x1, y1], outline=fill, width=2)
    if text:
        draw.text((x0 + 2, y0 + 2), text, fill=fill)


def draw_projected_points(
    draw: ImageDraw.ImageDraw,
    points: torch.Tensor,
    intrinsics: torch.Tensor,
    color: tuple[int, int, int],
    image_hw: tuple[int, int],
    label: str = "",
    radius: int = 3,
) -> None:
    pts = project_points(points, intrinsics).detach().float().cpu().numpy()
    h, w = image_hw
    first_valid: tuple[float, float] | None = None
    for x, y in pts:
        if not np.isfinite(x) or not np.isfinite(y) or x < 0 or y < 0 or x >= w or y >= h:
            continue
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
        if first_valid is None:
            first_valid = (float(x), float(y))
    if label and first_valid is not None:
        draw.text((first_valid[0] + 4, first_valid[1] + 4), label, fill=color)


def projected_bbox_iou(points: torch.Tensor, intrinsics: torch.Tensor, box_cxcywh: torch.Tensor, box_valid: torch.Tensor, image_hw: tuple[int, int]) -> float | None:
    if not bool(box_valid):
        return None
    projected = project_points(points, intrinsics).detach().float().cpu().numpy()
    h, w = image_hw
    ok = np.isfinite(projected).all(axis=1) & (projected[:, 0] >= 0) & (projected[:, 0] < w) & (projected[:, 1] >= 0) & (projected[:, 1] < h)
    if not ok.any():
        return None
    x0, y0 = projected[ok].min(axis=0)
    x1, y1 = projected[ok].max(axis=0)
    cx, cy, bw, bh = [float(v) for v in box_cxcywh.detach().cpu().reshape(-1).tolist()]
    bx0 = (cx - bw * 0.5) * w
    by0 = (cy - bh * 0.5) * h
    bx1 = (cx + bw * 0.5) * w
    by1 = (cy + bh * 0.5) * h
    return box_iou_xyxy((x0, y0, x1, y1), (bx0, by0, bx1, by1))


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


def colorize_depth(array: np.ndarray, valid: np.ndarray | None = None, robust: bool = True) -> Image.Image:
    arr = np.asarray(array, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(arr)
    else:
        valid = valid & np.isfinite(arr)
    lo, hi = robust_range(arr[valid], robust=robust)
    norm = np.zeros_like(arr, dtype=np.float32)
    norm[valid] = np.clip((arr[valid] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    r = (255.0 * norm).astype(np.uint8)
    g = (255.0 * (1.0 - np.abs(norm - 0.5) * 2.0)).astype(np.uint8)
    b = (255.0 * (1.0 - norm)).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[~valid] = np.array([0, 0, 0], dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def robust_range(values: np.ndarray, robust: bool = True) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    if robust and vals.size > 10:
        lo = float(np.percentile(vals, 2.0))
        hi = float(np.percentile(vals, 98.0))
    else:
        lo = float(vals.min())
        hi = float(vals.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return lo, hi


def depth_valid_mask(gt_depth: torch.Tensor, raw_depth: torch.Tensor, hsi_depth: torch.Tensor, depth_max_m: float) -> torch.Tensor:
    valid = torch.isfinite(gt_depth) & torch.isfinite(raw_depth) & torch.isfinite(hsi_depth) & (gt_depth > 1e-6)
    if depth_max_m > 0:
        valid = valid & (gt_depth <= float(depth_max_m))
    return valid


def depth_region_stats(raw_depth: torch.Tensor, hsi_depth: torch.Tensor, gt_depth: torch.Tensor, valid: torch.Tensor) -> dict[str, Any]:
    if not valid.any():
        return {"valid_pixels": 0}
    raw_err = torch.abs(raw_depth[valid] - gt_depth[valid])
    hsi_err = torch.abs(hsi_depth[valid] - gt_depth[valid])
    return {
        "valid_pixels": int(valid.sum().detach().cpu()),
        "raw_l1_mean_m": scalar(raw_err.mean()),
        "hsi_l1_mean_m": scalar(hsi_err.mean()),
        "raw_l1_median_m": scalar(raw_err.median()),
        "hsi_l1_median_m": scalar(hsi_err.median()),
        "raw_rmse_m": scalar(torch.sqrt(raw_err.square().mean())),
        "hsi_rmse_m": scalar(torch.sqrt(hsi_err.square().mean())),
        "hsi_minus_raw_median_l1_delta_m": scalar(hsi_err.median() - raw_err.median()),
    }


def array_stats(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach().float()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"count": 0}
    return {
        "count": int(values.numel()),
        "min": scalar(values.min()),
        "p05": quantile(values, 0.05),
        "median": scalar(values.median()),
        "p95": quantile(values, 0.95),
        "max": scalar(values.max()),
        "mean": scalar(values.mean()),
        "std": std_value(values),
    }


def gradient_stats(depth: torch.Tensor, valid: torch.Tensor) -> dict[str, Any]:
    dzdx = F.pad(depth[:, 2:] - depth[:, :-2], (1, 1, 0, 0)) * 0.5
    dzdy = F.pad(depth[2:, :] - depth[:-2, :], (0, 0, 1, 1)) * 0.5
    grad = torch.sqrt(dzdx.square() + dzdy.square())
    return array_stats(grad[valid])


def robust_affine_fit(x: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    valid = torch.isfinite(x) & torch.isfinite(y)
    x = x[valid].detach().float()
    y = y[valid].detach().float()
    if x.numel() < 8:
        return {"valid": False, "reason": "not_enough_points", "count": int(x.numel())}
    x_med = x.median()
    y_med = y.median()
    xm = x - x_med
    ym = y - y_med
    denom = (xm * xm).mean()
    if float(denom.detach().cpu()) <= 1e-12:
        return {"valid": False, "reason": "raw_depth_nearly_constant", "count": int(x.numel())}
    scale = (xm * ym).mean() / denom
    bias = y_med - scale * x_med
    pred = x * scale + bias
    err = torch.abs(pred - y)
    return {
        "valid": True,
        "count": int(x.numel()),
        "scale": scalar(scale),
        "bias": scalar(bias),
        "median_l1_m": scalar(err.median()),
        "mean_l1_m": scalar(err.mean()),
    }


def frame_scale_bias(predictions: dict[str, torch.Tensor], frame_idx: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "hsi_scene_scale" in predictions:
        out["hsi_scene_scale"] = scalar(predictions["hsi_scene_scale"][0, frame_idx].reshape(-1)[0])
    if "hsi_scene_depth_bias" in predictions:
        out["hsi_scene_depth_bias"] = scalar(predictions["hsi_scene_depth_bias"][0, frame_idx].reshape(-1)[0])
    for key in ("hsi_anchor_depth_residual", "hsi_per_query_scene_depth_bias"):
        if key in predictions:
            values = predictions[key][0, frame_idx].detach().float().reshape(-1)
            out[key] = array_stats(values)
    return out


def frame_alerts(depth_stats: dict[str, Any], scale_bias: dict[str, Any], projection_stats: dict[str, Any], args: argparse.Namespace) -> list[str]:
    alerts = []
    scale = scale_bias.get("hsi_scene_scale")
    if scale is not None and abs(float(scale)) < float(args.small_scale_threshold):
        alerts.append("hsi_scene_scale_is_too_small_depth_can_collapse_to_plane")
    std_ratio = depth_stats.get("hsi_vs_raw_std_ratio")
    if std_ratio is not None and float(std_ratio) < float(args.flat_std_ratio_threshold):
        alerts.append("hsi_depth_std_much_smaller_than_raw_depth_flattening_likely")
    grad_ratio = depth_stats.get("hsi_vs_raw_gradient_ratio")
    if grad_ratio is not None and float(grad_ratio) < float(args.flat_gradient_ratio_threshold):
        alerts.append("hsi_depth_gradient_much_smaller_than_raw_depth_flattening_likely")
    fit = depth_stats.get("robust_affine_roi_raw_to_gt", {})
    pred_scale = scale_bias.get("hsi_scene_scale")
    if fit.get("valid") and pred_scale is not None:
        fit_scale = fit.get("scale")
        if fit_scale is not None and abs(float(pred_scale) - float(fit_scale)) > max(0.5, abs(float(fit_scale)) * 0.75):
            alerts.append("predicted_hsi_scale_far_from_roi_robust_affine_fit")
    for person in projection_stats.get("persons", []):
        gt_dataset = person.get("gt_vertices_datasetK_gtDepth", {})
        gt_vggt = person.get("gt_vertices_vggtK_gtDepth", {})
        if gt_dataset.get("mean_abs_sample_minus_z_m") is not None and gt_vggt.get("mean_abs_sample_minus_z_m") is not None:
            if float(gt_dataset["mean_abs_sample_minus_z_m"]) < 0.25 and float(gt_vggt["mean_abs_sample_minus_z_m"]) > 0.75:
                alerts.append("gt_smpl_matches_depth_with_datasetK_but_not_vggtK_camera_mismatch_likely")
                break
    return sorted(set(alerts))


def collect_global_alerts(samples: list[dict[str, Any]]) -> list[str]:
    alerts = []
    for sample in samples:
        for frame in sample.get("frames", []):
            alerts.extend(frame.get("alerts", []))
    return sorted(set(alerts))


def intrinsics_diff(k_dataset: torch.Tensor, k_vggt: torch.Tensor) -> dict[str, Any]:
    kd = k_dataset.detach().float().cpu().numpy()
    kv = k_vggt.detach().float().cpu().numpy()
    labels = {"fx": (0, 0), "fy": (1, 1), "cx": (0, 2), "cy": (1, 2)}
    out = {}
    for name, (r, c) in labels.items():
        denom = abs(float(kd[r, c])) + 1e-6
        out[name] = {
            "dataset": float(kd[r, c]),
            "vggt": float(kv[r, c]),
            "abs_diff": float(kv[r, c] - kd[r, c]),
            "rel_diff": float((kv[r, c] - kd[r, c]) / denom),
        }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def squeeze_confs(confs: torch.Tensor) -> torch.Tensor:
    if confs.ndim == 4 and confs.shape[-1] == 1:
        return confs[..., 0]
    return confs


def scalar(value: torch.Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        value = float(value.detach().float().cpu().reshape(-1)[0])
    value = float(value)
    return value if math.isfinite(value) else None


def tensor_to_list(value: torch.Tensor) -> Any:
    return value.detach().float().cpu().tolist()


def quantile(values: torch.Tensor, q: float) -> float | None:
    if values.numel() == 0:
        return None
    return scalar(torch.quantile(values.detach().float(), float(q)))


def quantile_abs(values: torch.Tensor, q: float) -> float | None:
    if values.numel() == 0:
        return None
    return quantile(torch.abs(values.detach().float()), q)


def std_value(values: torch.Tensor) -> float | None:
    values = values.detach().float()
    values = values[torch.isfinite(values)]
    if values.numel() < 2:
        return None
    return scalar(values.std(unbiased=False))


def safe_ratio(numer: float | None, denom: float | None) -> float | None:
    if numer is None or denom is None or abs(float(denom)) < 1e-12:
        return None
    return float(numer) / float(denom)


def print_human_summary(summary: dict[str, Any]) -> None:
    print("========== NLF-HSI depth flattening debug ==========")
    print(f"samples: {summary.get('num_samples')}")
    alerts = summary.get("global_alerts", [])
    print(f"global alerts: {alerts if alerts else '<none>'}")
    for sample_idx, sample in enumerate(summary.get("samples", [])):
        for frame_idx, frame in enumerate(sample.get("frames", [])):
            scale = frame.get("hsi_affine", {}).get("hsi_scene_scale")
            bias = frame.get("hsi_affine", {}).get("hsi_scene_depth_bias")
            roi = frame.get("depth_stats", {}).get("human_roi", {})
            print(
                f"sample={sample_idx} frame={frame_idx} "
                f"scale={scale} bias={bias} "
                f"roi raw_med={roi.get('raw_l1_median_m')} hsi_med={roi.get('hsi_l1_median_m')} "
                f"alerts={frame.get('alerts', [])}"
            )


if __name__ == "__main__":
    main()
