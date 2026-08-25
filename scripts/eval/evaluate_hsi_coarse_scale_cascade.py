#!/usr/bin/env python
"""Compare analytic SMPL-depth scale, direct HSI, and coarse-to-HSI cascade."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_smpl import apply_overrides, build_model  # noqa: E402
from scripts.vis.serve_nlf_hsi_vggt_sequence_viewer import (  # noqa: E402
    canonical_depth,
    decode_people,
    load_sequence_images,
)
from scripts.vis.visualize_smpl_inference import (  # noqa: E402
    estimate_scene_to_smpl_scale,
    load_training_checkpoint,
    load_vggt_baseline_for_camera,
)
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.tracking.io import iter_image_files  # noqa: E402
from vggt_omega.training.config import deep_update, load_yaml_config, require_path  # noqa: E402


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = select_frames(resolve_path(args.frames_dir), args)
    if not frame_paths:
        raise RuntimeError(f"No input frames found: {args.frames_dir}")

    config = load_inference_config(args)
    patch_size = int(config["model"].get("patch_size", 16))
    image_resolution = int(config["data"].get("image_resolution", 512))
    images, _ = load_sequence_images(
        frame_paths,
        image_resolution,
        patch_size,
        str(config["data"].get("resize_mode", "balanced")),
    )
    image_sequence = images.unsqueeze(0).to(device)

    model = build_model(config).to(device).eval()
    load_vggt_baseline_for_camera(model, config, device)
    checkpoint = resolve_path(args.checkpoint)
    load_training_checkpoint(model, checkpoint, device)
    smpl = SMPLLayer(require_path(config, "assets.smpl_model_dir", allow_empty=False)).to(device).eval()

    with torch.inference_mode():
        direct = model(image_sequence)
        raw_depth = canonical_depth(direct["depth"]).detach().float()
        direct_depth = apply_scene_affine(raw_depth, direct)
        decoded = decode_people(direct, smpl, args, device)
        base_vertices = decoded["base_vertices_cam"]
        coarse_scales, coarse_records = estimate_sequence_scales(
            vertices=base_vertices,
            depth=raw_depth,
            pose_enc=direct["pose_enc"],
            confs=direct["pred_confs"],
            args=args,
        )
        coarse_depth = raw_depth * coarse_scales[..., None, None]
        residual = model(
            image_sequence,
            hsi_depth_override=coarse_depth,
            hsi_depth_is_metric=True,
            hsi_geometry_mode="smpl_coarse_metric",
        )
        cascade_depth = apply_scene_affine(coarse_depth, residual)

    variants = {
        "raw_vggt": raw_depth,
        "analytic_coarse": coarse_depth,
        "direct_hsi": direct_depth,
        "coarse_then_hsi": cascade_depth,
    }
    variant_records = {
        name: evaluate_variant(
            vertices=base_vertices,
            depth=depth,
            pose_enc=direct["pose_enc"],
            confs=direct["pred_confs"],
            args=args,
        )
        for name, depth in variants.items()
    }
    direct_scale = direct["hsi_scene_scale"].detach().float().reshape(1, len(frame_paths), -1)[..., 0]
    direct_bias = direct["hsi_scene_depth_bias"].detach().float().reshape(1, len(frame_paths), -1)[..., 0]
    residual_scale = residual["hsi_scene_scale"].detach().float().reshape(1, len(frame_paths), -1)[..., 0]
    residual_bias = residual["hsi_scene_depth_bias"].detach().float().reshape(1, len(frame_paths), -1)[..., 0]

    frame_rows = []
    for frame_idx, frame_path in enumerate(frame_paths):
        row = {
            "frame_index": frame_idx,
            "frame_id": frame_path.stem,
            "people": int((direct["pred_confs"][0, frame_idx, :, 0] >= args.conf_threshold).sum().cpu()),
            "coarse_scale": float(coarse_scales[0, frame_idx].cpu()),
            "coarse_estimator": coarse_records[frame_idx],
            "direct_model_scale": float(direct_scale[0, frame_idx].cpu()),
            "direct_model_bias": float(direct_bias[0, frame_idx].cpu()),
            "residual_model_scale": float(residual_scale[0, frame_idx].cpu()),
            "residual_model_bias": float(residual_bias[0, frame_idx].cpu()),
            "cascade_effective_scale": float((coarse_scales[0, frame_idx] * residual_scale[0, frame_idx]).cpu()),
            "variant_alignment": {
                name: records[frame_idx]
                for name, records in variant_records.items()
            },
        }
        frame_rows.append(row)

    summary = {
        "status": "ok",
        "checkpoint": str(checkpoint),
        "frames_dir": str(resolve_path(args.frames_dir)),
        "num_frames": len(frame_paths),
        "method": "nearest-SMPL-anchor-per-pixel median z_smpl/z_depth",
        "config": {
            "scale_min": args.scale_min,
            "scale_max": args.scale_max,
            "anchor_stride": args.anchor_stride,
            "min_anchor_pixels": args.min_anchor_pixels,
            "confidence_threshold": args.conf_threshold,
        },
        "aggregate": {
            "coarse_scale": summarize_values([row["coarse_scale"] for row in frame_rows]),
            "direct_model_scale": summarize_values([row["direct_model_scale"] for row in frame_rows]),
            "residual_model_scale": summarize_values([row["residual_model_scale"] for row in frame_rows]),
            "cascade_effective_scale": summarize_values([row["cascade_effective_scale"] for row in frame_rows]),
            "variants": {
                name: summarize_variant(records)
                for name, records in variant_records.items()
            },
        },
        "frames": frame_rows,
    }
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": str(output_path), "aggregate": summary["aggregate"]}, indent=2, ensure_ascii=False))


def load_inference_config(args: argparse.Namespace) -> dict[str, Any]:
    config = deep_update(load_yaml_config(resolve_path(args.path_config)), load_yaml_config(resolve_path(args.train_config)))
    config = apply_overrides(config, args.override)
    model_cfg = config.setdefault("model", {})
    model_cfg.update(
        {
            "enable_camera": True,
            "enable_depth": True,
            "enable_smpl": True,
            "enable_hsi_refine": True,
            "smpl_provider": "nlf",
            "num_smpl_queries": int(args.max_humans),
            "smpl_use_aggregator_queries": False,
            "smpl_query_box_prior": False,
            "smpl_query_patch_pool": False,
            "nlf_use_detector": True,
            "nlf_require_boxes": False,
            "smpl_track_assignment_mode": "none",
            "smpl_use_external_track_prior": False,
        }
    )
    return config


def estimate_sequence_scales(
    vertices: torch.Tensor,
    depth: torch.Tensor,
    pose_enc: torch.Tensor,
    confs: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    scales = depth.new_ones(depth.shape[:2])
    records: list[dict[str, Any]] = []
    for frame_idx in range(depth.shape[1]):
        valid_people = confs[0, frame_idx, :, 0] >= float(args.conf_threshold)
        if not bool(valid_people.any()):
            records.append({"applied": False, "reason": "no_confident_people", "scale": 1.0})
            continue
        record = estimate_scene_to_smpl_scale(
            smpl_vertices=vertices[0, frame_idx, valid_people],
            depth=depth[0, frame_idx],
            pose_enc=pose_enc[:, frame_idx : frame_idx + 1],
            input_size=max(depth.shape[-2:]),
            min_anchor_pixels=int(args.min_anchor_pixels),
            scale_min=float(args.scale_min),
            scale_max=float(args.scale_max),
            anchor_stride=int(args.anchor_stride),
        )
        if bool(record.get("applied", False)):
            scales[0, frame_idx] = float(record["scale"])
        records.append(record)
    return scales, records


def evaluate_variant(
    vertices: torch.Tensor,
    depth: torch.Tensor,
    pose_enc: torch.Tensor,
    confs: torch.Tensor,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    _, records = estimate_sequence_scales(vertices, depth, pose_enc, confs, args)
    return records


def apply_scene_affine(depth: torch.Tensor, predictions: dict[str, torch.Tensor]) -> torch.Tensor:
    scale = predictions["hsi_scene_scale"].detach().float().reshape(*depth.shape[:2], 1, 1)
    bias = predictions["hsi_scene_depth_bias"].detach().float().reshape(*depth.shape[:2], 1, 1)
    return depth * scale + bias


def summarize_variant(records: list[dict[str, Any]]) -> dict[str, Any]:
    applied = [record for record in records if bool(record.get("applied", False))]
    return {
        "applied_frames": len(applied),
        "required_residual_scale": summarize_values([float(record["scale"]) for record in applied]),
        "anchor_depth_l1_median_before": summarize_values(
            [float(record["anchor_depth_l1_median_before"]) for record in applied if "anchor_depth_l1_median_before" in record]
        ),
        "anchor_depth_l1_median_after_extra_fit": summarize_values(
            [float(record["anchor_depth_l1_median_after"]) for record in applied if "anchor_depth_l1_median_after" in record]
        ),
    }


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def select_frames(frames_dir: Path, args: argparse.Namespace) -> list[Path]:
    frames = iter_image_files(frames_dir)
    selected = frames[max(int(args.start_index), 0) :: max(int(args.frame_stride), 1)]
    return selected[: int(args.max_frames)] if int(args.max_frames) > 0 else selected


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_hsi_gt_depth_scale_scene_affine.yaml")
    parser.add_argument("--output-dir", default="outputs/eval/hsi_coarse_scale_cascade")
    parser.add_argument("--device", default="")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-humans", type=int, default=20)
    parser.add_argument("--conf-threshold", type=float, default=0.10)
    parser.add_argument("--scale-min", type=float, default=0.10)
    parser.add_argument("--scale-max", type=float, default=10.0)
    parser.add_argument("--anchor-stride", type=int, default=8)
    parser.add_argument("--min-anchor-pixels", type=int, default=32)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    main()
