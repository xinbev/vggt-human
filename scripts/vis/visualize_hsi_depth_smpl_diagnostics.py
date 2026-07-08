#!/usr/bin/env python
"""Single-image diagnostics for HSI-refined SMPL and HSI-aligned depth."""

from __future__ import annotations

import argparse
import json
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
    depth_triplet,
    get_foot_sole_indices,
    greedy_match,
    human_roi_depth_mask,
    load_training_checkpoint,
    load_vggt_baseline,
    project_points,
    sample_depth_at_points,
    sample_local_support_plane_signed_delta,
    scale_points_to_depth,
)
from scripts.train.train_smpl import apply_overrides, build_model, load_yaml_config  # noqa: E402
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
    image_path = Path(args.image).expanduser()
    batch, rgb = load_single_bedlam_batch(image_path, config, args)
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}

    model = build_model(config).to(device)
    load_vggt_baseline(model, config, device)
    load_training_checkpoint(model, Path(args.checkpoint).expanduser(), device)
    model.eval()
    smpl = SMPLLayer(require_path(config, "assets.smpl_model_dir", allow_empty=False)).to(device).eval()

    with torch.no_grad():
        predictions = model(
            batch["images"],
            smpl_query_boxes=batch["gt_boxes"] if args.use_gt_box_prior else None,
            smpl_query_boxes_mask=batch["boxes_mask"] if args.use_gt_box_prior else None,
        )

    depth_summary, depth_images = compute_depth_diagnostics(predictions, batch)
    smpl_summary, contact_points = compute_smpl_diagnostics(predictions, batch, smpl, config, args)
    diagnosis = infer_source(depth_summary, smpl_summary, args)

    stem = image_path.stem
    image_paths = save_depth_visuals(output_dir, stem, rgb, depth_images, depth_summary)
    image_paths.update(save_contact_overlays(output_dir, stem, rgb, contact_points, args))
    out_json = output_dir / f"{stem}_hsi_depth_smpl_diagnostics.json"
    out_json.write_text(
        json.dumps(
            {
                "image": str(image_path),
                "checkpoint": str(args.checkpoint),
                "use_gt_box_prior": bool(args.use_gt_box_prior),
                "depth": depth_summary,
                "smpl": smpl_summary,
                "diagnosis": diagnosis,
                "visualizations": image_paths,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print_human_summary(depth_summary, smpl_summary, diagnosis)
    print(json.dumps({"output_json": str(out_json), "visualizations": image_paths}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize HSI depth vs GT depth and refined SMPL vs GT SMPL")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_hsi_refine.yaml")
    parser.add_argument("--output-dir", default="outputs/vis/hsi_depth_smpl_diagnostics")
    parser.add_argument("--device", default="")
    parser.add_argument("--split", default="Training")
    parser.add_argument("--smpl-model-dir", default="")
    parser.add_argument("--conf-threshold", type=float, default=0.10)
    parser.add_argument("--use-gt-box-prior", action="store_true")
    parser.add_argument("--depth-bad-threshold-m", type=float, default=0.30)
    parser.add_argument("--smpl-good-threshold-m", type=float, default=0.08)
    parser.add_argument("--foot-contact-threshold-m", type=float, default=0.12)
    parser.add_argument("--foot-float-margin-m", type=float, default=0.04)
    parser.add_argument("--foot-penetration-margin-m", type=float, default=0.02)
    parser.add_argument("--foot-sole-num-vertices", type=int, default=80)
    parser.add_argument("--foot-sole-contact-threshold-m", type=float, default=0.08)
    parser.add_argument("--support-plane-window", type=int, default=9)
    parser.add_argument("--support-plane-min-points", type=int, default=6)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.train_config))
    config = apply_overrides(config, args.override)
    if args.baseline_checkpoint:
        config.setdefault("checkpoints", {})["vggt_baseline"] = args.baseline_checkpoint
    if args.smpl_model_dir:
        config.setdefault("assets", {})["smpl_model_dir"] = args.smpl_model_dir
    config.setdefault("model", {})["enable_camera"] = True
    config.setdefault("model", {})["enable_depth"] = True
    config.setdefault("model", {})["enable_hsi_refine"] = True
    config.setdefault("data", {})["require_depth"] = True
    config.setdefault("data", {})["require_boxes"] = True
    config.setdefault("data", {})["image_resolution"] = int(config.get("data", {}).get("image_resolution", 512))
    config.setdefault("data", {})["resize_mode"] = str(config.get("data", {}).get("resize_mode", "balanced"))
    return config


def load_single_bedlam_batch(image_path: Path, config: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, torch.Tensor], Image.Image]:
    data_cfg = config["data"]
    image_size, image_resolution = resolve_image_size_config(data_cfg)
    dataset_root = Path(require_path(config, data_cfg.get("root_key", "datasets.bedlam_root"), allow_empty=False)).expanduser()
    boxes_root = Path(require_path(config, data_cfg["boxes_root_key"], allow_empty=False)).expanduser()
    split = str(args.split)
    dataset = BedlamDataset(
        root=dataset_root,
        split=split,
        sequence_length=int(data_cfg["sequence_length"]),
        stride=int(data_cfg["stride"]),
        image_size=image_size,
        image_resolution=image_resolution,
        resize_mode=str(data_cfg.get("resize_mode", "balanced")),
        max_humans=int(data_cfg.get("max_humans", config.get("model", {}).get("num_smpl_queries", 20))),
        require_smpl=bool(data_cfg.get("require_smpl", True)),
        require_depth=bool(data_cfg.get("require_depth", True)),
        boxes_root=boxes_root,
        require_boxes=bool(data_cfg.get("require_boxes", True)),
        query_source=str(data_cfg.get("query_source", "persons")),
        patch_size=int(config.get("model", {}).get("patch_size", 16)),
        mask_patch_threshold=float(data_cfg.get("mask_patch_threshold", 0.10)),
        min_mask_patches=int(data_cfg.get("min_mask_patches", 4)),
    )
    dataset_index = find_bedlam_window_index(dataset, image_path, dataset_root, split)
    sample = dataset[dataset_index]
    batch = {key: value.unsqueeze(0) for key, value in sample.items() if isinstance(value, torch.Tensor)}
    rgb = tensor_image_to_pil(sample["images"][0])
    return batch, rgb


def find_bedlam_window_index(dataset: BedlamDataset, image_path: Path, dataset_root: Path, split: str) -> int:
    seq_dir = image_path.parent.parent
    frame_id = image_path.stem
    try:
        sequence_rel = seq_dir.relative_to(dataset_root / split)
    except ValueError as exc:
        raise ValueError(f"Image path must live under {dataset_root / split}: {image_path}") from exc
    for dataset_index, (seq_idx, start_idx) in enumerate(dataset._index):
        seq_path, frame_ids = dataset._sequences[seq_idx]
        if seq_path.relative_to(dataset_root / split) == sequence_rel and frame_ids[start_idx] == frame_id:
            return dataset_index
    raise ValueError(
        f"Could not find image as the first frame of a BEDLAM window: split={split} "
        f"sequence={sequence_rel} frame={frame_id}. Try an earlier IMAGE_PATH with enough following frames."
    )


def tensor_image_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().float().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((array * 255.0).round().astype("uint8"), mode="RGB")


def compute_depth_diagnostics(predictions: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
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
        hsi_depth = raw_depth
        scale_value = None
        bias_value = None
    else:
        scale = scale.to(device=raw_depth.device, dtype=raw_depth.dtype).reshape(*raw_depth.shape[:2], 1, 1)
        bias = bias.to(device=raw_depth.device, dtype=raw_depth.dtype).reshape(*raw_depth.shape[:2], 1, 1)
        hsi_depth = raw_depth * scale + bias
        scale_value = float(scale.reshape(-1)[0].detach().cpu())
        bias_value = float(bias.reshape(-1)[0].detach().cpu())

    valid = torch.isfinite(gt_depth) & (gt_depth > 1e-6) & torch.isfinite(raw_depth) & torch.isfinite(hsi_depth)
    near_valid = valid & (gt_depth <= 30.0)
    far_valid = valid & (gt_depth > 30.0)
    roi_mask = human_roi_depth_mask(
        batch["gt_boxes"].to(device=raw_depth.device, dtype=raw_depth.dtype),
        batch["boxes_mask"].to(device=raw_depth.device).bool(),
        raw_depth.shape[-2],
        raw_depth.shape[-1],
        expand=0.75,
    )
    roi_valid = valid & roi_mask
    full_stats = depth_region_summary(raw_depth, hsi_depth, gt_depth, valid)
    near_stats = depth_region_summary(raw_depth, hsi_depth, gt_depth, near_valid)
    far_stats = depth_region_summary(raw_depth, hsi_depth, gt_depth, far_valid)
    roi_stats = depth_region_summary(raw_depth, hsi_depth, gt_depth, roi_valid)
    summary = {
        "hsi_scene_scale": scale_value,
        "hsi_scene_depth_bias": bias_value,
        "valid_pixels": full_stats["valid_pixels"],
        "raw_depth_l1_mean_m": full_stats["raw_depth_l1_mean_m"],
        "hsi_depth_l1_mean_m": full_stats["hsi_depth_l1_mean_m"],
        "raw_depth_l1_median_m": full_stats["raw_depth_l1_median_m"],
        "hsi_depth_l1_median_m": full_stats["hsi_depth_l1_median_m"],
        "raw_depth_rmse_m": full_stats["raw_depth_rmse_m"],
        "hsi_depth_rmse_m": full_stats["hsi_depth_rmse_m"],
        "hsi_depth_improvement_median_percent": full_stats["hsi_depth_improvement_median_percent"],
        "regions": {
            "full": full_stats,
            "near_lt_30m": near_stats,
            "far_ge_30m": far_stats,
            "human_roi": roi_stats,
        },
    }
    images = {
        "gt_depth": gt_depth[0, 0].detach().cpu().numpy(),
        "raw_depth": raw_depth[0, 0].detach().cpu().numpy(),
        "hsi_depth": hsi_depth[0, 0].detach().cpu().numpy(),
        "raw_abs_error": torch.abs(raw_depth[0, 0] - gt_depth[0, 0]).detach().cpu().numpy(),
        "hsi_abs_error": torch.abs(hsi_depth[0, 0] - gt_depth[0, 0]).detach().cpu().numpy(),
        "valid": valid[0, 0].detach().cpu().numpy(),
    }
    return summary, images


def depth_region_summary(
    raw_depth: torch.Tensor,
    hsi_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, Any]:
    raw_abs = torch.abs(raw_depth[valid] - gt_depth[valid]) if valid.any() else raw_depth.new_empty(0)
    hsi_abs = torch.abs(hsi_depth[valid] - gt_depth[valid]) if valid.any() else raw_depth.new_empty(0)
    raw_median = scalar(raw_abs.median()) if raw_abs.numel() else None
    hsi_median = scalar(hsi_abs.median()) if hsi_abs.numel() else None
    return {
        "valid_pixels": int(valid.sum().detach().cpu()),
        "raw_depth_l1_mean_m": scalar(raw_abs.mean()) if raw_abs.numel() else None,
        "hsi_depth_l1_mean_m": scalar(hsi_abs.mean()) if hsi_abs.numel() else None,
        "raw_depth_l1_median_m": raw_median,
        "hsi_depth_l1_median_m": hsi_median,
        "raw_depth_rmse_m": scalar(torch.sqrt((raw_abs.square()).mean())) if raw_abs.numel() else None,
        "hsi_depth_rmse_m": scalar(torch.sqrt((hsi_abs.square()).mean())) if hsi_abs.numel() else None,
        "hsi_depth_improvement_median_percent": improvement(raw_median, hsi_median),
    }


def compute_smpl_diagnostics(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    smpl: SMPLLayer,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    base = decode_smpl_batch(predictions["pred_poses"], predictions["pred_betas"], predictions["pred_transl_cam"], smpl)
    hsi = decode_smpl_batch(
        predictions["hsi_refined_pred_poses"],
        predictions["hsi_refined_pred_betas"],
        predictions["hsi_refined_pred_transl_cam"],
        smpl,
    )
    gt_poses = rot6d_to_axis_angle(batch["gt_pose_6d"].reshape(-1, 24, 6)).reshape(*batch["gt_pose_6d"].shape[:3], 72)
    gt = decode_smpl_batch(gt_poses, batch["gt_betas"], batch["gt_transl_cam"], smpl)
    intrinsics = encoding_to_camera(
        predictions["pose_enc"],
        image_size_hw=(int(batch["images"].shape[-2]), int(batch["images"].shape[-1])),
        build_intrinsics=True,
    )[1]
    confs = predictions["pred_confs"].detach()
    if confs.ndim == 4 and confs.shape[-1] == 1:
        confs = confs[..., 0]
    _, hsi_depth, gt_depth = depth_triplet(predictions, batch)

    gt_idx = torch.nonzero(batch["smpl_mask"][0, 0].bool(), as_tuple=False).flatten()
    pred_idx = torch.nonzero(confs[0, 0] >= float(args.conf_threshold), as_tuple=False).flatten()
    matches = greedy_match(base["joints"][0, 0, pred_idx, :24], gt["joints"][0, 0, gt_idx, :24]) if gt_idx.numel() and pred_idx.numel() else []
    per_person = []
    contact_points: dict[str, list[dict[str, Any]]] = {"base": [], "hsi": []}
    sole_indices = get_foot_sole_indices(smpl, int(args.foot_sole_num_vertices), device=gt["vertices"].device)
    meters: dict[str, list[float]] = {
        "base_joints_mpjpe_m": [],
        "hsi_joints_mpjpe_m": [],
        "base_vertices_pve_m": [],
        "hsi_vertices_pve_m": [],
        "base_transl_l2_m": [],
        "hsi_transl_l2_m": [],
        "base_projected_joints_l2_px": [],
        "hsi_projected_joints_l2_px": [],
        "base_foot_abs_delta_m": [],
        "hsi_foot_abs_delta_m": [],
        "base_foot_float_m": [],
        "hsi_foot_float_m": [],
        "base_foot_penetration_m": [],
        "hsi_foot_penetration_m": [],
        "foot_contact_valid_count": [],
        "base_sole_abs_delta_m": [],
        "hsi_sole_abs_delta_m": [],
        "base_sole_float_m": [],
        "hsi_sole_float_m": [],
        "base_sole_penetration_m": [],
        "hsi_sole_penetration_m": [],
        "sole_contact_valid_count": [],
        "base_sole_plane_abs_signed_m": [],
        "hsi_sole_plane_abs_signed_m": [],
        "base_sole_plane_float_m": [],
        "hsi_sole_plane_float_m": [],
        "base_sole_plane_penetration_m": [],
        "hsi_sole_plane_penetration_m": [],
        "sole_plane_contact_valid_count": [],
    }
    for pred_local, gt_local in matches:
        q = int(pred_idx[pred_local].item())
        g = int(gt_idx[gt_local].item())
        values = human_metric_values(
            base,
            hsi,
            gt,
            intrinsics[0, 0],
            hsi_depth[0, 0],
            gt_depth[0, 0],
            (int(batch["images"].shape[-2]), int(batch["images"].shape[-1])),
            args,
            q,
            g,
            sole_indices,
        )
        person_contact_points = values.pop("_contact_points", {"base": [], "hsi": []})
        for prefix in ("base", "hsi"):
            contact_points[prefix].extend(person_contact_points.get(prefix, []))
        for key, value in values.items():
            meters[key].append(value)
        per_person.append({"pred_query": q, "gt_index": g, **values})

    summary = {key: mean_or_none(values) for key, values in meters.items()}
    summary.update(
        {
            "num_gt": int(gt_idx.numel()),
            "num_pred_conf": int(pred_idx.numel()),
            "num_matched": int(len(matches)),
            "conf_threshold": float(args.conf_threshold),
            "per_person": per_person,
        }
    )
    return summary, contact_points


def human_metric_values(
    base: dict[str, torch.Tensor],
    hsi: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    intrinsics: torch.Tensor,
    hsi_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    image_size: int,
    args: argparse.Namespace,
    q: int,
    g: int,
    sole_indices: torch.Tensor,
) -> dict[str, float | None]:
    gt_j = gt["joints"][0, 0, g, :24]
    base_j = base["joints"][0, 0, q, :24]
    hsi_j = hsi["joints"][0, 0, q, :24]
    base_2d = project_points(base_j, intrinsics)
    hsi_2d = project_points(hsi_j, intrinsics)
    gt_2d = project_points(gt_j, intrinsics)
    valid_base = (base_j[:, 2] > 1e-4) & (gt_j[:, 2] > 1e-4)
    valid_hsi = (hsi_j[:, 2] > 1e-4) & (gt_j[:, 2] > 1e-4)
    values = {
        "base_joints_mpjpe_m": scalar(torch.linalg.norm(base_j - gt_j, dim=-1).mean()),
        "hsi_joints_mpjpe_m": scalar(torch.linalg.norm(hsi_j - gt_j, dim=-1).mean()),
        "base_vertices_pve_m": scalar(torch.linalg.norm(base["vertices"][0, 0, q] - gt["vertices"][0, 0, g], dim=-1).mean()),
        "hsi_vertices_pve_m": scalar(torch.linalg.norm(hsi["vertices"][0, 0, q] - gt["vertices"][0, 0, g], dim=-1).mean()),
        "base_transl_l2_m": scalar(torch.linalg.norm(base["transl"][0, 0, q] - gt["transl"][0, 0, g])),
        "hsi_transl_l2_m": scalar(torch.linalg.norm(hsi["transl"][0, 0, q] - gt["transl"][0, 0, g])),
        "base_projected_joints_l2_px": scalar(torch.linalg.norm(base_2d[valid_base] - gt_2d[valid_base], dim=-1).mean()) if valid_base.any() else None,
        "hsi_projected_joints_l2_px": scalar(torch.linalg.norm(hsi_2d[valid_hsi] - gt_2d[valid_hsi], dim=-1).mean()) if valid_hsi.any() else None,
    }
    values.update(foot_contact_values(base_j, hsi_j, gt_j, intrinsics, hsi_depth, gt_depth, image_size, args))
    values.update(
        sole_contact_values(
            base["vertices"][0, 0, q],
            hsi["vertices"][0, 0, q],
            gt["vertices"][0, 0, g],
            sole_indices,
            intrinsics,
            hsi_depth,
            gt_depth,
            image_size,
            args,
        )
    )
    return values


def foot_contact_values(
    base_joints: torch.Tensor,
    hsi_joints: torch.Tensor,
    gt_joints: torch.Tensor,
    intrinsics: torch.Tensor,
    hsi_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    image_size: int,
    args: argparse.Namespace,
) -> dict[str, float | None]:
    foot_idx = torch.tensor([7, 8, 10, 11], dtype=torch.long, device=hsi_joints.device)
    gt_foot = gt_joints[foot_idx]
    gt_projected = scale_points_to_depth(project_points(gt_foot, intrinsics), image_size, gt_depth.shape[-2], gt_depth.shape[-1])
    sampled_gt, gt_valid = sample_depth_at_points(gt_depth, gt_projected)
    contact = (torch.abs(sampled_gt - gt_foot[:, 2].to(dtype=sampled_gt.dtype)) < float(args.foot_contact_threshold_m)) & gt_valid
    values: dict[str, float | None] = {
        "base_foot_abs_delta_m": None,
        "hsi_foot_abs_delta_m": None,
        "base_foot_float_m": None,
        "hsi_foot_float_m": None,
        "base_foot_penetration_m": None,
        "hsi_foot_penetration_m": None,
        "foot_contact_valid_count": float(contact.sum().detach().cpu()),
    }
    if not contact.any():
        return values
    for prefix, joints in [("base", base_joints), ("hsi", hsi_joints)]:
        foot = joints[foot_idx]
        projected = scale_points_to_depth(project_points(foot, intrinsics), image_size, hsi_depth.shape[-2], hsi_depth.shape[-1])
        sampled, valid = sample_depth_at_points(hsi_depth, projected)
        use = contact & valid & torch.isfinite(sampled) & torch.isfinite(foot[:, 2])
        if not use.any():
            continue
        depth_delta = sampled - foot[:, 2].to(dtype=sampled.dtype)
        float_amt = torch.relu(depth_delta - float(args.foot_float_margin_m))
        penetration_amt = torch.relu(-depth_delta - float(args.foot_penetration_margin_m))
        values[f"{prefix}_foot_abs_delta_m"] = scalar(torch.abs(depth_delta[use]).mean())
        values[f"{prefix}_foot_float_m"] = scalar(float_amt[use].mean())
        values[f"{prefix}_foot_penetration_m"] = scalar(penetration_amt[use].mean())
    return values


def sole_contact_values(
    base_vertices: torch.Tensor,
    hsi_vertices: torch.Tensor,
    gt_vertices: torch.Tensor,
    sole_indices: torch.Tensor,
    intrinsics: torch.Tensor,
    hsi_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    image_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    gt_sole = gt_vertices[sole_indices]
    gt_projected = scale_points_to_depth(project_points(gt_sole, intrinsics), image_size, gt_depth.shape[-2], gt_depth.shape[-1])
    sampled_gt, gt_valid = sample_depth_at_points(gt_depth, gt_projected)
    contact = (torch.abs(sampled_gt - gt_sole[:, 2].to(dtype=sampled_gt.dtype)) < float(args.foot_sole_contact_threshold_m)) & gt_valid
    values: dict[str, Any] = {
        "base_sole_abs_delta_m": None,
        "hsi_sole_abs_delta_m": None,
        "base_sole_float_m": None,
        "hsi_sole_float_m": None,
        "base_sole_penetration_m": None,
        "hsi_sole_penetration_m": None,
        "sole_contact_valid_count": float(contact.sum().detach().cpu()),
        "base_sole_plane_abs_signed_m": None,
        "hsi_sole_plane_abs_signed_m": None,
        "base_sole_plane_float_m": None,
        "hsi_sole_plane_float_m": None,
        "base_sole_plane_penetration_m": None,
        "hsi_sole_plane_penetration_m": None,
        "sole_plane_contact_valid_count": float(contact.sum().detach().cpu()),
        "_contact_points": {"base": [], "hsi": []},
    }
    if not contact.any():
        return values

    for prefix, vertices in [("base", base_vertices), ("hsi", hsi_vertices)]:
        sole = vertices[sole_indices]
        projected = scale_points_to_depth(project_points(sole, intrinsics), image_size, hsi_depth.shape[-2], hsi_depth.shape[-1])
        sampled, valid = sample_depth_at_points(hsi_depth, projected)
        use = contact & valid & torch.isfinite(sampled) & torch.isfinite(sole[:, 2])
        depth_delta = sampled - sole[:, 2].to(dtype=sampled.dtype)
        if use.any():
            float_amt = torch.relu(depth_delta - float(args.foot_float_margin_m))
            penetration_amt = torch.relu(-depth_delta - float(args.foot_penetration_margin_m))
            values[f"{prefix}_sole_abs_delta_m"] = scalar(torch.abs(depth_delta[use]).mean())
            values[f"{prefix}_sole_float_m"] = scalar(float_amt[use].mean())
            values[f"{prefix}_sole_penetration_m"] = scalar(penetration_amt[use].mean())

        signed, plane_valid = sample_local_support_plane_signed_delta(
            hsi_depth,
            projected,
            sole,
            intrinsics,
            image_size=image_size,
            window_size=int(args.support_plane_window),
            min_points=int(args.support_plane_min_points),
        )
        plane_use = contact & plane_valid & torch.isfinite(signed)
        if plane_use.any():
            plane_float = torch.relu(signed - float(args.foot_float_margin_m))
            plane_pen = torch.relu(-signed - float(args.foot_penetration_margin_m))
            values[f"{prefix}_sole_plane_abs_signed_m"] = scalar(torch.abs(signed[plane_use]).mean())
            values[f"{prefix}_sole_plane_float_m"] = scalar(plane_float[plane_use].mean())
            values[f"{prefix}_sole_plane_penetration_m"] = scalar(plane_pen[plane_use].mean())

        draw_mask = contact & (plane_valid | use)
        draw_indices = torch.nonzero(draw_mask, as_tuple=False).flatten()
        if draw_indices.numel() > 80:
            draw_indices = draw_indices[:: max(int(draw_indices.numel() // 80), 1)][:80]
        for idx in draw_indices:
            idx_int = int(idx.item())
            if bool(plane_valid[idx_int].detach().cpu()):
                delta = float(signed[idx_int].detach().cpu())
            elif bool(use[idx_int].detach().cpu()):
                delta = float(depth_delta[idx_int].detach().cpu())
            else:
                continue
            status = contact_status(delta, float(args.foot_float_margin_m), float(args.foot_penetration_margin_m))
            values["_contact_points"][prefix].append(
                {
                    "x": float(projected[idx_int, 0].detach().cpu()) * float(image_size) / float(hsi_depth.shape[-1]),
                    "y": float(projected[idx_int, 1].detach().cpu()) * float(image_size) / float(hsi_depth.shape[-2]),
                    "delta_m": delta,
                    "status": status,
                }
            )
    return values


def contact_status(delta: float, float_margin: float, penetration_margin: float) -> str:
    if delta > float_margin:
        return "floating"
    if delta < -penetration_margin:
        return "penetration"
    return "contact"


def save_contact_overlays(
    output_dir: Path,
    stem: str,
    rgb: Image.Image,
    contact_points: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    colors = {
        "penetration": (255, 48, 48),
        "floating": (48, 128, 255),
        "contact": (64, 220, 96),
    }
    radius = 3
    for prefix in ("base", "hsi"):
        overlay = rgb.copy()
        draw = ImageDraw.Draw(overlay)
        points = contact_points.get(prefix, [])
        for point in points:
            status = str(point.get("status", "contact"))
            color = colors.get(status, (240, 240, 240))
            x = float(point.get("x", 0.0))
            y = float(point.get("y", 0.0))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(20, 20, 20))
        draw.rectangle((6, 6, 304, 62), fill=(0, 0, 0))
        draw.text((14, 14), f"{prefix.upper()} sole contact | red pen blue float green contact", fill=(245, 245, 245))
        draw.text((14, 36), f"points={len(points)} window={args.support_plane_window}", fill=(245, 245, 245))
        path = output_dir / f"{stem}_{prefix}_sole_contact_overlay.png"
        overlay.save(path)
        paths[f"{prefix}_sole_contact_overlay"] = str(path)
    return paths


def save_depth_visuals(output_dir: Path, stem: str, rgb: Image.Image, images: dict[str, np.ndarray], summary: dict[str, Any]) -> dict[str, str]:
    valid = images["valid"].astype(bool)
    depth_stack = np.stack([images["gt_depth"], images["raw_depth"], images["hsi_depth"]])
    depth_valid = np.isfinite(depth_stack) & valid[None]
    depth_range = robust_range(depth_stack[depth_valid])
    error_stack = np.stack([images["raw_abs_error"], images["hsi_abs_error"]])
    error_valid = np.isfinite(error_stack) & valid[None]
    error_range = (0.0, max(float(np.percentile(error_stack[error_valid], 95)) if error_valid.any() else 1.0, 1e-6))

    panels = [
        ("RGB", rgb),
        ("GT depth", colorize(images["gt_depth"], depth_range, valid)),
        ("VGGT raw depth", colorize(images["raw_depth"], depth_range, valid)),
        ("HSI aligned depth", colorize(images["hsi_depth"], depth_range, valid)),
        (f"raw error mean={fmt(summary['raw_depth_l1_mean_m'])}m", colorize(images["raw_abs_error"], error_range, valid, heat=True)),
        (f"HSI error mean={fmt(summary['hsi_depth_l1_mean_m'])}m", colorize(images["hsi_abs_error"], error_range, valid, heat=True)),
    ]
    board = make_board(panels, panel_size=256)
    board_path = output_dir / f"{stem}_depth_gt_raw_hsi_compare.png"
    board.save(board_path)
    paths = {"depth_compare": str(board_path)}
    for name in ("gt_depth", "raw_depth", "hsi_depth", "raw_abs_error", "hsi_abs_error"):
        image = colorize(images[name], error_range if "error" in name else depth_range, valid, heat="error" in name)
        path = output_dir / f"{stem}_{name}.png"
        image.save(path)
        paths[name] = str(path)
    return paths


def colorize(values: np.ndarray, value_range: tuple[float, float], valid: np.ndarray, heat: bool = False) -> Image.Image:
    lo, hi = value_range
    norm = (values.astype(np.float32) - lo) / max(hi - lo, 1e-6)
    norm = np.clip(norm, 0.0, 1.0)
    if heat:
        rgb = np.stack([255 * norm, 255 * np.sqrt(norm), 40 * (1.0 - norm)], axis=-1)
    else:
        rgb = np.stack([255 * norm, 255 * (1.0 - np.abs(norm - 0.5) * 2.0), 255 * (1.0 - norm)], axis=-1)
    rgb[~valid] = np.array([0, 0, 0], dtype=np.float32)
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def make_board(panels: list[tuple[str, Image.Image]], panel_size: int = 256) -> Image.Image:
    label_h = 28
    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    board = Image.new("RGB", (cols * panel_size, rows * (panel_size + label_h)), (20, 20, 20))
    draw = ImageDraw.Draw(board)
    for idx, (label, image) in enumerate(panels):
        x = (idx % cols) * panel_size
        y = (idx // cols) * (panel_size + label_h)
        image = image.resize((panel_size, panel_size), Image.BILINEAR)
        board.paste(image, (x, y + label_h))
        draw.text((x + 6, y + 7), label, fill=(240, 240, 240))
    return board


def robust_range(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return (0.0, 1.0)
    lo = float(np.percentile(values, 2))
    hi = float(np.percentile(values, 98))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def infer_source(depth: dict[str, Any], smpl: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    hsi_mpjpe = smpl.get("hsi_joints_mpjpe_m")
    hsi_depth = depth.get("hsi_depth_l1_median_m")
    if hsi_mpjpe is None or hsi_depth is None:
        source = "insufficient_metrics"
    elif hsi_mpjpe <= args.smpl_good_threshold_m and hsi_depth >= args.depth_bad_threshold_m:
        source = "depth_or_scene_alignment_likely"
    elif hsi_mpjpe >= args.smpl_good_threshold_m and hsi_depth <= args.depth_bad_threshold_m:
        source = "smpl_alignment_likely"
    elif hsi_mpjpe <= args.smpl_good_threshold_m and hsi_depth <= args.depth_bad_threshold_m:
        source = "both_close_check_visualization_projection_or_export"
    else:
        source = "both_smpl_and_depth_have_residual_error"
    return {
        "likely_source": source,
        "smpl_good_threshold_m": float(args.smpl_good_threshold_m),
        "depth_bad_threshold_m": float(args.depth_bad_threshold_m),
    }


def improvement(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or abs(before) < 1e-12:
        return None
    return (before - after) / before * 100.0


def mean_or_none(values: list[float]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def scalar(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return float(value.detach().cpu())


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def print_human_summary(depth: dict[str, Any], smpl: dict[str, Any], diagnosis: dict[str, Any]) -> None:
    print("========== HSI depth / SMPL diagnostics ==========")
    print(f"SMPL HSI MPJPE      : {fmt(smpl.get('hsi_joints_mpjpe_m'))} m")
    print(f"SMPL HSI PVE        : {fmt(smpl.get('hsi_vertices_pve_m'))} m")
    print(f"SMPL HSI transl L2  : {fmt(smpl.get('hsi_transl_l2_m'))} m")
    print(
        "Foot float / pen    : "
        f"base={fmt(smpl.get('base_foot_float_m'))}/{fmt(smpl.get('base_foot_penetration_m'))} m "
        f"hsi={fmt(smpl.get('hsi_foot_float_m'))}/{fmt(smpl.get('hsi_foot_penetration_m'))} m"
    )
    print(
        "Sole float / pen    : "
        f"base={fmt(smpl.get('base_sole_float_m'))}/{fmt(smpl.get('base_sole_penetration_m'))} m "
        f"hsi={fmt(smpl.get('hsi_sole_float_m'))}/{fmt(smpl.get('hsi_sole_penetration_m'))} m"
    )
    print(
        "Sole plane float/pen: "
        f"base={fmt(smpl.get('base_sole_plane_float_m'))}/{fmt(smpl.get('base_sole_plane_penetration_m'))} m "
        f"hsi={fmt(smpl.get('hsi_sole_plane_float_m'))}/{fmt(smpl.get('hsi_sole_plane_penetration_m'))} m"
    )
    print(f"Raw depth median L1 : {fmt(depth.get('raw_depth_l1_median_m'))} m")
    print(f"HSI depth median L1 : {fmt(depth.get('hsi_depth_l1_median_m'))} m")
    regions = depth.get("regions", {})
    for label, key in [("Near <30m", "near_lt_30m"), ("Far >=30m", "far_ge_30m"), ("Human ROI", "human_roi")]:
        region = regions.get(key, {})
        print(
            f"{label:18s}: raw={fmt(region.get('raw_depth_l1_median_m'))}m "
            f"hsi={fmt(region.get('hsi_depth_l1_median_m'))}m "
            f"pixels={region.get('valid_pixels', 0)}"
        )
    print(f"HSI scale / bias    : {fmt(depth.get('hsi_scene_scale'))} / {fmt(depth.get('hsi_scene_depth_bias'))}")
    print(f"Likely source       : {diagnosis['likely_source']}")


if __name__ == "__main__":
    main()
