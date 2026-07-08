#!/usr/bin/env python
"""Evaluate base SMPL vs HSI-refined outputs against BEDLAM GT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

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

from scripts.train.train_smpl import apply_overrides, build_model, load_yaml_config
from vggt_omega.data import BedlamDataset, bedlam_collate_fn
from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.training.config import deep_update, require_path
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.utils.rotation import rot6d_to_axis_angle


class Meter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def add(self, value: float, count: int = 1) -> None:
        if np.isfinite(value) and count > 0:
            self.total += float(value) * int(count)
            self.count += int(count)

    @property
    def mean(self) -> float | None:
        return self.total / self.count if self.count else None


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args)
    model = build_model(config).to(device)
    load_vggt_baseline(model, config, device)
    load_training_checkpoint(model, Path(args.checkpoint), device)
    model.eval()

    loader = build_eval_loader(config, args)
    smpl = SMPLLayer(require_path(config, "assets.smpl_model_dir", allow_empty=False)).to(device).eval()
    metrics = init_metrics()
    hsi_scales = []
    hsi_biases = []
    examples = []

    processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = move_to_device(batch, device)
            predictions = model(
                batch["images"],
                smpl_query_boxes=batch["gt_boxes"] if args.use_gt_box_prior else None,
                smpl_query_boxes_mask=batch["boxes_mask"] if args.use_gt_box_prior else None,
                smpl_track_ids=batch.get("gt_track_ids", batch.get("person_ids")),
                smpl_track_mask=batch.get("gt_track_mask", batch.get("person_id_mask")),
            )
            evaluate_batch(predictions, batch, smpl, config, args, metrics, hsi_scales, hsi_biases)
            if len(examples) < 8:
                examples.append(example_summary(batch_idx, predictions, batch))
            processed += int(batch["images"].shape[0])
            if processed >= args.max_samples:
                break
            if args.log_interval > 0 and processed % args.log_interval == 0:
                print(f"[eval] processed={processed}")

    summary = build_summary(metrics, hsi_scales, hsi_biases, processed, args, examples)
    out_json = output_dir / "hsi_refine_metrics.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_human_summary(summary)
    print(json.dumps({"output_json": str(out_json), "num_samples": processed}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base vs HSI-refined SMPL predictions")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_hsi_refine.yaml")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--output-dir", default="outputs/eval/hsi_refine_metrics")
    parser.add_argument("--device", default="")
    parser.add_argument("--split", default="Training")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--conf-threshold", type=float, default=0.10)
    parser.add_argument("--use-gt-box-prior", action="store_true")
    parser.add_argument("--foot-contact-threshold-m", type=float, default=0.12)
    parser.add_argument("--foot-float-margin-m", type=float, default=0.04)
    parser.add_argument("--foot-penetration-margin-m", type=float, default=0.02)
    parser.add_argument("--foot-sole-num-vertices", type=int, default=80)
    parser.add_argument("--foot-sole-contact-threshold-m", type=float, default=0.08)
    parser.add_argument("--support-plane-window", type=int, default=9)
    parser.add_argument("--support-plane-min-points", type=int, default=6)
    parser.add_argument("--log-interval", type=int, default=8)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.train_config))
    config = apply_overrides(config, args.override)
    if args.baseline_checkpoint:
        config.setdefault("checkpoints", {})["vggt_baseline"] = args.baseline_checkpoint
    config.setdefault("model", {})["enable_camera"] = True
    config.setdefault("model", {})["enable_depth"] = True
    config.setdefault("model", {})["enable_hsi_refine"] = True
    config.setdefault("data", {})["require_depth"] = True
    config.setdefault("data", {})["require_boxes"] = True
    return config


def build_eval_loader(config: dict[str, Any], args: argparse.Namespace) -> DataLoader:
    data_cfg = config["data"]
    dataset = BedlamDataset(
        root=require_path(config, data_cfg.get("root_key", "datasets.bedlam_root")),
        split=args.split,
        sequence_length=int(data_cfg["sequence_length"]),
        stride=int(data_cfg["stride"]),
        image_size=int(data_cfg.get("image_size", data_cfg.get("image_resolution", 512))),
        image_resolution=int(data_cfg.get("image_resolution", data_cfg.get("image_size", 512))),
        resize_mode=str(data_cfg.get("resize_mode", "balanced")),
        max_humans=int(data_cfg["max_humans"]),
        require_smpl=True,
        require_depth=True,
        boxes_root=require_path(config, data_cfg["boxes_root_key"], allow_empty=False),
        require_boxes=True,
    )
    end = min(len(dataset), int(args.start_index) + int(args.max_samples))
    subset = Subset(dataset, list(range(int(args.start_index), end)))
    return DataLoader(
        subset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        collate_fn=bedlam_collate_fn,
        drop_last=False,
    )


def load_vggt_baseline(model: torch.nn.Module, config: dict[str, Any], device: torch.device) -> None:
    checkpoint_path = require_path(config, "checkpoints.vggt_baseline", allow_empty=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[ckpt] loaded VGGT baseline: {checkpoint_path}")
    print(f"[ckpt] baseline missing={len(missing)} unexpected={len(unexpected)}")


def load_training_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[ckpt] loaded training checkpoint: {checkpoint_path}")
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return {name.removeprefix("module."): tensor for name, tensor in value.items()}
    if isinstance(checkpoint, dict) and all(torch.is_tensor(value) for value in checkpoint.values()):
        return {name.removeprefix("module."): tensor for name, tensor in checkpoint.items()}
    raise ValueError("Could not find a model state_dict in checkpoint")


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def init_metrics() -> dict[str, Meter]:
    names = [
        "base_joints_mpjpe_m",
        "hsi_joints_mpjpe_m",
        "base_vertices_pve_m",
        "hsi_vertices_pve_m",
        "base_transl_l2_m",
        "hsi_transl_l2_m",
        "base_projected_joints_l2_px",
        "hsi_projected_joints_l2_px",
        "hsi_worse_than_base_ratio_2cm",
        "hsi_joint_error_delta_m",
        "raw_depth_l1_mean_m",
        "hsi_depth_l1_mean_m",
        "raw_depth_l1_median_m",
        "hsi_depth_l1_median_m",
        "raw_depth_near_l1_mean_m",
        "hsi_depth_near_l1_mean_m",
        "raw_depth_near_l1_median_m",
        "hsi_depth_near_l1_median_m",
        "raw_depth_far_l1_mean_m",
        "hsi_depth_far_l1_mean_m",
        "raw_depth_far_l1_median_m",
        "hsi_depth_far_l1_median_m",
        "raw_depth_human_roi_l1_mean_m",
        "hsi_depth_human_roi_l1_mean_m",
        "raw_depth_human_roi_l1_median_m",
        "hsi_depth_human_roi_l1_median_m",
        "depth_valid_pixels",
        "depth_near_valid_pixels",
        "depth_far_valid_pixels",
        "depth_human_roi_valid_pixels",
        "base_foot_abs_delta_m",
        "hsi_foot_abs_delta_m",
        "base_foot_float_m",
        "hsi_foot_float_m",
        "base_foot_penetration_m",
        "hsi_foot_penetration_m",
        "foot_contact_valid_count",
        "base_sole_abs_delta_m",
        "hsi_sole_abs_delta_m",
        "base_sole_float_m",
        "hsi_sole_float_m",
        "base_sole_penetration_m",
        "hsi_sole_penetration_m",
        "sole_contact_valid_count",
        "base_sole_plane_abs_signed_m",
        "hsi_sole_plane_abs_signed_m",
        "base_sole_plane_float_m",
        "hsi_sole_plane_float_m",
        "base_sole_plane_penetration_m",
        "hsi_sole_plane_penetration_m",
        "sole_plane_contact_valid_count",
    ]
    return {name: Meter() for name in names} | {"num_gt": Meter(), "num_matched": Meter()}


def evaluate_batch(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    smpl: SMPLLayer,
    config: dict[str, Any],
    args: argparse.Namespace,
    metrics: dict[str, Meter],
    hsi_scales: list[float],
    hsi_biases: list[float],
) -> None:
    base = decode_smpl_batch(predictions["pred_poses"], predictions["pred_betas"], predictions["pred_transl_cam"], smpl)
    hsi = decode_smpl_batch(
        predictions["hsi_refined_pred_poses"],
        predictions["hsi_refined_pred_betas"],
        predictions["hsi_refined_pred_transl_cam"],
        smpl,
    )
    gt_poses = rot6d_to_axis_angle(batch["gt_pose_6d"].reshape(-1, 24, 6)).reshape(*batch["gt_pose_6d"].shape[:3], 72)
    gt = decode_smpl_batch(gt_poses, batch["gt_betas"], batch["gt_transl_cam"], smpl)

    image_hw = (int(batch["images"].shape[-2]), int(batch["images"].shape[-1]))
    intrinsics = encoding_to_camera(
        predictions["pose_enc"],
        image_size_hw=image_hw,
        build_intrinsics=True,
    )[1]
    confs = predictions["pred_confs"].detach()
    if confs.ndim == 4 and confs.shape[-1] == 1:
        confs = confs[..., 0]
    raw_depth, hsi_depth, gt_depth = depth_triplet(predictions, batch)
    mask = batch["smpl_mask"].bool()
    batch_size, num_frames, num_queries = mask.shape
    sole_indices = get_foot_sole_indices(smpl, int(args.foot_sole_num_vertices), device=gt["vertices"].device)
    for b in range(batch_size):
        for s in range(num_frames):
            gt_idx = torch.nonzero(mask[b, s], as_tuple=False).flatten()
            pred_idx = torch.nonzero(confs[b, s] >= float(args.conf_threshold), as_tuple=False).flatten()
            metrics["num_gt"].add(float(gt_idx.numel()))
            if gt_idx.numel() == 0 or pred_idx.numel() == 0:
                continue
            matches = greedy_match(base["joints"][b, s, pred_idx, :24], gt["joints"][b, s, gt_idx, :24])
            metrics["num_matched"].add(float(len(matches)))
            for pred_local, gt_local in matches:
                q = pred_idx[pred_local]
                g = gt_idx[gt_local]
                add_human_metrics(
                    metrics,
                    base,
                    hsi,
                    gt,
                    intrinsics[b, s],
                    hsi_depth[b, s],
                    gt_depth[b, s],
                    image_hw,
                    args,
                    b,
                    s,
                    q,
                    g,
                    sole_indices,
                )

    add_depth_metrics(metrics, predictions, batch)
    if "hsi_scene_scale" in predictions:
        hsi_scales.extend(predictions["hsi_scene_scale"].detach().float().cpu().reshape(-1).tolist())
    if "hsi_scene_depth_bias" in predictions:
        hsi_biases.extend(predictions["hsi_scene_depth_bias"].detach().float().cpu().reshape(-1).tolist())


def decode_smpl_batch(poses: torch.Tensor, betas: torch.Tensor, transl: torch.Tensor, smpl: SMPLLayer) -> dict[str, torch.Tensor]:
    shape = poses.shape[:3]
    vertices, joints = smpl(poses.reshape(-1, 72).float(), betas.reshape(-1, betas.shape[-1]).float())
    vertices = vertices.reshape(*shape, vertices.shape[-2], 3).to(dtype=transl.dtype) + transl[..., None, :]
    joints = joints.reshape(*shape, joints.shape[-2], 3).to(dtype=transl.dtype) + transl[..., None, :]
    return {"vertices": vertices, "joints": joints, "transl": transl}


def greedy_match(pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> list[tuple[int, int]]:
    cost = torch.linalg.norm(pred_joints[:, None] - gt_joints[None], dim=-1).mean(dim=-1).detach().cpu()
    matches = []
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    while len(used_pred) < cost.shape[0] and len(used_gt) < cost.shape[1]:
        best = None
        for p in range(cost.shape[0]):
            if p in used_pred:
                continue
            for g in range(cost.shape[1]):
                if g in used_gt:
                    continue
                value = float(cost[p, g])
                if best is None or value < best[0]:
                    best = (value, p, g)
        if best is None:
            break
        _, p, g = best
        matches.append((p, g))
        used_pred.add(p)
        used_gt.add(g)
    return matches


def add_human_metrics(
    metrics: dict[str, Meter],
    base: dict[str, torch.Tensor],
    hsi: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    intrinsics: torch.Tensor,
    hsi_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    image_size: int | tuple[int, int],
    args: argparse.Namespace,
    b: int,
    s: int,
    q: torch.Tensor,
    g: torch.Tensor,
    sole_indices: torch.Tensor,
) -> None:
    q_int = int(q.item())
    g_int = int(g.item())
    gt_j = gt["joints"][b, s, g_int, :24]
    base_j = base["joints"][b, s, q_int, :24]
    hsi_j = hsi["joints"][b, s, q_int, :24]
    base_joint_error = torch.linalg.norm(base_j - gt_j, dim=-1).mean()
    hsi_joint_error = torch.linalg.norm(hsi_j - gt_j, dim=-1).mean()
    metrics["base_joints_mpjpe_m"].add(float(base_joint_error.detach().cpu()))
    metrics["hsi_joints_mpjpe_m"].add(float(hsi_joint_error.detach().cpu()))
    metrics["hsi_worse_than_base_ratio_2cm"].add(float((hsi_joint_error > base_joint_error + 0.02).detach().cpu()))
    metrics["hsi_joint_error_delta_m"].add(float((hsi_joint_error - base_joint_error).detach().cpu()))
    metrics["base_vertices_pve_m"].add(float(torch.linalg.norm(base["vertices"][b, s, q_int] - gt["vertices"][b, s, g_int], dim=-1).mean().detach().cpu()))
    metrics["hsi_vertices_pve_m"].add(float(torch.linalg.norm(hsi["vertices"][b, s, q_int] - gt["vertices"][b, s, g_int], dim=-1).mean().detach().cpu()))
    metrics["base_transl_l2_m"].add(float(torch.linalg.norm(base["transl"][b, s, q_int] - gt["transl"][b, s, g_int]).detach().cpu()))
    metrics["hsi_transl_l2_m"].add(float(torch.linalg.norm(hsi["transl"][b, s, q_int] - gt["transl"][b, s, g_int]).detach().cpu()))

    base_2d = project_points(base_j, intrinsics)
    hsi_2d = project_points(hsi_j, intrinsics)
    gt_2d = project_points(gt_j, intrinsics)
    valid_base = (base_j[:, 2] > 1e-4) & (gt_j[:, 2] > 1e-4)
    valid_hsi = (hsi_j[:, 2] > 1e-4) & (gt_j[:, 2] > 1e-4)
    if valid_base.any():
        metrics["base_projected_joints_l2_px"].add(float(torch.linalg.norm(base_2d[valid_base] - gt_2d[valid_base], dim=-1).mean().detach().cpu()))
    if valid_hsi.any():
        metrics["hsi_projected_joints_l2_px"].add(float(torch.linalg.norm(hsi_2d[valid_hsi] - gt_2d[valid_hsi], dim=-1).mean().detach().cpu()))
    add_foot_contact_metrics(metrics, base_j, hsi_j, gt_j, intrinsics, hsi_depth, gt_depth, image_size, args)
    add_foot_sole_contact_metrics(
        metrics,
        base["vertices"][b, s, q_int],
        hsi["vertices"][b, s, q_int],
        gt["vertices"][b, s, g_int],
        sole_indices,
        intrinsics,
        hsi_depth,
        gt_depth,
        image_size,
        args,
    )


def add_depth_metrics(metrics: dict[str, Meter], predictions: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
    depth, hsi_depth, gt_depth = depth_triplet(predictions, batch)
    valid = torch.isfinite(gt_depth) & (gt_depth > 1e-6) & torch.isfinite(depth) & torch.isfinite(hsi_depth)
    near_valid = valid & (gt_depth <= 30.0)
    far_valid = valid & (gt_depth > 30.0)
    roi_mask = human_roi_depth_mask(
        batch["gt_boxes"].to(device=depth.device, dtype=depth.dtype),
        batch["boxes_mask"].to(device=depth.device).bool(),
        depth.shape[-2],
        depth.shape[-1],
        expand=0.75,
    )
    roi_valid = valid & roi_mask
    for b in range(depth.shape[0]):
        for s in range(depth.shape[1]):
            add_depth_region_metrics(metrics, "depth", depth[b, s], hsi_depth[b, s], gt_depth[b, s], valid[b, s])
            add_depth_region_metrics(metrics, "depth_near", depth[b, s], hsi_depth[b, s], gt_depth[b, s], near_valid[b, s])
            add_depth_region_metrics(metrics, "depth_far", depth[b, s], hsi_depth[b, s], gt_depth[b, s], far_valid[b, s])
            add_depth_region_metrics(metrics, "depth_human_roi", depth[b, s], hsi_depth[b, s], gt_depth[b, s], roi_valid[b, s])


def depth_triplet(predictions: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    depth = canonical_depth(predictions["depth"]).float()
    gt_depth = canonical_depth(batch["gt_depth"]).to(device=depth.device, dtype=depth.dtype)
    if gt_depth.shape[-2:] != depth.shape[-2:]:
        gt_depth = F.interpolate(
            gt_depth.reshape(-1, 1, *gt_depth.shape[-2:]),
            size=depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape(*gt_depth.shape[:2], *depth.shape[-2:])
    hsi_depth = depth
    if "hsi_scene_scale" in predictions and "hsi_scene_depth_bias" in predictions:
        scale = predictions["hsi_scene_scale"].to(device=depth.device, dtype=depth.dtype).reshape(*depth.shape[:2], 1, 1)
        bias = predictions["hsi_scene_depth_bias"].to(device=depth.device, dtype=depth.dtype).reshape(*depth.shape[:2], 1, 1)
        hsi_depth = depth * scale + bias
    return depth, hsi_depth, gt_depth


def add_foot_contact_metrics(
    metrics: dict[str, Meter],
    base_joints: torch.Tensor,
    hsi_joints: torch.Tensor,
    gt_joints: torch.Tensor,
    intrinsics: torch.Tensor,
    hsi_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    image_size: int | tuple[int, int],
    args: argparse.Namespace,
) -> None:
    foot_idx = torch.tensor([7, 8, 10, 11], dtype=torch.long, device=hsi_joints.device)
    gt_foot = gt_joints[foot_idx]
    gt_projected = scale_points_to_depth(project_points(gt_foot, intrinsics), image_size, gt_depth.shape[-2], gt_depth.shape[-1])
    sampled_gt, gt_valid = sample_depth_at_points(gt_depth, gt_projected)
    contact = (torch.abs(sampled_gt - gt_foot[:, 2].to(dtype=sampled_gt.dtype)) < float(args.foot_contact_threshold_m)) & gt_valid
    if not contact.any():
        return

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
        metrics[f"{prefix}_foot_abs_delta_m"].add(float(torch.abs(depth_delta[use]).mean().detach().cpu()), int(use.sum().detach().cpu()))
        metrics[f"{prefix}_foot_float_m"].add(float(float_amt[use].mean().detach().cpu()), int(use.sum().detach().cpu()))
        metrics[f"{prefix}_foot_penetration_m"].add(float(penetration_amt[use].mean().detach().cpu()), int(use.sum().detach().cpu()))
    metrics["foot_contact_valid_count"].add(float(contact.sum().detach().cpu()))


def add_foot_sole_contact_metrics(
    metrics: dict[str, Meter],
    base_vertices: torch.Tensor,
    hsi_vertices: torch.Tensor,
    gt_vertices: torch.Tensor,
    sole_indices: torch.Tensor,
    intrinsics: torch.Tensor,
    hsi_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    image_size: int | tuple[int, int],
    args: argparse.Namespace,
) -> None:
    gt_sole = gt_vertices[sole_indices]
    gt_projected = scale_points_to_depth(project_points(gt_sole, intrinsics), image_size, gt_depth.shape[-2], gt_depth.shape[-1])
    sampled_gt, gt_valid = sample_depth_at_points(gt_depth, gt_projected)
    contact = (torch.abs(sampled_gt - gt_sole[:, 2].to(dtype=sampled_gt.dtype)) < float(args.foot_sole_contact_threshold_m)) & gt_valid
    if not contact.any():
        return

    for prefix, vertices in [("base", base_vertices), ("hsi", hsi_vertices)]:
        sole = vertices[sole_indices]
        projected = scale_points_to_depth(project_points(sole, intrinsics), image_size, hsi_depth.shape[-2], hsi_depth.shape[-1])
        sampled, valid = sample_depth_at_points(hsi_depth, projected)
        use = contact & valid & torch.isfinite(sampled) & torch.isfinite(sole[:, 2])
        if use.any():
            depth_delta = sampled - sole[:, 2].to(dtype=sampled.dtype)
            float_amt = torch.relu(depth_delta - float(args.foot_float_margin_m))
            penetration_amt = torch.relu(-depth_delta - float(args.foot_penetration_margin_m))
            metrics[f"{prefix}_sole_abs_delta_m"].add(float(torch.abs(depth_delta[use]).mean().detach().cpu()), int(use.sum().detach().cpu()))
            metrics[f"{prefix}_sole_float_m"].add(float(float_amt[use].mean().detach().cpu()), int(use.sum().detach().cpu()))
            metrics[f"{prefix}_sole_penetration_m"].add(float(penetration_amt[use].mean().detach().cpu()), int(use.sum().detach().cpu()))

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
            metrics[f"{prefix}_sole_plane_abs_signed_m"].add(float(torch.abs(signed[plane_use]).mean().detach().cpu()), int(plane_use.sum().detach().cpu()))
            metrics[f"{prefix}_sole_plane_float_m"].add(float(plane_float[plane_use].mean().detach().cpu()), int(plane_use.sum().detach().cpu()))
            metrics[f"{prefix}_sole_plane_penetration_m"].add(float(plane_pen[plane_use].mean().detach().cpu()), int(plane_use.sum().detach().cpu()))

    metrics["sole_contact_valid_count"].add(float(contact.sum().detach().cpu()))
    metrics["sole_plane_contact_valid_count"].add(float(contact.sum().detach().cpu()))


def scale_points_to_depth(points: torch.Tensor, image_size: int | tuple[int, int], depth_height: int, depth_width: int) -> torch.Tensor:
    if isinstance(image_size, int):
        image_h, image_w = int(image_size), int(image_size)
    else:
        image_h, image_w = int(image_size[0]), int(image_size[1])
    scale = points.new_tensor([depth_width / float(image_w), depth_height / float(image_h)])
    return points * scale


def sample_depth_at_points(depth: torch.Tensor, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = depth.shape[-2:]
    x = points[:, 0].round().long()
    y = points[:, 1].round().long()
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    sampled = depth.new_zeros(points.shape[0])
    if valid.any():
        sampled[valid] = depth[y[valid], x[valid]]
    valid = valid & torch.isfinite(sampled) & (sampled > 1e-6)
    return sampled, valid


def get_foot_sole_indices(smpl: SMPLLayer, count: int, device: torch.device) -> torch.Tensor:
    template = smpl.layer.v_template.detach().float().reshape(-1, 3)
    count = min(max(int(count), 1), int(template.shape[0]))
    return torch.argsort(template[:, 1])[:count].long().to(device=device)


def sample_local_support_plane_signed_delta(
    depth: torch.Tensor,
    points_2d: torch.Tensor,
    points_cam: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: int | tuple[int, int],
    window_size: int = 9,
    min_points: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = depth.shape[-2:]
    if isinstance(image_size, int):
        image_h, image_w = int(image_size), int(image_size)
    else:
        image_h, image_w = int(image_size[0]), int(image_size[1])
    window_size = max(int(window_size), 1)
    if window_size % 2 == 0:
        window_size += 1
    radius = window_size // 2
    min_points = max(int(min_points), 3)

    center_x = points_2d[:, 0].round().long()
    center_y = points_2d[:, 1].round().long()
    point_valid = (
        torch.isfinite(points_2d).all(dim=-1)
        & torch.isfinite(points_cam).all(dim=-1)
        & (points_cam[:, 2] > 1e-6)
        & (center_x >= 0)
        & (center_x < width)
        & (center_y >= 0)
        & (center_y < height)
    )

    offsets = torch.arange(-radius, radius + 1, device=points_2d.device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    ox = ox.reshape(1, -1)
    oy = oy.reshape(1, -1)
    xs = center_x[:, None] + ox
    ys = center_y[:, None] + oy
    local_valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs = xs.clamp(0, width - 1)
    ys = ys.clamp(0, height - 1)

    sampled_depth = depth[ys, xs]
    local_valid = local_valid & torch.isfinite(sampled_depth) & (sampled_depth > 1e-6)
    pixel_x = xs.to(dtype=sampled_depth.dtype) * (float(image_w) / float(width))
    pixel_y = ys.to(dtype=sampled_depth.dtype) * (float(image_h) / float(height))
    fx = intrinsics[0, 0].to(dtype=sampled_depth.dtype).clamp(min=1e-6)
    fy = intrinsics[1, 1].to(dtype=sampled_depth.dtype).clamp(min=1e-6)
    cx = intrinsics[0, 2].to(dtype=sampled_depth.dtype)
    cy = intrinsics[1, 2].to(dtype=sampled_depth.dtype)
    scene_x = (pixel_x - cx) * sampled_depth / fx
    scene_y = (pixel_y - cy) * sampled_depth / fy
    scene_xyz = torch.stack([scene_x, scene_y, sampled_depth], dim=-1)

    weights = local_valid.to(dtype=scene_xyz.dtype)
    valid_count = local_valid.sum(dim=-1)
    denom = weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
    center = (scene_xyz * weights[..., None]).sum(dim=-2) / denom
    centered = (scene_xyz - center[:, None, :]) * weights[..., None]
    cov = torch.matmul(centered.transpose(-1, -2), centered) / denom[:, :, None].clamp(min=1.0)
    cov = cov + torch.eye(3, dtype=cov.dtype, device=cov.device).reshape(1, 3, 3) * 1e-6
    _, evecs = torch.linalg.eigh(cov.float())
    normal = evecs[..., 0].to(dtype=points_cam.dtype)
    normal = normal / torch.linalg.norm(normal, dim=-1, keepdim=True).clamp(min=1e-6)
    normal = torch.where(normal[:, 2:3] > 0, -normal, normal)
    signed = ((points_cam - center.to(dtype=points_cam.dtype)) * normal).sum(dim=-1)
    valid = point_valid & (valid_count >= min_points) & torch.isfinite(signed)
    signed = torch.where(valid, signed, torch.zeros_like(signed))
    return signed, valid


def add_depth_region_metrics(
    metrics: dict[str, Meter],
    prefix: str,
    depth: torch.Tensor,
    hsi_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    metrics[f"{prefix}_valid_pixels"].add(float(valid.sum().detach().cpu()))
    if not valid.any():
        return
    raw_abs = torch.abs(depth[valid] - gt_depth[valid])
    hsi_abs = torch.abs(hsi_depth[valid] - gt_depth[valid])
    raw_mean_key, hsi_mean_key, raw_median_key, hsi_median_key = depth_metric_keys(prefix)
    metrics[raw_mean_key].add(float(raw_abs.mean().detach().cpu()), int(raw_abs.numel()))
    metrics[hsi_mean_key].add(float(hsi_abs.mean().detach().cpu()), int(hsi_abs.numel()))
    metrics[raw_median_key].add(float(raw_abs.median().detach().cpu()))
    metrics[hsi_median_key].add(float(hsi_abs.median().detach().cpu()))


def depth_metric_keys(prefix: str) -> tuple[str, str, str, str]:
    if prefix == "depth":
        return "raw_depth_l1_mean_m", "hsi_depth_l1_mean_m", "raw_depth_l1_median_m", "hsi_depth_l1_median_m"
    suffix = prefix.removeprefix("depth_")
    return (
        f"raw_depth_{suffix}_l1_mean_m",
        f"hsi_depth_{suffix}_l1_mean_m",
        f"raw_depth_{suffix}_l1_median_m",
        f"hsi_depth_{suffix}_l1_median_m",
    )


def human_roi_depth_mask(
    boxes: torch.Tensor,
    boxes_mask: torch.Tensor,
    depth_height: int,
    depth_width: int,
    expand: float = 0.75,
) -> torch.Tensor:
    mask = torch.zeros(*boxes.shape[:2], depth_height, depth_width, dtype=torch.bool, device=boxes.device)
    expand = max(float(expand), 0.0)
    for batch_idx in range(boxes.shape[0]):
        for frame_idx in range(boxes.shape[1]):
            valid_indices = torch.nonzero(boxes_mask[batch_idx, frame_idx], as_tuple=False).flatten()
            for box_idx in valid_indices:
                cx, cy, width, height = boxes[batch_idx, frame_idx, box_idx].unbind(dim=-1)
                width = width * (1.0 + expand)
                height = height * (1.0 + expand)
                x1 = int(torch.floor((cx - 0.5 * width).clamp(0.0, 1.0) * depth_width).item())
                x2 = int(torch.ceil((cx + 0.5 * width).clamp(0.0, 1.0) * depth_width).item())
                y1 = int(torch.floor((cy - 0.5 * height).clamp(0.0, 1.0) * depth_height).item())
                y2 = int(torch.ceil((cy + 0.5 * height).clamp(0.0, 1.0) * depth_height).item())
                if x2 > x1 and y2 > y1:
                    mask[batch_idx, frame_idx, y1:y2, x1:x2] = True
    return mask


def canonical_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 5 and depth.shape[-1] == 1:
        return depth[..., 0]
    if depth.ndim == 5 and depth.shape[2] == 1:
        return depth[:, :, 0]
    if depth.ndim == 4:
        return depth
    raise ValueError(f"Unsupported depth shape: {tuple(depth.shape)}")


def project_points(points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    z = points[..., 2].clamp(min=1e-6)
    x = intrinsics[0, 0] * points[..., 0] / z + intrinsics[0, 2]
    y = intrinsics[1, 1] * points[..., 1] / z + intrinsics[1, 2]
    return torch.stack([x, y], dim=-1)


def build_summary(
    metrics: dict[str, Meter],
    hsi_scales: list[float],
    hsi_biases: list[float],
    processed: int,
    args: argparse.Namespace,
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    values = {name: meter.mean for name, meter in metrics.items()}
    pairs = {
        "joints_mpjpe_m": ("base_joints_mpjpe_m", "hsi_joints_mpjpe_m"),
        "vertices_pve_m": ("base_vertices_pve_m", "hsi_vertices_pve_m"),
        "transl_l2_m": ("base_transl_l2_m", "hsi_transl_l2_m"),
        "projected_joints_l2_px": ("base_projected_joints_l2_px", "hsi_projected_joints_l2_px"),
        "depth_l1_mean_m": ("raw_depth_l1_mean_m", "hsi_depth_l1_mean_m"),
        "depth_l1_median_m": ("raw_depth_l1_median_m", "hsi_depth_l1_median_m"),
        "depth_near_l1_median_m": ("raw_depth_near_l1_median_m", "hsi_depth_near_l1_median_m"),
        "depth_far_l1_median_m": ("raw_depth_far_l1_median_m", "hsi_depth_far_l1_median_m"),
        "depth_human_roi_l1_median_m": ("raw_depth_human_roi_l1_median_m", "hsi_depth_human_roi_l1_median_m"),
        "sole_float_m": ("base_sole_float_m", "hsi_sole_float_m"),
        "sole_penetration_m": ("base_sole_penetration_m", "hsi_sole_penetration_m"),
        "sole_plane_float_m": ("base_sole_plane_float_m", "hsi_sole_plane_float_m"),
        "sole_plane_penetration_m": ("base_sole_plane_penetration_m", "hsi_sole_plane_penetration_m"),
    }
    improvements = {name: improvement(values[base], values[hsi]) for name, (base, hsi) in pairs.items()}
    scale_arr = np.asarray(hsi_scales, dtype=np.float64) if hsi_scales else np.asarray([], dtype=np.float64)
    bias_arr = np.asarray(hsi_biases, dtype=np.float64) if hsi_biases else np.asarray([], dtype=np.float64)
    return {
        "checkpoint": args.checkpoint,
        "num_samples": processed,
        "conf_threshold": float(args.conf_threshold),
        "use_gt_box_prior": bool(args.use_gt_box_prior),
        "metrics": values,
        "improvement_percent_lower_is_better": improvements,
        "hsi_scene_scale": describe_array(scale_arr),
        "hsi_scene_depth_bias": describe_array(bias_arr),
        "examples": examples,
    }


def improvement(base: float | None, hsi: float | None) -> float | None:
    if base is None or hsi is None or abs(base) < 1e-12:
        return None
    return (base - hsi) / base * 100.0


def describe_array(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
    }


def example_summary(batch_idx: int, predictions: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "batch_idx": int(batch_idx),
        "num_gt": int(batch["smpl_mask"].sum().detach().cpu()),
        "num_pred_conf": int((predictions["pred_confs"] >= 0.10).sum().detach().cpu()),
        "hsi_scene_scale": predictions.get("hsi_scene_scale", torch.empty(0)).detach().float().cpu().reshape(-1).tolist(),
        "hsi_scene_depth_bias": predictions.get("hsi_scene_depth_bias", torch.empty(0)).detach().float().cpu().reshape(-1).tolist(),
    }


def print_human_summary(summary: dict[str, Any]) -> None:
    print("========== HSI refine metrics ==========")
    metrics = summary["metrics"]
    improvements = summary["improvement_percent_lower_is_better"]
    for label, base_key, hsi_key, improvement_key in [
        ("3D joints MPJPE (m)", "base_joints_mpjpe_m", "hsi_joints_mpjpe_m", "joints_mpjpe_m"),
        ("Vertices PVE (m)", "base_vertices_pve_m", "hsi_vertices_pve_m", "vertices_pve_m"),
        ("Translation L2 (m)", "base_transl_l2_m", "hsi_transl_l2_m", "transl_l2_m"),
        ("Projected joints (px)", "base_projected_joints_l2_px", "hsi_projected_joints_l2_px", "projected_joints_l2_px"),
        ("Depth L1 mean (m)", "raw_depth_l1_mean_m", "hsi_depth_l1_mean_m", "depth_l1_mean_m"),
        ("Depth L1 median (m)", "raw_depth_l1_median_m", "hsi_depth_l1_median_m", "depth_l1_median_m"),
        ("Near depth median (m)", "raw_depth_near_l1_median_m", "hsi_depth_near_l1_median_m", "depth_near_l1_median_m"),
        ("Far depth median (m)", "raw_depth_far_l1_median_m", "hsi_depth_far_l1_median_m", "depth_far_l1_median_m"),
        ("Human ROI depth median", "raw_depth_human_roi_l1_median_m", "hsi_depth_human_roi_l1_median_m", "depth_human_roi_l1_median_m"),
    ]:
        imp = improvements.get(improvement_key)
        print(f"{label:24s} base={fmt(metrics.get(base_key))} hsi={fmt(metrics.get(hsi_key))} improvement={fmt(imp)}%")
    print(
        "depth valid pixels/frame: "
        f"full={fmt(metrics.get('depth_valid_pixels'))} "
        f"near={fmt(metrics.get('depth_near_valid_pixels'))} "
        f"far={fmt(metrics.get('depth_far_valid_pixels'))} "
        f"human_roi={fmt(metrics.get('depth_human_roi_valid_pixels'))}"
    )
    print(
        "HSI guard metrics       "
        f"worse>2cm={fmt(metrics.get('hsi_worse_than_base_ratio_2cm'))} "
        f"joint_delta={fmt(metrics.get('hsi_joint_error_delta_m'))}m"
    )
    print(
        "Foot contact metrics    "
        f"base_float={fmt(metrics.get('base_foot_float_m'))}m "
        f"hsi_float={fmt(metrics.get('hsi_foot_float_m'))}m "
        f"base_pen={fmt(metrics.get('base_foot_penetration_m'))}m "
        f"hsi_pen={fmt(metrics.get('hsi_foot_penetration_m'))}m "
        f"contacts={fmt(metrics.get('foot_contact_valid_count'))}"
    )
    print(
        "Sole contact metrics    "
        f"base_float={fmt(metrics.get('base_sole_float_m'))}m "
        f"hsi_float={fmt(metrics.get('hsi_sole_float_m'))}m "
        f"base_pen={fmt(metrics.get('base_sole_penetration_m'))}m "
        f"hsi_pen={fmt(metrics.get('hsi_sole_penetration_m'))}m "
        f"contacts={fmt(metrics.get('sole_contact_valid_count'))}"
    )
    print(
        "Sole plane metrics      "
        f"base_float={fmt(metrics.get('base_sole_plane_float_m'))}m "
        f"hsi_float={fmt(metrics.get('hsi_sole_plane_float_m'))}m "
        f"base_pen={fmt(metrics.get('base_sole_plane_penetration_m'))}m "
        f"hsi_pen={fmt(metrics.get('hsi_sole_plane_penetration_m'))}m "
        f"contacts={fmt(metrics.get('sole_plane_contact_valid_count'))}"
    )
    print(f"matched humans: {fmt(metrics.get('num_matched'))} / gt {fmt(metrics.get('num_gt'))}")
    print(f"hsi scale: {summary['hsi_scene_scale']}")
    print(f"hsi bias : {summary['hsi_scene_depth_bias']}")


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    main()
