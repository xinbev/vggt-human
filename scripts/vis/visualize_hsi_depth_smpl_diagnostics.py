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
    greedy_match,
    human_roi_depth_mask,
    load_training_checkpoint,
    load_vggt_baseline,
    project_points,
)
from scripts.train.train_smpl import apply_overrides, build_model, load_yaml_config  # noqa: E402
from vggt_omega.data.bedlam import (  # noqa: E402
    _build_box_targets,
    _build_smpl_targets,
    _load_box_persons,
    _load_depth_tensor,
    _load_persons,
    _load_rgb_tensor,
)
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
    smpl_summary = compute_smpl_diagnostics(predictions, batch, smpl, config, args)
    diagnosis = infer_source(depth_summary, smpl_summary, args)

    stem = image_path.stem
    image_paths = save_depth_visuals(output_dir, stem, rgb, depth_images, depth_summary)
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
    return config


def load_single_bedlam_batch(image_path: Path, config: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, torch.Tensor], Image.Image]:
    data_cfg = config["data"]
    image_size = int(data_cfg.get("image_size", 518))
    max_humans = int(data_cfg.get("max_humans", config.get("model", {}).get("num_smpl_queries", 20)))
    dataset_root = Path(require_path(config, data_cfg.get("root_key", "datasets.bedlam_root"), allow_empty=False)).expanduser()
    boxes_root = Path(require_path(config, data_cfg["boxes_root_key"], allow_empty=False)).expanduser()
    split = str(args.split)
    seq_dir = image_path.parent.parent
    frame_id = image_path.stem
    sequence_rel = seq_dir.relative_to(dataset_root / split)

    image_tensor, _ = _load_rgb_tensor(image_path, image_size)
    rgb = Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    depth = _load_depth_tensor(seq_dir / "depth" / f"{frame_id}.npy", image_size, require_depth=True)
    persons = _load_persons(seq_dir / "smpl" / f"{frame_id}.pkl", require_smpl=True)
    box_persons = _load_box_persons(boxes_root / split / sequence_rel / "smpl_boxes" / f"{frame_id}.pkl", require_boxes=True)
    smpl = _build_smpl_targets([persons], max_humans)
    boxes = _build_box_targets([box_persons], [persons], max_humans, require_boxes=True)
    batch = {
        "images": image_tensor.unsqueeze(0).unsqueeze(0),
        "gt_depth": depth.unsqueeze(0).unsqueeze(0),
        "gt_pose_6d": smpl["pose_6d"].unsqueeze(0),
        "gt_betas": smpl["betas"].unsqueeze(0),
        "gt_transl_cam": smpl["transl_cam"].unsqueeze(0),
        "gt_cam_trans": smpl["transl_cam"].unsqueeze(0),
        "smpl_mask": smpl["smpl_mask"].unsqueeze(0),
        "gt_boxes": boxes["boxes"].unsqueeze(0),
        "boxes_mask": boxes["boxes_mask"].unsqueeze(0),
        "person_ids": boxes["person_ids"].unsqueeze(0),
        "person_id_mask": boxes["person_id_mask"].unsqueeze(0),
    }
    return batch, rgb


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
) -> dict[str, Any]:
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
        image_size_hw=(int(config["data"]["image_size"]), int(config["data"]["image_size"])),
        build_intrinsics=True,
    )[1]
    confs = predictions["pred_confs"].detach()
    if confs.ndim == 4 and confs.shape[-1] == 1:
        confs = confs[..., 0]

    gt_idx = torch.nonzero(batch["smpl_mask"][0, 0].bool(), as_tuple=False).flatten()
    pred_idx = torch.nonzero(confs[0, 0] >= float(args.conf_threshold), as_tuple=False).flatten()
    matches = greedy_match(base["joints"][0, 0, pred_idx, :24], gt["joints"][0, 0, gt_idx, :24]) if gt_idx.numel() and pred_idx.numel() else []
    per_person = []
    meters: dict[str, list[float]] = {
        "base_joints_mpjpe_m": [],
        "hsi_joints_mpjpe_m": [],
        "base_vertices_pve_m": [],
        "hsi_vertices_pve_m": [],
        "base_transl_l2_m": [],
        "hsi_transl_l2_m": [],
        "base_projected_joints_l2_px": [],
        "hsi_projected_joints_l2_px": [],
    }
    for pred_local, gt_local in matches:
        q = int(pred_idx[pred_local].item())
        g = int(gt_idx[gt_local].item())
        values = human_metric_values(base, hsi, gt, intrinsics[0, 0], q, g)
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
    return summary


def human_metric_values(
    base: dict[str, torch.Tensor],
    hsi: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    intrinsics: torch.Tensor,
    q: int,
    g: int,
) -> dict[str, float]:
    gt_j = gt["joints"][0, 0, g, :24]
    base_j = base["joints"][0, 0, q, :24]
    hsi_j = hsi["joints"][0, 0, q, :24]
    base_2d = project_points(base_j, intrinsics)
    hsi_2d = project_points(hsi_j, intrinsics)
    gt_2d = project_points(gt_j, intrinsics)
    valid_base = (base_j[:, 2] > 1e-4) & (gt_j[:, 2] > 1e-4)
    valid_hsi = (hsi_j[:, 2] > 1e-4) & (gt_j[:, 2] > 1e-4)
    return {
        "base_joints_mpjpe_m": scalar(torch.linalg.norm(base_j - gt_j, dim=-1).mean()),
        "hsi_joints_mpjpe_m": scalar(torch.linalg.norm(hsi_j - gt_j, dim=-1).mean()),
        "base_vertices_pve_m": scalar(torch.linalg.norm(base["vertices"][0, 0, q] - gt["vertices"][0, 0, g], dim=-1).mean()),
        "hsi_vertices_pve_m": scalar(torch.linalg.norm(hsi["vertices"][0, 0, q] - gt["vertices"][0, 0, g], dim=-1).mean()),
        "base_transl_l2_m": scalar(torch.linalg.norm(base["transl"][0, 0, q] - gt["transl"][0, 0, g])),
        "hsi_transl_l2_m": scalar(torch.linalg.norm(hsi["transl"][0, 0, q] - gt["transl"][0, 0, g])),
        "base_projected_joints_l2_px": scalar(torch.linalg.norm(base_2d[valid_base] - gt_2d[valid_base], dim=-1).mean()) if valid_base.any() else None,
        "hsi_projected_joints_l2_px": scalar(torch.linalg.norm(hsi_2d[valid_hsi] - gt_2d[valid_hsi], dim=-1).mean()) if valid_hsi.any() else None,
    }


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
