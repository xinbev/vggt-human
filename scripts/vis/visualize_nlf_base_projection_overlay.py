#!/usr/bin/env python
"""Visualize NLF base SMPL projections on the processed VGGT image plane."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.evaluate_hsi_refine_metrics import load_vggt_baseline, project_points  # noqa: E402
from scripts.train.train_smpl import apply_overrides, build_model, load_yaml_config  # noqa: E402
from scripts.vis.visualize_hsi_depth_smpl_diagnostics import load_single_bedlam_batch  # noqa: E402
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.training.config import deep_update, require_path  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402
from vggt_omega.utils.rotation import rot6d_to_axis_angle  # noqa: E402


COLORS = [
    (255, 64, 64),
    (64, 180, 255),
    (80, 230, 120),
    (255, 190, 64),
    (190, 80, 255),
    (255, 80, 180),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw NLF base SMPL projections before HSI training.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_hsi_nlf_provider.yaml")
    parser.add_argument("--output-dir", default="outputs/vis/nlf_base_projection_overlay")
    parser.add_argument("--device", default="")
    parser.add_argument("--split", default="Training")
    parser.add_argument("--smpl-model-dir", default="")
    parser.add_argument("--conf-threshold", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


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
    model.eval()

    with torch.no_grad():
        predictions = model(
            batch["images"],
            smpl_query_boxes=batch["gt_boxes"],
            smpl_query_boxes_mask=batch["boxes_mask"],
        )

    smpl = SMPLLayer(require_path(config, "assets.smpl_model_dir", allow_empty=False)).to(device).eval()
    pred_joints = decode_pred_joints(predictions, smpl)
    gt_joints = decode_gt_joints(batch, smpl)
    intrinsics = encoding_to_camera(
        predictions["pose_enc"],
        image_size_hw=(int(batch["images"].shape[-2]), int(batch["images"].shape[-1])),
        build_intrinsics=True,
    )[1][0, 0].to(device=device, dtype=pred_joints.dtype)

    summary, overlay = build_overlay(args, rgb, predictions, batch, pred_joints, gt_joints, intrinsics)
    stem = image_path.stem
    overlay_path = output_dir / f"{stem}_nlf_base_projection_overlay.png"
    json_path = output_dir / f"{stem}_nlf_base_projection_overlay.json"
    overlay.save(overlay_path)
    summary.update(
        {
            "image": str(image_path),
            "overlay": str(overlay_path),
            "image_hw": [int(batch["images"].shape[-2]), int(batch["images"].shape[-1])],
            "nlf_image_hw": tensor_to_list(predictions.get("nlf_image_hw")),
            "nlf_intrinsics": tensor_to_list(predictions.get("nlf_intrinsics", torch.empty(0))[0, 0]),
        }
    )
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[ok] NLF base projection overlay written")
    print(json.dumps({"overlay": str(overlay_path), "summary": str(json_path)}, indent=2))


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.train_config))
    config = apply_overrides(config, args.override)
    if args.smpl_model_dir:
        config.setdefault("assets", {})["smpl_model_dir"] = args.smpl_model_dir
    model_cfg = config.setdefault("model", {})
    model_cfg["enable_camera"] = True
    model_cfg["enable_smpl"] = True
    model_cfg["enable_hsi_refine"] = False
    model_cfg["smpl_provider"] = "nlf"
    model_cfg["nlf_use_detector"] = False
    model_cfg["nlf_require_boxes"] = True
    data_cfg = config.setdefault("data", {})
    data_cfg["require_boxes"] = True
    data_cfg["require_depth"] = True
    return config


def decode_pred_joints(predictions: dict[str, torch.Tensor], smpl: SMPLLayer) -> torch.Tensor:
    poses = predictions["pred_poses"]
    betas = predictions["pred_betas"]
    transl = predictions["pred_transl_cam"]
    shape = poses.shape[:3]
    vertices, joints = smpl(poses.reshape(-1, 72).float(), betas.reshape(-1, betas.shape[-1]).float())
    _ = vertices
    joints = joints.reshape(*shape, joints.shape[-2], 3).to(dtype=transl.dtype) + transl[..., None, :]
    return joints


def decode_gt_joints(batch: dict[str, torch.Tensor], smpl: SMPLLayer) -> torch.Tensor | None:
    if "gt_pose_6d" not in batch or "gt_betas" not in batch or "gt_transl_cam" not in batch:
        return None
    poses = rot6d_to_axis_angle(batch["gt_pose_6d"].reshape(-1, 24, 6)).reshape(*batch["gt_pose_6d"].shape[:3], 72)
    betas = batch["gt_betas"]
    transl = batch["gt_transl_cam"]
    shape = poses.shape[:3]
    vertices, joints = smpl(poses.reshape(-1, 72).float(), betas.reshape(-1, betas.shape[-1]).float())
    _ = vertices
    joints = joints.reshape(*shape, joints.shape[-2], 3).to(dtype=transl.dtype) + transl[..., None, :]
    return joints


def build_overlay(
    args: argparse.Namespace,
    rgb,
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    pred_joints: torch.Tensor,
    gt_joints: torch.Tensor | None,
    intrinsics: torch.Tensor,
) -> tuple[dict[str, Any], Any]:
    image = rgb.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size
    pred_confs = predictions["pred_confs"][0, 0, :, 0].detach()
    order = torch.argsort(pred_confs, descending=True)
    selected = [int(idx.item()) for idx in order if float(pred_confs[idx].detach().cpu()) >= args.conf_threshold]
    selected = selected[: max(int(args.top_k), 1)]

    gt_boxes = batch["gt_boxes"][0, 0].detach().float().cpu()
    pred_boxes = predictions.get("pred_boxes", torch.zeros_like(batch["gt_boxes"]))[0, 0].detach().float().cpu()
    gt_mask = batch["boxes_mask"][0, 0].detach().bool().cpu()
    summary_people = []

    for rank, query_idx in enumerate(selected):
        color = COLORS[rank % len(COLORS)]
        if query_idx < gt_boxes.shape[0] and bool(gt_mask[query_idx]):
            draw_box(draw, cxcywh_to_xyxy(gt_boxes[query_idx], width, height), color, width_px=3)
        if query_idx < pred_boxes.shape[0]:
            draw_box(draw, cxcywh_to_xyxy(pred_boxes[query_idx], width, height), tuple(max(c - 70, 0) for c in color), width_px=2)

        joints = pred_joints[0, 0, query_idx, :24]
        projected = project_points(joints, intrinsics)
        mask = (
            (joints[:, 2] > 1e-4)
            & torch.isfinite(projected).all(dim=-1)
            & (projected[:, 0] >= 0)
            & (projected[:, 0] < float(width))
            & (projected[:, 1] >= 0)
            & (projected[:, 1] < float(height))
        )
        draw_joints(draw, projected.detach().cpu(), mask.detach().cpu(), color, radius=3)

        if gt_joints is not None and query_idx < gt_joints.shape[2]:
            gt = gt_joints[0, 0, query_idx, :24]
            gt_projected = project_points(gt, intrinsics)
            gt_mask_j = (
                (gt[:, 2] > 1e-4)
                & torch.isfinite(gt_projected).all(dim=-1)
                & (gt_projected[:, 0] >= 0)
                & (gt_projected[:, 0] < float(width))
                & (gt_projected[:, 1] >= 0)
                & (gt_projected[:, 1] < float(height))
            )
            draw_joints(draw, gt_projected.detach().cpu(), gt_mask_j.detach().cpu(), (255, 255, 255), radius=2)

        transl = predictions["pred_transl_cam"][0, 0, query_idx].detach().float().cpu()
        label_xy = label_position(gt_boxes[query_idx] if query_idx < gt_boxes.shape[0] else None, width, height)
        draw_label(draw, font, label_xy, f"q{query_idx} c={float(pred_confs[query_idx]):.2f} z={float(transl[2]):.2f}m", color)
        summary_people.append(
            {
                "query": query_idx,
                "confidence": float(pred_confs[query_idx].detach().cpu()),
                "pred_transl_cam": transl.tolist(),
                "valid_projected_joints": int(mask.sum().detach().cpu()),
                "gt_box_cxcywh": gt_boxes[query_idx].tolist() if query_idx < gt_boxes.shape[0] else None,
                "nlf_box_cxcywh": pred_boxes[query_idx].tolist() if query_idx < pred_boxes.shape[0] else None,
            }
        )

    draw_label(draw, font, (8, 8), "NLF base: colored=input boxes/joints, dark=NLF boxes, white=GT SMPL joints", (255, 255, 255))
    return {"num_selected": len(selected), "people": summary_people}, image


def cxcywh_to_xyxy(box: torch.Tensor, width: int, height: int) -> list[float]:
    cx, cy, bw, bh = [float(v) for v in box]
    x1 = (cx - 0.5 * bw) * float(width)
    y1 = (cy - 0.5 * bh) * float(height)
    x2 = (cx + 0.5 * bw) * float(width)
    y2 = (cy + 0.5 * bh) * float(height)
    return [x1, y1, x2, y2]


def draw_box(draw: ImageDraw.ImageDraw, box: list[float], color: tuple[int, int, int], width_px: int) -> None:
    draw.rectangle(box, outline=color, width=width_px)


def draw_joints(draw: ImageDraw.ImageDraw, joints: torch.Tensor, mask: torch.Tensor, color: tuple[int, int, int], radius: int) -> None:
    for idx, xy in enumerate(joints[:24]):
        if idx >= mask.numel() or not bool(mask[idx]):
            continue
        x, y = float(xy[0]), float(xy[1])
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=(0, 0, 0))


def label_position(box: torch.Tensor | None, width: int, height: int) -> tuple[float, float]:
    if box is None:
        return (8.0, 28.0)
    x1, y1, _, _ = cxcywh_to_xyxy(box, width, height)
    return (max(0.0, x1), max(0.0, y1 - 16.0))


def draw_label(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    xy: tuple[float, float],
    text: str,
    outline: tuple[int, int, int],
) -> None:
    bbox = draw.textbbox(xy, text, font=font)
    draw.rectangle([bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3], fill=(0, 0, 0), outline=outline, width=1)
    draw.text(xy, text, fill=(255, 255, 255), font=font)


def tensor_to_list(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


if __name__ == "__main__":
    main()
