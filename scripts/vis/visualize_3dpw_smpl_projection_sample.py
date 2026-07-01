from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.evaluate_3dpw_smpl_base_metrics import align_by_pelvis, procrustes_mpjpe
from scripts.train.train_smpl import apply_overrides, build_model, forward_model, load_initial_checkpoint
from vggt_omega.data import ThreeDPWDataset, threedpw_collate_fn
from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.training.config import deep_update, load_yaml_config, require_path
from vggt_omega.training.hungarian_losses import flatten_smpl_targets
from vggt_omega.training.smpl_matcher import HungarianSMPLMatcher, cxcywh_to_xyxy
from vggt_omega.utils.rotation import rot6d_to_axis_angle


GT_COLOR = (40, 210, 90)
PRED_COLOR = (245, 70, 65)
TEXT_COLOR = (25, 25, 25)
SKELETON_EDGES = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 12),
    (9, 13),
    (9, 14),
    (12, 15),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
    (20, 22),
    (21, 23),
)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.train_config))
    config = apply_overrides(config, args.override)
    model = build_model(config).to(device)
    load_initial_checkpoint(model, config, device)
    load_training_checkpoint(model, Path(args.checkpoint), device)
    model.eval()

    smpl = SMPLLayer(require_path(config, "assets.smpl_model_dir"), gender="neutral").to(device).eval()
    dataset = build_dataset(config, args)
    selected = select_indices(args, len(dataset))
    matcher = HungarianSMPLMatcher(cost_conf=0.5, cost_bbox=5.0, cost_giou=2.0, cost_kpts=0.0, require_boxes=True, require_j2ds=False)

    summary: list[dict[str, Any]] = []
    with torch.no_grad():
        for dataset_index in selected:
            item = dataset[int(dataset_index)]
            batch = move_to_device(threedpw_collate_fn([item]), device)
            predictions = forward_model(model, batch, config)
            sample = collect_sample_visuals(
                predictions=predictions,
                batch=batch,
                matcher=matcher,
                smpl=smpl,
                dataset_index=int(dataset_index),
            )
            out_path = output_dir / f"sample_{int(dataset_index):06d}_projection.jpg"
            draw_sample(item["images"][0], sample, out_path)
            sample_json = output_dir / f"sample_{int(dataset_index):06d}_projection.json"
            sample_json.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")
            summary.append({"dataset_index": int(dataset_index), "image": str(out_path), "json": str(sample_json), "matches": sample["matches"]})
            print(f"[vis] wrote {out_path}", flush=True)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "num_samples": len(summary), "summary": str(output_dir / "summary.json")}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize 3DPW GT/pred SMPL projection and per-person metrics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_base_3dpw_ray_refine.yaml")
    parser.add_argument("--output-dir", default="outputs/vis/3dpw_smpl_projection_sample")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="")
    parser.add_argument("--indices", default="", help="Comma-separated dataset indices to visualize.")
    parser.add_argument("--rows-csv", default="", help="Optional eval rows CSV; worst unique dataset indices are selected from it.")
    parser.add_argument("--sort-metric", default="transl_l2_mm")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def build_dataset(config: dict[str, Any], args: argparse.Namespace) -> ThreeDPWDataset:
    data_cfg = config["data"]
    return ThreeDPWDataset(
        root=require_path(config, data_cfg.get("root_key", "datasets.threedpw_root")),
        annotation_root=require_path(config, data_cfg.get("annotation_root_key", "datasets.threedpw_smpl_base_root")),
        split=args.split,
        sequence_length=int(data_cfg.get("sequence_length", 1)),
        stride=int(data_cfg.get("stride", 1)),
        image_size=int(data_cfg.get("image_size", 518)),
        max_humans=int(data_cfg.get("max_humans", 2)),
        require_boxes=True,
        require_smpl=True,
    )


def select_indices(args: argparse.Namespace, dataset_len: int) -> list[int]:
    if args.indices.strip():
        values = [int(value.strip()) for value in args.indices.split(",") if value.strip()]
    elif args.rows_csv.strip():
        values = indices_from_rows_csv(Path(args.rows_csv), args.sort_metric, int(args.top_k))
    else:
        values = list(range(int(args.start_index), int(args.start_index) + int(args.top_k)))
    values = [idx for idx in values if 0 <= idx < int(dataset_len)]
    if not values:
        raise ValueError("No valid dataset indices selected for visualization")
    return values


def indices_from_rows_csv(path: Path, sort_metric: str, top_k: int) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Rows CSV not found: {path}")
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or "dataset_index" not in reader.fieldnames or sort_metric not in reader.fieldnames:
            raise ValueError(f"CSV must contain dataset_index and {sort_metric!r}: {path}")
        rows.extend(reader)
    rows.sort(key=lambda row: float(row.get(sort_metric, "nan")), reverse=True)
    selected: list[int] = []
    seen: set[int] = set()
    for row in rows:
        dataset_index = int(float(row["dataset_index"]))
        if dataset_index in seen:
            continue
        seen.add(dataset_index)
        selected.append(dataset_index)
        if len(selected) >= int(top_k):
            break
    return selected


def load_training_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if state_dict is None:
        state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint does not contain a state_dict: {checkpoint_path}")
    missing, unexpected = model.load_state_dict({key.removeprefix("module."): value for key, value in state_dict.items()}, strict=False)
    print(f"[ckpt] loaded {checkpoint_path} missing={len(missing)} unexpected={len(unexpected)}", flush=True)


def collect_sample_visuals(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    matcher: HungarianSMPLMatcher,
    smpl: SMPLLayer,
    dataset_index: int,
) -> dict[str, Any]:
    pred_confs = flatten_prediction(predictions["pred_confs"], 3)
    pred_boxes = flatten_prediction(predictions["pred_boxes"], 3)
    pred_poses = flatten_prediction(predictions["pred_poses"], 3)
    pred_betas = flatten_prediction(predictions["pred_betas"], 3)
    pred_transl = flatten_prediction(predictions["pred_transl_cam"], 3)
    targets = flatten_smpl_targets(batch, device=pred_confs.device)
    indices = matcher({"pred_confs": pred_confs, "pred_boxes": pred_boxes}, targets)
    matched = collect_matches(indices, targets, pred_confs.device)
    image_size = int(batch["images"].shape[-1])
    intrinsics = flatten_batch_intrinsics(batch["K_scal3r"]).to(device=pred_confs.device, dtype=pred_confs.dtype)
    if matched["frame_idx"].numel() == 0:
        return {"dataset_index": dataset_index, "image_size": image_size, "matches": []}

    frame_idx = matched["frame_idx"]
    src_idx = matched["src_idx"]
    target_idx = matched["target_idx"]
    gt_pose = rot6d_to_axis_angle(matched["pose_6d"]).reshape(-1, 72)
    gt_betas = matched["betas"]
    pred_vertices, pred_joints = smpl(pred_poses[frame_idx, src_idx].reshape(-1, 72).float(), pred_betas[frame_idx, src_idx].float())
    gt_vertices, gt_joints = smpl(gt_pose.float(), gt_betas.float())
    pred_joints = pred_joints[:, :24].to(dtype=pred_vertices.dtype)
    gt_joints = gt_joints[:, :24].to(dtype=gt_vertices.dtype)
    pred_transl_matched = pred_transl[frame_idx, src_idx].to(dtype=pred_joints.dtype)
    gt_transl_matched = matched["transl_cam"].to(device=pred_transl_matched.device, dtype=pred_transl_matched.dtype)
    pred_joints_cam = pred_joints + pred_transl_matched[:, None, :]
    gt_joints_cam = gt_joints + gt_transl_matched[:, None, :]
    pred_joints_a, gt_joints_a, _, _ = align_by_pelvis(pred_joints, gt_joints, pred_vertices, gt_vertices)
    mpjpe = torch.linalg.norm(pred_joints_a - gt_joints_a, dim=-1).mean(dim=-1)
    pa = procrustes_mpjpe(pred_joints_a, gt_joints_a)
    transl_delta = pred_transl_matched - gt_transl_matched
    transl_l2 = torch.linalg.norm(transl_delta, dim=-1)
    transl_xy = torch.linalg.norm(transl_delta[:, :2], dim=-1)
    transl_z = transl_delta[:, 2].abs()
    cam_mpjpe = torch.linalg.norm(pred_joints_cam - gt_joints_cam, dim=-1).mean(dim=-1)
    pred_2d = project_points(pred_joints_cam, intrinsics[frame_idx].to(dtype=pred_joints_cam.dtype))
    gt_2d = project_points(gt_joints_cam, intrinsics[frame_idx].to(dtype=gt_joints_cam.dtype))
    pred_box_xyxy = cxcywh_to_xyxy(pred_boxes[frame_idx, src_idx].clamp(0.0, 1.0)) * float(image_size)
    gt_box_xyxy = cxcywh_to_xyxy(matched["boxes"].to(device=pred_boxes.device, dtype=pred_boxes.dtype).clamp(0.0, 1.0)) * float(image_size)

    matches: list[dict[str, Any]] = []
    for row in range(int(frame_idx.numel())):
        matches.append(
            {
                "frame_idx": int(frame_idx[row].detach().cpu()),
                "query_idx": int(src_idx[row].detach().cpu()),
                "gt_idx": int(target_idx[row].detach().cpu()),
                "person_id": int(matched["person_ids"][row].detach().cpu()),
                "mpjpe_mm": float(mpjpe[row].detach().cpu() * 1000.0),
                "pa_mpjpe_mm": float(pa[row].detach().cpu() * 1000.0),
                "transl_l2_mm": float(transl_l2[row].detach().cpu() * 1000.0),
                "transl_xy_l2_mm": float(transl_xy[row].detach().cpu() * 1000.0),
                "transl_z_abs_mm": float(transl_z[row].detach().cpu() * 1000.0),
                "cam_mpjpe_no_align_mm": float(cam_mpjpe[row].detach().cpu() * 1000.0),
                "gt_transl_cam": to_float_list(gt_transl_matched[row]),
                "pred_transl_cam": to_float_list(pred_transl_matched[row]),
                "gt_box_xyxy": to_float_list(gt_box_xyxy[row]),
                "pred_box_xyxy": to_float_list(pred_box_xyxy[row]),
                "gt_joints_2d": to_float_nested(gt_2d[row]),
                "pred_joints_2d": to_float_nested(pred_2d[row]),
                "gt_joints_pelvis_aligned": to_float_nested(gt_joints_a[row]),
                "pred_joints_pelvis_aligned": to_float_nested(pred_joints_a[row]),
            }
        )
    return {"dataset_index": dataset_index, "image_size": image_size, "matches": matches}


def collect_matches(indices, targets: list[dict[str, torch.Tensor]], device: torch.device) -> dict[str, torch.Tensor]:
    frame_indices = []
    src_indices = []
    target_indices = []
    parts: dict[str, list[torch.Tensor]] = {"pose_6d": [], "betas": [], "transl_cam": [], "boxes": [], "person_ids": []}
    for frame_idx, (src_idx, tgt_idx) in enumerate(indices):
        if src_idx.numel() == 0:
            continue
        frame_indices.append(torch.full_like(src_idx, frame_idx))
        src_indices.append(src_idx)
        target_indices.append(tgt_idx)
        target = targets[frame_idx]
        for key in parts:
            parts[key].append(target[key][tgt_idx])
    if not frame_indices:
        return {
            "frame_idx": torch.empty(0, dtype=torch.long, device=device),
            "src_idx": torch.empty(0, dtype=torch.long, device=device),
            "target_idx": torch.empty(0, dtype=torch.long, device=device),
        }
    out = {"frame_idx": torch.cat(frame_indices), "src_idx": torch.cat(src_indices), "target_idx": torch.cat(target_indices)}
    out.update({key: torch.cat(value) for key, value in parts.items()})
    return out


def draw_sample(image_tensor: torch.Tensor, sample: dict[str, Any], output_path: Path) -> None:
    image = tensor_to_image(image_tensor)
    width, height = image.size
    aligned_width = 420
    text_width = 430
    canvas = Image.new("RGB", (width + aligned_width + text_width, height), (255, 255, 255))
    projection_panel = image.copy()
    draw_projection = ImageDraw.Draw(projection_panel)
    for match in sample["matches"]:
        draw_box(draw_projection, match["gt_box_xyxy"], GT_COLOR, width=3)
        draw_box(draw_projection, match["pred_box_xyxy"], PRED_COLOR, width=3)
        draw_skeleton(draw_projection, match["gt_joints_2d"], GT_COLOR, width=3, radius=3)
        draw_skeleton(draw_projection, match["pred_joints_2d"], PRED_COLOR, width=3, radius=3)
    draw_projection.rectangle((8, 8, 260, 58), fill=(255, 255, 255))
    draw_projection.text((16, 14), "Projection: GT green / Pred red", fill=TEXT_COLOR)
    draw_projection.text((16, 34), f"dataset_index={sample['dataset_index']}", fill=TEXT_COLOR)
    canvas.paste(projection_panel, (0, 0))

    aligned_panel = Image.new("RGB", (aligned_width, height), (250, 250, 250))
    draw_aligned = ImageDraw.Draw(aligned_panel)
    draw_aligned.text((14, 14), "Pelvis-aligned body (MPJPE view)", fill=TEXT_COLOR)
    draw_aligned_bodies(draw_aligned, sample["matches"], aligned_width, height)
    canvas.paste(aligned_panel, (width, 0))

    text_panel = Image.new("RGB", (text_width, height), (255, 255, 255))
    draw_text = ImageDraw.Draw(text_panel)
    draw_metrics(draw_text, sample)
    canvas.paste(text_panel, (width + aligned_width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def draw_metrics(draw: ImageDraw.ImageDraw, sample: dict[str, Any]) -> None:
    y = 18
    draw.text((16, y), "Per-person metrics", fill=TEXT_COLOR)
    y += 28
    draw.text((16, y), "GT: green   Pred: red", fill=TEXT_COLOR)
    y += 32
    for idx, match in enumerate(sample["matches"]):
        lines = [
            f"person {idx}: query={match['query_idx']} gt={match['gt_idx']} id={match['person_id']}",
            f"MPJPE          {match['mpjpe_mm']:.1f} mm",
            f"PA-MPJPE       {match['pa_mpjpe_mm']:.1f} mm",
            f"transl L2      {match['transl_l2_mm']:.1f} mm",
            f"transl XY      {match['transl_xy_l2_mm']:.1f} mm",
            f"transl Z abs   {match['transl_z_abs_mm']:.1f} mm",
            f"cam MPJPE      {match['cam_mpjpe_no_align_mm']:.1f} mm",
            f"GT t     {format_vec(match['gt_transl_cam'])}",
            f"Pred t   {format_vec(match['pred_transl_cam'])}",
        ]
        for line in lines:
            draw.text((16, y), line, fill=TEXT_COLOR)
            y += 20
        y += 16


def draw_aligned_bodies(draw: ImageDraw.ImageDraw, matches: list[dict[str, Any]], width: int, height: int) -> None:
    if not matches:
        return
    slot_h = max((height - 60) // len(matches), 120)
    for idx, match in enumerate(matches):
        top = 48 + idx * slot_h
        bottom = min(top + slot_h - 8, height - 8)
        viewport = (20, top, width - 20, bottom)
        draw.rectangle(viewport, outline=(220, 220, 220))
        gt = np.asarray(match["gt_joints_pelvis_aligned"], dtype=np.float32)[:, [0, 1]]
        pred = np.asarray(match["pred_joints_pelvis_aligned"], dtype=np.float32)[:, [0, 1]]
        points = np.concatenate([gt, pred], axis=0)
        center = points.mean(axis=0)
        scale = float(np.max(np.linalg.norm(points - center[None], axis=1)))
        scale = max(scale, 1e-4)

        def map_points(values: np.ndarray) -> list[tuple[float, float]]:
            x0, y0, x1, y1 = viewport
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            factor = 0.42 * min(x1 - x0, y1 - y0) / scale
            return [(float(cx + (p[0] - center[0]) * factor), float(cy - (p[1] - center[1]) * factor)) for p in values]

        draw_skeleton(draw, map_points(gt), GT_COLOR, width=2, radius=2)
        draw_skeleton(draw, map_points(pred), PRED_COLOR, width=2, radius=2)
        draw.text((viewport[0] + 6, viewport[1] + 4), f"person {idx}", fill=TEXT_COLOR)


def draw_skeleton(
    draw: ImageDraw.ImageDraw,
    joints: list[list[float]] | list[tuple[float, float]],
    color: tuple[int, int, int],
    width: int = 2,
    radius: int = 2,
) -> None:
    points = [(float(p[0]), float(p[1])) for p in joints]
    for a, b in SKELETON_EDGES:
        if a >= len(points) or b >= len(points):
            continue
        if not all(np.isfinite(points[a])) or not all(np.isfinite(points[b])):
            continue
        draw.line((points[a][0], points[a][1], points[b][0], points[b][1]), fill=color, width=width)
    for x, y in points:
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def draw_box(draw: ImageDraw.ImageDraw, box_xyxy: list[float], color: tuple[int, int, int], width: int = 2) -> None:
    x0, y0, x1, y1 = [float(v) for v in box_xyxy]
    for offset in range(width):
        draw.rectangle((x0 - offset, y0 - offset, x1 + offset, y1 + offset), outline=color)


def tensor_to_image(image_tensor: torch.Tensor) -> Image.Image:
    arr = image_tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")


def project_points(points_cam: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    z = points_cam[..., 2:3].clamp_min(1e-6)
    xy = points_cam[..., :2] / z
    fx = intrinsics[:, 0, 0][:, None]
    fy = intrinsics[:, 1, 1][:, None]
    cx = intrinsics[:, 0, 2][:, None]
    cy = intrinsics[:, 1, 2][:, None]
    return torch.stack((xy[..., 0] * fx + cx, xy[..., 1] * fy + cy), dim=-1)


def flatten_prediction(value: torch.Tensor, unframed_ndim: int) -> torch.Tensor:
    if value.ndim == unframed_ndim:
        return value
    if value.ndim == unframed_ndim + 1:
        return value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
    raise ValueError(f"Unsupported prediction shape {tuple(value.shape)}")


def flatten_batch_intrinsics(intrinsics: torch.Tensor) -> torch.Tensor:
    if intrinsics.ndim == 4:
        return intrinsics.reshape(intrinsics.shape[0] * intrinsics.shape[1], 3, 3)
    if intrinsics.ndim == 3:
        return intrinsics
    raise ValueError(f"Unsupported intrinsics shape: {tuple(intrinsics.shape)}")


def to_float_list(value: torch.Tensor) -> list[float]:
    return [float(x) for x in value.detach().cpu().reshape(-1).tolist()]


def to_float_nested(value: torch.Tensor) -> list[list[float]]:
    return [[float(x) for x in row] for row in value.detach().cpu().tolist()]


def format_vec(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.2f}" for value in values[:3]) + "]"


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


if __name__ == "__main__":
    main()
