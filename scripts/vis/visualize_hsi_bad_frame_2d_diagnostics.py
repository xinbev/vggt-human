#!/usr/bin/env python
"""High-information 2D diagnostics for one HSI bad frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

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

from scripts.train.train_smpl import build_model  # noqa: E402
from scripts.vis.visualize_smpl_inference import (  # noqa: E402
    COLORS,
    load_config,
    load_training_checkpoint,
    load_vggt_baseline_for_camera,
    project_points,
    require_smpl_model_dir,
)
from vggt_omega.data import BedlamDataset  # noqa: E402
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.training.config import require_path  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402
from vggt_omega.utils.rotation import rot6d_to_axis_angle  # noqa: E402


SMPL_EDGES = [
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
]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir).expanduser()
    panels_dir = output_dir / "panels"
    output_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args)
    config.setdefault("model", {})
    config["model"]["enable_camera"] = True
    config["model"]["enable_depth"] = True
    config["model"]["enable_hsi_refine"] = True
    config.setdefault("data", {})
    config["data"]["sequence_length"] = int(args.num_frames)
    config["data"]["stride"] = int(args.stride)
    config["data"]["require_depth"] = True
    config["data"]["require_boxes"] = True

    dataset = build_dataset(config, args)
    dataset_index, seq_idx, start_idx = find_dataset_window(dataset, Path(args.image).expanduser())
    sample = dataset[dataset_index]
    frame_paths = frame_paths_for_window(dataset, seq_idx, start_idx)
    target_idx = resolve_target_frame(args, frame_paths)
    target_path = frame_paths[target_idx]

    batch = {key: value.unsqueeze(0).to(device) for key, value in sample.items() if isinstance(value, torch.Tensor)}

    model = build_model(config).to(device)
    load_vggt_baseline_for_camera(model, config, device)
    load_training_checkpoint(model, Path(args.checkpoint).expanduser(), device)
    model.eval()

    with torch.no_grad():
        predictions = model(
            batch["images"],
            smpl_query_boxes=batch["gt_boxes"] if args.use_gt_box_prior else None,
            smpl_query_boxes_mask=batch["boxes_mask"] if args.use_gt_box_prior else None,
            smpl_track_ids=batch.get("gt_track_ids") if args.use_track_ids else None,
            smpl_track_mask=batch.get("gt_track_mask") if args.use_track_ids else None,
        )

    input_size = int(config["data"].get("image_size", args.image_size))
    original = Image.open(target_path).convert("RGB")
    persons = collect_person_diagnostics(predictions, batch, target_idx, original.size, input_size, config, args, device)
    panels = make_panels(original, persons, int(args.panel_width))
    panel_files = []
    for name, panel in panels.items():
        path = panels_dir / f"{target_path.stem}_{name}.png"
        panel.save(path)
        panel_files.append(str(path))
    board = make_board(list(panels.values()), persons, target_path, int(args.panel_width))
    board_path = output_dir / f"{target_path.stem}_bad_frame_2d_diagnostics.png"
    board.save(board_path)

    payload = {
        "checkpoint": str(args.checkpoint),
        "image_start": str(Path(args.image).expanduser()),
        "target_frame_index": int(target_idx),
        "target_frame": str(target_path),
        "use_gt_box_prior": bool(args.use_gt_box_prior),
        "use_track_ids": bool(args.use_track_ids),
        "num_persons": len(persons),
        "output_image": str(board_path),
        "panel_files": panel_files,
        "persons": persons,
        "legend": {
            "gt_box": "per-person color rectangle",
            "pred_box": "white dashed rectangle",
            "gt_smpl_projection": "green skeleton",
            "base_smpl_projection": "red skeleton",
            "hsi_smpl_projection": "cyan skeleton",
        },
    }
    json_path = output_dir / f"{target_path.stem}_bad_frame_2d_diagnostics.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output_image": str(board_path), "output_json": str(json_path), "num_persons": len(persons)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw high-information 2D diagnostics for one bad frame")
    parser.add_argument("--image", required=True, help="Start RGB image path in a BEDLAM sequence")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_hsi_refine.yaml")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--output-dir", default="outputs/vis/hsi_bad_frame_2d_diagnostics")
    parser.add_argument("--device", default="")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--num-frames", type=int, default=27)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--target-frame-stem", default="seq_000000_0100")
    parser.add_argument("--target-frame-index", type=int, default=-1)
    parser.add_argument("--split", default="Training")
    parser.add_argument("--use-gt-box-prior", action="store_true")
    parser.add_argument("--disable-track-ids", action="store_true")
    parser.add_argument("--smpl-model-dir", default="")
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    args.use_track_ids = not bool(args.disable_track_ids)
    return args


def build_dataset(config: dict[str, Any], args: argparse.Namespace) -> BedlamDataset:
    data_cfg = config["data"]
    root = require_path(config, data_cfg.get("root_key", "datasets.bedlam_root"), allow_empty=False)
    boxes_root = require_path(config, data_cfg.get("boxes_root_key", "datasets.bedlam_boxes_root"), allow_empty=False)
    return BedlamDataset(
        root=root,
        split=str(args.split),
        sequence_length=int(args.num_frames),
        stride=int(args.stride),
        image_size=int(data_cfg.get("image_size", args.image_size)),
        max_humans=int(data_cfg.get("max_humans", config.get("model", {}).get("num_smpl_queries", 20))),
        require_smpl=True,
        require_depth=True,
        boxes_root=boxes_root,
        require_boxes=True,
    )


def find_dataset_window(dataset: BedlamDataset, image_path: Path) -> tuple[int, int, int]:
    image_path = image_path.resolve()
    if image_path.parent.name != "rgb":
        raise ValueError(f"--image must point to a BEDLAM rgb frame, got: {image_path}")
    sequence_dir = image_path.parent.parent.resolve()
    frame_stem = image_path.stem
    for seq_idx, (seq_dir, frame_ids) in enumerate(dataset._sequences):
        if seq_dir.resolve() != sequence_dir:
            continue
        if frame_stem not in frame_ids:
            raise ValueError(f"Frame {frame_stem} not found in sequence {seq_dir}")
        start_idx = frame_ids.index(frame_stem)
        for dataset_index, (index_seq_idx, index_start_idx) in enumerate(dataset._index):
            if index_seq_idx == seq_idx and index_start_idx == start_idx:
                return dataset_index, seq_idx, start_idx
        raise ValueError(
            f"No dataset window starts at {frame_stem}. "
            f"Check NUM_FRAMES={dataset.sequence_length} and stride={dataset.stride}."
        )
    raise ValueError(f"Sequence directory not found in dataset: {sequence_dir}")


def frame_paths_for_window(dataset: BedlamDataset, seq_idx: int, start_idx: int) -> list[Path]:
    seq_dir, frame_ids = dataset._sequences[seq_idx]
    paths = []
    for step in range(dataset.sequence_length):
        idx = start_idx + step * dataset.stride
        if idx >= len(frame_ids):
            break
        paths.append(seq_dir / "rgb" / f"{frame_ids[idx]}.png")
    return paths


def resolve_target_frame(args: argparse.Namespace, frame_paths: list[Path]) -> int:
    if int(args.target_frame_index) >= 0:
        idx = int(args.target_frame_index)
        if idx >= len(frame_paths):
            raise ValueError(f"target frame index out of range: {idx}, valid=[0,{len(frame_paths) - 1}]")
        return idx
    stem = str(args.target_frame_stem).strip()
    for idx, path in enumerate(frame_paths):
        if path.stem == stem:
            return idx
    raise ValueError(f"target frame stem not found in selected clip: {stem}")


def collect_person_diagnostics(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    frame_idx: int,
    image_size: tuple[int, int],
    input_size: int,
    config: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    width, height = image_size
    valid = batch["smpl_mask"][0, frame_idx].bool() & batch["boxes_mask"][0, frame_idx].bool()
    slots = torch.nonzero(valid, as_tuple=False).flatten()
    if slots.numel() == 0:
        return []

    smpl_model_dir = require_smpl_model_dir(config, args)
    smpl = SMPLLayer(smpl_model_dir).to(device).eval()
    with torch.no_grad():
        base_vertices, base_joints = smpl(
            predictions["pred_poses"][0, frame_idx, slots].reshape(-1, 72),
            predictions["pred_betas"][0, frame_idx, slots],
        )
        hsi_vertices, hsi_joints = smpl(
            predictions["hsi_refined_pred_poses"][0, frame_idx, slots].reshape(-1, 72),
            predictions["hsi_refined_pred_betas"][0, frame_idx, slots],
        )
        gt_pose = rot6d_to_axis_angle(batch["gt_pose_6d"][0, frame_idx, slots]).reshape(-1, 72)
        gt_vertices, gt_joints = smpl(gt_pose, batch["gt_betas"][0, frame_idx, slots])

    base_transl = predictions["pred_transl_cam"][0, frame_idx, slots]
    hsi_transl = predictions["hsi_refined_pred_transl_cam"][0, frame_idx, slots]
    gt_transl = batch["gt_transl_cam"][0, frame_idx, slots]
    base_vertices = base_vertices + base_transl[:, None, :]
    base_joints = base_joints + base_transl[:, None, :]
    hsi_vertices = hsi_vertices + hsi_transl[:, None, :]
    hsi_joints = hsi_joints + hsi_transl[:, None, :]
    gt_vertices = gt_vertices + gt_transl[:, None, :]
    gt_joints = gt_joints + gt_transl[:, None, :]

    _, intrinsics = encoding_to_camera(
        predictions["pose_enc"][:, frame_idx : frame_idx + 1],
        image_size_hw=(input_size, input_size),
        build_intrinsics=True,
    )
    intrinsics_0 = intrinsics[0, 0].to(device=device, dtype=base_joints.dtype)
    scale = base_joints.new_tensor([float(width) / float(input_size), float(height) / float(input_size)])

    base_xy = project_points(base_joints, intrinsics_0) * scale
    hsi_xy = project_points(hsi_joints, intrinsics_0) * scale
    gt_xy = project_points(gt_joints, intrinsics_0) * scale
    base_mask = projected_mask(base_joints, base_xy, width, height)
    hsi_mask = projected_mask(hsi_joints, hsi_xy, width, height)
    gt_mask = projected_mask(gt_joints, gt_xy, width, height)

    persons: list[dict[str, Any]] = []
    confs = predictions["pred_confs"][0, frame_idx, :, 0].detach().float().cpu()
    pred_boxes = predictions["pred_boxes"][0, frame_idx].detach().float().cpu()
    gt_boxes = batch["gt_boxes"][0, frame_idx].detach().float().cpu()
    track_ids = batch.get("gt_track_ids")
    track_quality = batch.get("gt_track_quality")
    for local_idx, slot_tensor in enumerate(slots):
        slot = int(slot_tensor.detach().cpu())
        base_t = torch.linalg.norm(base_transl[local_idx] - gt_transl[local_idx])
        hsi_t = torch.linalg.norm(hsi_transl[local_idx] - gt_transl[local_idx])
        base_mpjpe = torch.linalg.norm(base_joints[local_idx] - gt_joints[local_idx], dim=-1).mean()
        hsi_mpjpe = torch.linalg.norm(hsi_joints[local_idx] - gt_joints[local_idx], dim=-1).mean()
        base_pve = torch.linalg.norm(base_vertices[local_idx] - gt_vertices[local_idx], dim=-1).mean()
        hsi_pve = torch.linalg.norm(hsi_vertices[local_idx] - gt_vertices[local_idx], dim=-1).mean()
        track_id = int(track_ids[0, frame_idx, slot].detach().cpu()) if track_ids is not None else slot
        quality = float(track_quality[0, frame_idx, slot].detach().cpu()) if track_quality is not None else 0.0
        persons.append(
            {
                "slot": slot,
                "query_idx": slot,
                "track_id": track_id,
                "track_quality": quality,
                "pred_conf": float(confs[slot]),
                "gt_box_xyxy": cxcywh_to_xyxy(gt_boxes[slot].tolist(), width, height),
                "pred_box_xyxy": cxcywh_to_xyxy(pred_boxes[slot].tolist(), width, height),
                "base_transl_l2_m": float(base_t.detach().cpu()),
                "hsi_transl_l2_m": float(hsi_t.detach().cpu()),
                "hsi_transl_delta_m": float((hsi_t - base_t).detach().cpu()),
                "base_mpjpe_m": float(base_mpjpe.detach().cpu()),
                "hsi_mpjpe_m": float(hsi_mpjpe.detach().cpu()),
                "base_pve_m": float(base_pve.detach().cpu()),
                "hsi_pve_m": float(hsi_pve.detach().cpu()),
                "base_joints_2d": base_xy[local_idx, :24].detach().cpu().tolist(),
                "hsi_joints_2d": hsi_xy[local_idx, :24].detach().cpu().tolist(),
                "gt_joints_2d": gt_xy[local_idx, :24].detach().cpu().tolist(),
                "base_joints_mask": base_mask[local_idx, :24].detach().cpu().tolist(),
                "hsi_joints_mask": hsi_mask[local_idx, :24].detach().cpu().tolist(),
                "gt_joints_mask": gt_mask[local_idx, :24].detach().cpu().tolist(),
            }
        )
    return persons


def projected_mask(joints: torch.Tensor, xy: torch.Tensor, width: int, height: int) -> torch.Tensor:
    return (
        torch.isfinite(joints).all(dim=-1)
        & torch.isfinite(xy).all(dim=-1)
        & (joints[..., 2] > 1e-4)
        & (xy[..., 0] >= 0)
        & (xy[..., 0] < float(width))
        & (xy[..., 1] >= 0)
        & (xy[..., 1] < float(height))
    )


def cxcywh_to_xyxy(box: list[float], width: int, height: int) -> list[float]:
    cx, cy, bw, bh = [float(v) for v in box]
    return [
        (cx - 0.5 * bw) * float(width),
        (cy - 0.5 * bh) * float(height),
        (cx + 0.5 * bw) * float(width),
        (cy + 0.5 * bh) * float(height),
    ]


def make_panels(original: Image.Image, persons: list[dict[str, Any]], panel_width: int) -> dict[str, Image.Image]:
    panels = {
        "00_original": panel_base(original, panel_width, "Original RGB"),
        "01_gt_boxes": panel_base(original, panel_width, "GT input boxes + predicted boxes"),
        "02_gt_smpl_2d": panel_base(original, panel_width, "GT SMPL 2D projection"),
        "03_base_smpl_2d": panel_base(original, panel_width, "Base SMPL 2D projection"),
        "04_hsi_smpl_2d": panel_base(original, panel_width, "HSI refined SMPL 2D projection"),
        "05_combined": panel_base(original, panel_width, "Combined: GT green, base red, HSI cyan"),
    }
    scale = panel_width / float(original.width)
    for idx, person in enumerate(persons):
        color = COLORS[idx % len(COLORS)]
        draw_boxes(panels["01_gt_boxes"], person, color, scale)
        draw_boxes(panels["02_gt_smpl_2d"], person, color, scale, pred=False)
        draw_boxes(panels["03_base_smpl_2d"], person, color, scale, pred=False)
        draw_boxes(panels["04_hsi_smpl_2d"], person, color, scale, pred=False)
        draw_boxes(panels["05_combined"], person, color, scale, pred=False)
        draw_skeleton(panels["02_gt_smpl_2d"], person["gt_joints_2d"], person["gt_joints_mask"], (64, 255, 96), scale)
        draw_skeleton(panels["03_base_smpl_2d"], person["gt_joints_2d"], person["gt_joints_mask"], (64, 255, 96), scale, width=1, radius=2)
        draw_skeleton(panels["03_base_smpl_2d"], person["base_joints_2d"], person["base_joints_mask"], (255, 64, 64), scale)
        draw_skeleton(panels["04_hsi_smpl_2d"], person["gt_joints_2d"], person["gt_joints_mask"], (64, 255, 96), scale, width=1, radius=2)
        draw_skeleton(panels["04_hsi_smpl_2d"], person["hsi_joints_2d"], person["hsi_joints_mask"], (64, 220, 255), scale)
        draw_skeleton(panels["05_combined"], person["gt_joints_2d"], person["gt_joints_mask"], (64, 255, 96), scale, width=2, radius=2)
        draw_skeleton(panels["05_combined"], person["base_joints_2d"], person["base_joints_mask"], (255, 64, 64), scale, width=2, radius=2)
        draw_skeleton(panels["05_combined"], person["hsi_joints_2d"], person["hsi_joints_mask"], (64, 220, 255), scale, width=2, radius=2)
        draw_person_label(panels["01_gt_boxes"], person, color, scale)
        draw_error_label(panels["03_base_smpl_2d"], person, (255, 64, 64), scale, mode="base")
        draw_error_label(panels["04_hsi_smpl_2d"], person, (64, 220, 255), scale, mode="hsi")
    return panels


def panel_base(original: Image.Image, panel_width: int, title: str) -> Image.Image:
    scale = panel_width / float(original.width)
    panel_height = int(round(original.height * scale))
    image = original.resize((panel_width, panel_height), Image.BILINEAR).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw_label(draw, font, (8, 8), title, (255, 255, 255), fill=(0, 0, 0))
    return image


def make_board(panels: list[Image.Image], persons: list[dict[str, Any]], target_path: Path, panel_width: int) -> Image.Image:
    if not panels:
        raise ValueError("No panels to compose")
    cols = 2
    rows = 3
    panel_h = panels[0].height
    table_h = 36 + max(len(persons), 1) * 20
    board = Image.new("RGB", (cols * panel_width, rows * panel_h + table_h), (18, 18, 18))
    for idx, panel in enumerate(panels):
        x = (idx % cols) * panel_width
        y = (idx // cols) * panel_h
        board.paste(panel, (x, y))
    draw = ImageDraw.Draw(board)
    font = ImageFont.load_default()
    y0 = rows * panel_h + 8
    draw.text((8, y0), f"{target_path.stem} | GT boxes=color, pred boxes=white dashed | GT=green base=red HSI=cyan", fill=(255, 255, 255), font=font)
    y = y0 + 20
    header = "track/query/conf | base_t -> hsi_t | base_mpjpe -> hsi_mpjpe | delta_t"
    draw.text((8, y), header, fill=(220, 220, 220), font=font)
    y += 18
    for idx, person in enumerate(persons):
        color = COLORS[idx % len(COLORS)]
        text = (
            f"{person['track_id']} q{person['query_idx']} c={person['pred_conf']:.2f} | "
            f"{person['base_transl_l2_m']:.3f}->{person['hsi_transl_l2_m']:.3f}m | "
            f"{person['base_mpjpe_m']:.3f}->{person['hsi_mpjpe_m']:.3f}m | "
            f"{person['hsi_transl_delta_m']:+.3f}m"
        )
        draw.text((8, y), text, fill=color, font=font)
        y += 18
    return board


def draw_boxes(panel: Image.Image, person: dict[str, Any], color: tuple[int, int, int], scale: float, pred: bool = True) -> None:
    draw = ImageDraw.Draw(panel)
    gt_box = [float(v) * scale for v in person["gt_box_xyxy"]]
    draw.rectangle(gt_box, outline=color, width=4)
    if pred:
        pred_box = [float(v) * scale for v in person["pred_box_xyxy"]]
        draw_dashed_rectangle(draw, pred_box, outline=(255, 255, 255), width=2)


def draw_person_label(panel: Image.Image, person: dict[str, Any], color: tuple[int, int, int], scale: float) -> None:
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    box = [float(v) * scale for v in person["gt_box_xyxy"]]
    text = f"track={person['track_id']} q={person['query_idx']} conf={person['pred_conf']:.2f}"
    draw_label(draw, font, (box[0], max(0.0, box[1] - 18.0)), text, color)


def draw_error_label(panel: Image.Image, person: dict[str, Any], color: tuple[int, int, int], scale: float, mode: str) -> None:
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    box = [float(v) * scale for v in person["gt_box_xyxy"]]
    if mode == "base":
        text = f"q{person['query_idx']} t={person['base_transl_l2_m']:.2f} mpjpe={person['base_mpjpe_m']:.2f}"
    else:
        text = f"q{person['query_idx']} t={person['hsi_transl_l2_m']:.2f} mpjpe={person['hsi_mpjpe_m']:.2f}"
    draw_label(draw, font, (box[0], min(panel.height - 16.0, box[3] + 4.0)), text, color)


def draw_skeleton(
    panel: Image.Image,
    joints: list[list[float]],
    masks: list[bool],
    color: tuple[int, int, int],
    scale: float,
    width: int = 3,
    radius: int = 3,
) -> None:
    draw = ImageDraw.Draw(panel)
    pts = [(float(xy[0]) * scale, float(xy[1]) * scale) for xy in joints]
    valid = [bool(v) for v in masks]
    for a, b in SMPL_EDGES:
        if a < len(pts) and b < len(pts) and valid[a] and valid[b]:
            draw.line([pts[a], pts[b]], fill=color, width=width)
    for idx, (x, y) in enumerate(pts[:24]):
        if idx >= len(valid) or not valid[idx]:
            continue
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=(0, 0, 0))


def draw_dashed_rectangle(draw: ImageDraw.ImageDraw, box: list[float], outline: tuple[int, int, int], width: int = 1, dash: int = 8) -> None:
    x0, y0, x1, y1 = box
    draw_dashed_line(draw, (x0, y0), (x1, y0), outline, width, dash)
    draw_dashed_line(draw, (x1, y0), (x1, y1), outline, width, dash)
    draw_dashed_line(draw, (x1, y1), (x0, y1), outline, width, dash)
    draw_dashed_line(draw, (x0, y1), (x0, y0), outline, width, dash)


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    fill: tuple[int, int, int],
    width: int,
    dash: int,
) -> None:
    x0, y0 = p0
    x1, y1 = p1
    length = float(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
    if length <= 1e-6:
        return
    steps = max(int(length / float(dash)), 1)
    for idx in range(0, steps, 2):
        t0 = idx / steps
        t1 = min((idx + 1) / steps, 1.0)
        draw.line([(x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0), (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1)], fill=fill, width=width)


def draw_label(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    xy: tuple[float, float],
    text: str,
    outline: tuple[int, int, int],
    fill: tuple[int, int, int] = (32, 32, 32),
) -> None:
    bbox = draw.textbbox(xy, text, font=font)
    pad = 3
    draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=fill, outline=outline, width=1)
    draw.text(xy, text, fill=(255, 255, 255), font=font)


if __name__ == "__main__":
    main()
