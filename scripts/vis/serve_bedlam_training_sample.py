#!/usr/bin/env python3
"""Inspect one processed BEDLAM training window through the production loader.

The target sequence is selected with a one-line manifest, but every data field
and preprocessing option is otherwise constructed from the supplied training
YAML.  Consequently the rendered images, depth, boxes, queries, and SMPL
targets are exactly the tensors seen by the training loop for that window.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.data import BedlamDataset, bedlam_collate_fn  # noqa: E402
from vggt_omega.data.geometry import resolve_image_size_config  # noqa: E402
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.training.config import deep_update, load_yaml_config, require_path  # noqa: E402
from vggt_omega.utils.rotation import rot6d_to_axis_angle  # noqa: E402


PALETTE = (
    (41, 98, 255),
    (239, 71, 111),
    (6, 180, 162),
    (255, 176, 0),
    (131, 90, 241),
    (46, 204, 113),
    (236, 72, 153),
    (14, 165, 233),
)


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args)
    sequence_dir, dataset_root, split, sequence_name = validate_sequence(config, args.sequence_dir)
    manifest_path = output_dir / "selected_sequence_manifest.txt"
    manifest_path.write_text(f"{sequence_name}\n", encoding="utf-8")

    dataset = build_dataset(config, split=split, sequence_manifest=manifest_path)
    if not 0 <= args.window_index < len(dataset):
        raise IndexError(f"--window-index must be in [0, {len(dataset) - 1}], got {args.window_index}")
    loader = DataLoader(
        Subset(dataset, [args.window_index]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=bedlam_collate_fn,
    )
    batch = next(iter(loader))
    validate_batch(batch, config)

    frame_index = select_frame_index(batch, args.frame_offset)
    frame_id = selected_frame_id(dataset, args.window_index, frame_index)
    rendered = render_2d_panels(batch, frame_index, output_dir, frame_id)
    summary = build_summary(
        config=config,
        batch=batch,
        sequence_dir=sequence_dir,
        dataset_root=dataset_root,
        split=split,
        sequence_name=sequence_name,
        dataset_windows=len(dataset),
        window_index=args.window_index,
        frame_index=frame_index,
        frame_id=frame_id,
        rendered=rendered,
    )
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "contact_sheet": str(rendered["contact_sheet"])}, indent=2), flush=True)

    if args.smoke_only:
        print("[ok] BEDLAM training-sample loader and visualization smoke passed", flush=True)
        return

    serve_viser(batch, frame_index, config, args, summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl.yaml")
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", default="outputs/vis/bedlam_training_sample")
    parser.add_argument("--window-index", type=int, default=0, help="Window index within the selected sequence")
    parser.add_argument("--frame-offset", type=int, default=0, help="0 uses the middle frame; negative/positive offsets select around it")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--depth-stride", type=int, default=4)
    parser.add_argument("--max-depth-m", type=float, default=30.0)
    parser.add_argument("--point-size", type=float, default=0.008)
    parser.add_argument("--mesh-opacity", type=float, default=0.35)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--override", action="append", default=[], help="Same dotted key=value form accepted by training")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.train_config))
    for item in args.override:
        if "=" not in item:
            raise ValueError(f"Override must have key=value format, got: {item}")
        dotted_key, raw_value = item.split("=", 1)
        cursor = config
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            child = cursor.get(part)
            if not isinstance(child, dict):
                child = {}
                cursor[part] = child
            cursor = child
        cursor[parts[-1]] = parse_override_value(raw_value)
    if str(config.get("data", {}).get("dataset", "bedlam")).lower() != "bedlam":
        raise ValueError("This viewer only supports data.dataset=bedlam because it visualizes processed BEDLAM sequences.")
    return config


def parse_override_value(value: str) -> Any:
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"none", "null", "~"}:
        return None
    for converter in (int, float):
        try:
            return converter(value)
        except ValueError:
            pass
    return value


def validate_sequence(config: dict[str, Any], sequence_arg: Path) -> tuple[Path, Path, str, str]:
    sequence_dir = resolve_path(sequence_arg)
    data_cfg = config["data"]
    dataset_root = resolve_path(require_path(config, data_cfg.get("root_key", "datasets.bedlam_root")))
    split = str(data_cfg.get("train_split", "Training"))
    split_dir = dataset_root / split
    if not sequence_dir.is_dir() or not (sequence_dir / "rgb").is_dir():
        raise FileNotFoundError(f"BEDLAM sequence must contain rgb/: {sequence_dir}")
    try:
        sequence_name = sequence_dir.relative_to(split_dir).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Sequence must be below the training root configured by this YAML:\n"
            f"  expected root: {split_dir}\n  received: {sequence_dir}"
        ) from exc
    if not sequence_name or sequence_name.startswith("../"):
        raise ValueError(f"Invalid sequence path relative to split: {sequence_dir}")
    return sequence_dir, dataset_root, split, sequence_name


def build_dataset(config: dict[str, Any], split: str, sequence_manifest: Path) -> BedlamDataset:
    data_cfg = config["data"]
    image_size, image_resolution = resolve_image_size_config(data_cfg)
    boxes_root = None
    if data_cfg.get("boxes_root_key"):
        boxes_root = require_path(
            config,
            data_cfg["boxes_root_key"],
            allow_empty=not bool(data_cfg.get("require_boxes", False)),
        )
    contact_teacher_root = str(data_cfg.get("contact_teacher_root", "") or "").strip()
    if not contact_teacher_root and data_cfg.get("contact_teacher_root_key"):
        contact_teacher_root = require_path(
            config,
            data_cfg["contact_teacher_root_key"],
            allow_empty=not bool(data_cfg.get("require_contact_teacher", False)),
        )
    return BedlamDataset(
        root=require_path(config, data_cfg.get("root_key", "datasets.bedlam_root")),
        split=split,
        sequence_length=int(data_cfg["sequence_length"]),
        stride=int(data_cfg["stride"]),
        image_size=image_size,
        image_resolution=image_resolution,
        resize_mode=str(data_cfg.get("resize_mode", "balanced")),
        max_humans=int(data_cfg["max_humans"]),
        require_smpl=bool(data_cfg.get("require_smpl", True)),
        require_depth=bool(data_cfg.get("require_depth", False)),
        boxes_root=boxes_root,
        require_boxes=bool(data_cfg.get("require_boxes", False)),
        box_free_gt_slots=bool(data_cfg.get("box_free_gt_slots", False)),
        query_source=str(data_cfg.get("query_source", "persons")),
        patch_size=int(config.get("model", {}).get("patch_size", 16)),
        mask_patch_threshold=float(data_cfg.get("mask_patch_threshold", 0.10)),
        min_mask_patches=int(data_cfg.get("min_mask_patches", 4)),
        sequence_manifest=sequence_manifest,
        contact_teacher_root=contact_teacher_root or None,
        require_contact_teacher=bool(data_cfg.get("require_contact_teacher", False)),
        contact_only_windows=bool(data_cfg.get("train_contact_only", False)),
    )


def validate_batch(batch: dict[str, torch.Tensor], config: dict[str, Any]) -> None:
    required = {"images", "gt_depth", "K_scal3r", "gt_pose_6d", "gt_betas", "gt_transl_cam", "smpl_mask", "gt_boxes", "boxes_mask"}
    missing = sorted(required - batch.keys())
    if missing:
        raise KeyError(f"Production BEDLAM batch is missing required visual fields: {missing}")
    expected_length = int(config["data"]["sequence_length"])
    if batch["images"].shape[:2] != (1, expected_length):
        raise ValueError(f"Unexpected image batch shape: {tuple(batch['images'].shape)}")
    for key in ("images", "gt_depth", "K_scal3r", "gt_pose_6d", "gt_betas", "gt_transl_cam"):
        if not torch.isfinite(batch[key]).all():
            raise ValueError(f"Non-finite values after training preprocessing: {key}")


def select_frame_index(batch: dict[str, torch.Tensor], frame_offset: int) -> int:
    frames = int(batch["images"].shape[1])
    center = (frames - 1) // 2
    index = center + int(frame_offset)
    if not 0 <= index < frames:
        raise IndexError(f"--frame-offset selects frame {index}, but this training window has {frames} frames")
    return index


def selected_frame_id(dataset: BedlamDataset, window_index: int, frame_index: int) -> str:
    seq_index, start_index = dataset._index[window_index]  # noqa: SLF001 - report exact selected training window
    _, frame_ids = dataset._sequences[seq_index]  # noqa: SLF001 - dataset owns source frame ordering
    return frame_ids[start_index + frame_index * dataset.stride]


def render_2d_panels(batch: dict[str, torch.Tensor], frame: int, output_dir: Path, frame_id: str) -> dict[str, str]:
    image = tensor_to_image(batch["images"][0, frame])
    depth = depth_to_image(batch["gt_depth"][0, frame, 0])
    gt_overlay = draw_boxes(image.copy(), batch, frame, key="gt_boxes", mask_key="boxes_mask", label="gt")
    query_overlay = image.copy()
    if "smpl_query_boxes" in batch and "smpl_query_boxes_mask" in batch:
        query_overlay = draw_boxes(query_overlay, batch, frame, key="smpl_query_boxes", mask_key="smpl_query_boxes_mask", label="query")
    else:
        ImageDraw.Draw(query_overlay).text((8, 8), "No detection query tensors (query_source=persons)", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0), font=ImageFont.load_default())

    panels = (image, depth, gt_overlay, query_overlay)
    for name, panel in zip(("input", "depth", "gt_boxes", "queries"), panels, strict=True):
        panel.save(output_dir / f"{frame_id}_{name}.png")
    width, height = image.size
    sheet = Image.new("RGB", (width * 2, height * 2), "black")
    for idx, panel in enumerate(panels):
        sheet.paste(panel, ((idx % 2) * width, (idx // 2) * height))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), f"training loader window frame: {frame_id}", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0), font=ImageFont.load_default())
    sheet_path = output_dir / f"{frame_id}_contact_sheet.png"
    sheet.save(sheet_path)
    return {"contact_sheet": str(sheet_path), "input": str(output_dir / f"{frame_id}_input.png"), "depth": str(output_dir / f"{frame_id}_depth.png"), "gt_boxes": str(output_dir / f"{frame_id}_gt_boxes.png"), "queries": str(output_dir / f"{frame_id}_queries.png")}


def tensor_to_image(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")


def depth_to_image(depth: torch.Tensor) -> Image.Image:
    values = depth.detach().cpu().numpy().astype(np.float32)
    valid = np.isfinite(values) & (values > 0)
    normalized = np.zeros_like(values, dtype=np.float32)
    if valid.any():
        lo, hi = np.percentile(values[valid], (2.0, 98.0))
        hi = max(float(hi), float(lo) + 1e-6)
        normalized[valid] = np.clip((values[valid] - lo) / (hi - lo), 0.0, 1.0)
    rgb = np.stack((normalized, np.sqrt(normalized), 1.0 - normalized), axis=-1)
    rgb[~valid] = 0
    return Image.fromarray((rgb * 255.0).round().astype(np.uint8), mode="RGB")


def draw_boxes(image: Image.Image, batch: dict[str, torch.Tensor], frame: int, key: str, mask_key: str, label: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    boxes = batch[key][0, frame].detach().cpu()
    valid = batch[mask_key][0, frame].detach().cpu().bool()
    person_ids = batch.get("person_ids")
    for slot in torch.nonzero(valid, as_tuple=False).reshape(-1).tolist():
        cx, cy, box_w, box_h = [float(value) for value in boxes[slot]]
        color = PALETTE[slot % len(PALETTE)]
        xyxy = ((cx - box_w / 2.0) * width, (cy - box_h / 2.0) * height, (cx + box_w / 2.0) * width, (cy + box_h / 2.0) * height)
        draw.rectangle(xyxy, outline=color, width=3)
        suffix = ""
        if label == "gt" and isinstance(person_ids, torch.Tensor):
            suffix = f" id={int(person_ids[0, frame, slot].item())}"
        draw.text((xyxy[0], max(0.0, xyxy[1] - 12.0)), f"{label}[{slot}]{suffix}", fill=color, stroke_width=1, stroke_fill=(0, 0, 0), font=ImageFont.load_default())
    return image


def serve_viser(batch: dict[str, torch.Tensor], frame: int, config: dict[str, Any], args: argparse.Namespace, summary_path: Path) -> None:
    try:
        import viser  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("Viser is required for the interactive view. Run with --smoke-only for 2D output only, or install viser in the server environment.") from exc

    requested_device = torch.device(args.device)
    device = requested_device if requested_device.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    points, colors = depth_points_from_batch(batch, frame, args.depth_stride, args.max_depth_m)
    smpl_vertices, smpl_joints, smpl_faces = decode_gt_smpl(batch, frame, config, device)
    server = viser.ViserServer(port=int(args.port))
    scene = getattr(server, "scene", server)
    if hasattr(scene, "set_up_direction"):
        scene.set_up_direction("-y")
    scene.add_point_cloud(name="training_depth", points=points, colors=colors, point_size=float(args.point_size))
    for index, (vertices, joints) in enumerate(zip(smpl_vertices, smpl_joints, strict=True)):
        color = PALETTE[index % len(PALETTE)]
        scene.add_mesh_simple(name=f"gt_smpl_{index}", vertices=vertices, faces=smpl_faces, color=color, opacity=float(args.mesh_opacity))
        scene.add_point_cloud(name=f"gt_joints_{index}", points=joints, colors=np.repeat(np.asarray(color, dtype=np.uint8)[None], joints.shape[0], axis=0), point_size=0.015)
    scene.add_label(name="title", text="BEDLAM training-loader sample: depth + GT SMPL (camera coordinates)", position=np.zeros(3, dtype=np.float32))
    print(json.dumps({"viewer": f"http://127.0.0.1:{args.port}", "summary": str(summary_path), "depth_points": int(points.shape[0]), "smpl_people": len(smpl_vertices)}, indent=2), flush=True)
    while True:
        time.sleep(3600)


def depth_points_from_batch(batch: dict[str, torch.Tensor], frame: int, stride: int, max_depth: float) -> tuple[np.ndarray, np.ndarray]:
    depth = batch["gt_depth"][0, frame, 0].detach().cpu().numpy().astype(np.float32)
    image = batch["images"][0, frame].detach().cpu().permute(1, 2, 0).numpy()
    intrinsics = batch["K_scal3r"][0, frame].detach().cpu().numpy().astype(np.float32)
    ys, xs = np.mgrid[0:depth.shape[0]:max(1, stride), 0:depth.shape[1]:max(1, stride)]
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z > 1e-6)
    if max_depth > 0:
        valid &= z <= max_depth
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    x = (xs - intrinsics[0, 2]) * z / max(float(intrinsics[0, 0]), 1e-6)
    y = (ys - intrinsics[1, 2]) * z / max(float(intrinsics[1, 1]), 1e-6)
    points = np.stack((x, y, z), axis=1).astype(np.float32)
    colors = (np.clip(image[::max(1, stride), ::max(1, stride)][valid], 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return points, colors


def decode_gt_smpl(batch: dict[str, torch.Tensor], frame: int, config: dict[str, Any], device: torch.device) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    model_dir = require_path(config, "assets.smpl_model_dir", allow_empty=False)
    smpl = SMPLLayer(model_dir).to(device).eval()
    valid = batch["smpl_mask"][0, frame].to(device).bool()
    slots = torch.nonzero(valid, as_tuple=False).reshape(-1)
    if slots.numel() == 0:
        return [], [], np.asarray(smpl.faces, dtype=np.int64)
    pose_6d = batch["gt_pose_6d"][0, frame, slots].to(device).float()
    pose = rot6d_to_axis_angle(pose_6d.reshape(-1, 24, 6)).reshape(-1, 24, 3)
    betas = batch["gt_betas"][0, frame, slots].to(device).float()
    transl = batch["gt_transl_cam"][0, frame, slots].to(device).float()
    with torch.inference_mode():
        vertices, joints = smpl(pose, betas)
    vertices = (vertices + transl[:, None]).detach().cpu().numpy().astype(np.float32)
    joints = (joints + transl[:, None]).detach().cpu().numpy().astype(np.float32)
    return [vertices[i] for i in range(vertices.shape[0])], [joints[i] for i in range(joints.shape[0])], np.asarray(smpl.faces, dtype=np.int64)


def build_summary(**kwargs: Any) -> dict[str, Any]:
    batch = kwargs.pop("batch")
    config = kwargs.pop("config")
    return {
        **kwargs,
        "data_config": config["data"],
        "model_patch_size": int(config.get("model", {}).get("patch_size", 16)),
        "batch_tensor_shapes": {key: list(value.shape) for key, value in batch.items() if isinstance(value, torch.Tensor)},
        "gt_people_in_selected_frame": int(batch["smpl_mask"][0, kwargs["frame_index"]].sum().item()),
        "gt_boxes_in_selected_frame": int(batch["boxes_mask"][0, kwargs["frame_index"]].sum().item()),
        "query_boxes_in_selected_frame": int(batch.get("smpl_query_boxes_mask", torch.zeros(1))[0, kwargs["frame_index"]].sum().item()) if "smpl_query_boxes_mask" in batch else 0,
    }


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


if __name__ == "__main__":
    main()
