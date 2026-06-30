#!/usr/bin/env python3
"""Create real-data paper-figure elements for HSI local scene probing.

This script runs the project model on a single image, reconstructs the same
anchor/projection/depth-probe quantities used by HSIRefinementHead, and exports
small visual elements that can be composed into a paper architecture figure.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

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

from scripts.train.train_smpl import apply_overrides, build_model, load_yaml_config  # noqa: E402
from scripts.vis.create_arch_patch_pooling_elements import (  # noqa: E402
    auto_sam2_person_mask,
    resize_mask,
    save_mask_artifacts,
)
from scripts.vis.visualize_smpl_inference import (  # noqa: E402
    extract_state_dict,
    load_image,
    load_training_checkpoint,
)
from vggt_omega.models.heads.hsi_refinement_head import (  # noqa: E402
    _canonical_depth,
    _estimate_depth_normals,
    _flatten_intrinsics,
    _project_points,
    _scale_points_to_depth,
    _unproject_pixels,
)
from vggt_omega.training.config import deep_update, require_path  # noqa: E402


PALETTE = [
    {"person": (220, 64, 64), "light": (255, 205, 205), "scene": (37, 99, 180), "probe": (42, 168, 107)},
    {"person": (37, 99, 180), "light": (197, 225, 255), "scene": (220, 64, 64), "probe": (42, 168, 107)},
    {"person": (47, 125, 72), "light": (216, 242, 206), "scene": (126, 72, 168), "probe": (42, 168, 107)},
]
GRID = (203, 213, 225)
WHITE = (255, 255, 255)
PANEL = (248, 250, 252)
MUTED = (102, 112, 133)
RESIDUAL = (220, 64, 64)
NORMAL = (26, 166, 166)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Input RGB image.")
    parser.add_argument("--checkpoint", default="", help="Training checkpoint. Defaults to checkpoint.resume in train config.")
    parser.add_argument("--baseline-checkpoint", default="", help="Override checkpoints.vggt_baseline.")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_hsi_after_translation_ray_refine.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vis/paper_hsi_local_probe_real_elements"))
    parser.add_argument("--device", default="")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--person-index", type=int, default=-1, help="-1 exports all selected query priors/top predictions.")
    parser.add_argument("--anchor-index", type=int, default=-1, help="-1 chooses the visible anchor with max depth residual.")
    parser.add_argument("--top-k", type=int, default=2, help="Number of people to export when --person-index=-1.")
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument("--auto-person-prior", action="store_true", help="Use project YOLO+SAM2 to create query priors.")
    parser.add_argument("--auto-top-k", type=int, default=2, help="Top detected people used as query priors.")
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-iou", type=float, default=0.50)
    parser.add_argument("--detector-image-size", type=int, default=1024)
    parser.add_argument("--det-half", action="store_true")
    parser.add_argument("--yolo-checkpoint", default=None)
    parser.add_argument("--sam2-root", default=None)
    parser.add_argument("--sam2-checkpoint", default=None)
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-single-mask", action="store_true")
    parser.add_argument("--patch-mask-min-overlap", type=float, default=0.02)
    parser.add_argument("--side-view-window", type=int, default=13)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args)
    checkpoint = resolve_checkpoint(args, config)
    image_path = resolve_project_path(args.image)
    input_size = int(config.get("data", {}).get("image_size", args.image_size))
    patch_size = int(config.get("model", {}).get("patch_size", 16))

    priors = build_query_priors(args, image_path, input_size, patch_size, config, device, output_dir)

    model = build_model(config).to(device)
    load_vggt_baseline_for_camera(model, config, device)
    load_training_checkpoint(model, checkpoint, device)
    model.eval()

    image_tensor, orig_image = load_image(image_path, input_size)
    with torch.no_grad():
        predictions = model(
            image_tensor.to(device),
            smpl_query_boxes=priors["boxes"],
            smpl_query_boxes_mask=priors["mask"],
            smpl_query_patch_masks=priors["patch_masks"],
        )

    rgb = orig_image.resize((input_size, input_size), Image.BILINEAR).convert("RGB")
    probe = compute_hsi_probe(predictions, model, input_size)
    query_indices = choose_query_indices(predictions, priors, args)

    files: list[str] = []
    people_meta = []
    for out_rank, query_idx in enumerate(query_indices):
        anchor_idx = choose_anchor_index(probe, query_idx, args.anchor_index)
        prefix = f"person{out_rank}_q{query_idx}_a{anchor_idx}"
        colors = PALETTE[out_rank % len(PALETTE)]
        meta = export_person_assets(
            output_dir=output_dir,
            prefix=prefix,
            rgb=rgb,
            predictions=predictions,
            probe=probe,
            query_idx=query_idx,
            anchor_idx=anchor_idx,
            patch_size=patch_size,
            colors=colors,
            side_view_window=args.side_view_window,
        )
        files.extend(meta["files"])
        people_meta.append(meta["metadata"])

    manifest = {
        "purpose": "Real-data HSI local scene probing/body-scene residual elements for paper architecture figures.",
        "image": str(image_path),
        "checkpoint": str(checkpoint),
        "train_config": str(args.train_config),
        "path_config": str(args.path_config),
        "auto_person_prior": priors["meta"],
        "model_outputs_used": [
            "pred_pose_6d",
            "pred_betas",
            "pred_transl_cam",
            "pred_confs",
            "pred_boxes",
            "depth",
            "pose_enc",
            "hsi_anchor_depth_residual",
            "hsi_scene_scale",
            "hsi_scene_depth_bias",
        ],
        "hsi_logic": [
            "SMPL base pose/betas/translation -> 24 body anchors",
            "VGGT pose_enc -> intrinsics",
            "anchor projection -> depth pixel and 3x3 patch-token window",
            "VGGT depth at projected pixel -> scene xyz",
            "offset = scene xyz - anchor xyz",
            "depth_residual = scene_z - anchor_z",
        ],
        "people": people_meta,
        "files": files,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "files": files, "num_people": len(people_meta)}, indent=2))


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.train_config))
    config = apply_overrides(config, args.override)
    if args.baseline_checkpoint:
        config.setdefault("checkpoints", {})["vggt_baseline"] = args.baseline_checkpoint
    model_cfg = config.setdefault("model", {})
    model_cfg["enable_camera"] = True
    model_cfg["enable_depth"] = True
    model_cfg["enable_smpl"] = True
    model_cfg["enable_hsi_refine"] = True
    if args.auto_person_prior:
        model_cfg["smpl_query_box_prior"] = True
        model_cfg["smpl_query_patch_pool"] = True
        model_cfg["smpl_query_patch_pool_mode"] = "mask_intersection"
    config.setdefault("data", {})["image_size"] = int(config.get("data", {}).get("image_size", args.image_size))
    return config


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
        for marker in ("vggt-omega", "vggt-human"):
            if marker in path.parts:
                idx = path.parts.index(marker)
                candidates.append(ROOT / Path(*path.parts[idx + 1 :]))
                break
    else:
        candidates.append(ROOT / path)
        candidates.append(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_checkpoint(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    raw = args.checkpoint or str(config.get("checkpoint", {}).get("resume", ""))
    if not raw:
        raise ValueError("No checkpoint provided and checkpoint.resume is empty in train config.")
    return resolve_project_path(raw)


def load_vggt_baseline_for_camera(model: torch.nn.Module, config: dict[str, Any], device: torch.device) -> None:
    checkpoint_path = require_path(config, "checkpoints.vggt_baseline", allow_empty=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[ckpt] loaded VGGT baseline for camera/depth: {checkpoint_path}")
    print(f"[ckpt] baseline missing={len(missing)} unexpected={len(unexpected)}")


def build_query_priors(
    args: argparse.Namespace,
    image_path: Path,
    input_size: int,
    patch_size: int,
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    max_queries = int(config.get("model", {}).get("num_smpl_queries", config.get("data", {}).get("max_humans", 20)))
    if not args.auto_person_prior:
        return {"boxes": None, "mask": None, "patch_masks": None, "valid_indices": None, "meta": {"enabled": False}}

    auto_args = SimpleNamespace(
        image=image_path,
        path_config=args.path_config,
        yolo_checkpoint=args.yolo_checkpoint,
        sam2_root=args.sam2_root,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_model_cfg=args.sam2_model_cfg,
        sam2_single_mask=args.sam2_single_mask,
        device=str(device),
        detector_image_size=args.detector_image_size,
        det_conf=args.det_conf,
        det_iou=args.det_iou,
        det_half=args.det_half,
        auto_top_k=args.auto_top_k,
        auto_person_index=0,
    )
    pixel_mask, boxes, auto_meta, instance_masks = auto_sam2_person_mask(auto_args)
    save_mask_artifacts(output_dir, pixel_mask, auto_meta, "real_auto_sam2_mask_original", instance_masks=instance_masks)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    selected = list(range(min(len(boxes), max_queries)))

    boxes_tensor = torch.zeros(1, 1, max_queries, 4, dtype=torch.float32, device=device)
    mask_tensor = torch.zeros(1, 1, max_queries, dtype=torch.bool, device=device)
    grid_h = input_size // patch_size
    grid_w = input_size // patch_size
    patch_tensor = torch.zeros(1, 1, max_queries, grid_h * grid_w, dtype=torch.bool, device=device)

    resized_masks = []
    for idx in selected:
        box = boxes[idx].clipped(width, height)
        cx = ((box.x1 + box.x2) * 0.5) / max(float(width), 1.0)
        cy = ((box.y1 + box.y2) * 0.5) / max(float(height), 1.0)
        bw = (box.x2 - box.x1) / max(float(width), 1.0)
        bh = (box.y2 - box.y1) / max(float(height), 1.0)
        boxes_tensor[0, 0, idx] = torch.tensor([cx, cy, bw, bh], dtype=torch.float32, device=device).clamp(0.0, 1.0)
        mask_tensor[0, 0, idx] = True
        key = sorted(instance_masks.keys())[idx]
        resized = resize_mask(np.asarray(instance_masks[key]).astype(bool), (input_size, input_size))
        resized_masks.append(resized)
        patch_tensor[0, 0, idx] = torch.from_numpy(mask_to_patch_grid(resized, grid_h, grid_w, patch_size, args.patch_mask_min_overlap)).to(device)

    resized_union = np.logical_or.reduce(resized_masks) if resized_masks else np.zeros((input_size, input_size), dtype=bool)
    resized_instances = {f"person_auto_{idx}": mask for idx, mask in zip(selected, resized_masks, strict=False)}
    save_mask_artifacts(
        output_dir,
        resized_union,
        auto_meta,
        "real_auto_sam2_mask_resized",
        instance_masks=resized_instances,
    )
    auto_meta["query_prior"] = {
        "max_queries": max_queries,
        "valid_query_indices": selected,
        "patch_grid_hw": [grid_h, grid_w],
        "patch_pool_mode_for_visualization": "mask_intersection",
    }
    return {
        "boxes": boxes_tensor,
        "mask": mask_tensor,
        "patch_masks": patch_tensor,
        "valid_indices": selected,
        "meta": auto_meta,
    }


def mask_to_patch_grid(mask: np.ndarray, grid_h: int, grid_w: int, patch_size: int, min_overlap: float) -> np.ndarray:
    selected = np.zeros((grid_h, grid_w), dtype=bool)
    for row in range(grid_h):
        for col in range(grid_w):
            crop = mask[row * patch_size : (row + 1) * patch_size, col * patch_size : (col + 1) * patch_size]
            selected[row, col] = crop.size > 0 and float(crop.mean()) >= float(min_overlap)
    return selected.reshape(-1)


def compute_hsi_probe(predictions: dict[str, torch.Tensor], model: torch.nn.Module, input_size: int) -> dict[str, torch.Tensor]:
    if getattr(model, "hsi_refinement_head", None) is None:
        raise ValueError("Model was built without hsi_refinement_head.")
    required = ["pred_pose_6d", "pred_betas", "pred_transl_cam", "depth", "pose_enc"]
    for key in required:
        if key not in predictions:
            raise ValueError(f"Model predictions missing {key}.")
    head = model.hsi_refinement_head
    pose6d = predictions["pred_pose_6d"].float()
    betas = predictions["pred_betas"].float()
    transl = predictions["pred_transl_cam"].float()
    depth_hw = _canonical_depth(predictions["depth"]).float()
    height, width = depth_hw.shape[-2:]
    intrinsics = _flatten_intrinsics(predictions["pose_enc"], input_size).to(device=pose6d.device, dtype=pose6d.dtype)
    anchors = head._anchors_cam(pose6d, betas, transl)
    batch_size, num_frames, num_queries, _, _ = anchors.shape
    flat_anchors = anchors.reshape(batch_size * num_frames * num_queries, 24, 3)
    projected = _project_points(flat_anchors, intrinsics.repeat_interleave(num_queries, dim=0)).reshape(
        batch_size, num_frames, num_queries, 24, 2
    )
    projected_depth = _scale_points_to_depth(projected, input_size, height, width)
    px = projected_depth[..., 0].round().long().clamp(0, width - 1)
    py = projected_depth[..., 1].round().long().clamp(0, height - 1)
    frame_idx = torch.arange(batch_size * num_frames, device=pose6d.device).reshape(batch_size, num_frames, 1, 1)
    frame_idx = frame_idx.expand(-1, -1, num_queries, 24)
    flat_depth = depth_hw.reshape(batch_size * num_frames, height, width)
    z_scene = flat_depth[frame_idx.reshape(-1), py.reshape(-1), px.reshape(-1)].reshape(batch_size, num_frames, num_queries, 24)
    scene_points = _unproject_pixels(projected[..., 0], projected[..., 1], z_scene, intrinsics, num_queries)
    normals_hw = _estimate_depth_normals(depth_hw, intrinsics, height, width)
    scene_normals = normals_hw.reshape(batch_size * num_frames, height, width, 3)[
        frame_idx.reshape(-1), py.reshape(-1), px.reshape(-1)
    ].reshape(batch_size, num_frames, num_queries, 24, 3)
    offset = scene_points - anchors
    distance = torch.linalg.norm(offset, dim=-1)
    depth_residual = scene_points[..., 2] - anchors[..., 2]
    grid_h = height // 16
    grid_w = width // 16
    center_x = (projected_depth[..., 0] / float(width) * float(grid_w)).floor().long().clamp(0, grid_w - 1)
    center_y = (projected_depth[..., 1] / float(height) * float(grid_h)).floor().long().clamp(0, grid_h - 1)
    return {
        "anchors": anchors.detach(),
        "projected": projected.detach(),
        "projected_depth": projected_depth.detach(),
        "scene_points": scene_points.detach(),
        "scene_normals": scene_normals.detach(),
        "offset": offset.detach(),
        "distance": distance.detach(),
        "depth_residual": depth_residual.detach(),
        "depth_hw": depth_hw.detach(),
        "intrinsics": intrinsics.detach(),
        "patch_center_x": center_x.detach(),
        "patch_center_y": center_y.detach(),
        "patch_grid_hw": torch.tensor([grid_h, grid_w], device=pose6d.device),
        "image_size": torch.tensor(int(input_size), device=pose6d.device),
    }


def choose_query_indices(predictions: dict[str, torch.Tensor], priors: dict[str, Any], args: argparse.Namespace) -> list[int]:
    valid_indices = priors.get("valid_indices")
    if valid_indices:
        candidates = list(valid_indices)
    else:
        confs = predictions["pred_confs"][0, 0, :, 0].detach().float().cpu()
        candidates = [int(i) for i in torch.argsort(confs, descending=True).tolist() if float(confs[i]) >= args.conf_threshold]
    if args.person_index >= 0:
        if args.person_index >= len(candidates):
            raise IndexError(f"--person-index {args.person_index} out of range for {len(candidates)} candidates")
        return [int(candidates[args.person_index])]
    return [int(i) for i in candidates[: max(args.top_k, 1)]]


def choose_anchor_index(probe: dict[str, torch.Tensor], query_idx: int, requested: int) -> int:
    if requested >= 0:
        return int(requested)
    projected = probe["projected_depth"][0, 0, query_idx]
    depth = probe["depth_hw"][0, 0]
    height, width = depth.shape[-2:]
    anchors = probe["anchors"][0, 0, query_idx]
    residual = probe["depth_residual"][0, 0, query_idx].detach().float()
    valid = (
        torch.isfinite(residual)
        & torch.isfinite(anchors[:, 2])
        & (anchors[:, 2] > 1e-5)
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
    )
    if not bool(valid.any()):
        return 23
    score = torch.where(valid, residual.abs(), torch.full_like(residual, -1.0))
    return int(torch.argmax(score).detach().cpu())


def export_person_assets(
    output_dir: Path,
    prefix: str,
    rgb: Image.Image,
    predictions: dict[str, torch.Tensor],
    probe: dict[str, torch.Tensor],
    query_idx: int,
    anchor_idx: int,
    patch_size: int,
    colors: dict[str, tuple[int, int, int]],
    side_view_window: int,
) -> dict[str, Any]:
    depth = probe["depth_hw"][0, 0].detach().float().cpu().numpy()
    depth_img = depth_to_rgba(depth)
    box = predicted_box_pixels(predictions, query_idx, rgb.size)
    out_files = []

    depth_path = output_dir / f"05a_real_depth_patch_window_{prefix}.png"
    create_depth_patch_window(depth_img, probe, query_idx, anchor_idx, patch_size, colors).save(depth_path)
    out_files.append(depth_path.name)

    project_path = output_dir / f"05b_real_anchor_project_to_depth_{prefix}.png"
    create_anchor_projection(rgb, depth_img, probe, query_idx, anchor_idx, box, colors).save(project_path)
    out_files.append(project_path.name)

    composite_path = output_dir / f"05c_real_local_scene_probe_{prefix}.png"
    create_local_probe_composite(rgb, depth_img, probe, query_idx, anchor_idx, patch_size, box, colors).save(composite_path)
    out_files.append(composite_path.name)

    residual_path = output_dir / f"06a_real_body_scene_residual_{prefix}.png"
    create_body_scene_residual(probe, query_idx, anchor_idx, colors, side_view_window).save(residual_path)
    out_files.append(residual_path.name)

    metadata = build_person_metadata(predictions, probe, query_idx, anchor_idx, box)
    meta_path = output_dir / f"real_probe_values_{prefix}.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    out_files.append(meta_path.name)
    return {"files": out_files, "metadata": metadata}


def depth_to_rgba(depth: np.ndarray) -> Image.Image:
    valid = np.isfinite(depth) & (depth > 1e-6)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2.0, 98.0])
    else:
        lo, hi = 0.0, 1.0
    norm = np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    stops = np.array(
        [
            [245, 250, 255],
            [190, 220, 250],
            [94, 151, 246],
            [37, 99, 180],
            [31, 41, 55],
        ],
        dtype=np.float32,
    )
    pos = norm * (len(stops) - 1)
    idx = np.floor(pos).astype(np.int32).clip(0, len(stops) - 2)
    frac = (pos - idx)[..., None]
    rgb = ((1.0 - frac) * stops[idx] + frac * stops[idx + 1]).clip(0, 255).astype(np.uint8)
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")


def create_depth_patch_window(
    depth_img: Image.Image,
    probe: dict[str, torch.Tensor],
    query_idx: int,
    anchor_idx: int,
    patch_size: int,
    colors: dict[str, tuple[int, int, int]],
) -> Image.Image:
    out = depth_img.copy().convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    width, height = out.size
    del patch_size
    grid_h, grid_w = [int(v) for v in probe["patch_grid_hw"].detach().cpu().tolist()]
    cell_w = float(width) / max(float(grid_w), 1.0)
    cell_h = float(height) / max(float(grid_h), 1.0)
    for col in range(grid_w + 1):
        x = col * cell_w
        draw.line([(x, 0), (x, height)], fill=(*GRID, 130), width=1)
    for row in range(grid_h + 1):
        y = row * cell_h
        draw.line([(0, y), (width, y)], fill=(*GRID, 130), width=1)
    cx = int(probe["patch_center_x"][0, 0, query_idx, anchor_idx].detach().cpu())
    cy = int(probe["patch_center_y"][0, 0, query_idx, anchor_idx].detach().cpu())
    radius = 1
    for row in range(max(0, cy - radius), min(grid_h, cy + radius + 1)):
        for col in range(max(0, cx - radius), min(grid_w, cx + radius + 1)):
            rect = [col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h]
            draw.rectangle(rect, fill=(*colors["probe"], 82), outline=(*colors["probe"], 245), width=2)
    px, py = probe["projected_depth"][0, 0, query_idx, anchor_idx].detach().float().cpu().tolist()
    draw_dot(draw, (px, py), colors["scene"], radius=7)
    return out


def create_anchor_projection(
    rgb: Image.Image,
    depth_img: Image.Image,
    probe: dict[str, torch.Tensor],
    query_idx: int,
    anchor_idx: int,
    box: list[float] | None,
    colors: dict[str, tuple[int, int, int]],
) -> Image.Image:
    width, height = rgb.size
    out = Image.new("RGBA", (width * 2 + 48, height), (255, 255, 255, 0))
    left = fade_rgb(rgb, 0.60).convert("RGBA")
    right = depth_img.copy().convert("RGBA")
    out.alpha_composite(left, (0, 0))
    out.alpha_composite(right, (width + 48, 0))
    draw = ImageDraw.Draw(out, "RGBA")
    if box is not None:
        draw.rectangle(box, outline=(*colors["person"], 235), width=3)
    px, py = probe["projected_depth"][0, 0, query_idx, anchor_idx].detach().float().cpu().tolist()
    left_pt = (px, py)
    right_pt = (px + width + 48, py)
    draw_dot(draw, left_pt, colors["person"], radius=8)
    draw_dot(draw, right_pt, colors["scene"], radius=8)
    draw_arrow(draw, (left_pt[0] + 10, left_pt[1]), (right_pt[0] - 10, right_pt[1]), colors["probe"], width=4)
    draw_depth_window_on(draw, right_pt, colors)
    return out


def create_local_probe_composite(
    rgb: Image.Image,
    depth_img: Image.Image,
    probe: dict[str, torch.Tensor],
    query_idx: int,
    anchor_idx: int,
    patch_size: int,
    box: list[float] | None,
    colors: dict[str, tuple[int, int, int]],
) -> Image.Image:
    width, height = rgb.size
    out = Image.new("RGBA", (width * 2 + 48, height), (255, 255, 255, 0))
    left = fade_rgb(rgb, 0.50).convert("RGBA")
    right = create_depth_patch_window(depth_img, probe, query_idx, anchor_idx, patch_size, colors)
    out.alpha_composite(left, (0, 0))
    out.alpha_composite(right, (width + 48, 0))
    draw = ImageDraw.Draw(out, "RGBA")
    if box is not None:
        draw.rectangle(box, outline=(*colors["person"], 230), width=3)
    projected = probe["projected_depth"][0, 0, query_idx, anchor_idx].detach().float().cpu().tolist()
    left_pt = (projected[0], projected[1])
    right_pt = (projected[0] + width + 48, projected[1])
    draw_dot(draw, left_pt, colors["person"], radius=8)
    draw_arrow(draw, (left_pt[0] + 12, left_pt[1]), (right_pt[0] - 12, right_pt[1]), colors["probe"], width=3)
    return out


def create_body_scene_residual(
    probe: dict[str, torch.Tensor],
    query_idx: int,
    anchor_idx: int,
    colors: dict[str, tuple[int, int, int]],
    window: int,
) -> Image.Image:
    canvas_w, canvas_h = 760, 440
    out = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(out, "RGBA")
    draw.rounded_rectangle([12, 12, canvas_w - 12, canvas_h - 12], radius=14, fill=(*PANEL, 215), outline=(*GRID, 210), width=2)

    anchor = probe["anchors"][0, 0, query_idx, anchor_idx].detach().float().cpu().numpy()
    scene = probe["scene_points"][0, 0, query_idx, anchor_idx].detach().float().cpu().numpy()
    normal = probe["scene_normals"][0, 0, query_idx, anchor_idx].detach().float().cpu().numpy()
    local_points = local_depth_side_points(probe, query_idx, anchor_idx, window)
    all_points = np.vstack([local_points, anchor[None], scene[None]]) if len(local_points) else np.vstack([anchor[None], scene[None]])
    x = all_points[:, 0]
    z = all_points[:, 2]
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    z_min, z_max = float(np.nanmin(z)), float(np.nanmax(z))
    if abs(x_max - x_min) < 1e-4:
        x_min -= 0.5
        x_max += 0.5
    if abs(z_max - z_min) < 1e-4:
        z_min -= 0.5
        z_max += 0.5
    pad_x = 0.18 * (x_max - x_min)
    pad_z = 0.22 * (z_max - z_min)
    x_min -= pad_x
    x_max += pad_x
    z_min -= pad_z
    z_max += pad_z

    def to_canvas(point: np.ndarray) -> tuple[float, float]:
        cx = 78 + (float(point[0]) - x_min) / max(x_max - x_min, 1e-6) * (canvas_w - 156)
        cy = canvas_h - 72 - (float(point[2]) - z_min) / max(z_max - z_min, 1e-6) * (canvas_h - 144)
        return cx, cy

    if len(local_points) >= 2:
        surface = [to_canvas(p) for p in local_points]
        draw.line(surface, fill=(*colors["scene"], 205), width=4, joint="curve")
        for p in surface[:: max(1, len(surface) // 12)]:
            draw_dot(draw, p, colors["scene"], radius=4, alpha=180)

    anchor_xy = to_canvas(anchor)
    scene_xy = to_canvas(scene)
    draw_arrow(draw, anchor_xy, scene_xy, RESIDUAL, width=4)
    draw_dashed_polyline(draw, [anchor_xy, (anchor_xy[0], scene_xy[1]), scene_xy], (*MUTED, 180), width=2)
    normal_tip_3d = scene + normal / max(float(np.linalg.norm(normal)), 1e-6) * max(0.08, 0.18 * (z_max - z_min))
    normal_xy = to_canvas(normal_tip_3d)
    draw_arrow(draw, scene_xy, normal_xy, NORMAL, width=3)
    draw_dot(draw, anchor_xy, colors["person"], radius=11)
    draw_dot(draw, scene_xy, colors["scene"], radius=11)
    return out


def local_depth_side_points(probe: dict[str, torch.Tensor], query_idx: int, anchor_idx: int, window: int) -> np.ndarray:
    depth = probe["depth_hw"][0, 0]
    intrinsics = probe["intrinsics"]
    projected = probe["projected_depth"][0, 0, query_idx, anchor_idx]
    height, width = depth.shape[-2:]
    radius = max(int(window), 3) // 2
    py = int(round(float(projected[1].detach().cpu())))
    px = int(round(float(projected[0].detach().cpu())))
    xs = torch.arange(px - radius, px + radius + 1, device=depth.device).clamp(0, width - 1)
    ys = torch.full_like(xs, max(0, min(height - 1, py)))
    z = depth[ys, xs]
    image_size = float(probe["image_size"].detach().cpu())
    image_x = xs.to(dtype=depth.dtype) * (image_size / float(width))
    image_y = ys.to(dtype=depth.dtype) * (image_size / float(height))
    intr = intrinsics.reshape(-1, 3, 3)[0].to(device=depth.device, dtype=depth.dtype)
    fx = intr[0, 0].clamp(min=1e-6)
    fy = intr[1, 1].clamp(min=1e-6)
    cx = intr[0, 2]
    cy = intr[1, 2]
    points = torch.stack([(image_x - cx) / fx * z, (image_y - cy) / fy * z, z], dim=-1)
    points = points.detach().float().cpu().numpy()
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-6)
    return points[valid]


def predicted_box_pixels(predictions: dict[str, torch.Tensor], query_idx: int, image_size: tuple[int, int]) -> list[float] | None:
    if "pred_boxes" not in predictions:
        return None
    width, height = image_size
    box = predictions["pred_boxes"][0, 0, query_idx].detach().float().cpu().clamp(0.0, 1.0).tolist()
    cx, cy, bw, bh = [float(v) for v in box]
    return [
        (cx - 0.5 * bw) * width,
        (cy - 0.5 * bh) * height,
        (cx + 0.5 * bw) * width,
        (cy + 0.5 * bh) * height,
    ]


def build_person_metadata(
    predictions: dict[str, torch.Tensor],
    probe: dict[str, torch.Tensor],
    query_idx: int,
    anchor_idx: int,
    box: list[float] | None,
) -> dict[str, Any]:
    def tensor_list(t: torch.Tensor) -> list[float]:
        return [float(v) for v in t.detach().float().cpu().reshape(-1).tolist()]

    conf = float(predictions["pred_confs"][0, 0, query_idx, 0].detach().float().cpu()) if "pred_confs" in predictions else None
    data = {
        "query_index": int(query_idx),
        "anchor_index": int(anchor_idx),
        "confidence": conf,
        "pred_box_xyxy_input_pixels": box,
        "anchor_cam_xyz": tensor_list(probe["anchors"][0, 0, query_idx, anchor_idx]),
        "projected_uv_input": tensor_list(probe["projected"][0, 0, query_idx, anchor_idx]),
        "projected_uv_depth": tensor_list(probe["projected_depth"][0, 0, query_idx, anchor_idx]),
        "patch_center_xy": [
            int(probe["patch_center_x"][0, 0, query_idx, anchor_idx].detach().cpu()),
            int(probe["patch_center_y"][0, 0, query_idx, anchor_idx].detach().cpu()),
        ],
        "scene_cam_xyz": tensor_list(probe["scene_points"][0, 0, query_idx, anchor_idx]),
        "scene_normal": tensor_list(probe["scene_normals"][0, 0, query_idx, anchor_idx]),
        "offset_scene_minus_anchor_xyz": tensor_list(probe["offset"][0, 0, query_idx, anchor_idx]),
        "distance_m": float(probe["distance"][0, 0, query_idx, anchor_idx].detach().float().cpu()),
        "depth_residual_m": float(probe["depth_residual"][0, 0, query_idx, anchor_idx].detach().float().cpu()),
    }
    for key in ("hsi_anchor_depth_residual", "hsi_scene_scale", "hsi_scene_depth_bias", "hsi_refine_gate"):
        if key in predictions:
            value = predictions[key][0, 0]
            if key == "hsi_anchor_depth_residual":
                data[key] = float(value[query_idx, anchor_idx, 0].detach().float().cpu())
            elif key == "hsi_refine_gate":
                data[key] = float(value[query_idx, 0].detach().float().cpu())
            else:
                data[key] = float(value.reshape(-1)[0].detach().float().cpu())
    return data


def fade_rgb(image: Image.Image, alpha: float) -> Image.Image:
    white = Image.new("RGB", image.size, WHITE)
    return Image.blend(white, image.convert("RGB"), alpha)


def draw_dot(draw: ImageDraw.ImageDraw, xy: tuple[float, float], color: tuple[int, int, int], radius: int = 6, alpha: int = 255) -> None:
    x, y = xy
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(*color, alpha), outline=(*WHITE, min(alpha, 245)), width=2)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    width: int = 3,
    alpha: int = 235,
) -> None:
    draw.line([start, end], fill=(*color, alpha), width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max(float(np.hypot(dx, dy)), 1e-6)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head = 13.0
    base = (end[0] - ux * head, end[1] - uy * head)
    p1 = (base[0] + px * head * 0.45, base[1] + py * head * 0.45)
    p2 = (base[0] - px * head * 0.45, base[1] - py * head * 0.45)
    draw.polygon([end, p1, p2], fill=(*color, alpha))


def draw_dashed_polyline(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color: tuple[int, int, int, int], width: int = 2) -> None:
    for start, end in zip(points[:-1], points[1:], strict=True):
        length = max(float(np.hypot(end[0] - start[0], end[1] - start[1])), 1e-6)
        steps = max(2, int(length // 10))
        for idx in range(steps):
            if idx % 2:
                continue
            t0 = idx / steps
            t1 = min((idx + 0.62) / steps, 1.0)
            p0 = (start[0] + (end[0] - start[0]) * t0, start[1] + (end[1] - start[1]) * t0)
            p1 = (start[0] + (end[0] - start[0]) * t1, start[1] + (end[1] - start[1]) * t1)
            draw.line([p0, p1], fill=color, width=width)


def draw_depth_window_on(draw: ImageDraw.ImageDraw, center: tuple[float, float], colors: dict[str, tuple[int, int, int]]) -> None:
    cx, cy = center
    cell = 16
    for row in range(-1, 2):
        for col in range(-1, 2):
            x1 = cx + col * cell - cell / 2
            y1 = cy + row * cell - cell / 2
            draw.rectangle([x1, y1, x1 + cell, y1 + cell], outline=(*colors["probe"], 230), fill=(*colors["probe"], 58), width=2)


if __name__ == "__main__":
    main()
