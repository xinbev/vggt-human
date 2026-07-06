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
import math
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
from scripts.vis.create_hsi_paper_ply_elements import (  # noqa: E402
    MeshBuilder,
    add_arrow as add_ply_arrow,
    add_uv_sphere,
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
from vggt_omega.utils.rotation import rot6d_to_axis_angle  # noqa: E402


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
    parser.add_argument("--person-index", type=int, default=-1, help="Index after person selection candidates; overrides --person-select when >=0.")
    parser.add_argument("--person-select", choices=("rightmost", "leftmost", "confidence", "all"), default="all")
    parser.add_argument("--anchor-index", type=int, default=-1, help="-1 chooses according to --anchor-mode.")
    parser.add_argument("--anchor-mode", choices=("foot", "max_residual", "nearest", "full_body"), default="foot")
    parser.add_argument("--top-k", type=int, default=2, help="Number of people to export when --person-select=all/confidence.")
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument("--auto-person-prior", action="store_true", help="Use project YOLO+SAM2 to create query priors.")
    parser.add_argument("--auto-top-k", type=int, default=2, help="Top detected people used as query priors.")
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-iou", type=float, default=0.50)
    parser.add_argument("--detector-image-size", type=int, default=640)
    parser.add_argument("--det-half", action="store_true")
    parser.add_argument("--yolo-checkpoint", default=None)
    parser.add_argument("--sam2-root", default=None)
    parser.add_argument("--sam2-checkpoint", default=None)
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-single-mask", action="store_true")
    parser.add_argument("--patch-mask-min-overlap", type=float, default=0.02)
    parser.add_argument("--side-view-window", type=int, default=13)
    parser.add_argument("--ply-scene-stride", type=int, default=4, help="Stride for VGGT depth point cloud vertices.")
    parser.add_argument("--ply-depth-upsample", type=int, default=2, help="Visualization-only depth/RGB upsampling factor for denser PLY surfaces.")
    parser.add_argument("--ply-max-scene-depth", type=float, default=30.0)
    parser.add_argument("--ply-depth-source", choices=("hsi", "raw"), default="hsi")
    parser.add_argument("--no-export-ply", action="store_true")
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
        anchor_idx = choose_anchor_index(probe, query_idx, args.anchor_index, args.anchor_mode)
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
            priors=priors,
            export_ply=not args.no_export_ply,
            scene_stride=args.ply_scene_stride,
            depth_upsample=args.ply_depth_upsample,
            max_scene_depth=args.ply_max_scene_depth,
            depth_source=args.ply_depth_source,
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
        "default_visualization_choice": {
            "person": args.person_select,
            "anchor": args.anchor_mode,
            "main_asset": "PLY depth point cloud + base SMPL mesh + foot-to-scene arrow",
            "box_source": "SAM2 mask bbox when auto-person-prior is enabled; model pred_boxes only as fallback",
            "coordinate_note": "Model projection uses resized input coordinates. RGB colors are sampled after resizing the original image to the depth grid through the model input image.",
        },
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
    try:
        pixel_mask, boxes, auto_meta, instance_masks = auto_sam2_person_mask(auto_args)
    except RuntimeError as exc:
        message = str(exc)
        fixed_size_mismatch = "The size of tensor" in message and "must match the size of tensor" in message
        if not fixed_size_mismatch or int(auto_args.detector_image_size) == 640:
            raise
        print(
            "[auto-prior] YOLO TorchScript failed at "
            f"detector_image_size={auto_args.detector_image_size}; retrying with 640. "
            "This usually means the TorchScript export has fixed anchors."
        )
        auto_args.detector_image_size = 640
        pixel_mask, boxes, auto_meta, instance_masks = auto_sam2_person_mask(auto_args)
        auto_meta.setdefault("detector", {})["fallback_from_image_size"] = int(args.detector_image_size)
    save_mask_artifacts(output_dir, pixel_mask, auto_meta, "real_auto_sam2_mask_original", instance_masks=instance_masks)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    selected = list(range(min(len(boxes), max_queries)))
    prior_boxes_xyxy_input: dict[int, list[float]] = {}

    boxes_tensor = torch.zeros(1, 1, max_queries, 4, dtype=torch.float32, device=device)
    mask_tensor = torch.zeros(1, 1, max_queries, dtype=torch.bool, device=device)
    grid_h = input_size // patch_size
    grid_w = input_size // patch_size
    patch_tensor = torch.zeros(1, 1, max_queries, grid_h * grid_w, dtype=torch.bool, device=device)

    resized_masks = []
    resized_masks_by_query: dict[int, np.ndarray] = {}
    for idx in selected:
        key = sorted(instance_masks.keys())[idx]
        box = mask_bbox(np.asarray(instance_masks[key]).astype(bool)).clipped(width, height)
        if box.area <= 0:
            box = boxes[idx].clipped(width, height)
        cx = ((box.x1 + box.x2) * 0.5) / max(float(width), 1.0)
        cy = ((box.y1 + box.y2) * 0.5) / max(float(height), 1.0)
        bw = (box.x2 - box.x1) / max(float(width), 1.0)
        bh = (box.y2 - box.y1) / max(float(height), 1.0)
        boxes_tensor[0, 0, idx] = torch.tensor([cx, cy, bw, bh], dtype=torch.float32, device=device).clamp(0.0, 1.0)
        mask_tensor[0, 0, idx] = True
        prior_boxes_xyxy_input[idx] = [
            float(box.x1) * float(input_size) / max(float(width), 1.0),
            float(box.y1) * float(input_size) / max(float(height), 1.0),
            float(box.x2) * float(input_size) / max(float(width), 1.0),
            float(box.y2) * float(input_size) / max(float(height), 1.0),
        ]
        resized = resize_mask(np.asarray(instance_masks[key]).astype(bool), (input_size, input_size))
        resized_masks.append(resized)
        resized_masks_by_query[int(idx)] = resized
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
        "prior_boxes_xyxy_input_pixels": {str(k): v for k, v in prior_boxes_xyxy_input.items()},
    }
    return {
        "boxes": boxes_tensor,
        "mask": mask_tensor,
        "patch_masks": patch_tensor,
        "valid_indices": selected,
        "person_mask_input": resized_union,
        "person_masks_input_by_query": resized_masks_by_query,
        "prior_boxes_xyxy_input_pixels": prior_boxes_xyxy_input,
        "meta": auto_meta,
    }


def mask_bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        from scripts.vis.create_arch_patch_pooling_elements import Box

        return Box(0.0, 0.0, 0.0, 0.0)
    from scripts.vis.create_arch_patch_pooling_elements import Box

    return Box(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


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
    hsi_scale = predictions.get("hsi_scene_scale")
    hsi_bias = predictions.get("hsi_scene_depth_bias")
    if isinstance(hsi_scale, torch.Tensor) and isinstance(hsi_bias, torch.Tensor):
        hsi_scale_hw = hsi_scale.to(device=depth_hw.device, dtype=depth_hw.dtype).reshape(*depth_hw.shape[:2], 1, 1)
        hsi_bias_hw = hsi_bias.to(device=depth_hw.device, dtype=depth_hw.dtype).reshape(*depth_hw.shape[:2], 1, 1)
        hsi_depth_hw = depth_hw * hsi_scale_hw + hsi_bias_hw
    else:
        hsi_scale_hw = torch.ones(*depth_hw.shape[:2], 1, 1, device=depth_hw.device, dtype=depth_hw.dtype)
        hsi_bias_hw = torch.zeros(*depth_hw.shape[:2], 1, 1, device=depth_hw.device, dtype=depth_hw.dtype)
        hsi_depth_hw = depth_hw
    intrinsics = _flatten_intrinsics(predictions["pose_enc"], input_size).to(device=pose6d.device, dtype=pose6d.dtype)
    anchors = head._anchors_cam(pose6d, betas, transl)
    poses_aa = rot6d_to_axis_angle(pose6d.reshape(-1, 24, 6)).reshape(-1, 72)
    smpl_vertices, smpl_joints = head.smpl(poses_aa.float(), betas.reshape(-1, betas.shape[-1]).float())
    smpl_vertices = smpl_vertices.to(device=pose6d.device, dtype=pose6d.dtype) + transl.reshape(-1, 1, 3)
    smpl_joints = smpl_joints[:, :24].to(device=pose6d.device, dtype=pose6d.dtype) + transl.reshape(-1, 1, 3)
    smpl_vertices = smpl_vertices.reshape(*pose6d.shape[:3], smpl_vertices.shape[-2], 3)
    smpl_joints = smpl_joints.reshape(*pose6d.shape[:3], smpl_joints.shape[-2], 3)
    foot_sole_indices = torch.argsort(head.smpl.layer.v_template.detach().float().reshape(-1, 3)[:, 1])[:160].long().to(device=pose6d.device)
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
        "hsi_depth_hw": hsi_depth_hw.detach(),
        "hsi_scene_scale_hw": hsi_scale_hw.detach(),
        "hsi_scene_depth_bias_hw": hsi_bias_hw.detach(),
        "intrinsics": intrinsics.detach(),
        "smpl_vertices": smpl_vertices.detach(),
        "smpl_joints": smpl_joints.detach(),
        "smpl_faces": torch.as_tensor(head.smpl.faces, dtype=torch.long, device=pose6d.device),
        "foot_sole_indices": foot_sole_indices.detach(),
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
    prior_boxes = priors.get("prior_boxes_xyxy_input_pixels") or {}
    if args.person_select in {"rightmost", "leftmost"} and prior_boxes:
        scored = []
        for query_idx in candidates:
            box = prior_boxes.get(int(query_idx))
            if box is None:
                continue
            scored.append((0.5 * (float(box[0]) + float(box[2])), int(query_idx)))
        if scored:
            scored = sorted(scored, reverse=args.person_select == "rightmost")
            return [scored[0][1]]
    if args.person_select == "all":
        return [int(i) for i in candidates[: max(args.top_k, 1)]]
    return [int(i) for i in candidates[: max(args.top_k, 1)]]


def choose_anchor_index(probe: dict[str, torch.Tensor], query_idx: int, requested: int, mode: str = "foot") -> int:
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
    if mode == "full_body":
        return 23
    if mode == "foot":
        foot_candidates = torch.tensor([6, 7, 9, 10], device=residual.device)
        foot_valid = valid[foot_candidates]
        if bool(foot_valid.any()):
            foot_scores = torch.where(foot_valid, residual[foot_candidates].abs(), torch.full_like(foot_candidates, float("inf"), dtype=residual.dtype))
            return int(foot_candidates[torch.argmin(foot_scores)].detach().cpu())
    if mode == "nearest":
        distance = probe["distance"][0, 0, query_idx].detach().float()
        score = torch.where(valid, distance, torch.full_like(distance, float("inf")))
        return int(torch.argmin(score).detach().cpu())
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
    priors: dict[str, Any],
    export_ply: bool,
    scene_stride: int,
    depth_upsample: int,
    max_scene_depth: float,
    depth_source: str,
) -> dict[str, Any]:
    depth = probe["depth_hw"][0, 0].detach().float().cpu().numpy()
    depth_img = depth_to_rgba(depth)
    box = prior_box_pixels(priors, query_idx) or predicted_box_pixels(predictions, query_idx, rgb.size)
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

    if export_ply:
        ply_path = output_dir / f"05_06_real_hsi_foot_scene_{prefix}.ply"
        ply_stats = export_hsi_foot_scene_ply(
            ply_path,
            rgb,
            probe,
            query_idx,
            anchor_idx,
            scene_stride=scene_stride,
            depth_upsample=depth_upsample,
            max_scene_depth=max_scene_depth,
            depth_source=depth_source,
            remove_mask_input=None,
        )
        out_files.append(ply_path.name)
        ply_no_people_path = output_dir / f"05_06_real_hsi_foot_scene_no_people_depth_{prefix}.ply"
        ply_no_people_stats = export_hsi_foot_scene_ply(
            ply_no_people_path,
            rgb,
            probe,
            query_idx,
            anchor_idx,
            scene_stride=scene_stride,
            depth_upsample=depth_upsample,
            max_scene_depth=max_scene_depth,
            depth_source=depth_source,
            remove_mask_input=priors.get("person_mask_input"),
        )
        out_files.append(ply_no_people_path.name)
        env_path = output_dir / f"environment_{depth_source}_depth_{prefix}.ply"
        env_stats = export_environment_ply(
            env_path,
            rgb,
            probe,
            scene_stride=scene_stride,
            depth_upsample=depth_upsample,
            max_scene_depth=max_scene_depth,
            depth_source=depth_source,
            remove_mask_input=None,
        )
        out_files.append(env_path.name)
        env_no_people_path = output_dir / f"environment_{depth_source}_depth_no_people_{prefix}.ply"
        env_no_people_stats = export_environment_ply(
            env_no_people_path,
            rgb,
            probe,
            scene_stride=scene_stride,
            depth_upsample=depth_upsample,
            max_scene_depth=max_scene_depth,
            depth_source=depth_source,
            remove_mask_input=priors.get("person_mask_input"),
        )
        out_files.append(env_no_people_path.name)
        smpl_path = output_dir / f"smpl_only_{prefix}.ply"
        smpl_stats = export_smpl_only_ply(smpl_path, probe, query_idx)
        out_files.append(smpl_path.name)

    metadata = build_person_metadata(predictions, probe, query_idx, anchor_idx, box)
    if export_ply:
        metadata["ply"] = {
            "combined_file": ply_path.name,
            "combined_no_people_depth_file": ply_no_people_path.name,
            "environment_file": env_path.name,
            "environment_no_people_file": env_no_people_path.name,
            "smpl_file": smpl_path.name,
            "combined": ply_stats,
            "combined_no_people_depth": ply_no_people_stats,
            "environment": env_stats,
            "environment_no_people": env_no_people_stats,
            "smpl": smpl_stats,
        }
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


def prior_box_pixels(priors: dict[str, Any], query_idx: int) -> list[float] | None:
    boxes = priors.get("prior_boxes_xyxy_input_pixels") or {}
    box = boxes.get(int(query_idx))
    if box is None:
        box = boxes.get(str(int(query_idx)))
    if box is None:
        return None
    return [float(v) for v in box]


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


def export_hsi_foot_scene_ply(
    path: Path,
    rgb: Image.Image,
    probe: dict[str, torch.Tensor],
    query_idx: int,
    anchor_idx: int,
    scene_stride: int = 4,
    depth_upsample: int = 1,
    max_scene_depth: float = 30.0,
    depth_source: str = "hsi",
    remove_mask_input: np.ndarray | None = None,
) -> dict[str, Any]:
    mesh = MeshBuilder()
    depth_key = "hsi_depth_hw" if depth_source == "hsi" and "hsi_depth_hw" in probe else "depth_hw"
    scene_points, scene_colors, scene_faces = depth_to_surface_mesh(
        depth=probe[depth_key][0, 0],
        intrinsics=probe["intrinsics"],
        rgb=rgb,
        image_size=int(probe["image_size"].detach().cpu()),
        stride=max(int(scene_stride), 1),
        upsample=max(int(depth_upsample), 1),
        max_depth=float(max_scene_depth),
        remove_mask_input=remove_mask_input,
    )
    if scene_points.size:
        mesh.add_mesh(scene_points, scene_faces, scene_colors)

    smpl_vertices = probe["smpl_vertices"][0, 0, query_idx].detach().float().cpu().numpy()
    smpl_faces = probe["smpl_faces"].detach().cpu().numpy()
    mesh.add_mesh(smpl_vertices, smpl_faces, (232, 142, 82))

    visual = select_visual_body_scene_points(probe, query_idx, anchor_idx, depth_key)
    anchor = visual["body_point"]
    scene = visual["scene_point"]
    normal = visual["normal"]
    body_scale = max(float(np.linalg.norm(np.nanmax(smpl_vertices, axis=0) - np.nanmin(smpl_vertices, axis=0))), 1e-3)
    sphere_radius = body_scale * 0.025
    arrow_radius = body_scale * 0.006
    add_uv_sphere(mesh, anchor, sphere_radius, (220, 64, 64))
    add_uv_sphere(mesh, scene, sphere_radius, (37, 99, 180))
    add_ply_arrow(mesh, anchor, scene, arrow_radius, (220, 64, 64), head_radius=arrow_radius * 3.0, head_length=body_scale * 0.055)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm > 1e-6:
        add_ply_arrow(
            mesh,
            scene,
            scene + normal / normal_norm * body_scale * 0.12,
            arrow_radius * 0.65,
            (26, 166, 166),
            head_radius=arrow_radius * 2.1,
            head_length=body_scale * 0.035,
        )
    mesh.write(path)
    vertices, _, faces = mesh.as_arrays()
    return {
        "scene_vertices": int(scene_points.shape[0]),
        "scene_faces": int(scene_faces.shape[0]),
        "total_vertices": int(vertices.shape[0]),
        "total_faces": int(faces.shape[0]),
        "depth_source": "hsi" if depth_key == "hsi_depth_hw" else "raw",
        "depth_upsample": int(max(depth_upsample, 1)),
        "person_mask_removed": bool(remove_mask_input is not None),
        "visual_vertex_index": int(visual.get("vertex_index", -1)),
        "visual_body_point_source": str(visual.get("source", "")),
    }


def export_environment_ply(
    path: Path,
    rgb: Image.Image,
    probe: dict[str, torch.Tensor],
    scene_stride: int,
    depth_upsample: int,
    max_scene_depth: float,
    depth_source: str,
    remove_mask_input: np.ndarray | None,
) -> dict[str, Any]:
    depth_key = "hsi_depth_hw" if depth_source == "hsi" and "hsi_depth_hw" in probe else "depth_hw"
    vertices, colors, faces = depth_to_surface_mesh(
        depth=probe[depth_key][0, 0],
        intrinsics=probe["intrinsics"],
        rgb=rgb,
        image_size=int(probe["image_size"].detach().cpu()),
        stride=max(int(scene_stride), 1),
        upsample=max(int(depth_upsample), 1),
        max_depth=float(max_scene_depth),
        remove_mask_input=remove_mask_input,
    )
    mesh = MeshBuilder()
    if vertices.size:
        mesh.add_mesh(vertices, faces, colors)
    mesh.write(path)
    return {
        "file": path.name,
        "depth_source": "hsi" if depth_key == "hsi_depth_hw" else "raw",
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
        "color_source": "input_rgb_resized_to_depth_grid",
        "depth_upsample": int(max(depth_upsample, 1)),
        "person_mask_removed": bool(remove_mask_input is not None),
    }


def export_smpl_only_ply(path: Path, probe: dict[str, torch.Tensor], query_idx: int) -> dict[str, Any]:
    vertices = probe["smpl_vertices"][0, 0, query_idx].detach().float().cpu().numpy()
    faces = probe["smpl_faces"].detach().cpu().numpy()
    mesh = MeshBuilder()
    mesh.add_mesh(vertices, faces, (232, 142, 82))
    mesh.write(path)
    return {
        "file": path.name,
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
    }


def select_visual_body_scene_points(
    probe: dict[str, torch.Tensor],
    query_idx: int,
    anchor_idx: int,
    depth_key: str,
) -> dict[str, Any]:
    depth = probe[depth_key][0, 0]
    height, width = depth.shape[-2:]
    vertices = probe["smpl_vertices"][0, 0, query_idx]
    sole_indices = probe["foot_sole_indices"].to(device=vertices.device).long()
    sole = vertices[sole_indices]
    intr = probe["intrinsics"].reshape(-1, 3, 3)[0].to(device=vertices.device, dtype=vertices.dtype)
    projected = _project_points(sole.reshape(1, -1, 3), intr.reshape(1, 3, 3))[0]
    projected_depth = _scale_points_to_depth(
        projected.reshape(1, 1, 1, -1, 2),
        int(probe["image_size"].detach().cpu()),
        height,
        width,
    ).reshape(-1, 2)
    px = projected_depth[:, 0].round().long()
    py = projected_depth[:, 1].round().long()
    valid = (
        torch.isfinite(sole).all(dim=-1)
        & (sole[:, 2] > 1e-6)
        & torch.isfinite(projected_depth).all(dim=-1)
        & (px >= 0)
        & (px < width)
        & (py >= 0)
        & (py < height)
    )
    sampled = depth.new_zeros(sole.shape[0])
    if bool(valid.any()):
        sampled[valid] = depth[py[valid], px[valid]]
    valid = valid & torch.isfinite(sampled) & (sampled > 1e-6)
    if bool(valid.any()):
        delta = torch.abs(sampled - sole[:, 2])
        score = torch.where(valid, delta, torch.full_like(delta, float("inf")))
        local_idx = int(torch.argmin(score).detach().cpu())
        body_point = sole[local_idx]
        image_uv = projected[local_idx]
        scene_z = sampled[local_idx]
        scene_point = unproject_one(image_uv, scene_z, intr)
        normal = estimate_normal_at(depth, int(px[local_idx].detach().cpu()), int(py[local_idx].detach().cpu()))
        return {
            "body_point": body_point.detach().float().cpu().numpy(),
            "scene_point": scene_point.detach().float().cpu().numpy(),
            "normal": normal.detach().float().cpu().numpy(),
            "vertex_index": int(sole_indices[local_idx].detach().cpu()),
            "source": "foot_sole_vertex",
        }

    body_point = probe["anchors"][0, 0, query_idx, anchor_idx]
    projected = probe["projected"][0, 0, query_idx, anchor_idx]
    projected_depth = probe["projected_depth"][0, 0, query_idx, anchor_idx]
    px0 = int(round(float(projected_depth[0].detach().cpu())))
    py0 = int(round(float(projected_depth[1].detach().cpu())))
    px0 = max(0, min(width - 1, px0))
    py0 = max(0, min(height - 1, py0))
    scene_z = depth[py0, px0]
    scene_point = unproject_one(projected, scene_z, intr)
    normal = estimate_normal_at(depth, px0, py0)
    return {
        "body_point": body_point.detach().float().cpu().numpy(),
        "scene_point": scene_point.detach().float().cpu().numpy(),
        "normal": normal.detach().float().cpu().numpy(),
        "vertex_index": -1,
        "source": "hsi_anchor_fallback",
    }


def unproject_one(image_uv: torch.Tensor, z: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    fx = intrinsics[0, 0].clamp(min=1e-6)
    fy = intrinsics[1, 1].clamp(min=1e-6)
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    x = (image_uv[0].to(dtype=z.dtype) - cx.to(dtype=z.dtype)) / fx.to(dtype=z.dtype) * z
    y = (image_uv[1].to(dtype=z.dtype) - cy.to(dtype=z.dtype)) / fy.to(dtype=z.dtype) * z
    return torch.stack([x, y, z])


def estimate_normal_at(depth: torch.Tensor, px: int, py: int) -> torch.Tensor:
    height, width = depth.shape[-2:]
    x0 = max(0, min(width - 1, px - 1))
    x1 = max(0, min(width - 1, px + 1))
    y0 = max(0, min(height - 1, py - 1))
    y1 = max(0, min(height - 1, py + 1))
    dzdx = (depth[py, x1] - depth[py, x0]) * 0.5
    dzdy = (depth[y1, px] - depth[y0, px]) * 0.5
    normal = torch.stack([-dzdx, -dzdy, torch.ones_like(dzdx)])
    return normal / torch.linalg.norm(normal).clamp(min=1e-6)


def depth_to_surface_mesh(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    rgb: Image.Image,
    image_size: int,
    stride: int,
    upsample: int,
    max_depth: float,
    remove_mask_input: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth_hw = depth.detach().float()
    if int(upsample) > 1:
        depth_hw = torch.nn.functional.interpolate(
            depth_hw.reshape(1, 1, *depth_hw.shape[-2:]),
            scale_factor=int(upsample),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    height, width = depth_hw.shape[-2:]
    ys = torch.arange(0, height, stride, device=depth_hw.device)
    xs = torch.arange(0, width, stride, device=depth_hw.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    z = depth_hw[yy, xx]
    valid = torch.isfinite(z) & (z > 1e-6) & (z <= float(max_depth))
    if remove_mask_input is not None:
        remove_mask = resize_bool_mask(np.asarray(remove_mask_input).astype(bool), (width, height))
        remove_tensor = torch.from_numpy(remove_mask).to(device=depth_hw.device, dtype=torch.bool)
        valid = valid & (~remove_tensor[yy, xx])
    if not bool(valid.any()):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), np.empty((0, 3), dtype=np.int64)
    xx_f = xx.to(dtype=depth_hw.dtype) * (float(image_size) / float(width))
    yy_f = yy.to(dtype=depth_hw.dtype) * (float(image_size) / float(height))
    intr = intrinsics.reshape(-1, 3, 3)[0].to(device=depth_hw.device, dtype=depth_hw.dtype)
    fx = intr[0, 0].clamp(min=1e-6)
    fy = intr[1, 1].clamp(min=1e-6)
    cx = intr[0, 2]
    cy = intr[1, 2]
    points = torch.stack([(xx_f - cx) / fx * z, (yy_f - cy) / fy * z, z], dim=-1)
    points_grid = points.detach().float().cpu().numpy().astype(np.float32, copy=False)
    valid_np = valid.detach().cpu().numpy()
    index_map = -np.ones(valid_np.shape, dtype=np.int64)
    index_map[valid_np] = np.arange(int(valid_np.sum()), dtype=np.int64)
    points_np = points_grid[valid_np]

    rgb_small = np.asarray(rgb.resize((width, height), Image.BILINEAR).convert("RGB"), dtype=np.uint8)
    colors = rgb_small[yy.detach().cpu().numpy(), xx.detach().cpu().numpy()]
    colors = colors[valid_np]

    faces: list[list[int]] = []
    depth_np = z.detach().float().cpu().numpy()
    rows, cols = valid_np.shape
    for row in range(rows - 1):
        for col in range(cols - 1):
            ids = [
                index_map[row, col],
                index_map[row, col + 1],
                index_map[row + 1, col],
                index_map[row + 1, col + 1],
            ]
            if min(ids) < 0:
                continue
            d = np.asarray(
                [
                    depth_np[row, col],
                    depth_np[row, col + 1],
                    depth_np[row + 1, col],
                    depth_np[row + 1, col + 1],
                ],
                dtype=np.float32,
            )
            d_mean = max(float(np.mean(np.abs(d))), 1e-6)
            if float(d.max() - d.min()) / d_mean > 0.08:
                continue
            a, b, c, d_idx = [int(v) for v in ids]
            faces.append([a, c, b])
            faces.append([b, c, d_idx])
    faces_np = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if faces_np.size == 0 and points_np.shape[0] > 0:
        points_np, colors, faces_np = point_splat_mesh(points_np, colors, radius=estimate_splat_radius(points_np))
    return points_np, colors, faces_np


def estimate_splat_radius(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.01
    extent = np.nanmax(points, axis=0) - np.nanmin(points, axis=0)
    return max(float(np.linalg.norm(extent)) * 0.0015, 0.003)


def resize_bool_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    mask_image = Image.fromarray((np.asarray(mask).astype(np.uint8) * 255), mode="L")
    mask_image = mask_image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask_image) > 0


def point_splat_mesh(points: np.ndarray, colors: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback visible point cloud as tiny triangular billboards."""
    verts = []
    faces = []
    out_colors = []
    r = float(radius)
    for idx, (point, color) in enumerate(zip(points, colors, strict=True)):
        base = len(verts)
        verts.extend(
            [
                [point[0] - r, point[1], point[2]],
                [point[0] + r, point[1], point[2]],
                [point[0], point[1] + r, point[2]],
            ]
        )
        faces.append([base, base + 1, base + 2])
        out_colors.extend([color.tolist(), color.tolist(), color.tolist()])
        if idx >= 6000:
            break
    return (
        np.asarray(verts, dtype=np.float32).reshape(-1, 3),
        np.asarray(out_colors, dtype=np.uint8).reshape(-1, 3),
        np.asarray(faces, dtype=np.int64).reshape(-1, 3),
    )


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
        "box_xyxy_input_pixels": box,
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
        "coordinate_frames": {
            "model_input_size": int(probe["image_size"].detach().cpu()),
            "raw_depth_hw": [int(v) for v in probe["depth_hw"].shape[-2:]],
            "hsi_depth_hw": [int(v) for v in probe["hsi_depth_hw"].shape[-2:]] if "hsi_depth_hw" in probe else None,
            "projection_uv_units": "model_input_pixels",
            "depth_uv_units": "depth_grid_pixels",
            "environment_color_source": "original RGB resized to model input/depth grid",
        },
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
