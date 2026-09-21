from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.utils.pose_enc import encoding_to_camera


@dataclass(frozen=True, slots=True)
class CascadeConfig:
    confidence_threshold: float = 0.05
    coarse_min_anchor_pixels: int = 32
    coarse_scale_min: float = 0.10
    coarse_scale_max: float = 10.0
    coarse_anchor_stride: int = 8
    coarse_fallback: str = "sequence_median"
    effective_affine_mode: str = "clip_median"


@dataclass(frozen=True, slots=True)
class GroundEstimatorConfig:
    up_axis: str = "negative_y"
    scene_point_stride: int = 2
    max_scene_depth_m: float = 80.0
    bbox_expand_ratio: float = 0.10
    footprint_margin_m: float = 0.35
    vertical_window_m: float = 0.75
    ground_quantile: float = 0.90
    min_support_points: int = 64
    tolerance_m: float = 0.005
    target_min_iou: float = 0.30
    target_min_confidence: float = 0.05
    target_selection: str = "rich_box_iou"


def configure_reference_cascade_model(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    max_humans: int = 8,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the model settings used by the accepted cascade inference script."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Stage-2 checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    restored = dict(config)
    model_config = dict(config.get("model", {}))
    model_config.update(
        {
            "enable_camera": True,
            "enable_depth": True,
            "enable_smpl": True,
            "enable_hsi_refine": True,
            "smpl_provider": "nlf",
            "num_smpl_queries": int(max_humans),
            "smpl_use_aggregator_queries": False,
            "smpl_query_box_prior": False,
            "smpl_query_patch_pool": False,
            "nlf_use_detector": True,
            "nlf_require_boxes": False,
            "nlf_detector_threshold": 0.30,
            "hsi_scene_affine_mode": "per_frame",
            "hsi_align_feature_version": "legacy_scale_bias_v0",
        }
    )

    state_dict = _extract_state_dict(checkpoint)
    align_weight = state_dict.get("hsi_human_scene_align_head.mlp.0.weight")
    if not isinstance(align_weight, torch.Tensor) or align_weight.ndim != 2:
        raise RuntimeError(
            "Stage-2 checkpoint is missing hsi_human_scene_align_head.mlp.0.weight: "
            f"{checkpoint_path}"
        )
    checkpoint_input_dim = int(align_weight.shape[1])
    if checkpoint_input_dim != 25:
        raise ValueError(
            "The checkpoint does not match the accepted downtown_upstairs_00 cascade: "
            f"expected HSI alignment input_dim=25 for legacy_scale_bias_v0, got {checkpoint_input_dim}."
        )
    restored["model"] = model_config
    report = {
        "reference": "scripts/vis/serve_stage2_walking_coarse_scale_hsi_cascade.sh",
        "checkpoint": str(checkpoint_path),
        "checkpoint_align_input_dim": checkpoint_input_dim,
        "configured_align_feature_version": "legacy_scale_bias_v0",
        "hsi_scene_affine_mode": "per_frame",
        "smpl_use_aggregator_queries": False,
        "num_smpl_queries": int(max_humans),
        "target_selection": "configured by the evaluation protocol",
    }
    del checkpoint
    return restored, report


def load_evaluation_weights(
    model: torch.nn.Module,
    baseline_checkpoint: str | Path,
    stage2_checkpoint: str | Path,
    scale_checkpoint: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    baseline = torch.load(Path(baseline_checkpoint), map_location=device)
    baseline_state, shape_report = _make_state_dict_loadable(
        _extract_state_dict(baseline), model.state_dict()
    )
    missing, unexpected = model.load_state_dict(baseline_state, strict=False)
    report["baseline"] = {
        "path": str(baseline_checkpoint),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "shape_skipped": len(shape_report["skipped"]),
    }

    stage2 = torch.load(Path(stage2_checkpoint), map_location=device)
    stage2_state = _extract_state_dict(stage2)
    missing, unexpected = model.load_state_dict(stage2_state, strict=False)
    report["stage2"] = {
        "path": str(stage2_checkpoint),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "tensors": len(stage2_state),
    }

    overlay = torch.load(Path(scale_checkpoint), map_location=device)
    overlay_state = {
        key: value
        for key, value in _extract_state_dict(overlay).items()
        if key.startswith("hsi_refinement_head.")
    }
    if not overlay_state:
        raise RuntimeError(f"No hsi_refinement_head tensors in scale checkpoint: {scale_checkpoint}")
    missing, unexpected = model.load_state_dict(overlay_state, strict=False)
    report["scale_overlay"] = {
        "path": str(scale_checkpoint),
        "prefix": "hsi_refinement_head.",
        "tensors": len(overlay_state),
        "missing": len(missing),
        "unexpected": len(unexpected),
    }
    return report


@torch.no_grad()
def run_metric_cascade(
    model: torch.nn.Module,
    smpl: SMPLLayer,
    images: torch.Tensor,
    config: CascadeConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the metric-depth cascade without importing any visualization code."""
    kwargs: dict[str, torch.Tensor] = {}
    disabled_names = (
        "hsi_refinement_head",
        "hsi_trstr_head",
        "hsi_human_scene_align_head",
        "hsi_translation_refine_v4_head",
        "hsi_contact_refine_head",
        "hsi_foot_contact_intent_head",
        "hsi_grounding_head",
    )
    disabled: dict[str, Any] = {}
    for name in disabled_names:
        if hasattr(model, name):
            disabled[name] = getattr(model, name)
            setattr(model, name, None)
    try:
        first_pass = model(images, **kwargs)
    finally:
        for name, module in disabled.items():
            setattr(model, name, module)

    if disabled.get("hsi_trstr_head") is not None:
        raise RuntimeError("The dedicated RICH evaluator does not yet support hsi_trstr_head checkpoints")
    raw_depth = canonical_depth(first_pass["depth"]).detach().float()
    base_vertices = decode_vertices(first_pass, smpl, refined=False)
    confidences = first_pass["pred_confs"].detach().float()[..., 0]
    coarse_scale = raw_depth.new_ones(raw_depth.shape[:2])
    coarse_records: list[dict[str, Any]] = []
    for frame_index in range(raw_depth.shape[1]):
        valid_frame = confidences[0, frame_index] >= config.confidence_threshold
        if not bool(valid_frame.any()):
            coarse_records.append({"applied": False, "reason": "no_valid_target", "scale": 1.0})
            continue
        record = estimate_scene_to_smpl_scale(
            smpl_vertices=base_vertices[0, frame_index, valid_frame],
            depth=raw_depth[0, frame_index],
            pose_enc=first_pass["pose_enc"][:, frame_index : frame_index + 1],
            min_anchor_pixels=config.coarse_min_anchor_pixels,
            scale_min=config.coarse_scale_min,
            scale_max=config.coarse_scale_max,
            anchor_stride=config.coarse_anchor_stride,
        )
        if bool(record["applied"]):
            coarse_scale[0, frame_index] = float(record["scale"])
        coarse_records.append(record)

    applied = torch.tensor(
        [bool(record["applied"]) for record in coarse_records], device=coarse_scale.device, dtype=torch.bool
    )
    if config.coarse_fallback == "sequence_median" and bool(applied.any()):
        median_scale = torch.exp(torch.log(coarse_scale[0, applied].clamp(min=1e-6)).median())
        for frame_index, record in enumerate(coarse_records):
            if not bool(record["applied"]):
                coarse_scale[0, frame_index] = median_scale
                record["fallback"] = "sequence_median"
                record["scale"] = float(median_scale.cpu())

    coarse_depth = raw_depth * coarse_scale[..., None, None]
    predictions = model(
        images,
        **kwargs,
        hsi_depth_override=coarse_depth,
        hsi_depth_is_metric=True,
        hsi_geometry_mode="smpl_coarse_metric",
    )
    residual_scale = predictions["hsi_scene_scale"].detach().float()
    residual_bias = predictions["hsi_scene_depth_bias"].detach().float()
    frame_scale = predictions.get("hsi_frame_scene_scale", residual_scale).detach().float()
    frame_bias = predictions.get("hsi_frame_scene_depth_bias", residual_bias).detach().float()
    coarse_scale_3d = coarse_scale[..., None]
    frame_effective_scale = coarse_scale_3d * frame_scale
    if config.effective_affine_mode == "clip_median":
        log_scale = torch.log(frame_effective_scale.clamp(min=1e-6)).median(dim=1, keepdim=True).values
        bias = frame_bias.median(dim=1, keepdim=True).values
        effective_scale = torch.exp(log_scale).expand_as(frame_effective_scale)
        effective_bias = bias.expand_as(frame_bias)
    elif config.effective_affine_mode == "per_frame":
        effective_scale = coarse_scale_3d * residual_scale
        effective_bias = residual_bias
    else:
        raise ValueError(f"Unsupported effective_affine_mode: {config.effective_affine_mode}")

    predictions["hsi_coarse_scene_scale"] = coarse_scale_3d
    predictions["hsi_residual_scene_scale"] = residual_scale
    predictions["hsi_residual_scene_depth_bias"] = residual_bias
    predictions["hsi_frame_scene_scale"] = frame_effective_scale
    predictions["hsi_frame_scene_depth_bias"] = frame_bias
    predictions["hsi_scene_scale"] = effective_scale
    predictions["hsi_scene_depth_bias"] = effective_bias
    predictions["hsi_translation_depth"] = raw_depth * effective_scale[..., None] + effective_bias[..., None]
    diagnostics = {
        "cascade_config": asdict(config),
        "coarse_records": coarse_records,
        "coarse_scale": _tensor_list(coarse_scale),
        "residual_scale": _tensor_list(residual_scale),
        "residual_bias_m": _tensor_list(residual_bias),
        "effective_scale": _tensor_list(effective_scale),
        "effective_bias_m": _tensor_list(effective_bias),
    }
    return predictions, diagnostics


def evaluate_prediction_chunk(
    predictions: dict[str, Any],
    smpl: SMPLLayer,
    target_boxes: torch.Tensor,
    target_mask: torch.Tensor,
    config: GroundEstimatorConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    depth = predictions["hsi_translation_depth"].detach().float()
    vertices_cam = decode_vertices(predictions, smpl, refined=True).detach().float()
    image_hw = (int(depth.shape[-2]), int(depth.shape[-1]))
    extrinsics, intrinsics = encoding_to_camera(
        predictions["pose_enc"].detach().float(), image_size_hw=image_hw, build_intrinsics=True
    )
    scale = predictions["hsi_scene_scale"].detach().float().reshape(*depth.shape[:2], -1)[..., 0]
    extrinsics = extrinsics.detach().float().clone()
    extrinsics[..., :3, 3] *= scale[..., None]

    frame_payloads: list[dict[str, Any]] = []
    frame_ground_estimates = []
    for frame_index in range(depth.shape[1]):
        predicted_boxes = predictions["pred_boxes"][0, frame_index].detach().float()
        predicted_confidences = predictions["pred_confs"][0, frame_index, :, 0].detach().float()
        valid_detections = (
            torch.isfinite(predicted_boxes).all(dim=-1)
            & (predicted_boxes[:, 2:] > 0.0).all(dim=-1)
            & torch.isfinite(predicted_confidences)
            & (predicted_confidences >= config.target_min_confidence)
        )
        if not bool(valid_detections.any()):
            frame_payloads.append(
                {"valid": False, "invalid_reason": "no_valid_detector_prediction"}
            )
            frame_ground_estimates.append(None)
            continue

        if config.target_selection == "rich_box_iou":
            if not bool(target_mask[0, frame_index, 0]):
                frame_payloads.append(
                    {"valid": False, "invalid_reason": "missing_rich_target_box"}
                )
                frame_ground_estimates.append(None)
                continue
            selection_box = target_boxes[0, frame_index, 0]
            ious = normalized_cxcywh_iou(predicted_boxes, selection_box)
            ious = torch.where(valid_detections, ious, ious.new_full(ious.shape, -1.0))
            query_index = int(torch.argmax(ious).item())
            target_iou: float | None = float(ious[query_index].cpu())
            if target_iou < config.target_min_iou:
                frame_payloads.append(
                    {
                        "valid": False,
                        "invalid_reason": "target_iou_below_threshold",
                        "query_index": query_index,
                        "target_iou": target_iou,
                        "confidence": float(predicted_confidences[query_index].cpu()),
                    }
                )
                frame_ground_estimates.append(None)
                continue
        elif config.target_selection == "detector_confidence":
            scores = torch.where(
                valid_detections,
                predicted_confidences,
                predicted_confidences.new_full(predicted_confidences.shape, -1.0),
            )
            query_index = int(torch.argmax(scores).item())
            target_iou = None
            selection_box = predicted_boxes[query_index]
        else:
            raise ValueError(f"Unsupported target selection: {config.target_selection}")

        confidence = float(predicted_confidences[query_index].cpu())
        vertices_world = camera_points_to_world(
            vertices_cam[0, frame_index, query_index], extrinsics[0, frame_index]
        )
        points_world = depth_to_world_points(
            depth[0, frame_index],
            intrinsics[0, frame_index],
            extrinsics[0, frame_index],
            stride=config.scene_point_stride,
            max_depth_m=config.max_scene_depth_m,
            exclusion_box=selection_box,
            bbox_expand_ratio=config.bbox_expand_ratio,
        )
        body_low_y = float(vertices_world[:, 1].max().cpu())
        support_y = select_local_support_y(points_world, vertices_world, config)
        if support_y.numel() < config.min_support_points:
            frame_payloads.append(
                {
                    "valid": False,
                    "invalid_reason": "insufficient_local_scene_points",
                    "query_index": query_index,
                    "target_iou": target_iou,
                    "confidence": confidence,
                    "body_low_y": body_low_y,
                    "support_points": int(support_y.numel()),
                }
            )
            frame_ground_estimates.append(None)
            continue
        estimate = float(torch.quantile(support_y, config.ground_quantile).cpu())
        frame_ground_estimates.append(estimate)
        frame_payloads.append(
            {
                "valid": True,
                "query_index": query_index,
                "target_iou": target_iou,
                "confidence": confidence,
                "body_low_y": body_low_y,
                "frame_ground_y": estimate,
                "support_points": int(support_y.numel()),
            }
        )

    valid_estimates = [value for value in frame_ground_estimates if value is not None]
    if not valid_estimates:
        return frame_payloads, {
            "valid": False,
            "reason": "no_valid_ground_estimate",
            "config": asdict(config),
        }
    robust_ground_y = float(np.median(np.asarray(valid_estimates, dtype=np.float64)))
    for payload in frame_payloads:
        if not payload["valid"]:
            continue
        clearance = robust_ground_y - float(payload["body_low_y"])
        payload["ground_y"] = robust_ground_y
        payload["clearance_m"] = clearance
        payload["collision"] = clearance < -config.tolerance_m
        payload["floating"] = clearance >= config.tolerance_m
    return frame_payloads, {
        "valid": True,
        "method": "local_scene_y_quantile_then_chunk_median",
        "coordinate_convention": "VGGT world coordinates; -Y is up",
        "ground_y": robust_ground_y,
        "valid_frame_estimates": len(valid_estimates),
        "config": asdict(config),
    }


def compute_physical_metrics(clearances_m: list[float] | np.ndarray, tolerance_m: float = 0.005) -> dict[str, Any]:
    values = np.asarray(clearances_m, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    penetrating = values < -float(tolerance_m)
    floating = values >= float(tolerance_m)
    penetration_depths = -values[penetrating]
    float_heights = values[floating]
    return {
        "valid_frames": int(values.size),
        "collision_frames": int(penetrating.sum()),
        "floating_frames": int(floating.sum()),
        "neutral_frames": int(values.size - penetrating.sum() - floating.sum()),
        "collision_ratio_pct": float(100.0 * penetrating.mean()) if values.size else 0.0,
        "penetrate_cm": float(100.0 * penetration_depths.mean()) if penetration_depths.size else 0.0,
        "float_cm": float(100.0 * float_heights.mean()) if float_heights.size else 0.0,
        "penetration_max_cm": float(100.0 * penetration_depths.max()) if penetration_depths.size else 0.0,
        "tolerance_m": float(tolerance_m),
    }


def canonical_depth(value: torch.Tensor) -> torch.Tensor:
    depth = value
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 5 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 4:
        raise ValueError(f"Expected depth [B,S,H,W], got {tuple(value.shape)}")
    return depth


def decode_vertices(predictions: dict[str, Any], smpl: SMPLLayer, refined: bool) -> torch.Tensor:
    if refined:
        keys = (
            "hsi_refined_pred_poses",
            "hsi_refined_pred_betas",
            "hsi_refined_pred_transl_cam",
        )
        if not all(isinstance(predictions.get(key), torch.Tensor) for key in keys):
            keys = ("pred_poses", "pred_betas", "pred_transl_cam")
    else:
        keys = ("pred_poses", "pred_betas", "pred_transl_cam")
    poses, betas, translation = (predictions[key].detach() for key in keys)
    shape = poses.shape[:3]
    vertices, _ = smpl(poses.reshape(-1, 72).float(), betas.reshape(-1, betas.shape[-1]).float())
    return vertices.reshape(*shape, vertices.shape[-2], 3).to(translation) + translation[..., None, :]


def estimate_scene_to_smpl_scale(
    smpl_vertices: torch.Tensor,
    depth: torch.Tensor,
    pose_enc: torch.Tensor,
    min_anchor_pixels: int,
    scale_min: float,
    scale_max: float,
    anchor_stride: int,
) -> dict[str, Any]:
    vertices = smpl_vertices[:, :: max(int(anchor_stride), 1)].reshape(-1, 3).to(depth)
    _, intrinsics = encoding_to_camera(
        pose_enc.detach().float(), image_size_hw=(int(depth.shape[-2]), int(depth.shape[-1])), build_intrinsics=True
    )
    projected = project_points(vertices, intrinsics[0, 0].to(depth))
    px = projected[:, 0].round().long()
    py = projected[:, 1].round().long()
    height, width = depth.shape[-2:]
    valid = (
        torch.isfinite(vertices).all(dim=-1)
        & torch.isfinite(projected).all(dim=-1)
        & (vertices[:, 2] > 1e-6)
        & (px >= 0)
        & (px < width)
        & (py >= 0)
        & (py < height)
    )
    if not bool(valid.any()):
        return {"applied": False, "reason": "no_projected_anchors", "scale": 1.0, "anchor_pixels": 0}
    px, py, z_smpl = px[valid], py[valid], vertices[valid, 2]
    z_scene = depth[py, px]
    valid_depth = torch.isfinite(z_scene) & (z_scene > 1e-6)
    px, py, z_smpl, z_scene = px[valid_depth], py[valid_depth], z_smpl[valid_depth], z_scene[valid_depth]
    if z_scene.numel() == 0:
        return {"applied": False, "reason": "no_valid_scene_depth", "scale": 1.0, "anchor_pixels": 0}
    px, py, z_smpl, z_scene = nearest_anchor_per_pixel(px, py, z_smpl, z_scene, int(width))
    ratios = z_smpl / z_scene
    ratios = ratios[torch.isfinite(ratios) & (ratios >= scale_min) & (ratios <= scale_max)]
    if ratios.numel() < int(min_anchor_pixels):
        return {
            "applied": False,
            "reason": "insufficient_anchor_pixels",
            "scale": float(ratios.median().cpu()) if ratios.numel() else 1.0,
            "anchor_pixels": int(ratios.numel()),
        }
    return {
        "applied": True,
        "reason": "ok",
        "scale": float(ratios.median().cpu()),
        "anchor_pixels": int(ratios.numel()),
    }


def nearest_anchor_per_pixel(
    px: torch.Tensor,
    py: torch.Tensor,
    z_body: torch.Tensor,
    z_scene: torch.Tensor,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pixel = (py * int(width) + px).detach().cpu().numpy()
    depth = z_body.detach().float().cpu().numpy()
    order = np.lexsort((depth, pixel))
    sorted_pixel = pixel[order]
    keep = np.ones(order.shape[0], dtype=bool)
    keep[1:] = sorted_pixel[1:] != sorted_pixel[:-1]
    indices = torch.as_tensor(order[keep], device=px.device, dtype=torch.long)
    return px[indices], py[indices], z_body[indices], z_scene[indices]


def project_points(points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    z = points[:, 2].clamp(min=1e-6)
    x = intrinsics[0, 0] * points[:, 0] / z + intrinsics[0, 2]
    y = intrinsics[1, 1] * points[:, 1] / z + intrinsics[1, 2]
    return torch.stack((x, y), dim=-1)


def normalized_cxcywh_iou(boxes: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    boxes = boxes.float()
    target = target.to(boxes).float()
    boxes_min = boxes[:, :2] - 0.5 * boxes[:, 2:].clamp(min=0.0)
    boxes_max = boxes[:, :2] + 0.5 * boxes[:, 2:].clamp(min=0.0)
    target_min = target[:2] - 0.5 * target[2:].clamp(min=0.0)
    target_max = target[:2] + 0.5 * target[2:].clamp(min=0.0)
    intersection_min = torch.maximum(boxes_min, target_min)
    intersection_max = torch.minimum(boxes_max, target_max)
    intersection = (intersection_max - intersection_min).clamp(min=0.0).prod(dim=-1)
    boxes_area = (boxes_max - boxes_min).clamp(min=0.0).prod(dim=-1)
    target_area = (target_max - target_min).clamp(min=0.0).prod()
    return intersection / (boxes_area + target_area - intersection).clamp(min=1e-8)


def camera_points_to_world(points: torch.Tensor, extrinsic_w2c: torch.Tensor) -> torch.Tensor:
    rotation = extrinsic_w2c[:3, :3].to(points)
    translation = extrinsic_w2c[:3, 3].to(points)
    return (points - translation) @ rotation


def depth_to_world_points(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsic_w2c: torch.Tensor,
    stride: int,
    max_depth_m: float,
    exclusion_box: torch.Tensor,
    bbox_expand_ratio: float,
) -> torch.Tensor:
    height, width = int(depth.shape[-2]), int(depth.shape[-1])
    step = max(int(stride), 1)
    ys, xs = torch.meshgrid(
        torch.arange(0, height, step, device=depth.device, dtype=torch.float32),
        torch.arange(0, width, step, device=depth.device, dtype=torch.float32),
        indexing="ij",
    )
    z = depth[ys.long(), xs.long()]
    fx = intrinsics[0, 0].clamp(min=1e-6)
    fy = intrinsics[1, 1].clamp(min=1e-6)
    x = (xs - intrinsics[0, 2]) / fx * z
    y = (ys - intrinsics[1, 2]) / fy * z
    points = torch.stack((x, y, z), dim=-1)
    valid = torch.isfinite(points).all(dim=-1) & (z > 1e-6)
    if max_depth_m > 0:
        valid &= z <= float(max_depth_m)
    cx, cy, bw, bh = exclusion_box.to(depth).unbind(dim=-1)
    bw = bw * (1.0 + 2.0 * float(bbox_expand_ratio))
    bh = bh * (1.0 + 2.0 * float(bbox_expand_ratio))
    nx = xs / max(float(width), 1.0)
    ny = ys / max(float(height), 1.0)
    inside = (nx >= cx - 0.5 * bw) & (nx <= cx + 0.5 * bw) & (ny >= cy - 0.5 * bh) & (ny <= cy + 0.5 * bh)
    valid &= ~inside
    return camera_points_to_world(points[valid], extrinsic_w2c)


def select_local_support_y(
    scene_points_world: torch.Tensor,
    body_vertices_world: torch.Tensor,
    config: GroundEstimatorConfig,
) -> torch.Tensor:
    if config.up_axis != "negative_y":
        raise ValueError(f"Unsupported up_axis: {config.up_axis}")
    if scene_points_world.numel() == 0:
        return scene_points_world.new_zeros((0,))
    margin = float(config.footprint_margin_m)
    min_x, max_x = body_vertices_world[:, 0].min() - margin, body_vertices_world[:, 0].max() + margin
    min_z, max_z = body_vertices_world[:, 2].min() - margin, body_vertices_world[:, 2].max() + margin
    body_low_y = body_vertices_world[:, 1].max()
    vertical = float(config.vertical_window_m)
    mask = (
        (scene_points_world[:, 0] >= min_x)
        & (scene_points_world[:, 0] <= max_x)
        & (scene_points_world[:, 2] >= min_z)
        & (scene_points_world[:, 2] <= max_z)
        & (scene_points_world[:, 1] >= body_low_y - vertical)
        & (scene_points_world[:, 1] <= body_low_y + vertical)
    )
    return scene_points_world[mask, 1]


def _tensor_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().float().reshape(-1).cpu().tolist()]


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if isinstance(checkpoint.get(key), dict):
                return {
                    name.removeprefix("module."): value
                    for name, value in checkpoint[key].items()
                }
    if isinstance(checkpoint, dict) and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return {name.removeprefix("module."): value for name, value in checkpoint.items()}
    raise ValueError("Could not find a model state_dict in checkpoint")


def _make_state_dict_loadable(
    checkpoint_state: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    loadable: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in checkpoint_state.items():
        target = model_state.get(key)
        if target is not None and isinstance(value, torch.Tensor) and tuple(value.shape) != tuple(target.shape):
            skipped.append(f"{key}: {tuple(value.shape)} -> {tuple(target.shape)}")
            continue
        loadable[key] = value
    return loadable, {"skipped": skipped}
