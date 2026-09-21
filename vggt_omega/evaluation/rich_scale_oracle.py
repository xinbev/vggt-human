from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vggt_omega.evaluation.rich_physical_grounding import (
    GroundEstimatorConfig,
    camera_points_to_world,
    canonical_depth,
    compute_physical_metrics,
    decode_vertices,
    normalized_cxcywh_iou,
    select_local_support_y,
)
from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.utils.pose_enc import encoding_to_camera


CACHE_FORMAT = "vggt_omega_rich_manual_scale_window_v1"


def build_scale_window_cache(
    predictions: dict[str, Any],
    smpl: SMPLLayer,
    images: torch.Tensor,
    target_boxes: torch.Tensor,
    target_mask: torch.Tensor,
    config: GroundEstimatorConfig,
) -> dict[str, np.ndarray]:
    """Freeze one inference window before applying a viewer-only scale multiplier."""
    depth = canonical_depth(predictions["hsi_translation_depth"]).detach().float()
    vertices = decode_vertices(predictions, smpl, refined=True).detach().float()
    height, width = int(depth.shape[-2]), int(depth.shape[-1])
    extrinsics, intrinsics = encoding_to_camera(
        predictions["pose_enc"].detach().float(),
        image_size_hw=(height, width),
        build_intrinsics=True,
    )
    model_scale = predictions["hsi_scene_scale"].detach().float().reshape(*depth.shape[:2], -1)[..., 0]
    extrinsics = extrinsics.detach().float().clone()
    extrinsics[..., :3, 3] *= model_scale[..., None]

    frame_count = int(depth.shape[1])
    query_indices = torch.full((frame_count,), -1, dtype=torch.int16, device=depth.device)
    confidences = torch.zeros((frame_count,), dtype=torch.float32, device=depth.device)
    selection_boxes = torch.zeros((frame_count, 4), dtype=torch.float32, device=depth.device)
    selected_vertices = torch.zeros(
        (frame_count, vertices.shape[-2], 3), dtype=torch.float32, device=depth.device
    )
    selected_valid = torch.zeros((frame_count,), dtype=torch.bool, device=depth.device)

    for frame_index in range(frame_count):
        boxes = predictions["pred_boxes"][0, frame_index].detach().float()
        scores = predictions["pred_confs"][0, frame_index, :, 0].detach().float()
        valid = (
            torch.isfinite(boxes).all(dim=-1)
            & (boxes[:, 2:] > 0.0).all(dim=-1)
            & torch.isfinite(scores)
            & (scores >= config.target_min_confidence)
        )
        if not bool(valid.any()):
            continue
        if config.target_selection == "detector_confidence":
            ranked = torch.where(valid, scores, scores.new_full(scores.shape, -1.0))
            query_index = int(torch.argmax(ranked).item())
        elif config.target_selection == "rich_box_iou":
            if not bool(target_mask[0, frame_index, 0]):
                continue
            ious = normalized_cxcywh_iou(boxes, target_boxes[0, frame_index, 0])
            ious = torch.where(valid, ious, ious.new_full(ious.shape, -1.0))
            query_index = int(torch.argmax(ious).item())
            if float(ious[query_index].cpu()) < config.target_min_iou:
                continue
        else:
            raise ValueError(f"Unsupported target selection: {config.target_selection}")
        query_indices[frame_index] = query_index
        confidences[frame_index] = scores[query_index]
        selection_boxes[frame_index] = boxes[query_index]
        selected_vertices[frame_index] = vertices[0, frame_index, query_index]
        selected_valid[frame_index] = True

    stride = max(1, int(config.scene_point_stride))
    sampled_depth = depth[0, :, ::stride, ::stride]
    rgb = images.detach().float()
    if rgb.ndim == 5:
        rgb = rgb[0]
    sampled_rgb = (rgb[:, :, ::stride, ::stride].permute(0, 2, 3, 1).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    return {
        "format": np.asarray(CACHE_FORMAT),
        "depth": sampled_depth.cpu().numpy().astype(np.float32, copy=False),
        "rgb": sampled_rgb.cpu().numpy().astype(np.uint8, copy=False),
        "intrinsics": intrinsics[0].cpu().numpy().astype(np.float32, copy=False),
        "extrinsics": extrinsics[0].cpu().numpy().astype(np.float32, copy=False),
        "vertices_cam": selected_vertices.cpu().numpy().astype(np.float32, copy=False),
        "selection_boxes": selection_boxes.cpu().numpy().astype(np.float32, copy=False),
        "selected_valid": selected_valid.cpu().numpy().astype(np.uint8, copy=False),
        "query_indices": query_indices.cpu().numpy().astype(np.int16, copy=False),
        "confidences": confidences.cpu().numpy().astype(np.float32, copy=False),
        "model_scale": model_scale[0].cpu().numpy().astype(np.float32, copy=False),
        "image_hw": np.asarray((height, width), dtype=np.int32),
        "scene_point_stride": np.asarray(stride, dtype=np.int32),
    }


def save_scale_window_cache(path: str | Path, cache: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **cache)
    temporary.replace(path)


def load_scale_window_cache(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        cache = {key: payload[key] for key in payload.files}
    if str(cache.get("format", "")) != CACHE_FORMAT:
        raise ValueError(f"Unsupported manual-scale cache format in {path}")
    return cache


def evaluate_scale_window_cache(
    cache: dict[str, np.ndarray],
    multiplier: float,
    config: GroundEstimatorConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    multiplier = _validate_multiplier(multiplier)
    frame_payloads: list[dict[str, Any]] = []
    ground_estimates: list[float | None] = []
    for frame_index in range(int(cache["depth"].shape[0])):
        if not bool(cache["selected_valid"][frame_index]):
            frame_payloads.append({"valid": False, "invalid_reason": "no_valid_detector_prediction"})
            ground_estimates.append(None)
            continue
        points_world, _, vertices_world = reconstruct_scale_frame(
            cache, frame_index, multiplier, config, exclude_person=True
        )
        body_low_y = float(vertices_world[:, 1].max().item())
        support_y = select_local_support_y(points_world, vertices_world, config)
        if support_y.numel() < config.min_support_points:
            frame_payloads.append(
                {
                    "valid": False,
                    "invalid_reason": "insufficient_local_scene_points",
                    "query_index": int(cache["query_indices"][frame_index]),
                    "confidence": float(cache["confidences"][frame_index]),
                    "body_low_y": body_low_y,
                    "support_points": int(support_y.numel()),
                }
            )
            ground_estimates.append(None)
            continue
        estimate = float(torch.quantile(support_y, config.ground_quantile).item())
        ground_estimates.append(estimate)
        frame_payloads.append(
            {
                "valid": True,
                "query_index": int(cache["query_indices"][frame_index]),
                "confidence": float(cache["confidences"][frame_index]),
                "body_low_y": body_low_y,
                "frame_ground_y": estimate,
                "support_points": int(support_y.numel()),
            }
        )

    valid_estimates = [value for value in ground_estimates if value is not None]
    if not valid_estimates:
        return frame_payloads, {
            "valid": False,
            "reason": "no_valid_ground_estimate",
            "manual_scale_multiplier": multiplier,
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
    clearances = [float(row["clearance_m"]) for row in frame_payloads if row.get("valid")]
    metrics = compute_physical_metrics(clearances, tolerance_m=config.tolerance_m)
    metrics["mean_abs_clearance_cm"] = (
        float(100.0 * np.mean(np.abs(np.asarray(clearances, dtype=np.float64)))) if clearances else 0.0
    )
    return frame_payloads, {
        "valid": True,
        "method": "cached_metric_geometry_with_window_scale_multiplier",
        "manual_scale_multiplier": multiplier,
        "ground_y": robust_ground_y,
        "valid_frame_estimates": len(valid_estimates),
        "metrics": metrics,
        "config": asdict(config),
    }


def reconstruct_scale_frame(
    cache: dict[str, np.ndarray],
    frame_index: int,
    multiplier: float,
    config: GroundEstimatorConfig,
    exclude_person: bool,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    multiplier = _validate_multiplier(multiplier)
    index = int(frame_index)
    depth = torch.from_numpy(np.asarray(cache["depth"][index], dtype=np.float32)) * multiplier
    intrinsic = torch.from_numpy(np.asarray(cache["intrinsics"][index], dtype=np.float32))
    extrinsic = torch.from_numpy(np.asarray(cache["extrinsics"][index], dtype=np.float32)).clone()
    extrinsic[:3, 3] *= multiplier
    vertices_cam = torch.from_numpy(np.asarray(cache["vertices_cam"][index], dtype=np.float32))
    vertices_world = camera_points_to_world(vertices_cam, extrinsic)

    height, width = (int(value) for value in np.asarray(cache["image_hw"]).reshape(-1)[:2])
    stride = int(np.asarray(cache["scene_point_stride"]).reshape(-1)[0])
    ys = torch.arange(0, height, stride, dtype=torch.float32)
    xs = torch.arange(0, width, stride, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    if tuple(depth.shape) != tuple(grid_y.shape):
        raise ValueError(f"Cached depth grid {tuple(depth.shape)} does not match {(height, width, stride)}")
    z = depth
    x = (grid_x - intrinsic[0, 2]) / intrinsic[0, 0].clamp(min=1e-6) * z
    y = (grid_y - intrinsic[1, 2]) / intrinsic[1, 1].clamp(min=1e-6) * z
    points_cam = torch.stack((x, y, z), dim=-1)
    valid = torch.isfinite(points_cam).all(dim=-1) & (z > 1e-6)
    if config.max_scene_depth_m > 0:
        valid &= z <= float(config.max_scene_depth_m)
    if exclude_person:
        cx, cy, bw, bh = torch.from_numpy(
            np.asarray(cache["selection_boxes"][index], dtype=np.float32)
        ).unbind(dim=-1)
        bw *= 1.0 + 2.0 * float(config.bbox_expand_ratio)
        bh *= 1.0 + 2.0 * float(config.bbox_expand_ratio)
        nx = grid_x / max(float(width), 1.0)
        ny = grid_y / max(float(height), 1.0)
        valid &= ~(
            (nx >= cx - 0.5 * bw)
            & (nx <= cx + 0.5 * bw)
            & (ny >= cy - 0.5 * bh)
            & (ny <= cy + 0.5 * bh)
        )
    points_world = camera_points_to_world(points_cam[valid], extrinsic)
    colors = np.asarray(cache["rgb"][index], dtype=np.uint8)[valid.numpy()]
    return points_world, colors, vertices_world


def _validate_multiplier(multiplier: float) -> float:
    value = float(multiplier)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"Scale multiplier must be finite and positive, got {multiplier}")
    return value
