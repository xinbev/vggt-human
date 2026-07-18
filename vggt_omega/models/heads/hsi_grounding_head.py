from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.utils.contact_geometry import (
    build_sole_vertex_indices,
    canonical_depth,
    estimate_local_support_planes,
)
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.utils.rotation import rot6d_to_axis_angle


class HSIGroundingHead(nn.Module):
    """Geometry-first root grounding with a learned apply gate.

    The support plane determines the only translation candidate.  The network
    cannot invent a new displacement; it only decides whether that candidate
    is trustworthy for the current person.
    """

    def __init__(
        self,
        smpl_model_dir: str,
        hidden_dim: int = 192,
        sole_vertices_per_foot: int = 48,
        exclusion_vertices: int = 0,
        support_window: int = 31,
        support_min_points: int = 32,
        support_max_rmse_m: float = 0.05,
        support_max_depth_m: float = 20.0,
        support_max_point_depth_delta_m: float = 0.75,
        target_clearance_m: float = 0.0,
        clearance_deadzone_m: float = 0.025,
        max_root_delta_m: float = 0.12,
        gate_threshold: float = 0.5,
        hard_gate_eval: bool = True,
        overwrite_refined: bool = True,
        image_size: int = 518,
        min_depth_confidence: float = 0.0,
    ) -> None:
        super().__init__()
        if not smpl_model_dir:
            raise ValueError("HSIGroundingHead requires smpl_model_dir")
        self.smpl = SMPLLayer(smpl_model_dir).eval()
        for parameter in self.smpl.parameters():
            parameter.requires_grad = False
        sole = build_sole_vertex_indices(self.smpl.layer.v_template.detach(), sole_vertices_per_foot)
        self.register_buffer("sole_vertex_indices", sole, persistent=False)
        template = self.smpl.layer.v_template.detach().float().reshape(-1, 3)
        if int(exclusion_vertices) <= 0 or int(exclusion_vertices) >= int(template.shape[0]):
            exclusion = torch.arange(template.shape[0], dtype=torch.long)
        else:
            count = max(int(exclusion_vertices), 16)
            exclusion = torch.linspace(0, template.shape[0] - 1, count).round().long()
        self.register_buffer("exclusion_vertex_indices", exclusion, persistent=False)
        self.support_window = int(support_window)
        self.support_min_points = int(support_min_points)
        self.support_max_rmse_m = float(support_max_rmse_m)
        self.support_max_depth_m = float(support_max_depth_m)
        self.support_max_point_depth_delta_m = float(support_max_point_depth_delta_m)
        self.target_clearance_m = float(target_clearance_m)
        self.clearance_deadzone_m = max(float(clearance_deadzone_m), 0.0)
        self.max_root_delta_m = float(max_root_delta_m)
        self.gate_threshold = min(max(float(gate_threshold), 0.05), 0.95)
        self.hard_gate_eval = bool(hard_gate_eval)
        self.overwrite_refined = bool(overwrite_refined)
        self.image_size = int(image_size)
        self.min_depth_confidence = float(min_depth_confidence)

        # lower-body pose + geometric reliability and support features
        feature_dim = 36 + 2 + 2 + 2 + 3 + 2 + 1 + 1 + 1 + 3
        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gate_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, 0.0)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        depth: torch.Tensor,
        pose_enc: torch.Tensor,
        image_size_hw: tuple[int, int] | None = None,
        intrinsics_override: torch.Tensor | None = None,
        depth_is_metric: bool = False,
        person_boxes: torch.Tensor | None = None,
        depth_confidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pose6d = predictions.get("hsi_refined_pred_pose_6d", predictions.get("pred_pose_6d"))
        poses = predictions.get("hsi_refined_pred_poses", predictions.get("pred_poses"))
        betas = predictions.get("hsi_refined_pred_betas", predictions.get("pred_betas"))
        base_transl = predictions.get("hsi_refined_pred_transl_cam", predictions.get("pred_transl_cam"))
        confs = predictions.get("pred_confs")
        if not all(isinstance(value, torch.Tensor) for value in (pose6d, poses, betas, base_transl, confs)):
            raise ValueError("HSI grounding requires pose, betas, and translation")
        if pose_enc is None:
            raise ValueError("HSI grounding requires pose_enc")
        pose6d = pose6d.float()
        poses = poses.float()
        betas = betas.float()
        base_transl = base_transl.float()
        _validate_smpl_inputs(pose6d, poses, betas, base_transl, confs)
        batch_size, num_frames, num_queries = base_transl.shape[:3]
        image_size_hw = image_size_hw or (self.image_size, self.image_size)
        intrinsics = _resolve_intrinsics(
            pose_enc, image_size_hw, batch_size, num_frames, base_transl.device, base_transl.dtype, intrinsics_override
        )
        depth_hw = canonical_depth(depth).float()
        if depth_hw.shape[:2] != (batch_size, num_frames):
            raise ValueError(f"Grounding depth must have [B,S,H,W], got {tuple(depth_hw.shape)}")
        if not depth_is_metric:
            depth_hw = _apply_scene_affine(depth_hw, predictions)
        confidence_hw = canonical_depth(depth_confidence).float() if isinstance(depth_confidence, torch.Tensor) else None

        vertices, _ = self._decode(pose6d, betas)
        sole = vertices[:, self.sole_vertex_indices].mean(dim=-2)
        sole_cam = sole + base_transl.reshape(-1, 1, 3)
        exclusion_points = vertices[:, self.exclusion_vertex_indices] + base_transl.reshape(-1, 1, 3)
        exclusion_mask = _rasterize_exclusion_mask(
            exclusion_points,
            intrinsics,
            depth_hw.shape[-2:],
            image_size_hw,
            batch_size * num_frames,
            num_queries,
            person_valid=(confs[..., 0] > 0.0).reshape(-1),
        )
        frame_idx = torch.arange(batch_size * num_frames, device=base_transl.device).repeat_interleave(num_queries)
        planes = estimate_local_support_planes(
            depth_hw.reshape(batch_size * num_frames, *depth_hw.shape[-2:]),
            intrinsics,
            sole_cam,
            frame_idx,
            image_size_hw=image_size_hw,
            window_size=self.support_window,
            min_points=self.support_min_points,
            max_rmse_m=self.support_max_rmse_m,
            max_depth_m=self.support_max_depth_m,
            max_point_depth_delta_m=self.support_max_point_depth_delta_m,
            exclusion_mask=exclusion_mask,
            depth_confidence=(confidence_hw.reshape(batch_size * num_frames, *confidence_hw.shape[-2:]) if confidence_hw is not None else None),
            min_depth_confidence=self.min_depth_confidence,
        )
        signed = planes["signed"]
        box_valid = _foot_inside_person_box(
            sole_cam.reshape(batch_size, num_frames, num_queries, 2, 3),
            intrinsics,
            person_boxes,
            image_size_hw,
        ).reshape(-1, 2)
        slot_valid = (confs[..., 0] > 0.0).reshape(-1, 1)
        valid = planes["valid"] & torch.isfinite(signed) & box_valid & slot_valid
        # Estimate a person-level root offset only when the valid feet agree.
        # A grounded support foot plus a swing foot must not pull the whole
        # person; a globally floating/penetrating person has both feet outside
        # the deadzone with the same signed direction.
        active_positive = valid & (signed > self.clearance_deadzone_m)
        active_negative = valid & (signed < -self.clearance_deadzone_m)
        valid_count = valid.sum(dim=-1, keepdim=True)
        positive_count = active_positive.sum(dim=-1, keepdim=True)
        negative_count = active_negative.sum(dim=-1, keepdim=True)
        common_positive = (positive_count == valid_count) & (positive_count > 0)
        common_negative = (negative_count == valid_count) & (negative_count > 0)
        candidate_valid = valid_count > 0
        candidate_use = candidate_valid & (common_positive | common_negative | (valid_count == 1))
        candidate_weights = valid.to(dtype=base_transl.dtype) * candidate_use.to(dtype=base_transl.dtype)
        weight_sum = candidate_weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
        support_signed = (signed * candidate_weights).sum(dim=-1, keepdim=True) / weight_sum
        selected_normal = (planes["normal"] * candidate_weights[..., None]).sum(dim=1) / weight_sum
        selected_normal = torch.nn.functional.normalize(selected_normal, dim=-1, eps=1e-6)
        selected_normal = selected_normal * candidate_use.to(dtype=selected_normal.dtype)
        candidate_valid = candidate_use
        raw_delta_scalar = self.target_clearance_m - support_signed
        delta_scalar = raw_delta_scalar.clamp(-self.max_root_delta_m, self.max_root_delta_m)
        delta_scalar = torch.where(
            raw_delta_scalar.abs() > self.clearance_deadzone_m,
            delta_scalar,
            torch.zeros_like(delta_scalar),
        )
        delta_scalar = torch.where(candidate_valid, delta_scalar, torch.zeros_like(delta_scalar))
        candidate_delta = delta_scalar * selected_normal
        candidate_transl = base_transl.reshape(-1, 3) + candidate_delta

        lower_pose = pose6d.reshape(-1, 24, 6)[:, [1, 2, 4, 5, 7, 8]].reshape(-1, 36)
        support_rmse = torch.where(valid, planes["rmse"], torch.zeros_like(planes["rmse"]))
        point_count = planes["point_count"].to(dtype=base_transl.dtype).clamp(max=256.0) / 256.0
        foot_range = torch.where(valid, signed, torch.zeros_like(signed)).amax(dim=-1) - torch.where(
            valid, signed, torch.zeros_like(signed)
        ).amin(dim=-1)
        feature_parts = {
            "lower_pose": lower_pose,
            "signed": signed,
            "support_rmse": support_rmse,
            "valid": valid.to(dtype=base_transl.dtype),
            "selected_normal": selected_normal,
            "point_count": point_count,
            "delta_scalar": delta_scalar,
            "delta_abs": delta_scalar.abs(),
            "foot_range": foot_range.unsqueeze(-1),
            "base_transl": base_transl.reshape(-1, 3),
        }
        _validate_feature_parts(feature_parts, expected_rows=batch_size * num_frames * num_queries)
        features = torch.cat(list(feature_parts.values()), dim=-1)
        if features.shape[-1] != self.gate_mlp[0].in_features:
            raise RuntimeError(
                f"Grounding gate feature width {features.shape[-1]} does not match "
                f"configured width {self.gate_mlp[0].in_features}"
            )
        hidden = self.gate_mlp(torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0))
        probability = torch.sigmoid(self.gate_head(hidden)) * candidate_valid.to(dtype=base_transl.dtype)
        if self.training or not self.hard_gate_eval:
            apply_gate = probability
        else:
            apply_gate = (probability >= self.gate_threshold).to(dtype=probability.dtype)
        refined_transl = base_transl.reshape(-1, 3) + apply_gate * candidate_delta
        refined_signed = signed + (
            apply_gate.unsqueeze(-1) * candidate_delta[:, None, :] * planes["normal"]
        ).sum(dim=-1)
        expected_flat = batch_size * num_frames * num_queries
        _require_shape(candidate_delta, (expected_flat, 3), "candidate_delta")
        _require_shape(probability, (expected_flat, 1), "gate_probability")
        _require_shape(refined_transl, (expected_flat, 3), "refined_transl")
        _require_shape(refined_signed, (expected_flat, 2), "refined_signed")
        outputs = {
            "hsi_grounding_base_pred_transl_cam": base_transl,
            "hsi_grounding_candidate_pred_transl_cam": candidate_transl.reshape(batch_size, num_frames, num_queries, 3),
            "hsi_grounding_candidate_delta": candidate_delta.reshape(batch_size, num_frames, num_queries, 3),
            "hsi_grounding_refined_pred_transl_cam": refined_transl.reshape(batch_size, num_frames, num_queries, 3),
            "hsi_grounding_gate_probability": probability.reshape(batch_size, num_frames, num_queries, 1),
            "hsi_grounding_gate": apply_gate.reshape(batch_size, num_frames, num_queries, 1),
            "hsi_grounding_support_signed_m": signed.reshape(batch_size, num_frames, num_queries, 2),
            "hsi_grounding_refined_signed_m": refined_signed.reshape(batch_size, num_frames, num_queries, 2),
            "hsi_grounding_support_valid": valid.reshape(batch_size, num_frames, num_queries, 2),
            "hsi_grounding_candidate_valid": candidate_valid.reshape(batch_size, num_frames, num_queries, 1),
            "hsi_grounding_support_rmse": planes["rmse"].reshape(batch_size, num_frames, num_queries, 2),
            "hsi_grounding_support_normal": planes["normal"].reshape(batch_size, num_frames, num_queries, 2, 3),
            "hsi_grounding_support_point_count": planes["point_count"].reshape(batch_size, num_frames, num_queries, 2),
        }
        if self.overwrite_refined:
            outputs.update(
                {
                    "hsi_refined_pred_pose_6d": pose6d,
                    "hsi_refined_pred_poses": poses,
                    "hsi_refined_pred_betas": betas,
                    "hsi_refined_pred_transl_cam": refined_transl.reshape(batch_size, num_frames, num_queries, 3),
                }
            )
        return outputs

    def _decode(self, pose6d: torch.Tensor, betas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        aa = rot6d_to_axis_angle(pose6d.reshape(-1, 24, 6)).reshape(-1, 72)
        vertices, joints = self.smpl(aa.float(), betas.reshape(-1, betas.shape[-1]).float())
        return vertices.to(dtype=pose6d.dtype), joints.to(dtype=pose6d.dtype)


def _resolve_intrinsics(pose_enc, image_size_hw, batch_size, num_frames, device, dtype, override):
    if override is not None:
        return override.to(device=device, dtype=dtype).reshape(batch_size * num_frames, 3, 3)
    _, intrinsics = encoding_to_camera(pose_enc, image_size_hw=image_size_hw, build_intrinsics=True)
    return intrinsics.reshape(batch_size * num_frames, 3, 3).to(device=device, dtype=dtype)


def _apply_scene_affine(depth: torch.Tensor, predictions: dict[str, torch.Tensor]) -> torch.Tensor:
    scale = predictions.get("hsi_scene_scale")
    bias = predictions.get("hsi_scene_depth_bias")
    if scale is None or bias is None:
        return depth
    while scale.ndim < depth.ndim:
        scale = scale.unsqueeze(-1)
    while bias.ndim < depth.ndim:
        bias = bias.unsqueeze(-1)
    return depth * scale.to(dtype=depth.dtype) + bias.to(dtype=depth.dtype)


def _rasterize_exclusion_mask(
    points,
    intrinsics,
    depth_hw,
    image_size_hw,
    flat_frames,
    num_queries,
    person_valid,
):
    height, width = int(depth_hw[0]), int(depth_hw[1])
    image_h, image_w = int(image_size_hw[0]), int(image_size_hw[1])
    points = points.reshape(flat_frames * num_queries, points.shape[1], 3)
    z = points[..., 2].clamp(min=1e-6)
    intr = intrinsics.repeat_interleave(num_queries, dim=0)
    x = intr[:, None, 0, 0] * points[..., 0] / z + intr[:, None, 0, 2]
    y = intr[:, None, 1, 1] * points[..., 1] / z + intr[:, None, 1, 2]
    x = (x * width / max(image_w, 1)).round().long()
    y = (y * height / max(image_h, 1)).round().long()
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (points[..., 2] > 1e-6)
    if person_valid.shape != (flat_frames * num_queries,):
        raise ValueError(
            f"Grounding exclusion person_valid shape {tuple(person_valid.shape)} "
            f"!= {(flat_frames * num_queries,)}"
        )
    valid = valid & person_valid.to(device=points.device).bool()[:, None]
    frame = torch.arange(flat_frames, device=points.device).repeat_interleave(num_queries)
    frame = frame[:, None].expand_as(x)
    flat = (frame * height + y.clamp(0, height - 1)) * width + x.clamp(0, width - 1)
    mask = torch.zeros(flat_frames * height * width, dtype=torch.bool, device=points.device)
    mask.scatter_(0, flat[valid], torch.ones_like(flat[valid], dtype=torch.bool))
    mask = mask.reshape(flat_frames, 1, height, width).float()
    return (F.max_pool2d(mask, kernel_size=5, stride=1, padding=2) > 0.0)[:, 0]


def _foot_inside_person_box(sole_cam, intrinsics, person_boxes, image_size_hw):
    batch_size, num_frames, num_queries = sole_cam.shape[:3]
    if not isinstance(person_boxes, torch.Tensor):
        return torch.ones(*sole_cam.shape[:-1], dtype=torch.bool, device=sole_cam.device)
    if person_boxes.shape != (batch_size, num_frames, num_queries, 4):
        raise ValueError(
            f"person_boxes must have shape {(batch_size, num_frames, num_queries, 4)}, "
            f"got {tuple(person_boxes.shape)}"
        )
    points = sole_cam.reshape(batch_size * num_frames * num_queries, 2, 3)
    intr = intrinsics.repeat_interleave(num_queries, dim=0)
    z = points[..., 2].clamp(min=1e-6)
    x = intr[:, None, 0, 0] * points[..., 0] / z + intr[:, None, 0, 2]
    y = intr[:, None, 1, 1] * points[..., 1] / z + intr[:, None, 1, 2]
    image_h, image_w = float(image_size_hw[0]), float(image_size_hw[1])
    projected = torch.stack([x / max(image_w, 1.0), y / max(image_h, 1.0)], dim=-1)
    boxes = person_boxes.to(device=sole_cam.device, dtype=sole_cam.dtype).reshape(-1, 4)
    box_min = boxes[:, None, :2] - 0.60 * boxes[:, None, 2:]
    box_max = boxes[:, None, :2] + 0.60 * boxes[:, None, 2:]
    inside = (projected >= box_min).all(dim=-1) & (projected <= box_max).all(dim=-1)
    return inside.reshape(batch_size, num_frames, num_queries, 2)


def _validate_feature_parts(parts: dict[str, torch.Tensor], expected_rows: int) -> None:
    problems = []
    for name, value in parts.items():
        if value.ndim != 2:
            problems.append(f"{name}={tuple(value.shape)} (expected [M,C])")
        elif value.shape[0] != expected_rows:
            problems.append(f"{name}={tuple(value.shape)} (expected M={expected_rows})")
    if problems:
        raise RuntimeError("Invalid grounding gate feature shapes: " + "; ".join(problems))


def _validate_smpl_inputs(pose6d, poses, betas, transl, confs) -> None:
    if transl.ndim != 4 or transl.shape[-1] != 3:
        raise ValueError(f"Grounding translation must have [B,S,Q,3], got {tuple(transl.shape)}")
    prefix = tuple(transl.shape[:3])
    expected = {
        "pose6d": (prefix, 144),
        "poses": (prefix, 72),
        "betas": (prefix, None),
        "confs": (prefix, 1),
    }
    values = {"pose6d": pose6d, "poses": poses, "betas": betas, "confs": confs}
    for name, value in values.items():
        expected_prefix, expected_width = expected[name]
        if value.ndim != 4 or tuple(value.shape[:3]) != expected_prefix:
            raise ValueError(
                f"Grounding {name} must start with [B,S,Q]={expected_prefix}, got {tuple(value.shape)}"
            )
        if expected_width is not None and value.shape[-1] != expected_width:
            raise ValueError(f"Grounding {name} width must be {expected_width}, got {tuple(value.shape)}")


def _require_shape(value: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    if tuple(value.shape) != tuple(expected):
        raise RuntimeError(f"Grounding {name} shape {tuple(value.shape)} != expected {tuple(expected)}")
