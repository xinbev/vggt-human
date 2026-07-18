from __future__ import annotations

import torch
import torch.nn as nn

from vggt_omega.models.heads.hsi_human_scene_align_head import (
    _camera_basis,
    _canonical_depth,
    _deterministic_fps_indices,
    _local_scene_points,
    _masked_quantile,
    _project_points,
    _resolve_image_size_hw,
    _resolve_intrinsics,
)
from vggt_omega.models.smpl_layer import SMPLLayer


class HSITranslationRefineV4Head(nn.Module):
    """Decoupled translation correction and application gate.

    Stage2 V4-A1 trains only `correction_trunk` and `correction_head`. The gate
    has a separate trunk so later gate training cannot alter candidate deltas.
    """

    def __init__(
        self,
        smpl_model_dir: str,
        hidden_dim: int = 256,
        num_sample_vertices: int = 128,
        local_window: int = 7,
        min_correspondences: int = 12,
        max_ray_ratio: float = 0.25,
        ray_parameterization: str = "residual_gain",
        max_ray_gain: float = 4.0,
        max_tangent_delta_m: float = 0.12,
        max_correspondence_distance_m: float = 3.5,
        residual_mad_multiplier: float = 3.0,
        max_depth_m: float = 20.0,
        phase: str = "correction",
        gate_threshold: float = 0.5,
        overwrite_refined: bool = True,
        image_size: int = 518,
    ) -> None:
        super().__init__()
        if not smpl_model_dir:
            raise ValueError("HSITranslationRefineV4Head requires smpl_model_dir")
        if local_window % 2 != 1:
            raise ValueError(f"hsi_v4_local_window must be odd, got {local_window}")
        self.smpl = SMPLLayer(smpl_model_dir).eval()
        for parameter in self.smpl.parameters():
            parameter.requires_grad = False
        self.local_window = int(local_window)
        self.min_correspondences = max(int(min_correspondences), 1)
        self.max_ray_ratio = float(max_ray_ratio)
        self.ray_parameterization = str(ray_parameterization or "residual_gain").lower()
        if self.ray_parameterization not in {"residual_gain", "signed_magnitude"}:
            raise ValueError(f"Unsupported V4 ray parameterization: {self.ray_parameterization!r}")
        self.max_ray_gain = max(float(max_ray_gain), 0.0)
        self.max_tangent_delta_m = float(max_tangent_delta_m)
        self.max_correspondence_distance_m = float(max_correspondence_distance_m)
        self.residual_mad_multiplier = float(residual_mad_multiplier)
        self.max_depth_m = float(max_depth_m)
        self.phase = str(phase or "correction").lower()
        if self.phase not in {"correction", "gate", "combined"}:
            raise ValueError(f"Unsupported HSI V4 phase: {self.phase!r}")
        self.gate_threshold = min(max(float(gate_threshold), 0.0), 1.0)
        self.overwrite_refined = bool(overwrite_refined)
        self.image_size = int(image_size)

        indices = _deterministic_fps_indices(self.smpl.layer.v_template.detach().float(), int(num_sample_vertices))
        self.register_buffer("sample_vertex_indices", indices, persistent=False)

        feature_dim = 3 + 1 + 1 + 1 + 1 + 12 + 12 + 3 + 1 + 1 + 2 + 2
        self.correction_trunk = _make_mlp(feature_dim, hidden_dim)
        self.correction_head = nn.Linear(hidden_dim, 3)
        nn.init.normal_(self.correction_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.correction_head.bias)
        with torch.no_grad():
            self.correction_head.bias[0] = -2.0

        gate_feature_dim = feature_dim + 4
        self.gate_trunk = _make_mlp(gate_feature_dim, hidden_dim)
        self.gate_head = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.gate_head.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.gate_head.bias, -2.2)

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
        pose6d = predictions.get("pred_pose_6d")
        poses = predictions.get("pred_poses")
        betas = predictions.get("pred_betas")
        base_transl = predictions.get("pred_transl_cam")
        confs = predictions.get("pred_confs")
        if any(value is None for value in (pose6d, poses, betas, base_transl, confs)):
            raise ValueError("HSI V4 requires SMPL pose/betas/transl/confs")
        if depth is None or pose_enc is None:
            raise ValueError("HSI V4 requires depth and camera encoding")

        pose6d = pose6d.float()
        poses = poses.float()
        betas = betas.float()
        base_transl = base_transl.float()
        confs = confs.float()
        batch_size, num_frames, num_queries = base_transl.shape[:3]
        depth_hw = _canonical_depth(depth).float()
        if depth_hw.shape[:2] != (batch_size, num_frames):
            raise ValueError(f"Expected depth [B,S,H,W], got {tuple(depth_hw.shape)}")
        image_size_hw = _resolve_image_size_hw(image_size_hw, self.image_size)
        intrinsics = _resolve_intrinsics(
            pose_enc,
            image_size_hw=image_size_hw,
            batch_size=batch_size,
            num_frames=num_frames,
            device=base_transl.device,
            dtype=base_transl.dtype,
            intrinsics_override=intrinsics_override,
        )
        metric_depth = depth_hw if depth_is_metric else _metric_depth(predictions, depth_hw)
        sample_points = self._sample_smpl_points(poses, betas, base_transl)
        scene_points, valid = _local_scene_points(
            depth_hw=metric_depth,
            points_cam=sample_points,
            intrinsics=intrinsics,
            image_size_hw=image_size_hw,
            local_window=self.local_window,
            person_boxes=person_boxes,
            depth_confidence=depth_confidence,
            min_depth_confidence=0.0,
            max_correspondence_distance_m=self.max_correspondence_distance_m,
            residual_mad_multiplier=self.residual_mad_multiplier,
            max_depth_m=self.max_depth_m,
        )
        valid = valid & (confs[..., :1].unsqueeze(-2) > 0.0)
        valid_mask = valid[..., 0]
        valid_f = valid.to(dtype=base_transl.dtype)
        valid_count = valid_f.sum(dim=-2)
        valid_ratio = valid_f.mean(dim=-2)
        eligible = (valid_count >= float(self.min_correspondences)) & (confs[..., :1] > 0.0)

        residual = scene_points - sample_points
        ray, tangent_x, tangent_y = _camera_basis(base_transl)
        residual_basis = torch.stack(
            [
                (residual * ray.unsqueeze(-2)).sum(dim=-1),
                (residual * tangent_x.unsqueeze(-2)).sum(dim=-1),
                (residual * tangent_y.unsqueeze(-2)).sum(dim=-1),
            ],
            dim=-1,
        )
        p10 = _masked_quantile(residual_basis, valid, 0.10)
        median = _masked_quantile(residual_basis, valid, 0.50)
        p90 = _masked_quantile(residual_basis, valid, 0.90)
        mad = _masked_quantile((residual_basis - median.unsqueeze(-2)).abs(), valid, 0.50)
        denom = valid_f.sum(dim=-2).clamp(min=1.0)
        mean_abs = (residual_basis.abs() * valid_f).sum(dim=-2) / denom
        ray_values = residual_basis[..., 0]
        median_ray = median[..., :1]
        sign_agree = ((ray_values * median_ray) > 0.0) & valid_mask
        sign_agreement = sign_agree.to(dtype=base_transl.dtype).sum(dim=-1, keepdim=True) / valid_count.clamp(min=1.0)
        ray_inlier = (ray_values - median_ray).abs() <= (3.0 * mad[..., :1].clamp(min=0.005))
        inlier_ratio = (ray_inlier & valid_mask).to(dtype=base_transl.dtype).sum(dim=-1, keepdim=True) / valid_count.clamp(min=1.0)

        base_norm = torch.linalg.norm(base_transl, dim=-1, keepdim=True).clamp(min=1e-3)
        robust = torch.cat([p10, median, p90, mad], dim=-1)
        robust_normalized = robust / base_norm
        projected_center = _project_points(
            base_transl.reshape(-1, 1, 3), intrinsics.repeat_interleave(num_queries, dim=0)
        ).reshape(batch_size, num_frames, num_queries, 2)
        projected_center = projected_center / base_transl.new_tensor(
            [max(float(image_size_hw[1] - 1), 1.0), max(float(image_size_hw[0] - 1), 1.0)]
        )
        box_size = base_transl.new_zeros(*base_transl.shape[:3], 2)
        if isinstance(person_boxes, torch.Tensor):
            box_size = person_boxes[..., 2:].to(device=base_transl.device, dtype=base_transl.dtype)
        features = torch.cat(
            [
                base_transl / max(self.max_depth_m, 1.0),
                base_norm / max(self.max_depth_m, 1.0),
                confs.clamp(0.0, 1.0),
                valid_ratio,
                valid_count / float(sample_points.shape[-2]),
                robust,
                robust_normalized,
                mean_abs,
                sign_agreement,
                inlier_ratio,
                projected_center,
                box_size,
            ],
            dim=-1,
        )
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        correction_hidden = self.correction_trunk(features)
        correction_raw = self.correction_head(correction_hidden)
        if self.ray_parameterization == "residual_gain":
            ray_gain = self.max_ray_gain * torch.sigmoid(correction_raw[..., :1])
            ray_delta = median_ray * ray_gain
        else:
            ray_ratio = self.max_ray_ratio * torch.sigmoid(correction_raw[..., :1])
            ray_delta = torch.sign(median_ray) * ray_ratio * base_norm
        tangent_coeff = self.max_tangent_delta_m * torch.tanh(correction_raw[..., 1:])
        candidate_coeff = torch.cat([ray_delta, tangent_coeff], dim=-1)
        candidate_delta = (
            ray_delta * ray
            + tangent_coeff[..., :1] * tangent_x
            + tangent_coeff[..., 1:] * tangent_y
        )
        candidate_delta = candidate_delta * eligible.to(dtype=candidate_delta.dtype)
        candidate_transl = base_transl + candidate_delta

        gate_features = torch.cat(
            [features.detach(), candidate_coeff.detach(), torch.linalg.norm(candidate_delta.detach(), dim=-1, keepdim=True)],
            dim=-1,
        )
        gate_logit = self.gate_head(self.gate_trunk(gate_features))
        gate_probability = torch.sigmoid(gate_logit)
        hard_apply = (gate_probability >= self.gate_threshold) & eligible
        if self.phase == "correction":
            apply = eligible
        else:
            hard = hard_apply.to(dtype=base_transl.dtype)
            apply = hard + gate_probability - gate_probability.detach() if self.phase == "combined" else hard
        refined_transl = base_transl + apply.to(dtype=base_transl.dtype) * candidate_delta

        outputs = {
            "hsi_v4_base_pred_transl_cam": base_transl,
            "hsi_v4_candidate_delta_transl_cam": candidate_delta,
            "hsi_v4_candidate_pred_transl_cam": candidate_transl,
            "hsi_v4_candidate_coeff": candidate_coeff,
            "hsi_v4_ray_parameterization": torch.tensor(
                0.0 if self.ray_parameterization == "residual_gain" else 1.0,
                device=base_transl.device,
                dtype=base_transl.dtype,
            ),
            "hsi_v4_gate_logit": gate_logit,
            "hsi_v4_gate_probability": gate_probability,
            "hsi_v4_hard_apply": hard_apply.to(dtype=base_transl.dtype),
            "hsi_v4_geometry_eligible": eligible.to(dtype=base_transl.dtype),
            "hsi_v4_valid_count": valid_count,
            "hsi_v4_valid_ratio": valid_ratio,
            "hsi_v4_residual_basis_p10": p10.detach(),
            "hsi_v4_residual_basis_median": median.detach(),
            "hsi_v4_residual_basis_p90": p90.detach(),
            "hsi_v4_residual_basis_mad": mad.detach(),
            "hsi_v4_residual_sign_agreement": sign_agreement.detach(),
            "hsi_v4_residual_inlier_ratio": inlier_ratio.detach(),
            "hsi_v4_refined_pred_transl_cam": refined_transl,
        }
        if self.overwrite_refined:
            outputs["hsi_refined_pred_pose_6d"] = pose6d
            outputs["hsi_refined_pred_poses"] = poses
            outputs["hsi_refined_pred_betas"] = betas
            outputs["hsi_refined_pred_transl_cam"] = refined_transl
        return outputs

    def _sample_smpl_points(self, poses: torch.Tensor, betas: torch.Tensor, transl: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, num_queries = transl.shape[:3]
        vertices, joints = self.smpl(poses.reshape(-1, 72).float(), betas.reshape(-1, betas.shape[-1]).float())
        vertices = vertices.to(device=transl.device, dtype=transl.dtype)
        joints = joints[:, :24].to(device=transl.device, dtype=transl.dtype)
        sampled = vertices[:, self.sample_vertex_indices.to(vertices.device)]
        points = torch.cat([joints, sampled], dim=1) + transl.reshape(-1, 1, 3)
        return points.reshape(batch_size, num_frames, num_queries, points.shape[1], 3)


def _make_mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
    )


def _metric_depth(predictions: dict[str, torch.Tensor], depth_hw: torch.Tensor) -> torch.Tensor:
    scale = predictions.get("hsi_scene_scale")
    bias = predictions.get("hsi_scene_depth_bias")
    if scale is None or bias is None:
        return depth_hw
    while scale.ndim < depth_hw.ndim:
        scale = scale.unsqueeze(-1)
    while bias.ndim < depth_hw.ndim:
        bias = bias.unsqueeze(-1)
    return depth_hw * scale.to(device=depth_hw.device, dtype=depth_hw.dtype) + bias.to(
        device=depth_hw.device, dtype=depth_hw.dtype
    )
