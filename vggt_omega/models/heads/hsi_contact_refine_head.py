from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.utils.contact_geometry import build_sole_vertex_indices, canonical_depth, estimate_local_support_planes
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.utils.rotation import axis_angle_to_rotmat, rot6d_to_axis_angle, rot6d_to_rotmat


LOWER_BODY_JOINTS = (1, 2, 4, 5, 7, 8)


class HSIContactRefineHead(nn.Module):
    """Bounded root-normal and lower-leg contact correction."""

    def __init__(
        self,
        smpl_model_dir: str,
        hidden_dim: int = 256,
        sole_vertices_per_foot: int = 48,
        support_window: int = 21,
        support_min_points: int = 24,
        support_max_rmse_m: float = 0.05,
        support_max_depth_m: float = 20.0,
        max_root_normal_delta_m: float = 0.12,
        max_hip_delta_deg: float = 4.0,
        max_knee_delta_deg: float = 8.0,
        max_ankle_delta_deg: float = 10.0,
        use_temporal_velocity: bool = False,
        max_velocity_m: float = 0.25,
        overwrite_refined: bool = True,
        image_size: int = 518,
    ) -> None:
        super().__init__()
        self.smpl = SMPLLayer(smpl_model_dir).eval()
        for parameter in self.smpl.parameters():
            parameter.requires_grad = False
        sole = build_sole_vertex_indices(self.smpl.layer.v_template.detach(), sole_vertices_per_foot)
        self.register_buffer("sole_vertex_indices", sole, persistent=False)
        self.support_window = int(support_window)
        self.support_min_points = int(support_min_points)
        self.support_max_rmse_m = float(support_max_rmse_m)
        self.support_max_depth_m = float(support_max_depth_m)
        self.max_root_normal_delta_m = float(max_root_normal_delta_m)
        self.use_temporal_velocity = bool(use_temporal_velocity)
        self.max_velocity_m = max(float(max_velocity_m), 1e-3)
        self.overwrite_refined = bool(overwrite_refined)
        self.image_size = int(image_size)
        max_degrees = [max_hip_delta_deg, max_hip_delta_deg, max_knee_delta_deg, max_knee_delta_deg, max_ankle_delta_deg, max_ankle_delta_deg]
        self.register_buffer("max_pose_delta_rad", torch.tensor(max_degrees).float() * (math.pi / 180.0), persistent=False)

        feature_dim = 3 + 36 + 2 + 2 + 2 + 6 + 2
        if self.use_temporal_velocity:
            feature_dim += 2 + 2
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.root_normal_head = _zero_last(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)))
        self.lower_pose_head = _zero_last(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 18)))
        self.contact_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        depth: torch.Tensor,
        pose_enc: torch.Tensor,
        image_size_hw: tuple[int, int] | None = None,
        intrinsics_override: torch.Tensor | None = None,
        depth_is_metric: bool = False,
    ) -> dict[str, torch.Tensor]:
        pose6d = predictions.get("hsi_refined_pred_pose_6d", predictions.get("pred_pose_6d"))
        betas = predictions.get("hsi_refined_pred_betas", predictions.get("pred_betas"))
        transl = predictions.get("hsi_refined_pred_transl_cam", predictions.get("pred_transl_cam"))
        if pose6d is None or betas is None or transl is None:
            raise ValueError("HSI contact refinement requires pose, betas, and translation")
        pose6d = pose6d.float()
        betas = betas.float()
        transl = transl.float()
        batch_size, num_frames, num_queries = transl.shape[:3]
        image_size_hw = image_size_hw or (self.image_size, self.image_size)
        intrinsics = _resolve_intrinsics(
            pose_enc,
            image_size_hw,
            batch_size,
            num_frames,
            transl.device,
            transl.dtype,
            intrinsics_override,
        )
        depth_hw = canonical_depth(depth).float()
        if not depth_is_metric:
            depth_hw = _apply_scene_affine(depth_hw, predictions)

        vertices, _ = self._decode(pose6d, betas)
        sole = vertices[:, self.sole_vertex_indices].mean(dim=-2)
        flat_transl = transl.reshape(-1, 1, 3)
        if sole.shape != (flat_transl.shape[0], 2, 3):
            raise ValueError(
                f"Expected sole centers [B*S*Q,2,3], got {tuple(sole.shape)} "
                f"for translations {tuple(flat_transl.shape)}"
            )
        sole_cam = sole + flat_transl
        flat_frames = batch_size * num_frames
        frame_idx = torch.arange(flat_frames, device=transl.device).repeat_interleave(num_queries)
        planes = estimate_local_support_planes(
            depth_hw.reshape(flat_frames, *depth_hw.shape[-2:]),
            intrinsics,
            sole_cam,
            frame_idx,
            image_size_hw=image_size_hw,
            window_size=self.support_window,
            min_points=self.support_min_points,
            max_rmse_m=self.support_max_rmse_m,
            max_depth_m=self.support_max_depth_m,
        )
        if self.use_temporal_velocity:
            temporal_velocity, temporal_velocity_valid = _compute_temporal_foot_velocity(
                sole_cam=sole_cam.reshape(batch_size, num_frames, num_queries, 2, 3),
                pose_enc=pose_enc,
                track_ids=predictions.get("assigned_track_ids"),
                track_mask=predictions.get("assigned_track_mask"),
            )
        else:
            temporal_velocity = sole_cam.new_zeros(batch_size, num_frames, num_queries, 2)
            temporal_velocity_valid = torch.zeros_like(temporal_velocity, dtype=torch.bool)
        lower_pose = pose6d.reshape(-1, 24, 6)[:, list(LOWER_BODY_JOINTS)].reshape(-1, 36)
        feature_parts = [
            transl.reshape(-1, 3),
            lower_pose,
            planes["signed"],
            planes["rmse"],
            planes["valid"].to(dtype=transl.dtype),
            planes["normal"].reshape(-1, 6),
            planes["point_count"].to(dtype=transl.dtype).clamp(max=256.0) / 256.0,
        ]
        if self.use_temporal_velocity:
            feature_parts.extend(
                [
                    temporal_velocity.reshape(-1, 2).clamp(max=self.max_velocity_m) / self.max_velocity_m,
                    temporal_velocity_valid.reshape(-1, 2).to(dtype=transl.dtype),
                ]
            )
        features = torch.cat(feature_parts, dim=-1)
        hidden = self.mlp(features)
        contact_logits = self.contact_head(hidden)
        valid_f = planes["valid"].to(dtype=transl.dtype)
        contact_probability = torch.sigmoid(contact_logits) * valid_f
        normal_weights = contact_probability[..., None]
        normal = (planes["normal"] * normal_weights).sum(dim=1) / normal_weights.sum(dim=1).clamp(min=1e-6)
        normal = F.normalize(normal, dim=-1, eps=1e-6)
        any_valid = planes["valid"].any(dim=1, keepdim=True).to(dtype=transl.dtype)
        person_contact_gate = contact_probability.amax(dim=1, keepdim=True) * any_valid
        root_scalar = torch.tanh(self.root_normal_head(hidden)) * self.max_root_normal_delta_m * person_contact_gate
        root_delta = root_scalar * normal
        refined_transl = transl.reshape(-1, 3) + root_delta

        raw_pose_delta = torch.tanh(self.lower_pose_head(hidden)).reshape(-1, len(LOWER_BODY_JOINTS), 3)
        pose_delta = raw_pose_delta * self.max_pose_delta_rad.to(device=raw_pose_delta.device, dtype=raw_pose_delta.dtype)[None, :, None]
        leg_contact_gate = contact_probability[:, [0, 1, 0, 1, 0, 1]].unsqueeze(-1)
        pose_delta = pose_delta * leg_contact_gate
        refined_pose6d = _compose_lower_pose_delta(pose6d.reshape(-1, 24, 6), pose_delta)
        refined_pose6d = refined_pose6d.reshape(batch_size, num_frames, num_queries, 144)
        refined_transl = refined_transl.reshape(batch_size, num_frames, num_queries, 3)
        refined_poses = rot6d_to_axis_angle(refined_pose6d.reshape(-1, 24, 6)).reshape(batch_size, num_frames, num_queries, 72)

        refined_vertices, _ = self._decode(refined_pose6d, betas)
        refined_sole = refined_vertices[:, self.sole_vertex_indices].mean(dim=-2) + refined_transl.reshape(-1, 1, 3)
        refined_signed = ((refined_sole - planes["center"]) * planes["normal"]).sum(dim=-1)
        outputs = {
            "hsi_contact_base_pred_pose_6d": pose6d,
            "hsi_contact_base_pred_transl_cam": transl,
            "hsi_contact_refined_pred_pose_6d": refined_pose6d,
            "hsi_contact_refined_pred_poses": refined_poses,
            "hsi_contact_refined_pred_transl_cam": refined_transl,
            "hsi_contact_root_normal_delta": root_delta.reshape(batch_size, num_frames, num_queries, 3),
            "hsi_contact_lower_pose_delta_aa": pose_delta.reshape(batch_size, num_frames, num_queries, len(LOWER_BODY_JOINTS), 3),
            "hsi_contact_foot_logits": contact_logits.reshape(batch_size, num_frames, num_queries, 2),
            "hsi_contact_foot_probability": contact_probability.reshape(batch_size, num_frames, num_queries, 2),
            "hsi_contact_person_gate": person_contact_gate.reshape(batch_size, num_frames, num_queries, 1),
            "hsi_contact_support_center": planes["center"].reshape(batch_size, num_frames, num_queries, 2, 3),
            "hsi_contact_support_normal": planes["normal"].reshape(batch_size, num_frames, num_queries, 2, 3),
            "hsi_contact_support_rmse": planes["rmse"].reshape(batch_size, num_frames, num_queries, 2),
            "hsi_contact_support_valid": planes["valid"].reshape(batch_size, num_frames, num_queries, 2),
            "hsi_contact_base_signed_m": planes["signed"].reshape(batch_size, num_frames, num_queries, 2),
            "hsi_contact_refined_signed_m": refined_signed.reshape(batch_size, num_frames, num_queries, 2),
            "hsi_contact_foot_velocity_m": temporal_velocity,
            "hsi_contact_foot_velocity_valid": temporal_velocity_valid.to(dtype=transl.dtype),
        }
        if self.overwrite_refined:
            outputs.update(
                {
                    "hsi_refined_pred_pose_6d": refined_pose6d,
                    "hsi_refined_pred_poses": refined_poses,
                    "hsi_refined_pred_betas": betas,
                    "hsi_refined_pred_transl_cam": refined_transl,
                }
            )
        return outputs

    def _decode(self, pose6d: torch.Tensor, betas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        aa = rot6d_to_axis_angle(pose6d.reshape(-1, 24, 6)).reshape(-1, 72)
        vertices, joints = self.smpl(aa.float(), betas.reshape(-1, betas.shape[-1]).float())
        return vertices.to(dtype=pose6d.dtype), joints.to(dtype=pose6d.dtype)


def _compose_lower_pose_delta(base_pose6d: torch.Tensor, delta_aa: torch.Tensor) -> torch.Tensor:
    base_rot = rot6d_to_rotmat(base_pose6d)
    delta_rot = axis_angle_to_rotmat(delta_aa)
    out = base_rot.clone()
    indices = torch.tensor(LOWER_BODY_JOINTS, device=base_pose6d.device, dtype=torch.long)
    out[:, indices] = delta_rot @ base_rot[:, indices]
    return out[..., :2, :].reshape(*base_pose6d.shape[:-2], 24, 6)


def _compute_temporal_foot_velocity(
    sole_cam: torch.Tensor,
    pose_enc: torch.Tensor,
    track_ids: torch.Tensor | None,
    track_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute inference-available per-foot velocity in world coordinates.

    Contact teachers use neighboring frames and track-consistent sole motion.
    This reproduces that observable at inference time without reading GT
    contact velocity or labels.
    """
    if sole_cam.ndim != 5 or sole_cam.shape[-2:] != (2, 3):
        raise ValueError(f"sole_cam must have shape [B,S,Q,2,3], got {tuple(sole_cam.shape)}")
    batch_size, num_frames, num_queries = sole_cam.shape[:3]
    velocity = sole_cam.new_zeros(batch_size, num_frames, num_queries, 2)
    velocity_count = sole_cam.new_zeros(batch_size, num_frames, num_queries, 2)
    if num_frames < 2 or not isinstance(track_ids, torch.Tensor):
        return velocity, velocity_count.bool()
    if track_ids.shape[:3] != (batch_size, num_frames, num_queries):
        raise ValueError(
            f"assigned_track_ids shape must start with {(batch_size, num_frames, num_queries)}, "
            f"got {tuple(track_ids.shape)}"
        )
    ids = track_ids.to(device=sole_cam.device, dtype=torch.long)
    valid_tracks = ids >= 0
    if isinstance(track_mask, torch.Tensor):
        if track_mask.shape[:3] != (batch_size, num_frames, num_queries):
            raise ValueError(
                f"assigned_track_mask shape must start with {(batch_size, num_frames, num_queries)}, "
                f"got {tuple(track_mask.shape)}"
            )
        valid_tracks = valid_tracks & track_mask.to(device=sole_cam.device).bool()

    extrinsics, _ = encoding_to_camera(pose_enc, image_size_hw=(1, 1), build_intrinsics=False)
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    centered = sole_cam - translation[:, :, None, None, :]
    sole_world = torch.einsum("bsij,bsqfj->bsqfi", rotation.transpose(-1, -2), centered)

    for frame in range(num_frames):
        for offset in (-1, 1):
            neighbor = frame + offset
            if neighbor < 0 or neighbor >= num_frames:
                continue
            matches = ids[:, frame, :, None] == ids[:, neighbor, None, :]
            matches = matches & valid_tracks[:, frame, :, None] & valid_tracks[:, neighbor, None, :]
            has_match = matches.any(dim=-1)
            match_index = matches.to(dtype=torch.int64).argmax(dim=-1)
            gather_index = match_index[:, :, None, None].expand(-1, -1, 2, 3)
            neighbor_world = sole_world[:, neighbor].gather(dim=1, index=gather_index)
            distance = torch.linalg.norm(sole_world[:, frame] - neighbor_world, dim=-1)
            valid_distance = has_match[:, :, None].expand(-1, -1, 2)
            velocity = velocity + torch.where(valid_distance, distance, torch.zeros_like(distance))
            velocity_count = velocity_count + valid_distance.to(dtype=velocity_count.dtype)

    valid = velocity_count > 0.0
    velocity = velocity / velocity_count.clamp(min=1.0)
    return velocity.detach(), valid.detach()


def _resolve_intrinsics(pose_enc, image_size_hw, batch_size, num_frames, device, dtype, override):
    if override is not None:
        intrinsics = override.to(device=device, dtype=dtype)
        return intrinsics.reshape(batch_size * num_frames, 3, 3)
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


def _zero_last(module: nn.Sequential) -> nn.Sequential:
    last = module[-1]
    if isinstance(last, nn.Linear):
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    return module
