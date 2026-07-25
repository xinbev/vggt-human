from __future__ import annotations

import math

import torch
import torch.nn as nn

from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.utils.contact_geometry import build_sole_vertex_indices
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.utils.rotation import rot6d_to_axis_angle


LOWER_BODY_JOINTS = (1, 2, 4, 5, 7, 8)
LEFT_LEG_POSITIONS = (0, 2, 4)
RIGHT_LEG_POSITIONS = (1, 3, 5)


class HSIFootContactIntentHead(nn.Module):
    """Classify per-foot support intent without reading scene height or absolute position."""

    def __init__(
        self,
        smpl_model_dir: str,
        hidden_dim: int = 192,
        sole_vertices_per_foot: int = 48,
        max_velocity_m: float = 0.50,
        max_acceleration_m: float = 0.50,
        initial_contact_probability: float = 0.25,
    ) -> None:
        super().__init__()
        self.smpl = SMPLLayer(smpl_model_dir).eval()
        for parameter in self.smpl.parameters():
            parameter.requires_grad = False
        sole = build_sole_vertex_indices(self.smpl.layer.v_template.detach(), sole_vertices_per_foot)
        self.register_buffer("sole_vertex_indices", sole, persistent=False)
        self.max_velocity_m = max(float(max_velocity_m), 1e-3)
        self.max_acceleration_m = max(float(max_acceleration_m), 1e-3)

        # 50 shared body features + 37 side-specific leg/foot features.
        feature_dim = 87
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        probability = min(max(float(initial_contact_probability), 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.constant_(self.classifier[-1].bias, math.log(probability / (1.0 - probability)))

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        pose_enc: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pose6d = predictions.get("hsi_refined_pred_pose_6d", predictions.get("pred_pose_6d"))
        betas = predictions.get("hsi_refined_pred_betas", predictions.get("pred_betas"))
        transl = predictions.get("hsi_refined_pred_transl_cam", predictions.get("pred_transl_cam"))
        if pose6d is None or betas is None or transl is None:
            raise ValueError("HSI foot contact intent requires pose, betas, and translation")
        if transl.ndim != 4 or transl.shape[-1] != 3:
            raise ValueError(f"Expected translation [B,S,Q,3], got {tuple(transl.shape)}")

        pose6d = pose6d.float()
        betas = betas.float()
        transl = transl.float()
        batch_size, num_frames, num_queries = transl.shape[:3]
        flat_count = batch_size * num_frames * num_queries

        with torch.no_grad():
            aa = rot6d_to_axis_angle(pose6d.reshape(-1, 24, 6)).reshape(-1, 72)
            vertices, _ = self.smpl(aa.float(), betas.reshape(-1, betas.shape[-1]).float())
            local_sole = vertices[:, self.sole_vertex_indices].mean(dim=-2).to(dtype=pose6d.dtype)
            local_sole = local_sole.reshape(batch_size, num_frames, num_queries, 2, 3)
            sole_cam = local_sole + transl[..., None, :]
            root_world, sole_world = _camera_points_to_world(transl, sole_cam, pose_enc)

            track_ids = predictions.get("assigned_track_ids")
            track_mask = predictions.get("assigned_track_mask")
            root_velocity, root_velocity_valid, root_accel, root_accel_valid = _track_temporal_differences(
                root_world, track_ids, track_mask
            )
            local_velocity, local_velocity_valid, local_accel, local_accel_valid = _track_temporal_differences(
                local_sole, track_ids, track_mask
            )
            world_foot_velocity, world_foot_velocity_valid, world_foot_accel, world_foot_accel_valid = (
                _track_temporal_differences(sole_world, track_ids, track_mask)
            )

        lower_pose = pose6d.reshape(flat_count, 24, 6)[:, list(LOWER_BODY_JOINTS)]
        leg_positions = torch.tensor(
            [LEFT_LEG_POSITIONS, RIGHT_LEG_POSITIONS], device=pose6d.device, dtype=torch.long
        )
        leg_pose = lower_pose[:, leg_positions].reshape(flat_count, 2, 18)
        root_pose = pose6d.reshape(flat_count, 24, 6)[:, 0]

        root_velocity_raw = root_velocity
        root_accel_raw = root_accel
        root_velocity_feature = _bounded(root_velocity.reshape(flat_count, 3), self.max_velocity_m)
        root_accel_feature = _bounded(root_accel.reshape(flat_count, 3), self.max_acceleration_m)
        shared = torch.cat(
            [
                root_pose,
                lower_pose.reshape(flat_count, 36),
                root_velocity_feature,
                root_accel_feature,
                root_velocity_valid.reshape(flat_count, 1).to(dtype=pose6d.dtype),
                root_accel_valid.reshape(flat_count, 1).to(dtype=pose6d.dtype),
            ],
            dim=-1,
        ).unsqueeze(1).expand(-1, 2, -1)

        foot_velocity_valid_lr = world_foot_velocity_valid[..., None].expand(-1, -1, -1, 2)
        foot_accel_valid_lr = world_foot_accel_valid[..., None].expand(-1, -1, -1, 2)
        local_velocity_valid_lr = local_velocity_valid[..., None].expand(-1, -1, -1, 2)
        local_accel_valid_lr = local_accel_valid[..., None].expand(-1, -1, -1, 2)
        side_specific = torch.cat(
            [
                leg_pose,
                local_sole.reshape(flat_count, 2, 3),
                _bounded(local_velocity.reshape(flat_count, 2, 3), self.max_velocity_m),
                _bounded(local_accel.reshape(flat_count, 2, 3), self.max_acceleration_m),
                _bounded(world_foot_velocity.reshape(flat_count, 2, 3), self.max_velocity_m),
                _bounded(world_foot_accel.reshape(flat_count, 2, 3), self.max_acceleration_m),
                local_velocity_valid_lr.reshape(flat_count, 2, 1).to(dtype=pose6d.dtype),
                local_accel_valid_lr.reshape(flat_count, 2, 1).to(dtype=pose6d.dtype),
                foot_velocity_valid_lr.reshape(flat_count, 2, 1).to(dtype=pose6d.dtype),
                foot_accel_valid_lr.reshape(flat_count, 2, 1).to(dtype=pose6d.dtype),
            ],
            dim=-1,
        )
        features = torch.cat([shared, side_specific], dim=-1)
        if features.shape != (flat_count, 2, 87):
            raise RuntimeError(f"Unexpected contact-intent feature shape: {tuple(features.shape)}")
        logits = self.classifier(features).squeeze(-1).reshape(batch_size, num_frames, num_queries, 2)
        probability = torch.sigmoid(logits)
        temporal_valid = (
            root_velocity_valid[..., None]
            & local_velocity_valid[..., None]
            & world_foot_velocity_valid[..., None]
        ).expand(-1, -1, -1, 2)
        return {
            "hsi_foot_contact_intent_logits": logits,
            "hsi_foot_contact_intent_probability": probability,
            "hsi_foot_contact_intent_temporal_valid": temporal_valid.to(dtype=pose6d.dtype),
            "hsi_foot_contact_intent_root_velocity_m": root_velocity_raw,
            "hsi_foot_contact_intent_root_acceleration_m": root_accel_raw,
            "hsi_foot_contact_intent_foot_velocity_m": world_foot_velocity,
            "hsi_foot_contact_intent_foot_acceleration_m": world_foot_accel,
        }


def _camera_points_to_world(
    root_cam: torch.Tensor,
    feet_cam: torch.Tensor,
    pose_enc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    extrinsics, _ = encoding_to_camera(pose_enc, image_size_hw=(1, 1), build_intrinsics=False)
    rotation = extrinsics[..., :3, :3].to(device=root_cam.device, dtype=root_cam.dtype)
    translation = extrinsics[..., :3, 3].to(device=root_cam.device, dtype=root_cam.dtype)
    root_centered = root_cam - translation[:, :, None, :]
    feet_centered = feet_cam - translation[:, :, None, None, :]
    root_world = torch.einsum("bsij,bsqj->bsqi", rotation.transpose(-1, -2), root_centered)
    feet_world = torch.einsum("bsij,bsqfj->bsqfi", rotation.transpose(-1, -2), feet_centered)
    return root_world, feet_world


def _track_temporal_differences(
    values: torch.Tensor,
    track_ids: torch.Tensor | None,
    track_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return forward-oriented velocity and centered acceleration for tracked values."""
    batch_size, num_frames, num_queries = values.shape[:3]
    velocity = torch.zeros_like(values)
    acceleration = torch.zeros_like(values)
    velocity_valid = torch.zeros(batch_size, num_frames, num_queries, dtype=torch.bool, device=values.device)
    acceleration_valid = torch.zeros_like(velocity_valid)
    if num_frames < 2 or not isinstance(track_ids, torch.Tensor):
        return velocity, velocity_valid, acceleration, acceleration_valid
    if track_ids.shape[:3] != (batch_size, num_frames, num_queries):
        raise ValueError(
            f"assigned_track_ids shape must start with {(batch_size, num_frames, num_queries)}, "
            f"got {tuple(track_ids.shape)}"
        )
    ids = track_ids.to(device=values.device, dtype=torch.long)
    valid_tracks = ids >= 0
    if isinstance(track_mask, torch.Tensor):
        if track_mask.shape[:3] != (batch_size, num_frames, num_queries):
            raise ValueError(
                f"assigned_track_mask shape must start with {(batch_size, num_frames, num_queries)}, "
                f"got {tuple(track_mask.shape)}"
            )
        valid_tracks &= track_mask.to(device=values.device).bool()

    prev_values: list[torch.Tensor | None] = [None] * num_frames
    next_values: list[torch.Tensor | None] = [None] * num_frames
    prev_valid: list[torch.Tensor | None] = [None] * num_frames
    next_valid: list[torch.Tensor | None] = [None] * num_frames
    for frame in range(num_frames):
        for neighbor, storage_values, storage_valid in (
            (frame - 1, prev_values, prev_valid),
            (frame + 1, next_values, next_valid),
        ):
            if neighbor < 0 or neighbor >= num_frames:
                continue
            matches = ids[:, frame, :, None] == ids[:, neighbor, None, :]
            matches &= valid_tracks[:, frame, :, None] & valid_tracks[:, neighbor, None, :]
            has_match = matches.any(dim=-1)
            match_index = matches.to(dtype=torch.int64).argmax(dim=-1)
            gather_shape = (batch_size, num_queries) + (1,) * (values.ndim - 3)
            gather_index = match_index.reshape(gather_shape).expand(batch_size, num_queries, *values.shape[3:])
            storage_values[frame] = values[:, neighbor].gather(dim=1, index=gather_index)
            storage_valid[frame] = has_match

    for frame in range(num_frames):
        deltas = []
        masks = []
        if prev_values[frame] is not None:
            deltas.append(values[:, frame] - prev_values[frame])
            masks.append(prev_valid[frame])
        if next_values[frame] is not None:
            deltas.append(next_values[frame] - values[:, frame])
            masks.append(next_valid[frame])
        if deltas:
            count = values.new_zeros(batch_size, num_queries)
            total = torch.zeros_like(values[:, frame])
            for delta, mask in zip(deltas, masks):
                expanded_mask = mask.reshape(batch_size, num_queries, *((1,) * (values.ndim - 3)))
                total += torch.where(expanded_mask, delta, torch.zeros_like(delta))
                count += mask.to(dtype=count.dtype)
            velocity[:, frame] = total / count.clamp(min=1.0).reshape(
                batch_size, num_queries, *((1,) * (values.ndim - 3))
            )
            velocity_valid[:, frame] = count > 0.0
        if prev_values[frame] is not None and next_values[frame] is not None:
            valid = prev_valid[frame] & next_valid[frame]
            expanded_valid = valid.reshape(batch_size, num_queries, *((1,) * (values.ndim - 3)))
            centered = next_values[frame] - 2.0 * values[:, frame] + prev_values[frame]
            acceleration[:, frame] = torch.where(expanded_valid, centered, torch.zeros_like(centered))
            acceleration_valid[:, frame] = valid
    return velocity.detach(), velocity_valid.detach(), acceleration.detach(), acceleration_valid.detach()


def _bounded(value: torch.Tensor, maximum: float) -> torch.Tensor:
    return value.clamp(min=-maximum, max=maximum) / maximum
