"""Standalone offline SMPL sequence denoiser/refiner.

This module deliberately has no dependency on VGGT, HSI, TRSTR, SMPL-X, or
image tensors.  It is trained first as a residual temporal prior on tracked
SMPL sequences, then can be connected to the inference pipeline through the
adapter in :mod:`vggt_omega.integrations.smpl_temporal_refiner`.

The model is bidirectional: a frame may attend to past and future frames.
Consequently it is intended for offline clips, not live causal inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from vggt_omega.utils.rotation import axis_angle_to_rotmat, rot6d_to_rotmat


NUM_SMPL_JOINTS = 24
POSE_6D_DIM = NUM_SMPL_JOINTS * 6


def _rotmat_to_6d(rotation: torch.Tensor) -> torch.Tensor:
    return rotation[..., :2, :].reshape(*rotation.shape[:-2], 6)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(dtype=value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    return (value * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


@dataclass(frozen=True)
class TemporalRefinerConfig:
    """Architecture and conservative residual bounds for the refiner."""

    window_size: int = 9
    hidden_dim: int = 384
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    max_translation_delta_m: float = 0.25
    max_pose_delta_rad: float = 0.35
    gate_bias: float = -1.5


class TemporalSMPLRefiner(nn.Module):
    """Predict gated residual corrections for a tracked SMPL sequence.

    Inputs are the *noisy single-frame predictions* rather than ground truth.
    ``pose_6d`` is ``[B, S, 144]`` and ``transl`` is ``[B, S, 3]``.  The
    refiner never writes beta; it is passed through to make that invariant
    explicit for callers that keep beta alongside pose/translation.
    """

    def __init__(self, config: TemporalRefinerConfig | None = None) -> None:
        super().__init__()
        self.config = config or TemporalRefinerConfig()
        if self.config.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.config.hidden_dim % self.config.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        # Translation is represented both relative to the clip centre and as
        # its finite-difference velocity.  This reduces sensitivity to global
        # scene origin while preserving the information needed to correct drift.
        input_dim = POSE_6D_DIM + 3 + 3 + 1
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.position = nn.Parameter(torch.zeros(1, self.config.window_size, self.config.hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.hidden_dim * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.config.num_layers)
        self.output_norm = nn.LayerNorm(self.config.hidden_dim)
        self.pose_head = nn.Linear(self.config.hidden_dim, NUM_SMPL_JOINTS * 3)
        self.transl_head = nn.Linear(self.config.hidden_dim, 3)
        self.pose_gate_head = nn.Linear(self.config.hidden_dim, NUM_SMPL_JOINTS)
        self.transl_gate_head = nn.Linear(self.config.hidden_dim, 1)
        self.uncertainty_head = nn.Linear(self.config.hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.zeros_(self.pose_head.weight)
        nn.init.zeros_(self.pose_head.bias)
        nn.init.zeros_(self.transl_head.weight)
        nn.init.zeros_(self.transl_head.bias)
        nn.init.zeros_(self.pose_gate_head.weight)
        nn.init.constant_(self.pose_gate_head.bias, self.config.gate_bias)
        nn.init.zeros_(self.transl_gate_head.weight)
        nn.init.constant_(self.transl_gate_head.bias, self.config.gate_bias)

    def forward(
        self,
        pose_6d: torch.Tensor,
        transl: torch.Tensor,
        betas: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Refine an offline window.

        ``valid_mask`` has shape ``[B, S]``.  Invalid positions are accepted
        for short tracker gaps but are left unchanged in the returned tensors.
        """
        if pose_6d.ndim != 3 or pose_6d.shape[-1] != POSE_6D_DIM:
            raise ValueError(f"pose_6d must be [B,S,{POSE_6D_DIM}], got {tuple(pose_6d.shape)}")
        if transl.shape != (*pose_6d.shape[:2], 3):
            raise ValueError(f"transl must be [B,S,3], got {tuple(transl.shape)}")
        batch, steps = pose_6d.shape[:2]
        if steps > self.config.window_size:
            raise ValueError(f"input steps={steps} exceed configured window_size={self.config.window_size}")
        if valid_mask is None:
            valid_mask = torch.ones(batch, steps, device=pose_6d.device, dtype=torch.bool)
        else:
            valid_mask = valid_mask.to(device=pose_6d.device, dtype=torch.bool)
            if valid_mask.shape != (batch, steps):
                raise ValueError(f"valid_mask must be [B,S], got {tuple(valid_mask.shape)}")
        if confidence is None:
            confidence = valid_mask.to(dtype=pose_6d.dtype)
        else:
            confidence = confidence.to(device=pose_6d.device, dtype=pose_6d.dtype)
            if confidence.shape == (batch, steps, 1):
                confidence = confidence[..., 0]
            if confidence.shape != (batch, steps):
                raise ValueError(f"confidence must be [B,S] or [B,S,1], got {tuple(confidence.shape)}")
            confidence = confidence.clamp(0.0, 1.0)

        origin = _masked_mean(transl, valid_mask, dim=1).unsqueeze(1)
        transl_relative = transl - origin
        velocity = torch.zeros_like(transl)
        velocity[:, 1:] = transl[:, 1:] - transl[:, :-1]
        features = torch.cat((pose_6d, transl_relative, velocity, confidence.unsqueeze(-1)), dim=-1)
        hidden = self.input_proj(self.input_norm(features)) + self.position[:, :steps]
        hidden = self.encoder(hidden, src_key_padding_mask=~valid_mask)
        hidden = self.output_norm(hidden)

        pose_gate = torch.sigmoid(self.pose_gate_head(hidden)).unsqueeze(-1)
        transl_gate = torch.sigmoid(self.transl_gate_head(hidden))
        raw_pose_delta = torch.tanh(self.pose_head(hidden)).reshape(batch, steps, NUM_SMPL_JOINTS, 3)
        pose_delta_axis_angle = raw_pose_delta * (self.config.max_pose_delta_rad * pose_gate)
        raw_transl_delta = torch.tanh(self.transl_head(hidden))
        transl_delta = raw_transl_delta * (self.config.max_translation_delta_m * transl_gate)

        base_rotmat = rot6d_to_rotmat(pose_6d.reshape(batch, steps, NUM_SMPL_JOINTS, 6))
        residual_rotmat = axis_angle_to_rotmat(pose_delta_axis_angle)
        refined_pose_6d = _rotmat_to_6d(residual_rotmat @ base_rotmat).reshape(batch, steps, POSE_6D_DIM)
        refined_transl = transl + transl_delta

        valid_float = valid_mask.to(dtype=pose_6d.dtype).unsqueeze(-1)
        refined_pose_6d = refined_pose_6d * valid_float + pose_6d * (1.0 - valid_float)
        refined_transl = refined_transl * valid_float + transl * (1.0 - valid_float)
        return {
            "refined_pose_6d": refined_pose_6d,
            "refined_transl": refined_transl,
            "refined_betas": betas if betas is not None else torch.empty(0, device=pose_6d.device),
            "pose_delta_axis_angle": pose_delta_axis_angle,
            "transl_delta": transl_delta,
            "pose_gate": pose_gate,
            "transl_gate": transl_gate,
            "uncertainty": F.softplus(self.uncertainty_head(hidden)),
        }


def rotation_geodesic(pred_rotmat: torch.Tensor, target_rotmat: torch.Tensor) -> torch.Tensor:
    relative = pred_rotmat @ target_rotmat.transpose(-1, -2)
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cosine)


class TemporalSMPLRefinerLoss(nn.Module):
    """Accuracy-first losses; temporal terms follow GT motion, not zero motion."""

    def __init__(
        self,
        pose_weight: float = 1.0,
        transl_weight: float = 2.0,
        velocity_weight: float = 0.5,
        acceleration_weight: float = 0.25,
        no_worse_weight: float = 0.5,
        clean_residual_weight: float = 0.1,
        no_worse_margin: float = 0.002,
    ) -> None:
        super().__init__()
        self.pose_weight = float(pose_weight)
        self.transl_weight = float(transl_weight)
        self.velocity_weight = float(velocity_weight)
        self.acceleration_weight = float(acceleration_weight)
        self.no_worse_weight = float(no_worse_weight)
        self.clean_residual_weight = float(clean_residual_weight)
        self.no_worse_margin = float(no_worse_margin)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target_pose_6d: torch.Tensor,
        target_transl: torch.Tensor,
        valid_mask: torch.Tensor,
        base_pose_6d: torch.Tensor,
        base_transl: torch.Tensor,
        clean_clip_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        valid = valid_mask.to(dtype=torch.bool)
        zero = target_transl.sum() * 0.0
        refined_pose = outputs["refined_pose_6d"]
        refined_transl = outputs["refined_transl"]
        b, s = valid.shape
        target_rot = rot6d_to_rotmat(target_pose_6d.reshape(b, s, NUM_SMPL_JOINTS, 6))
        refined_rot = rot6d_to_rotmat(refined_pose.reshape(b, s, NUM_SMPL_JOINTS, 6))
        base_rot = rot6d_to_rotmat(base_pose_6d.reshape(b, s, NUM_SMPL_JOINTS, 6))
        pose_error = rotation_geodesic(refined_rot, target_rot).mean(dim=-1)
        base_pose_error = rotation_geodesic(base_rot, target_rot).mean(dim=-1)
        transl_error = (refined_transl - target_transl).abs().mean(dim=-1)
        base_transl_error = (base_transl - target_transl).abs().mean(dim=-1)
        frame_weight = valid.to(dtype=target_transl.dtype)
        denom = frame_weight.sum().clamp_min(1.0)
        pose_loss = (pose_error * frame_weight).sum() / denom
        transl_loss = (transl_error * frame_weight).sum() / denom

        pair_valid = valid[:, 1:] & valid[:, :-1]
        if s > 1 and bool(pair_valid.any()):
            pred_velocity = refined_transl[:, 1:] - refined_transl[:, :-1]
            target_velocity = target_transl[:, 1:] - target_transl[:, :-1]
            velocity_error = (pred_velocity - target_velocity).abs().mean(dim=-1)
            velocity_loss = (velocity_error * pair_valid.to(velocity_error.dtype)).sum() / pair_valid.sum().clamp_min(1)
        else:
            velocity_loss = zero

        triple_valid = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
        if s > 2 and bool(triple_valid.any()):
            pred_accel = refined_transl[:, 2:] - 2.0 * refined_transl[:, 1:-1] + refined_transl[:, :-2]
            target_accel = target_transl[:, 2:] - 2.0 * target_transl[:, 1:-1] + target_transl[:, :-2]
            acceleration_error = (pred_accel - target_accel).abs().mean(dim=-1)
            acceleration_loss = (acceleration_error * triple_valid.to(acceleration_error.dtype)).sum() / triple_valid.sum().clamp_min(1)
        else:
            acceleration_loss = zero

        output_error = pose_error + transl_error
        base_error = base_pose_error + base_transl_error
        no_worse_loss = (F.relu(output_error - base_error + self.no_worse_margin) * frame_weight).sum() / denom
        if clean_clip_mask is not None and bool(clean_clip_mask.any()):
            clean = clean_clip_mask.to(device=valid.device, dtype=torch.bool).unsqueeze(-1) & valid
            clean_weight = clean.to(dtype=target_transl.dtype)
            clean_denom = clean_weight.sum().clamp_min(1.0)
            clean_residual = ((outputs["transl_delta"].abs().mean(dim=-1) + outputs["pose_delta_axis_angle"].abs().mean(dim=(-1, -2))) * clean_weight).sum() / clean_denom
        else:
            clean_residual = zero
        total = (
            self.pose_weight * pose_loss
            + self.transl_weight * transl_loss
            + self.velocity_weight * velocity_loss
            + self.acceleration_weight * acceleration_loss
            + self.no_worse_weight * no_worse_loss
            + self.clean_residual_weight * clean_residual
        )
        return {
            "loss_total": total,
            "loss_pose": pose_loss,
            "loss_transl": transl_loss,
            "loss_velocity": velocity_loss,
            "loss_acceleration": acceleration_loss,
            "loss_no_worse": no_worse_loss,
            "loss_clean_residual": clean_residual,
            "metric_pose_geodesic_rad": pose_loss.detach(),
            "metric_transl_l1_m": transl_loss.detach(),
            "metric_base_pose_geodesic_rad": (base_pose_error * frame_weight).sum().detach() / denom.detach(),
            "metric_base_transl_l1_m": (base_transl_error * frame_weight).sum().detach() / denom.detach(),
            "metric_translation_improvement": (base_transl_error - transl_error).mul(frame_weight).sum().detach() / denom.detach(),
            "metric_no_worse_rate": (((output_error <= base_error + self.no_worse_margin) & valid).sum().detach().to(target_transl.dtype) / denom.detach()),
            "metric_pose_gate_mean": outputs["pose_gate"].detach().mean(),
            "metric_transl_gate_mean": outputs["transl_gate"].detach().mean(),
        }
