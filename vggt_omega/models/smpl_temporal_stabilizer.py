"""Conservative translation-only temporal stabilizer (V2 E0).

The module deliberately cannot invent an arbitrary translation residual.  For
an internal frame t it first predicts a motion proposal only from t-2, t-1,
t+1, t+2; a separate gate then blends the observed single-frame translation
towards that proposal by at most ``max_blend``.  This is the minimal
implementation of the observation-versus-temporal-motion design.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from vggt_omega.utils.rotation import axis_angle_to_rotmat, rotation_matrix_to_axis_angle, rot6d_to_rotmat


@dataclass(frozen=True)
class TranslationStabilizerConfig:
    window_size: int = 9
    proposal_hidden_dim: int = 128
    gate_hidden_dim: int = 64
    max_motion_residual_m: float = 0.25
    max_blend: float = 0.5


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class TranslationTemporalStabilizer(nn.Module):
    """Fuse current translation observations with a neighbour-only proposal."""

    def __init__(self, config: TranslationStabilizerConfig | None = None) -> None:
        super().__init__()
        self.config = config or TranslationStabilizerConfig()
        if self.config.window_size < 5 or self.config.window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer >= 5")
        if not 0.0 < self.config.max_blend <= 1.0:
            raise ValueError("max_blend must be in (0,1]")
        self.proposal_head = _mlp(12, self.config.proposal_hidden_dim, 3)
        self.gate_head = _mlp(9, self.config.gate_hidden_dim, 1)
        # At start, the neighbour midpoint is the proposal and fusion is 25%.
        # This permits learning in both directions and avoids V1's closed gate.
        nn.init.zeros_(self.proposal_head[-1].weight)
        nn.init.zeros_(self.proposal_head[-1].bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.zeros_(self.gate_head[-1].bias)

    def forward(self, observed_transl: torch.Tensor, valid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Stabilize ``[B,S,3]`` translation with a two-frame context each side.

        Boundary frames and any frame with an invalid required neighbour remain
        exactly equal to the input.  ``motion_proposal`` does not depend on the
        observed centre frame for any frame where ``context_valid`` is true.
        """
        if observed_transl.ndim != 3 or observed_transl.shape[-1] != 3:
            raise ValueError(f"observed_transl must be [B,S,3], got {tuple(observed_transl.shape)}")
        batch, steps, _ = observed_transl.shape
        if steps != self.config.window_size:
            raise ValueError(f"expected window_size={self.config.window_size}, got {steps}")
        if valid_mask is None:
            valid_mask = torch.ones(batch, steps, dtype=torch.bool, device=observed_transl.device)
        else:
            valid_mask = valid_mask.to(device=observed_transl.device, dtype=torch.bool)
            if valid_mask.shape != (batch, steps):
                raise ValueError(f"valid_mask must be [B,S], got {tuple(valid_mask.shape)}")

        proposal = observed_transl.clone()
        blend = observed_transl.new_zeros(batch, steps, 1)
        context_valid = torch.zeros(batch, steps, dtype=torch.bool, device=observed_transl.device)
        # t=2...S-3 has four actual neighbours.  S=9 yields five supervised
        # centres per window, enough for the fixed-batch E0 test.
        centres = torch.arange(2, steps - 2, device=observed_transl.device)
        if centres.numel() == 0:
            return {
                "refined_transl": observed_transl,
                "motion_proposal": proposal,
                "blend": blend,
                "context_valid": context_valid,
            }
        neighbours = torch.stack(
            (
                observed_transl[:, centres - 2],
                observed_transl[:, centres - 1],
                observed_transl[:, centres + 1],
                observed_transl[:, centres + 2],
            ),
            dim=2,
        )
        neighbour_valid = torch.stack(
            (
                valid_mask[:, centres - 2],
                valid_mask[:, centres - 1],
                valid_mask[:, centres + 1],
                valid_mask[:, centres + 2],
            ),
            dim=2,
        ).all(dim=2)
        midpoint = 0.5 * (neighbours[:, :, 1] + neighbours[:, :, 2])
        relative_neighbours = (neighbours - midpoint.unsqueeze(2)).reshape(batch, centres.numel(), 12)
        proposal_residual = torch.tanh(self.proposal_head(relative_neighbours)).to(dtype=observed_transl.dtype)
        proposal_residual = proposal_residual * self.config.max_motion_residual_m
        proposal_centres = midpoint + proposal_residual
        observed_centres = observed_transl[:, centres]
        neighbour_velocity = 0.5 * (neighbours[:, :, 3] - neighbours[:, :, 0])
        gate_features = torch.cat(
            (
                observed_centres - proposal_centres,
                (observed_centres - proposal_centres).abs(),
                neighbour_velocity,
            ),
            dim=-1,
        )
        blend_centres = torch.sigmoid(self.gate_head(gate_features)).to(dtype=observed_transl.dtype)
        blend_centres = blend_centres * self.config.max_blend
        blend_centres = blend_centres * neighbour_valid.to(dtype=blend_centres.dtype).unsqueeze(-1)
        proposal[:, centres] = torch.where(neighbour_valid.unsqueeze(-1), proposal_centres, observed_centres)
        blend[:, centres] = blend_centres
        context_valid[:, centres] = neighbour_valid
        refined = observed_transl + blend * (proposal - observed_transl)
        refined = torch.where(valid_mask.unsqueeze(-1), refined, observed_transl)
        return {
            "refined_transl": refined,
            "motion_proposal": proposal,
            "blend": blend,
            "context_valid": context_valid,
        }


class TranslationStabilizerLoss(nn.Module):
    """Losses for a constrained, directly supervised observation-motion blend."""

    def __init__(
        self,
        final_weight: float = 2.0,
        proposal_weight: float = 1.0,
        blend_weight: float = 1.0,
        velocity_weight: float = 0.25,
        oracle_positive_threshold: float = 0.02,
        max_blend: float = 0.5,
    ) -> None:
        super().__init__()
        self.final_weight = float(final_weight)
        self.proposal_weight = float(proposal_weight)
        self.blend_weight = float(blend_weight)
        self.velocity_weight = float(velocity_weight)
        self.oracle_positive_threshold = float(oracle_positive_threshold)
        self.max_blend = float(max_blend)

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(dtype=value.dtype)
        while weight.ndim < value.ndim:
            weight = weight.unsqueeze(-1)
        return (value * weight).sum() / weight.sum().clamp_min(1.0)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target_transl: torch.Tensor,
        observed_transl: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        context = outputs["context_valid"] & valid_mask
        refined = outputs["refined_transl"]
        proposal = outputs["motion_proposal"]
        blend = outputs["blend"]
        zero = target_transl.sum() * 0.0
        direction = proposal - observed_transl
        numerator = ((target_transl - observed_transl) * direction).sum(dim=-1, keepdim=True)
        denominator = direction.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
        oracle_blend = (numerator / denominator).clamp(0.0, 1.0)
        # The model's allowed blend is intentionally capped; clamp the target
        # to the same feasible interval before supervising the gate.
        oracle_blend = oracle_blend.clamp_max(self.max_blend).detach()
        final_l1_frame = (refined - target_transl).abs().mean(dim=-1)
        base_l1_frame = (observed_transl - target_transl).abs().mean(dim=-1)
        proposal_l1_frame = (proposal - target_transl).abs().mean(dim=-1)
        final_loss = self._masked_mean(final_l1_frame, context)
        proposal_loss = self._masked_mean(proposal_l1_frame, context)
        blend_loss = self._masked_mean(F.smooth_l1_loss(blend, oracle_blend, reduction="none").squeeze(-1), context)
        pair_mask = context[:, 1:] & context[:, :-1]
        if bool(pair_mask.any()):
            velocity_error = ((refined[:, 1:] - refined[:, :-1]) - (target_transl[:, 1:] - target_transl[:, :-1])).abs().mean(dim=-1)
            velocity_loss = self._masked_mean(velocity_error, pair_mask)
        else:
            velocity_loss = zero
        total = (
            self.final_weight * final_loss
            + self.proposal_weight * proposal_loss
            + self.blend_weight * blend_loss
            + self.velocity_weight * velocity_loss
        )
        positive = (oracle_blend[..., 0] > self.oracle_positive_threshold) & context
        zero_oracle = ~positive & context
        improvement = base_l1_frame - final_l1_frame
        return {
            "loss_total": total,
            "loss_final_l1": final_loss,
            "loss_proposal_l1": proposal_loss,
            "loss_blend": blend_loss,
            "loss_velocity": velocity_loss,
            "metric_base_l1_m": self._masked_mean(base_l1_frame, context).detach(),
            "metric_proposal_l1_m": self._masked_mean(proposal_l1_frame, context).detach(),
            "metric_final_l1_m": final_loss.detach(),
            "metric_improvement_m": self._masked_mean(improvement, context).detach(),
            "metric_improvement_rate": self._masked_mean((improvement > 0.0).to(target_transl.dtype), context).detach(),
            "metric_blend_mean": self._masked_mean(blend[..., 0], context).detach(),
            "metric_oracle_blend_mean": self._masked_mean(oracle_blend[..., 0], context).detach(),
            "metric_blend_l1": self._masked_mean((blend - oracle_blend).abs().squeeze(-1), context).detach(),
            "metric_oracle_positive_fraction": self._masked_mean(positive.to(target_transl.dtype), context).detach(),
            "metric_false_apply_zero_oracle": self._masked_mean((blend[..., 0] > self.oracle_positive_threshold).to(target_transl.dtype), zero_oracle).detach(),
            "metric_context_frame_fraction": context.to(target_transl.dtype).mean().detach(),
        }


def _rotmat_to_6d(rotation: torch.Tensor) -> torch.Tensor:
    return rotation[..., :2, :].reshape(*rotation.shape[:-2], 6)


def _rotation_log(rotation: torch.Tensor) -> torch.Tensor:
    return rotation_matrix_to_axis_angle(rotation)


def _rotation_midpoint(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Geodesic midpoint from ``left`` to ``right`` on SO(3)."""
    relative = right @ left.transpose(-1, -2)
    return axis_angle_to_rotmat(0.5 * _rotation_log(relative)) @ left


@dataclass(frozen=True)
class PoseStabilizerConfig:
    window_size: int = 9
    proposal_hidden_dim: int = 128
    gate_hidden_dim: int = 64
    max_motion_residual_rad: float = 0.35
    max_blend: float = 0.5


class PoseTemporalStabilizer(nn.Module):
    """Per-joint, neighbour-only SO(3) motion proposal and conservative blend.

    MLP weights are shared by all 24 SMPL joints.  Thus this E0 test checks
    whether pose jitter is recoverable from local temporal evidence without
    allowing the network a free all-joint residual output.
    """

    def __init__(self, config: PoseStabilizerConfig | None = None) -> None:
        super().__init__()
        self.config = config or PoseStabilizerConfig()
        if self.config.window_size < 5 or self.config.window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer >= 5")
        self.proposal_head = _mlp(12, self.config.proposal_hidden_dim, 3)
        self.gate_head = _mlp(9, self.config.gate_hidden_dim, 1)
        nn.init.zeros_(self.proposal_head[-1].weight)
        nn.init.zeros_(self.proposal_head[-1].bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.zeros_(self.gate_head[-1].bias)

    def forward(self, observed_pose_6d: torch.Tensor, valid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if observed_pose_6d.ndim != 3 or observed_pose_6d.shape[-1] != 144:
            raise ValueError(f"observed_pose_6d must be [B,S,144], got {tuple(observed_pose_6d.shape)}")
        batch, steps, _ = observed_pose_6d.shape
        if steps != self.config.window_size:
            raise ValueError(f"expected window_size={self.config.window_size}, got {steps}")
        if valid_mask is None:
            valid_mask = torch.ones(batch, steps, dtype=torch.bool, device=observed_pose_6d.device)
        else:
            valid_mask = valid_mask.to(device=observed_pose_6d.device, dtype=torch.bool)
        observed = rot6d_to_rotmat(observed_pose_6d.reshape(batch, steps, 24, 6))
        proposal = observed.clone()
        blend = observed_pose_6d.new_zeros(batch, steps, 24, 1)
        context_valid = torch.zeros(batch, steps, dtype=torch.bool, device=observed_pose_6d.device)
        centres = torch.arange(2, steps - 2, device=observed_pose_6d.device)
        neighbours = torch.stack(
            (observed[:, centres - 2], observed[:, centres - 1], observed[:, centres + 1], observed[:, centres + 2]), dim=2
        )
        neighbour_valid = torch.stack(
            (valid_mask[:, centres - 2], valid_mask[:, centres - 1], valid_mask[:, centres + 1], valid_mask[:, centres + 2]), dim=2
        ).all(dim=2)
        midpoint = _rotation_midpoint(neighbours[:, :, 1], neighbours[:, :, 2])
        relative_neighbours = _rotation_log(neighbours @ midpoint.unsqueeze(2).transpose(-1, -2))
        proposal_input = relative_neighbours.reshape(batch * centres.numel() * 24, 12)
        proposal_residual = torch.tanh(self.proposal_head(proposal_input)).reshape(batch, centres.numel(), 24, 3)
        proposal_residual = proposal_residual.to(dtype=observed_pose_6d.dtype) * self.config.max_motion_residual_rad
        proposal_centres = axis_angle_to_rotmat(proposal_residual) @ midpoint
        observed_centres = observed[:, centres]
        deviation = _rotation_log(proposal_centres @ observed_centres.transpose(-1, -2))
        neighbour_velocity = 0.5 * _rotation_log(neighbours[:, :, 3] @ neighbours[:, :, 0].transpose(-1, -2))
        gate_input = torch.cat((deviation, deviation.abs(), neighbour_velocity), dim=-1).reshape(batch * centres.numel() * 24, 9)
        blend_centres = torch.sigmoid(self.gate_head(gate_input)).reshape(batch, centres.numel(), 24, 1)
        blend_centres = blend_centres.to(dtype=observed_pose_6d.dtype) * self.config.max_blend
        blend_centres = blend_centres * neighbour_valid.to(dtype=blend_centres.dtype).reshape(batch, centres.numel(), 1, 1)
        proposal[:, centres] = torch.where(neighbour_valid[..., None, None, None], proposal_centres, observed_centres)
        blend[:, centres] = blend_centres
        context_valid[:, centres] = neighbour_valid
        final_delta = _rotation_log(proposal @ observed.transpose(-1, -2))
        refined = axis_angle_to_rotmat(blend * final_delta) @ observed
        refined = torch.where(valid_mask[..., None, None, None], refined, observed)
        return {
            "refined_pose_6d": _rotmat_to_6d(refined).reshape(batch, steps, 144),
            "motion_proposal_pose_6d": _rotmat_to_6d(proposal).reshape(batch, steps, 144),
            "blend": blend,
            "context_valid": context_valid,
        }


class PoseStabilizerLoss(nn.Module):
    def __init__(
        self,
        final_weight: float = 2.0,
        proposal_weight: float = 1.0,
        blend_weight: float = 1.0,
        angular_velocity_weight: float = 0.25,
        oracle_positive_threshold: float = 0.02,
        max_blend: float = 0.5,
    ) -> None:
        super().__init__()
        self.final_weight = float(final_weight)
        self.proposal_weight = float(proposal_weight)
        self.blend_weight = float(blend_weight)
        self.angular_velocity_weight = float(angular_velocity_weight)
        self.oracle_positive_threshold = float(oracle_positive_threshold)
        self.max_blend = float(max_blend)

    @staticmethod
    def _mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(dtype=value.dtype)
        while weight.ndim < value.ndim:
            weight = weight.unsqueeze(-1)
        return (value * weight).sum() / weight.sum().clamp_min(1.0)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target_pose_6d: torch.Tensor,
        observed_pose_6d: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, steps, _ = target_pose_6d.shape
        target = rot6d_to_rotmat(target_pose_6d.reshape(batch, steps, 24, 6))
        observed = rot6d_to_rotmat(observed_pose_6d.reshape(batch, steps, 24, 6))
        refined = rot6d_to_rotmat(outputs["refined_pose_6d"].reshape(batch, steps, 24, 6))
        proposal = rot6d_to_rotmat(outputs["motion_proposal_pose_6d"].reshape(batch, steps, 24, 6))
        context = outputs["context_valid"] & valid_mask
        direction = _rotation_log(proposal @ observed.transpose(-1, -2))
        target_delta = _rotation_log(target @ observed.transpose(-1, -2))
        oracle = ((target_delta * direction).sum(dim=-1, keepdim=True) / direction.square().sum(dim=-1, keepdim=True).clamp_min(1e-8))
        oracle = oracle.clamp(0.0, self.max_blend).detach()
        final_error = _rotation_log(refined @ target.transpose(-1, -2)).norm(dim=-1)
        base_error = _rotation_log(observed @ target.transpose(-1, -2)).norm(dim=-1)
        proposal_error = _rotation_log(proposal @ target.transpose(-1, -2)).norm(dim=-1)
        final_loss = self._mean(final_error.mean(dim=-1), context)
        proposal_loss = self._mean(proposal_error.mean(dim=-1), context)
        blend_loss = self._mean(F.smooth_l1_loss(outputs["blend"], oracle, reduction="none").mean(dim=-2).squeeze(-1), context)
        pair_mask = context[:, 1:] & context[:, :-1]
        if bool(pair_mask.any()):
            final_velocity = _rotation_log(refined[:, 1:] @ refined[:, :-1].transpose(-1, -2))
            target_velocity = _rotation_log(target[:, 1:] @ target[:, :-1].transpose(-1, -2))
            velocity_loss = self._mean((final_velocity - target_velocity).norm(dim=-1).mean(dim=-1), pair_mask)
        else:
            velocity_loss = target_pose_6d.sum() * 0.0
        total = self.final_weight * final_loss + self.proposal_weight * proposal_loss + self.blend_weight * blend_loss + self.angular_velocity_weight * velocity_loss
        improvement = base_error - final_error
        positive = (oracle[..., 0] > self.oracle_positive_threshold) & context[..., None]
        zero_oracle = ~positive & context[..., None]
        return {
            "loss_total": total,
            "loss_final_geodesic": final_loss,
            "loss_proposal_geodesic": proposal_loss,
            "loss_blend": blend_loss,
            "loss_angular_velocity": velocity_loss,
            "metric_base_geodesic_rad": self._mean(base_error.mean(dim=-1), context).detach(),
            "metric_proposal_geodesic_rad": self._mean(proposal_error.mean(dim=-1), context).detach(),
            "metric_final_geodesic_rad": final_loss.detach(),
            "metric_improvement_rad": self._mean(improvement.mean(dim=-1), context).detach(),
            "metric_improvement_rate": self._mean((improvement > 0.0).to(target_pose_6d.dtype).mean(dim=-1), context).detach(),
            "metric_blend_mean": self._mean(outputs["blend"].mean(dim=-2).squeeze(-1), context).detach(),
            "metric_oracle_blend_mean": self._mean(oracle.mean(dim=-2).squeeze(-1), context).detach(),
            "metric_blend_l1": self._mean((outputs["blend"] - oracle).abs().mean(dim=-2).squeeze(-1), context).detach(),
            "metric_oracle_positive_fraction": self._mean(positive.to(target_pose_6d.dtype).mean(dim=-1), context).detach(),
            "metric_context_frame_fraction": context.to(target_pose_6d.dtype).mean().detach(),
        }
