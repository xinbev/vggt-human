"""Realistic synthetic single-frame-SMPL errors for temporal-refiner training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vggt_omega.models.smpl_temporal_refiner import NUM_SMPL_JOINTS, _rotmat_to_6d
from vggt_omega.utils.rotation import axis_angle_to_rotmat, rot6d_to_rotmat


@dataclass(frozen=True)
class TemporalSMPLNoiseConfig:
    """Noise magnitudes in metres/radians, matched to a jittery framewise HMR prior."""

    translation_drift_std_m: float = 0.06
    translation_jitter_std_m: float = 0.025
    translation_outlier_std_m: float = 0.10
    translation_outlier_prob: float = 0.05
    pose_drift_std_rad: float = 0.06
    pose_jitter_std_rad: float = 0.025
    root_pose_multiplier: float = 1.5
    clean_clip_prob: float = 0.15


def corrupt_smpl_sequence(
    pose_6d: torch.Tensor,
    transl: torch.Tensor,
    valid_mask: torch.Tensor,
    config: TemporalSMPLNoiseConfig,
) -> dict[str, torch.Tensor]:
    """Corrupt every frame in a window, not only its centre frame.

    The composed low-frequency drift plus independent innovation is deliberately
    closer to framewise HMR behaviour than IID perturbations.  A clean-clip
    bucket teaches the learned residual module to leave correct outputs alone.
    """
    if pose_6d.ndim != 3 or pose_6d.shape[-1] != 144:
        raise ValueError(f"pose_6d must be [B,S,144], got {tuple(pose_6d.shape)}")
    if transl.shape != (*pose_6d.shape[:2], 3):
        raise ValueError(f"transl must be [B,S,3], got {tuple(transl.shape)}")
    batch, steps = pose_6d.shape[:2]
    device, dtype = pose_6d.device, pose_6d.dtype
    valid = valid_mask.to(device=device, dtype=dtype).unsqueeze(-1)
    clean_clip = torch.rand(batch, device=device) < config.clean_clip_prob
    active = (~clean_clip).to(dtype=dtype).reshape(batch, 1, 1)

    # Slowly changing bias represents tracking/scale drift; jitter represents
    # frame-specific detector instability.  The first frame receives the same
    # distribution as later frames and all tracks stay independently sampled.
    drift_steps = torch.randn(batch, steps, 3, device=device, dtype=dtype)
    drift = torch.cumsum(drift_steps, dim=1)
    drift = drift / drift.std(dim=1, keepdim=True).clamp_min(1e-5)
    drift = drift * config.translation_drift_std_m
    jitter = torch.randn(batch, steps, 3, device=device, dtype=dtype) * config.translation_jitter_std_m
    outlier_mask = torch.rand(batch, steps, 1, device=device) < config.translation_outlier_prob
    outlier = torch.randn(batch, steps, 3, device=device, dtype=dtype) * config.translation_outlier_std_m * outlier_mask
    translation_noise = (drift + jitter + outlier) * active * valid

    pose_drift_steps = torch.randn(batch, steps, NUM_SMPL_JOINTS, 3, device=device, dtype=dtype)
    pose_drift = torch.cumsum(pose_drift_steps, dim=1)
    pose_drift = pose_drift / pose_drift.std(dim=1, keepdim=True).clamp_min(1e-5)
    pose_noise = pose_drift * config.pose_drift_std_rad
    pose_noise = pose_noise + torch.randn_like(pose_noise) * config.pose_jitter_std_rad
    pose_noise[:, :, 0] *= config.root_pose_multiplier
    pose_noise = pose_noise * active.unsqueeze(-1) * valid.unsqueeze(-1)

    base_rot = rot6d_to_rotmat(pose_6d.reshape(batch, steps, NUM_SMPL_JOINTS, 6))
    base_pose_6d = _rotmat_to_6d(axis_angle_to_rotmat(pose_noise) @ base_rot).reshape(batch, steps, 144)
    base_transl = transl + translation_noise
    # A conservative pseudo-confidence allows the refiner API to be exercised
    # before real NLF confidence is exported.  It is not treated as GT.
    confidence = torch.exp(-translation_noise.norm(dim=-1) / 0.10) * valid_mask.to(dtype=dtype)
    return {
        "base_pose_6d": base_pose_6d,
        "base_transl": base_transl,
        "confidence": confidence,
        "clean_clip_mask": clean_clip,
    }
