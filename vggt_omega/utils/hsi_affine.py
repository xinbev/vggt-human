from __future__ import annotations

import torch


def apply_hsi_scene_affine_mode(
    predictions: dict[str, torch.Tensor],
    mode: str = "per_frame",
    ema_alpha: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Attach sequence-level HSI scene affine variants and select one as primary.

    The HSI head naturally predicts per-frame scale/bias. Video inference can be
    made more stable by replacing those per-frame values with a robust clip-level
    aggregate while preserving the raw predictions for diagnostics.
    """
    scale = predictions.get("hsi_scene_scale")
    bias = predictions.get("hsi_scene_depth_bias")
    if scale is None or bias is None:
        return predictions
    if scale.ndim < 3 or bias.ndim < 3 or scale.shape[:2] != bias.shape[:2]:
        return predictions

    mode = str(mode or "per_frame").lower()
    if mode not in {"per_frame", "clip_median", "ema"}:
        raise ValueError(f"Unsupported hsi_scene_affine_mode: {mode}")

    frame_scale = scale
    frame_bias = bias
    predictions["hsi_frame_scene_scale"] = frame_scale
    predictions["hsi_frame_scene_depth_bias"] = frame_bias

    clip_scale, clip_bias = clip_median_affine(frame_scale, frame_bias)
    ema_scale, ema_bias = ema_affine(frame_scale, frame_bias, alpha=float(ema_alpha))
    predictions["hsi_clip_scene_scale"] = clip_scale
    predictions["hsi_clip_scene_depth_bias"] = clip_bias
    predictions["hsi_ema_scene_scale"] = ema_scale
    predictions["hsi_ema_scene_depth_bias"] = ema_bias

    if mode == "clip_median":
        predictions["hsi_scene_scale"] = clip_scale
        predictions["hsi_scene_depth_bias"] = clip_bias
    elif mode == "ema":
        predictions["hsi_scene_scale"] = ema_scale
        predictions["hsi_scene_depth_bias"] = ema_bias
    return predictions


def clip_median_affine(scale: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    log_scale = torch.log(scale.float().clamp(min=1e-6))
    clip_log_scale = log_scale.median(dim=1, keepdim=True).values
    clip_bias = bias.float().median(dim=1, keepdim=True).values
    return torch.exp(clip_log_scale).to(dtype=scale.dtype).expand_as(scale), clip_bias.to(dtype=bias.dtype).expand_as(bias)


def ema_affine(scale: torch.Tensor, bias: torch.Tensor, alpha: float = 0.25) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = min(max(float(alpha), 0.0), 1.0)
    if scale.shape[1] <= 1:
        return scale, bias
    log_scale = torch.log(scale.float().clamp(min=1e-6))
    bias_float = bias.float()
    ema_log_frames = []
    ema_bias_frames = []
    prev_log = log_scale[:, 0]
    prev_bias = bias_float[:, 0]
    for frame_idx in range(scale.shape[1]):
        if frame_idx > 0:
            prev_log = alpha * log_scale[:, frame_idx] + (1.0 - alpha) * prev_log
            prev_bias = alpha * bias_float[:, frame_idx] + (1.0 - alpha) * prev_bias
        ema_log_frames.append(prev_log)
        ema_bias_frames.append(prev_bias)
    ema_scale = torch.exp(torch.stack(ema_log_frames, dim=1)).to(dtype=scale.dtype)
    ema_bias = torch.stack(ema_bias_frames, dim=1).to(dtype=bias.dtype)
    return ema_scale, ema_bias
