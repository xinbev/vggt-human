"""Fixed, translation-only corruption for the V2 E0 overfit test."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TranslationNoiseConfig:
    drift_std_m: float = 0.06
    jitter_std_m: float = 0.025
    outlier_std_m: float = 0.10
    outlier_probability: float = 0.05


def corrupt_translation_sequence(
    target_transl: torch.Tensor,
    valid_mask: torch.Tensor,
    config: TranslationNoiseConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Make a stable per-track drift plus frame-specific single-frame jitter."""
    if target_transl.ndim != 3 or target_transl.shape[-1] != 3:
        raise ValueError(f"target_transl must be [B,S,3], got {tuple(target_transl.shape)}")
    batch, steps, _ = target_transl.shape
    dtype, device = target_transl.dtype, target_transl.device
    # The random walk is normalized per sequence so its scale is explicitly
    # controlled by drift_std_m even for the short 9-frame E0 window.
    drift_steps = torch.randn(batch, steps, 3, device=device, dtype=dtype, generator=generator)
    drift = torch.cumsum(drift_steps, dim=1)
    drift = drift - drift.mean(dim=1, keepdim=True)
    drift = drift / drift.std(dim=1, keepdim=True).clamp_min(1e-5)
    drift = drift * float(config.drift_std_m)
    jitter = torch.randn(batch, steps, 3, device=device, dtype=dtype, generator=generator) * float(config.jitter_std_m)
    outlier_mask = torch.rand(batch, steps, 1, device=device, dtype=dtype, generator=generator) < float(config.outlier_probability)
    outlier = torch.randn(batch, steps, 3, device=device, dtype=dtype, generator=generator) * float(config.outlier_std_m)
    noise = (drift + jitter + outlier * outlier_mask) * valid_mask.to(dtype=dtype).unsqueeze(-1)
    return target_transl + noise
