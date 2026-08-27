#!/usr/bin/env python3
"""Lightweight import/shape check for the standalone temporal refiner."""

import torch

from vggt_omega.models import TemporalRefinerConfig, TemporalSMPLRefiner, TemporalSMPLRefinerLoss
from vggt_omega.training import TemporalSMPLNoiseConfig, corrupt_smpl_sequence


def main() -> None:
    torch.manual_seed(0)
    model = TemporalSMPLRefiner(TemporalRefinerConfig(window_size=9, hidden_dim=96, num_heads=4, num_layers=2))
    target_pose = torch.randn(2, 9, 144)
    target_transl = torch.randn(2, 9, 3)
    valid = torch.ones(2, 9, dtype=torch.bool)
    noisy = corrupt_smpl_sequence(target_pose, target_transl, valid, TemporalSMPLNoiseConfig())
    outputs = model(noisy["base_pose_6d"], noisy["base_transl"], valid_mask=valid, confidence=noisy["confidence"])
    losses = TemporalSMPLRefinerLoss()(outputs, target_pose, target_transl, valid, noisy["base_pose_6d"], noisy["base_transl"], noisy["clean_clip_mask"])
    assert outputs["refined_pose_6d"].shape == (2, 9, 144)
    assert outputs["refined_transl"].shape == (2, 9, 3)
    assert torch.isfinite(losses["loss_total"])
    losses["loss_total"].backward()
    print("[OK] temporal refiner shape, finite-loss, and backward checks passed")


if __name__ == "__main__":
    main()
