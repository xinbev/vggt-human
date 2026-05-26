from __future__ import annotations

from typing import Any

import torch


FLATTEN_FRAME_KEYS = {
    "img_mhmr",
    "mhmr_letterbox_scale",
    "mhmr_letterbox_pad",
    "mhmr_orig_hw",
}


def bedlam_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate ``BedlamDataset`` items and flatten per-frame MHMR inputs.

    Schema after collation:
      - images: [B, S, 3, img_size, img_size]
      - img_mhmr: [B*S, 3, mhmr_size, mhmr_size]
      - gt_depth: [B, S, 1, img_size, img_size]
      - K_scal3r/K_mhmr: [B, S, 3, 3]
      - mhmr_letterbox_scale/pad/orig_hw: [B*S, 2]
      - joints3d_cam: [B, S, M, 24, 3]
      - gt_pose: [B, S, M, 144]
      - gt_betas: [B, S, M, 10]
      - gt_cam_trans: [B, S, M, 3]
      - smpl_mask: [B, S, M]
    """
    if not batch:
        raise ValueError("Cannot collate an empty BEDLAM batch")

    out: dict[str, torch.Tensor] = {}
    for key in batch[0].keys():
        tensors = [_require_tensor(item[key], key) for item in batch]
        stacked = torch.stack(tensors, dim=0)
        if key in FLATTEN_FRAME_KEYS:
            batch_size, sequence_len = stacked.shape[:2]
            stacked = stacked.reshape(batch_size * sequence_len, *stacked.shape[2:])
        out[key] = stacked
    return out


def _require_tensor(value: Any, key: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"BEDLAM batch field {key!r} must be a torch.Tensor, got {type(value)!r}")
    return value
