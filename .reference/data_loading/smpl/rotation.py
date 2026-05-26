from __future__ import annotations

import torch


def aa_to_rotmat(aa: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors to rotation matrices via Rodrigues' formula."""
    shape = aa.shape[:-1]
    aa = aa.reshape(-1, 3).float()
    angle = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa / angle
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    t = 1.0 - cos_a
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    R = torch.stack(
        [
            cos_a.squeeze() + t.squeeze() * x * x,
            t.squeeze() * x * y - sin_a.squeeze() * z,
            t.squeeze() * x * z + sin_a.squeeze() * y,
            t.squeeze() * x * y + sin_a.squeeze() * z,
            cos_a.squeeze() + t.squeeze() * y * y,
            t.squeeze() * y * z - sin_a.squeeze() * x,
            t.squeeze() * x * z - sin_a.squeeze() * y,
            t.squeeze() * y * z + sin_a.squeeze() * x,
            cos_a.squeeze() + t.squeeze() * z * z,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    return R.reshape(*shape, 3, 3)


def aa_to_6d(aa: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors to Zhou et al. 6D rotation representation."""
    R = aa_to_rotmat(aa)
    return R[..., :2].reshape(*R.shape[:-2], 6)


IDENTITY_6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
