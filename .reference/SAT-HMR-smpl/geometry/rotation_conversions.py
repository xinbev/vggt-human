"""Rotation conversion utilities used by the SMPL regression head.

The head predicts 6D rotations because they are easier to regress than
axis-angle rotations. SMPL/smplx expects axis-angle, so the final pose tensor
must be converted before calling the SMPL layer.

SAT-HMR source reference:
- utils/transforms.py: rot6d_to_axis_angle
"""

import torch
import torch.nn.functional as F


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to rotation matrices.

    Args:
        x: Tensor with shape `(..., 6)`.

    Returns:
        Tensor with shape `(..., 3, 3)`.
    """
    original_shape = x.shape[:-1]
    x = x.reshape(-1, 3, 2)

    a1 = x[:, :, 0]
    a2 = x[:, :, 1]

    b1 = F.normalize(a1, dim=1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)

    return torch.stack((b1, b2, b3), dim=-1).reshape(*original_shape, 3, 3)


def rotation_matrix_to_axis_angle(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to axis-angle.

    Args:
        rotation_matrix: Tensor with shape `(..., 3, 3)`.

    Returns:
        Tensor with shape `(..., 3)`.
    """
    matrix = rotation_matrix.reshape(-1, 3, 3)
    cos_angle = ((matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]) - 1.0) * 0.5
    cos_angle = cos_angle.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    angle = torch.acos(cos_angle)

    axis = torch.stack(
        [
            matrix[:, 2, 1] - matrix[:, 1, 2],
            matrix[:, 0, 2] - matrix[:, 2, 0],
            matrix[:, 1, 0] - matrix[:, 0, 1],
        ],
        dim=1,
    )
    axis = axis / (2.0 * torch.sin(angle).unsqueeze(1) + 1e-6)
    axis_angle = axis * angle.unsqueeze(1)
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle.reshape(*rotation_matrix.shape[:-2], 3)


def rot6d_to_axis_angle(x: torch.Tensor) -> torch.Tensor:
    """Convert batched 6D SMPL pose to axis-angle SMPL pose.

    Args:
        x: Tensor with shape `(batch_size, num_queries, num_poses * 6)`.

    Returns:
        Tensor with shape `(batch_size, num_queries, num_poses * 3)`.
    """
    batch_size, num_queries, _ = x.shape
    rot_mat = rot6d_to_rotmat(x.reshape(batch_size, num_queries, -1, 6))
    axis_angle = rotation_matrix_to_axis_angle(rot_mat)
    return axis_angle.reshape(batch_size, num_queries, -1)
