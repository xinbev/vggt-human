"""Camera projection helper for SMPL outputs.

This mirrors SAT-HMR's `process_smpl` camera convention:

- The model predicts `cam_xys` with shape `(B, Q, 3)`.
- `cam_xys[..., :2]` controls x/y translation after scaling.
- `cam_xys[..., 2]` controls depth scale through a sigmoid.
- Depth is computed as `2 * focal / (scale * input_size)`.

TODO[target-project]: adapt this file if your camera convention differs.

SAT-HMR source reference:
- models/sat_model.py: process_smpl
"""

from typing import Optional, Tuple, Union

import torch


def project_smpl_outputs(
    verts: torch.Tensor,
    joints: torch.Tensor,
    cam_xys: torch.Tensor,
    cam_intrinsics: torch.Tensor,
    input_size: int,
    focal: Optional[Union[float, torch.Tensor]] = None,
    default_focal: Optional[float] = None,
    detach_joints_for_projection: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply camera translation and project SMPL joints.

    Args:
        verts: `(B, Q, V, 3)` SMPL vertices in body/root coordinates.
        joints: `(B, Q, J, 3)` SMPL joints in body/root coordinates.
        cam_xys: `(B, Q, 3)` predicted camera parameters.
        cam_intrinsics: `(B, 1, 3, 3)` or `(B, Q, 3, 3)` camera intrinsic matrices.
        input_size: network input size used by the SAT-HMR depth formula.
        focal: optional focal length. Can be scalar, `(B,)`, or `(B, 1)`.
        default_focal: fallback focal length when `focal` is None.
        detach_joints_for_projection: match SAT-HMR's optional detached 2D projection path.

    Returns:
        verts_cam: `(B, Q, V, 3)` camera-space vertices.
        joints_cam: `(B, Q, J, 3)` camera-space joints.
        joints_2d: `(B, Q, J, 2)` projected image-space joints.
        depths: `(B, Q, 2)`, containing raw root depth and normalized depth.
        transl: `(B, Q, 3)` camera translation.
    """
    if focal is None:
        if default_focal is None:
            raise ValueError("Either `focal` or `default_focal` must be provided.")
        focal_for_depth = default_focal
    else:
        focal_for_depth = focal

    if isinstance(focal_for_depth, torch.Tensor):
        if focal_for_depth.dim() == 1:
            focal_for_depth = focal_for_depth.unsqueeze(1)
        focal_for_depth = focal_for_depth.unsqueeze(-1)

    scale = 2.0 * cam_xys[:, :, 2:].sigmoid() + 1e-6
    t_xy = cam_xys[:, :, :2] / scale
    t_z = (2.0 * focal_for_depth) / (scale * input_size)
    transl = torch.cat([t_xy, t_z], dim=2)[:, :, None, :]

    verts_cam = verts + transl
    joints_cam = joints + transl

    project_source = joints.detach() + transl if detach_joints_for_projection else joints_cam
    joints_homo = torch.matmul(project_source, cam_intrinsics.transpose(2, 3))
    joints_2d = joints_homo[..., :2] / (joints_homo[..., 2, None] + 1e-6)

    focal_scalar = focal_for_depth.mean() if isinstance(focal_for_depth, torch.Tensor) else focal_for_depth
    depths = joints_cam[:, :, 0, 2:]
    depths = torch.cat([depths, depths / focal_scalar], dim=-1)

    return verts_cam, joints_cam, joints_2d, depths, transl.flatten(2)
