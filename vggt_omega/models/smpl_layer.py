from pathlib import Path

import torch
import torch.nn as nn
#================================
import numpy as np
_LEGACY_NUMPY_ALIASES = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}

for name, value in _LEGACY_NUMPY_ALIASES.items():
    try:
        getattr(np, name)
    except AttributeError:
        setattr(np, name, value)
#================================
class SMPLLayer(nn.Module):
    """Project-local SMPL wrapper for visualization and geometry checks.

    This adapts the SAT-HMR SMPL wrapper idea to this repository without runtime
    imports from `.reference`. It is intentionally thin: callers provide SMPL
    axis-angle poses and betas, and the layer returns vertices and joints in the
    SMPL root/body coordinate system.
    """

    def __init__(self, model_path: str | Path, gender: str = "neutral") -> None:
        super().__init__()
        try:
            import smplx
        except ImportError as exc:
            raise ImportError("SMPLLayer requires the optional 'smplx' package for mesh/joint decoding") from exc

        self.layer = smplx.create(
            str(model_path),
            model_type="smpl",
            gender=gender,
            create_global_orient=False,
            create_body_pose=False,
            create_betas=False,
            create_transl=False,
        )
        self.faces = self.layer.faces

    def forward(self, poses_axis_angle: torch.Tensor, betas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if poses_axis_angle.ndim == 2:
            poses_axis_angle = poses_axis_angle.reshape(poses_axis_angle.shape[0], -1, 3)
        if poses_axis_angle.ndim != 3 or poses_axis_angle.shape[1:] != (24, 3):
            raise ValueError(f"Expected poses shape (N, 24, 3) or (N, 72), got {poses_axis_angle.shape}")
        if betas.ndim != 2 or betas.shape[0] != poses_axis_angle.shape[0]:
            raise ValueError(f"Expected betas shape (N, C) matching poses, got {betas.shape}")

        output = self.layer(
            global_orient=poses_axis_angle[:, :1],
            body_pose=poses_axis_angle[:, 1:],
            betas=betas,
        )
        return output.vertices, output.joints
