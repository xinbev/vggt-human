"""Portable SMPL layer wrapper.

SAT-HMR source reference:
- models/human_models/smpl_models.py
"""

from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import smplx
import torch
from torch import nn


class SMPLLayer(nn.Module):
    """Thin wrapper around `smplx.create(..., model_type='smpl')`.

    Args:
        model_path: Path to the SMPL asset root accepted by smplx.
        with_genders: Whether to instantiate neutral, male, and female layers.
        load_extra_regressors: Whether to load SAT-HMR optional regressor files.
        **kwargs: Extra keyword arguments passed to `smplx.create`.
    """

    def __init__(
        self,
        model_path: str | Path,
        with_genders: bool = False,
        load_extra_regressors: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        model_path = str(model_path)
        smpl_kwargs = {
            "create_global_orient": False,
            "create_body_pose": False,
            "create_betas": False,
            "create_transl": False,
        }
        smpl_kwargs.update(kwargs)

        self.with_genders = with_genders
        self.layer_n = smplx.create(model_path, "smpl", gender="neutral", **smpl_kwargs)
        self.layers = {"neutral": self.layer_n}

        if with_genders:
            self.layer_m = smplx.create(model_path, "smpl", gender="male", **smpl_kwargs)
            self.layer_f = smplx.create(model_path, "smpl", gender="female", **smpl_kwargs)
            self.layers.update({"male": self.layer_m, "female": self.layer_f})

        self.vertex_num = 6890
        self.faces = self.layer_n.faces

        self.body_vertex_idx = None
        self.smpl2h36m_regressor = None
        if load_extra_regressors:
            smpl_dir = Path(model_path) / "smpl"
            self.body_vertex_idx = np.load(smpl_dir / "body_verts_smpl.npy")
            self.smpl2h36m_regressor = np.load(smpl_dir / "J_regressor_h36m_correct.npy")

    def forward_single_gender(
        self,
        poses: torch.Tensor,
        betas: torch.Tensor,
        gender: str = "neutral",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = poses.shape[0]
        if poses.ndim == 2:
            poses = poses.view(batch_size, -1, 3)

        if poses.shape[1] != 24:
            raise ValueError(f"Expected 24 SMPL joints, got {poses.shape[1]}.")

        pose_params = {
            "global_orient": poses[:, :1, :],
            "body_pose": poses[:, 1:, :],
        }
        smpl_output = self.layers[gender](betas=betas, **pose_params)
        return smpl_output.vertices, smpl_output.joints

    def forward(
        self,
        poses: torch.Tensor,
        betas: torch.Tensor,
        genders: Optional[Iterable[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if poses.shape[0] != betas.shape[0]:
            raise ValueError("`poses` and `betas` must have the same batch dimension.")

        if genders is None:
            return self.forward_single_gender(poses, betas, gender="neutral")

        genders = list(genders)
        if len(genders) != poses.shape[0]:
            raise ValueError("Length of `genders` must match batch size.")
        if not self.with_genders:
            raise ValueError("Gendered SMPL layers were not initialized.")
        if not set(genders) <= {"male", "female"}:
            raise ValueError("Supported genders are only 'male' and 'female'.")

        male_idx = [i for i, gender in enumerate(genders) if gender == "male"]
        if len(male_idx) == poses.shape[0]:
            return self.forward_single_gender(poses, betas, gender="male")
        if len(male_idx) == 0:
            return self.forward_single_gender(poses, betas, gender="female")

        vertices, joints = self.forward_single_gender(poses, betas, gender="female")
        male_vertices, male_joints = self.forward_single_gender(
            poses[male_idx],
            betas[male_idx],
            gender="male",
        )
        vertices[male_idx] = male_vertices
        joints[male_idx] = male_joints
        return vertices, joints
