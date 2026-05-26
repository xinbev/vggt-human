"""End-to-end example: decoder hidden states -> SMPL mesh outputs.

This file shows the migration glue code. It is not meant to replace your full
model; copy the relevant parts into your model's forward method.

SAT-HMR source reference:
- models/sat_model.py: process_smpl and final output dictionary
"""

from typing import Dict, Optional, Union

import numpy as np
import torch
from torch import nn

try:
    from reference_smpl_regression.config.smpl_paths import SMPL_MEAN_PARAMS_PATH, SMPL_MODEL_PATH
    from reference_smpl_regression.geometry.camera_projection import project_smpl_outputs
    from reference_smpl_regression.heads.smpl_regression_head import SMPLRegressionHead
    from reference_smpl_regression.smpl_layer.smpl_layer import SMPLLayer
except ImportError:  # Allows copying this folder into another package layout.
    from config.smpl_paths import SMPL_MEAN_PARAMS_PATH, SMPL_MODEL_PATH
    from geometry.camera_projection import project_smpl_outputs
    from heads.smpl_regression_head import SMPLRegressionHead
    from smpl_layer.smpl_layer import SMPLLayer


class DecoderToSMPL(nn.Module):
    """Minimal wrapper that turns decoder states into SMPL outputs."""

    def __init__(
        self,
        hidden_dim: int,
        num_decoder_layers: int,
        input_size: int,
        default_focal: float,
        smpl_model_path=SMPL_MODEL_PATH,
        smpl_mean_params_path=SMPL_MEAN_PARAMS_PATH,
    ) -> None:
        super().__init__()
        mean_params = np.load(smpl_mean_params_path, allow_pickle=True)
        mean_pose = torch.from_numpy(mean_params["pose"]).float()
        mean_shape = torch.from_numpy(mean_params["shape"]).float()

        self.input_size = input_size
        self.default_focal = default_focal
        self.smpl_head = SMPLRegressionHead(
            hidden_dim=hidden_dim,
            num_decoder_layers=num_decoder_layers,
            mean_pose_6d=mean_pose,
            mean_shape=mean_shape,
        )
        self.human_model = SMPLLayer(model_path=smpl_model_path, with_genders=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cam_intrinsics: torch.Tensor,
        focal: Optional[Union[float, torch.Tensor]] = None,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run SMPL regression and projection.

        Args:
            hidden_states: `(L, B, Q, C)` decoder outputs.
            cam_intrinsics: `(B, 1, 3, 3)` camera intrinsics.
            focal: optional focal length for depth conversion.
            return_aux: whether to return auxiliary decoder-layer predictions.
        """
        head_outputs = self.smpl_head(hidden_states, return_aux=return_aux)
        poses = head_outputs["pred_poses"]
        betas = head_outputs["pred_betas"]
        cam = head_outputs["pred_cam"]

        batch_size, num_queries, _ = poses.shape
        flat_poses = poses.flatten(0, 1)
        flat_betas = betas.flatten(0, 1)

        verts, joints = self.human_model(poses=flat_poses, betas=flat_betas)
        verts = verts.reshape(batch_size, num_queries, verts.shape[1], 3)
        joints = joints.reshape(batch_size, num_queries, joints.shape[1], 3)

        verts_cam, joints_cam, joints_2d, depths, transl = project_smpl_outputs(
            verts=verts,
            joints=joints,
            cam_xys=cam,
            cam_intrinsics=cam_intrinsics,
            input_size=self.input_size,
            focal=focal,
            default_focal=self.default_focal,
        )

        output = {
            "pred_poses": poses,
            "pred_betas": betas,
            "pred_confs": head_outputs["pred_confs"],
            "pred_verts": verts_cam,
            "pred_j3ds": joints_cam,
            "pred_j2ds": joints_2d,
            "pred_depths": depths,
            "pred_transl": transl,
            "pred_intrinsics": cam_intrinsics,
        }
        if return_aux:
            output["aux_outputs"] = head_outputs["aux_outputs"]
        return output
