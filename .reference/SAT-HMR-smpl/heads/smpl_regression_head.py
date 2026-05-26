"""SMPL regression heads extracted from SAT-HMR.

This file intentionally does not include SAT-HMR encoder/decoder code. It only
contains the output-head pattern that can be connected to any decoder producing
`(num_layers, batch_size, num_queries, hidden_dim)` hidden states.

SAT-HMR source reference:
- models/sat_model.py: _get_clones, MLP, pose_head, shape_head, cam_head, conf_head
- models/sat_model.py: forward loop around lines 754-821
"""

import copy
import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

try:
    from reference_smpl_regression.geometry.rotation_conversions import rot6d_to_axis_angle
except ImportError:  # Allows copying this file into another package layout.
    from geometry.rotation_conversions import rot6d_to_axis_angle


def get_clones(module: nn.Module, num_copies: int) -> nn.ModuleList:
    """Return independent deep copies of a module.

    SAT-HMR uses this for one independent pose/shape head per decoder layer.
    The architecture is shared, but parameters are not shared.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_copies)])


class MLP(nn.Module):
    """Simple multi-layer perceptron used by SAT-HMR heads."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        hidden_dims = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip([input_dim] + hidden_dims, hidden_dims + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SMPLRegressionHead(nn.Module):
    """Regress SMPL pose, shape, confidence, and camera parameters.

    This mirrors SAT-HMR's residual iterative refinement:

    ```text
    pose_6d = mean_pose_6d
    shape = mean_shape
    for each decoder layer:
        pose_6d = pose_6d + pose_head[layer](hidden_states[layer])
        shape = shape + shape_head[layer](hidden_states[layer])
    ```

    The final 6D pose is converted to SMPL axis-angle before returning.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_decoder_layers: int,
        mean_pose_6d: torch.Tensor,
        mean_shape: torch.Tensor,
        num_poses: int = 24,
        dim_shape: int = 10,
        pose_head_layers: int = 6,
        shape_head_layers: int = 5,
        cam_head_layers: int = 3,
        prior_prob: float = 0.01,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_decoder_layers = num_decoder_layers
        self.num_poses = num_poses
        self.dim_shape = dim_shape

        if mean_pose_6d.numel() != num_poses * 6:
            raise ValueError(f"mean_pose_6d must contain {num_poses * 6} values.")
        if mean_shape.numel() != dim_shape:
            raise ValueError(f"mean_shape must contain {dim_shape} values.")

        self.register_buffer("mean_pose", mean_pose_6d.float().reshape(1, 1, num_poses * 6))
        self.register_buffer("mean_shape", mean_shape.float().reshape(1, 1, dim_shape))

        pose_template = MLP(hidden_dim, hidden_dim, num_poses * 6, pose_head_layers)
        shape_template = MLP(hidden_dim, hidden_dim, dim_shape, shape_head_layers)

        self.pose_head = get_clones(pose_template, num_decoder_layers)
        self.shape_head = get_clones(shape_template, num_decoder_layers)
        self.cam_head = MLP(hidden_dim, hidden_dim // 2, 3, cam_head_layers)
        self.conf_head = nn.Linear(hidden_dim, 1)

        bias_value = -math.log((1.0 - prior_prob) / prior_prob)
        self.conf_head.bias.data = torch.ones(1) * bias_value

    def forward(self, hidden_states: torch.Tensor, return_aux: bool = False) -> Dict[str, torch.Tensor]:
        """Predict SMPL parameters from decoder hidden states.

        Args:
            hidden_states: `(L, B, Q, C)` decoder outputs.
            return_aux: Whether to include all intermediate layer outputs.
        """
        if hidden_states.ndim != 4:
            raise ValueError("hidden_states must have shape (num_layers, batch_size, num_queries, hidden_dim).")
        if hidden_states.shape[0] != self.num_decoder_layers:
            raise ValueError(
                f"Expected {self.num_decoder_layers} decoder layers, got {hidden_states.shape[0]}."
            )
        if hidden_states.shape[-1] != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, got {hidden_states.shape[-1]}.")

        pose_6d = self.mean_pose.expand(hidden_states.shape[1], hidden_states.shape[2], -1)
        shape = self.mean_shape.expand(hidden_states.shape[1], hidden_states.shape[2], -1)

        all_poses = []
        all_betas = []
        all_confs = []
        all_cams = []

        for layer_idx in range(hidden_states.shape[0]):
            layer_hidden = hidden_states[layer_idx]
            pose_6d = pose_6d + self.pose_head[layer_idx](layer_hidden)
            shape = shape + self.shape_head[layer_idx](layer_hidden)

            pose_axis_angle = rot6d_to_axis_angle(pose_6d)
            conf = self.conf_head(layer_hidden).sigmoid()
            cam = self.cam_head(layer_hidden)

            all_poses.append(pose_axis_angle)
            all_betas.append(shape)
            all_confs.append(conf)
            all_cams.append(cam)

        output = {
            "pred_poses": all_poses[-1],
            "pred_betas": all_betas[-1],
            "pred_confs": all_confs[-1],
            "pred_cam": all_cams[-1],
        }
        if return_aux:
            output["aux_outputs"] = {
                "pred_poses": torch.stack(all_poses),
                "pred_betas": torch.stack(all_betas),
                "pred_confs": torch.stack(all_confs),
                "pred_cam": torch.stack(all_cams),
            }
        return output
