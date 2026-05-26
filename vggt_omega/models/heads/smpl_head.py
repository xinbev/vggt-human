import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.token_layout import AggregatorTokenLayout
from vggt_omega.utils.rotation import rot6d_to_axis_angle


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        layers = []
        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            out_dim = output_dim if layer_idx == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_idx < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _get_clones(module: nn.Module, num_layers: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])


class SMPLRegressionHead(nn.Module):
    """SAT-HMR-style residual regression from mean SMPL pose and shape."""

    def __init__(
        self,
        dim_in: int = 2048,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_joints: int = 24,
        num_betas: int = 10,
        mlp_layers: int = 3,
        mean_pose_6d: torch.Tensor | None = None,
        mean_shape: torch.Tensor | None = None,
        return_aux: bool = False,
        predict_boxes: bool = False,
        predict_id_embed: bool = False,
        id_embed_dim: int = 256,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        self.num_layers = num_layers
        self.num_joints = num_joints
        self.num_betas = num_betas
        self.return_aux = return_aux
        self.predict_boxes = predict_boxes
        self.predict_id_embed = predict_id_embed
        pose_dim = num_joints * 6

        self.norm = nn.LayerNorm(dim_in, eps=1e-5)
        self.pose_heads = _get_clones(MLP(dim_in, hidden_dim, pose_dim, mlp_layers), num_layers)
        self.shape_heads = _get_clones(MLP(dim_in, hidden_dim, num_betas, mlp_layers), num_layers)
        self.transl_cam_heads = _get_clones(MLP(dim_in, hidden_dim, 3, mlp_layers), num_layers)
        self.conf_heads = _get_clones(MLP(dim_in, hidden_dim, 1, mlp_layers), num_layers)
        if predict_boxes:
            self.box_heads = _get_clones(MLP(dim_in, hidden_dim, 4, mlp_layers), num_layers)
        else:
            self.box_heads = None
        if predict_id_embed:
            id_hidden_dim = max(hidden_dim // 2, 1)
            self.id_embed_head = nn.Sequential(
                nn.Linear(dim_in, id_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(id_hidden_dim),
                nn.Linear(id_hidden_dim, id_embed_dim),
            )
        else:
            self.id_embed_head = None

        if mean_pose_6d is None:
            mean_pose_6d = _identity_pose_6d(num_joints)
        if mean_shape is None:
            mean_shape = torch.zeros(num_betas)
        if mean_pose_6d.numel() != pose_dim:
            raise ValueError(f"mean_pose_6d must contain {pose_dim} values, got {mean_pose_6d.numel()}")
        if mean_shape.numel() != num_betas:
            raise ValueError(f"mean_shape must contain {num_betas} values, got {mean_shape.numel()}")

        self.register_buffer("mean_pose_6d", mean_pose_6d.reshape(1, 1, pose_dim).float(), persistent=False)
        self.register_buffer("mean_shape", mean_shape.reshape(1, 1, num_betas).float(), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if hidden_states.ndim != 4:
            raise ValueError(f"Expected hidden_states shape (L, B, Q, C), got {hidden_states.shape}")
        num_layers, batch_size, num_queries, _ = hidden_states.shape
        if num_layers != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} hidden-state layers, got {num_layers}")

        pose_6d = self.mean_pose_6d.to(dtype=hidden_states.dtype, device=hidden_states.device).expand(batch_size, num_queries, -1)
        shape = self.mean_shape.to(dtype=hidden_states.dtype, device=hidden_states.device).expand(batch_size, num_queries, -1)
        aux_outputs: dict[str, list[torch.Tensor]] = {
            "pred_poses": [],
            "pred_pose_6d": [],
            "pred_betas": [],
            "pred_confs": [],
            "pred_transl_cam": [],
        }
        if self.predict_boxes:
            aux_outputs["pred_boxes"] = []

        pred_transl_cam = None
        pred_confs = None
        pred_poses = None
        pred_boxes = None
        last_hidden = None
        for layer_idx in range(self.num_layers):
            hidden = self.norm(hidden_states[layer_idx].float())
            last_hidden = hidden
            pose_6d = pose_6d + self.pose_heads[layer_idx](hidden)
            shape = shape + self.shape_heads[layer_idx](hidden)
            pred_transl_cam = self.transl_cam_heads[layer_idx](hidden)
            pred_confs = torch.sigmoid(self.conf_heads[layer_idx](hidden))
            if self.box_heads is not None:
                pred_boxes = torch.sigmoid(self.box_heads[layer_idx](hidden))
            pred_poses = rot6d_to_axis_angle(pose_6d)

            aux_outputs["pred_poses"].append(pred_poses)
            aux_outputs["pred_pose_6d"].append(pose_6d)
            aux_outputs["pred_betas"].append(shape)
            aux_outputs["pred_confs"].append(pred_confs)
            aux_outputs["pred_transl_cam"].append(pred_transl_cam)
            if pred_boxes is not None:
                aux_outputs["pred_boxes"].append(pred_boxes)

        if pred_poses is None or pred_transl_cam is None or pred_confs is None:
            raise RuntimeError("SMPLRegressionHead produced no predictions")

        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "pred_poses": pred_poses,
            "pred_pose_6d": pose_6d,
            "pred_betas": shape,
            "pred_confs": pred_confs,
            "pred_transl_cam": pred_transl_cam,
            # Temporary alias for older callers/checkpoints. New code should use pred_transl_cam.
            "pred_cam": pred_transl_cam,
        }
        if pred_boxes is not None:
            outputs["pred_boxes"] = pred_boxes
        if self.id_embed_head is not None:
            if last_hidden is None:
                raise RuntimeError("SMPLRegressionHead produced no hidden states for ID embeddings")
            outputs["pred_id_embed"] = F.normalize(self.id_embed_head(last_hidden), dim=-1)
        if self.return_aux:
            outputs["aux_outputs"] = {key: torch.stack(values, dim=0) for key, values in aux_outputs.items()}
        return outputs


class AggregatorSMPLHead(nn.Module):
    def __init__(
        self,
        dim_in: int = 2048,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        intermediate_layer_idx: tuple[int, ...] = (4, 11, 17, 23),
        return_aux: bool = False,
        mean_pose_6d: torch.Tensor | None = None,
        mean_shape: torch.Tensor | None = None,
        predict_boxes: bool = False,
        predict_id_embed: bool = False,
        id_embed_dim: int = 256,
    ) -> None:
        super().__init__()
        if len(intermediate_layer_idx) != num_layers:
            raise ValueError("intermediate_layer_idx length must match num_layers")
        self.intermediate_layer_idx = intermediate_layer_idx
        self.regression_head = SMPLRegressionHead(
            dim_in=dim_in,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mean_pose_6d=mean_pose_6d,
            mean_shape=mean_shape,
            return_aux=return_aux,
            predict_boxes=predict_boxes,
            predict_id_embed=predict_id_embed,
            id_embed_dim=id_embed_dim,
        )

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        token_layout: AggregatorTokenLayout,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if token_layout.num_smpl_queries <= 0:
            raise ValueError("AggregatorSMPLHead requires token_layout with at least one SMPL query")

        hidden_states = []
        batch_size = num_frames = None
        for layer_idx in self.intermediate_layer_idx:
            tokens = aggregated_tokens_list[layer_idx]
            if tokens is None:
                raise ValueError(f"Aggregator did not cache layer {layer_idx}, which AggregatorSMPLHead needs.")
            batch_size, num_frames, _, dim = tokens.shape
            smpl_tokens = tokens[:, :, token_layout.smpl_start : token_layout.smpl_end]
            hidden_states.append(smpl_tokens.reshape(batch_size * num_frames, token_layout.num_smpl_queries, dim))

        regression_outputs = self.regression_head(torch.stack(hidden_states, dim=0))
        if batch_size is None or num_frames is None:
            raise RuntimeError("AggregatorSMPLHead produced no hidden states")

        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}
        for key, value in regression_outputs.items():
            if key == "aux_outputs":
                outputs[key] = {
                    aux_key: aux_value.reshape(
                        aux_value.shape[0],
                        batch_size,
                        num_frames,
                        *aux_value.shape[2:],
                    )
                    for aux_key, aux_value in value.items()
                }
            else:
                outputs[key] = value.reshape(batch_size, num_frames, *value.shape[1:])
        return outputs


def _identity_pose_6d(num_joints: int) -> torch.Tensor:
    identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    return identity_6d.repeat(num_joints)
