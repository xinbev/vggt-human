# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
import torch.nn as nn

from vggt_omega.models.aggregator import Aggregator
from vggt_omega.models.heads import AggregatorSMPLHead, CameraHead, DenseHead, TextAlignmentHead


class VGGTOmega(nn.Module):
    """Minimal VGGT-Omega inference model for camera and depth prediction."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_alignment: bool = False,
        enable_smpl: bool = False,
        num_smpl_queries: int = 0,
        smpl_num_layers: int = 4,
        smpl_intermediate_layer_idx: tuple[int, ...] = (4, 11, 17, 23),
        smpl_predict_boxes: bool = False,
        smpl_predict_id_embed: bool = False,
        smpl_id_embed_dim: int = 256,
        smpl_return_aux: bool = False,
        freeze_aggregator_forward: bool = False,
    ) -> None:
        super().__init__()
        if enable_smpl and num_smpl_queries <= 0:
            raise ValueError("enable_smpl=True requires num_smpl_queries > 0")

        self.aggregator = Aggregator(
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_smpl_queries=num_smpl_queries if enable_smpl else 0,
        )
        self.freeze_aggregator_forward = freeze_aggregator_forward
        _warn_if_rope_not_max(self.aggregator)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.dense_head = DenseHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_depth else None
        self.text_alignment_head = TextAlignmentHead(dim_in=2 * embed_dim) if enable_alignment else None
        self.smpl_head = (
            AggregatorSMPLHead(
                dim_in=2 * embed_dim,
                num_layers=smpl_num_layers,
                intermediate_layer_idx=smpl_intermediate_layer_idx,
                predict_boxes=smpl_predict_boxes,
                predict_id_embed=smpl_predict_id_embed,
                id_embed_dim=smpl_id_embed_dim,
                return_aux=smpl_return_aux,
            )
            if enable_smpl
            else None
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            if self.freeze_aggregator_forward:
                with torch.no_grad():
                    aggregated_tokens_list, token_layout = self.aggregator(images)
            else:
                aggregated_tokens_list, token_layout = self.aggregator(images)

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which VGGTOmega needs.")

        predictions = {
            "camera_and_register_tokens": final_tokens[:, :, : token_layout.register_end].contiguous(),
        }
        with torch.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                predictions["pose_enc"] = self.camera_head(
                    aggregated_tokens_list,
                    token_layout=token_layout,
                )

            if self.dense_head is not None:
                depth, depth_conf = self.dense_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=token_layout.patch_start,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.text_alignment_head is not None:
                predictions.update(
                    self.text_alignment_head(
                        aggregated_tokens_list,
                        token_layout=token_layout,
                    )
                )

            if self.smpl_head is not None:
                predictions.update(
                    self.smpl_head(
                        aggregated_tokens_list,
                        token_layout=token_layout,
                    )
                )

        if not self.training:
            predictions["images"] = images
        return predictions


def _warn_if_rope_not_max(aggregator: nn.Module) -> None:
    for name, module in (("aggregator.patch_embed", aggregator.patch_embed), ("aggregator", aggregator)):
        rope_embed = getattr(module, "rope_embed", None)
        normalize_coords = getattr(rope_embed, "normalize_coords", None)
        if normalize_coords != "max":
            warnings.warn(
                f"{name} RoPE normalize_coords is {normalize_coords!r}; "
                "the released VGGT-Omega checkpoint was trained with 'max'.",
                stacklevel=2,
            )
