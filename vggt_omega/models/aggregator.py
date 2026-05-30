# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from vggt_omega.models.layers import Mlp, RopePositionEmbedding, SelfAttentionBlock
from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer
from vggt_omega.models.token_layout import AggregatorTokenLayout


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """Alternating-attention encoder over video frames."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 16,
        num_smpl_queries: int = 0,
        smpl_query_box_prior: bool = False,
        register_attention_block_indices: list[int] = [2, 6, 9, 14, 20],
        cached_layer_indices: tuple[int, ...] = (4, 11, 17, 23),
    ) -> None:
        super().__init__()
        if num_smpl_queries < 0:
            raise ValueError(f"num_smpl_queries must be non-negative, got {num_smpl_queries}")

        self.patch_embed = _build_patch_embed(patch_size=patch_size, embed_dim=embed_dim)
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100,
            normalize_coords="max",
            dtype=torch.float32,
        )

        self.frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )
        self.inter_frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.patch_size = patch_size
        self.num_smpl_queries = num_smpl_queries
        self.smpl_query_box_prior = smpl_query_box_prior
        self.cached_layer_indices = set(cached_layer_indices)
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.empty(1, 2, num_register_tokens, embed_dim))
        self.smpl_query_token = nn.Parameter(torch.empty(1, 2, num_smpl_queries, embed_dim))
        if smpl_query_box_prior and num_smpl_queries > 0:
            self.smpl_box_prior_embed = nn.Sequential(
                nn.Linear(5, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
            )
            self.smpl_fallback_boxes = nn.Parameter(_init_reference_boxes(num_smpl_queries).view(1, 1, num_smpl_queries, 4))
        else:
            self.smpl_box_prior_embed = None
            self.smpl_fallback_boxes = None
        self.token_layout = AggregatorTokenLayout(
            camera_start=0,
            camera_end=1,
            register_start=1,
            register_end=1 + num_register_tokens,
            smpl_start=1 + num_register_tokens,
            smpl_end=1 + num_register_tokens + num_smpl_queries,
            patch_start=1 + num_register_tokens + num_smpl_queries,
        )
        self.patch_token_start = self.token_layout.patch_start

        self.inter_frame_attention_types = ["global"] * depth
        for idx in register_attention_block_indices:
            if idx < 0 or idx >= depth:
                raise ValueError(f"register_attention_block_indices contains invalid block index {idx}")
            self.inter_frame_attention_types[idx] = "register"

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.camera_token, std=1e-3)
        nn.init.normal_(self.register_token, std=1e-3)
        nn.init.normal_(self.smpl_query_token, std=1e-3)

    def forward(
        self,
        images: torch.Tensor,
        smpl_query_boxes: torch.Tensor | None = None,
        smpl_query_boxes_mask: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor | None], AggregatorTokenLayout, torch.Tensor | None]:
        batch_size, num_frames, num_channels, height, width = images.shape
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(batch_size * num_frames, num_channels, height, width)

        camera_token = slice_expand_and_flatten(self.camera_token, batch_size, num_frames)
        register_token = slice_expand_and_flatten(self.register_token, batch_size, num_frames)
        smpl_query_token = slice_expand_and_flatten(self.smpl_query_token, batch_size, num_frames)
        smpl_reference_boxes = self._build_smpl_reference_boxes(
            batch_size,
            num_frames,
            smpl_query_token.device,
            smpl_query_token.dtype,
            smpl_query_boxes,
            smpl_query_boxes_mask,
        )
        smpl_query_token = self._add_smpl_box_prior(
            smpl_query_token,
            batch_size,
            num_frames,
            smpl_reference_boxes,
            smpl_query_boxes_mask,
        )

        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        tokens = torch.cat([camera_token, register_token, smpl_query_token, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape

        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
            frame_rope = (
                rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
            )

        outputs = []
        for block_idx in range(self.depth):
            tokens, frame_tokens = self._run_frame_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                frame_rope,
            )
            tokens = self._run_inter_frame_attention_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self.inter_frame_attention_types[block_idx],
            )
            if block_idx in self.cached_layer_indices:
                outputs.append(torch.cat([frame_tokens, tokens], dim=-1))
            else:
                outputs.append(None)

        return outputs, self.token_layout, smpl_reference_boxes

    def _build_smpl_reference_boxes(
        self,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
        smpl_query_boxes: torch.Tensor | None,
        smpl_query_boxes_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.num_smpl_queries == 0:
            return None
        if smpl_query_boxes is None:
            if self.smpl_fallback_boxes is None:
                return None
            return self.smpl_fallback_boxes.to(device=device, dtype=dtype).expand(batch_size, num_frames, -1, -1)
        if smpl_query_boxes.shape[:3] != (batch_size, num_frames, self.num_smpl_queries):
            raise ValueError(
                "smpl_query_boxes must have shape "
                f"(B, S, {self.num_smpl_queries}, 4), got {smpl_query_boxes.shape}"
            )
        boxes = smpl_query_boxes.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        fallback = (
            self.smpl_fallback_boxes.to(device=device, dtype=dtype).expand(batch_size, num_frames, -1, -1)
            if self.smpl_fallback_boxes is not None
            else torch.zeros_like(boxes)
        )
        if smpl_query_boxes_mask is None:
            mask = torch.ones(batch_size, num_frames, self.num_smpl_queries, dtype=torch.bool, device=device)
        else:
            mask = smpl_query_boxes_mask.to(device=device).bool()
            if mask.shape != (batch_size, num_frames, self.num_smpl_queries):
                raise ValueError(
                    "smpl_query_boxes_mask must have shape "
                    f"(B, S, {self.num_smpl_queries}), got {mask.shape}"
                )
        return torch.where(mask[..., None], boxes, fallback)

    def _add_smpl_box_prior(
        self,
        smpl_query_token: torch.Tensor,
        batch_size: int,
        num_frames: int,
        smpl_reference_boxes: torch.Tensor | None,
        smpl_query_boxes_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.smpl_box_prior_embed is None or self.num_smpl_queries == 0:
            return smpl_query_token
        if smpl_reference_boxes is None:
            return smpl_query_token
        boxes = smpl_reference_boxes.to(device=smpl_query_token.device, dtype=smpl_query_token.dtype).clamp(0.0, 1.0)
        if smpl_query_boxes_mask is None:
            mask = torch.zeros(batch_size, num_frames, self.num_smpl_queries, dtype=torch.bool, device=boxes.device)
        else:
            mask = smpl_query_boxes_mask.to(device=boxes.device).bool()
        prior_input = torch.cat([boxes, mask[..., None].to(dtype=boxes.dtype)], dim=-1)
        prior_embed = self.smpl_box_prior_embed(prior_input).reshape(batch_size * num_frames, self.num_smpl_queries, -1)
        return smpl_query_token + prior_embed.to(dtype=smpl_query_token.dtype)

    def _run_frame_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        rope_sincos: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokens.view(batch_size * num_frames, num_tokens, embed_dim)
        tokens = self.frame_blocks[block_idx](tokens, rope_sincos)
        return tokens, tokens.view(batch_size, num_frames, num_tokens, embed_dim)

    def _run_inter_frame_attention_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        attention_type: str,
    ) -> torch.Tensor:
        tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type == "global":
            tokens = tokens.view(batch_size, num_frames * num_tokens, embed_dim)
            tokens = self.inter_frame_blocks[block_idx](tokens, None)
            return tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type != "register":
            raise ValueError(f"Unknown inter-frame attention type: {attention_type}")

        patch_token_start = self.token_layout.patch_start
        context_tokens = tokens[:, :, :patch_token_start].reshape(
            batch_size,
            num_frames * patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, :, patch_token_start:].reshape(
            batch_size,
            num_frames * (num_tokens - patch_token_start),
            embed_dim,
        )

        context_tokens = self.inter_frame_blocks[block_idx](context_tokens, None)
        tokens = torch.cat([context_tokens, patch_tokens], dim=1)

        context_tokens = tokens[:, : num_frames * patch_token_start].view(
            batch_size,
            num_frames,
            patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, num_frames * patch_token_start :].view(
            batch_size,
            num_frames,
            num_tokens - patch_token_start,
            embed_dim,
        )
        return torch.cat([context_tokens, patch_tokens], dim=2)


def _build_patch_embed(patch_size: int, embed_dim: int) -> DinoVisionTransformer:
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=patch_size,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="max",
        pos_embed_rope_dtype="fp32",
        embed_dim=embed_dim,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-5,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    )
    model.init_weights()
    return model


def _init_reference_boxes(num_queries: int) -> torch.Tensor:
    if num_queries == 0:
        return torch.empty(0, 4)
    num_cols = int(num_queries**0.5)
    while num_cols > 1 and num_queries % num_cols != 0:
        num_cols -= 1
    num_rows = (num_queries + num_cols - 1) // num_cols
    xs = torch.linspace(0.1, 0.9, num_cols)
    ys = torch.linspace(0.2, 0.8, num_rows)
    boxes = []
    for y in ys:
        for x in xs:
            boxes.append(torch.tensor([x, y, 0.25, 0.50]))
            if len(boxes) == num_queries:
                return torch.stack(boxes, dim=0)
    return torch.stack(boxes, dim=0)


def slice_expand_and_flatten(token_tensor: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
    first_frame_token = token_tensor[:, 0:1].expand(batch_size, 1, *token_tensor.shape[2:])
    other_frame_tokens = token_tensor[:, 1:].expand(batch_size, num_frames - 1, *token_tensor.shape[2:])
    tokens = torch.cat([first_frame_token, other_frame_tokens], dim=1)
    return tokens.view(batch_size * num_frames, *tokens.shape[2:])
