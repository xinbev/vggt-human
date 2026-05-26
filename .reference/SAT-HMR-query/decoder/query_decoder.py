"""Portable query decoder inspired by SAT-HMR.

This is not a byte-for-byte copy of SAT-HMR's decoder. It keeps the query
mechanism while replacing xformers attention with standard PyTorch attention.

SAT-HMR source reference:
- models/decoder.py: TransformerDecoder, XformerDecoder, XformerDecoderLayer
"""

import copy
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

try:
    from reference_query_mechanism.bbox_update.bbox_refinement import MLP, iterative_reference_update
    from reference_query_mechanism.position.reference_position_encoding import reference_box_sine_embed
    from reference_query_mechanism.query_init.query_initializer import inverse_sigmoid
except ImportError:
    from bbox_update.bbox_refinement import MLP, iterative_reference_update
    from position.reference_position_encoding import reference_box_sine_embed
    from query_init.query_initializer import inverse_sigmoid


def get_clones(module: nn.Module, num_copies: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_copies)])


def pad_flattened_sequence(flattened: torch.Tensor, lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert flattened variable-length memory into padded batch-first tensor.

    Args:
        flattened: `(sum(lengths), C)`.
        lengths: token lengths for each batch item.

    Returns:
        padded: `(B, max_len, C)`.
        padding_mask: `(B, max_len)`, True for padded positions.
    """
    batch_size = len(lengths)
    max_len = max(lengths)
    hidden_dim = flattened.shape[-1]
    padded = flattened.new_zeros(batch_size, max_len, hidden_dim)
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=flattened.device)

    cursor = 0
    for batch_idx, length in enumerate(lengths):
        padded[batch_idx, :length] = flattened[cursor : cursor + length]
        padding_mask[batch_idx, :length] = False
        cursor += length
    return padded, padding_mask


class QueryDecoderLayer(nn.Module):
    """One query decoder layer with self-attention, cross-attention, and FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "relu",
        keep_query_pos: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.keep_query_pos = keep_query_pos

        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.ca_qpos_sine_proj = nn.Linear(hidden_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        query_pos: torch.Tensor,
        query_sine_embed: torch.Tensor,
        is_first: bool,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = k = self.norm1(tgt) + query_pos
        v = self.norm1(tgt)
        tgt2, _ = self.self_attn(q, k, v, attn_mask=self_attn_mask, need_weights=False)
        tgt = tgt + self.dropout1(tgt2)

        tgt_norm = self.norm2(tgt)
        if is_first or self.keep_query_pos:
            q = tgt_norm + query_pos
        else:
            q = tgt_norm

        # SAT-HMR concatenates sine query information in attention channels.
        # This portable version injects it additively after projection.
        q = q + self.ca_qpos_sine_proj(query_sine_embed)
        k = memory + memory_pos
        v = memory
        tgt2, _ = self.cross_attn(
            q,
            k,
            v,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(self.norm3(tgt)))))
        tgt = tgt + tgt2
        return tgt


class QueryDecoder(nn.Module):
    """DAB-DETR-style query decoder with reference-box positional queries."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        query_dim: int = 4,
        query_scale_type: str = "cond_elewise",
        modulate_hw_attn: bool = True,
        keep_query_pos: bool = False,
        return_intermediate: bool = True,
    ) -> None:
        super().__init__()
        if query_dim != 4:
            raise ValueError("This reference follows SAT-HMR and expects query_dim=4.")
        if query_scale_type not in {"cond_elewise", "cond_scalar", "fix_elewise"}:
            raise ValueError("Unsupported query_scale_type.")

        layer = QueryDecoderLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            keep_query_pos=keep_query_pos,
        )
        self.layers = get_clones(layer, num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.query_dim = query_dim
        self.return_intermediate = return_intermediate
        self.query_scale_type = query_scale_type
        self.modulate_hw_attn = modulate_hw_attn

        if query_scale_type == "cond_elewise":
            self.query_scale = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        elif query_scale_type == "cond_scalar":
            self.query_scale = MLP(hidden_dim, hidden_dim, 1, 2)
        else:
            self.query_scale = nn.Embedding(num_layers, hidden_dim)

        self.ref_point_head = MLP(query_dim // 2 * hidden_dim, hidden_dim, hidden_dim, 2)
        self.ref_anchor_head = MLP(hidden_dim, hidden_dim, 2, 2) if modulate_hw_attn else None
        self.bbox_embed = None
        self.bbox_embed_diff_each_layer = True

    def attach_bbox_heads(self, bbox_heads: nn.ModuleList, diff_each_layer: bool = True) -> None:
        """Attach bbox heads for iterative reference updates inside decoder."""
        self.bbox_embed = bbox_heads
        self.bbox_embed_diff_each_layer = diff_each_layer

    def _bbox_head_for_layer(self, layer_idx: int) -> nn.Module:
        if self.bbox_embed is None:
            raise RuntimeError("bbox heads are not attached.")
        if self.bbox_embed_diff_each_layer:
            return self.bbox_embed[layer_idx]
        return self.bbox_embed

    def forward(
        self,
        memory: torch.Tensor,
        memory_lens: List[int],
        tgt: torch.Tensor,
        refpoint_embed: torch.Tensor,
        pos_embed: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode object/person queries.

        Args:
            memory: `(sum_tokens, C)` flattened encoder tokens.
            memory_lens: token count per batch item.
            tgt: `(B, Q, C)` content queries.
            refpoint_embed: `(B, Q, 4)` inverse-sigmoid reference queries.
            pos_embed: `(sum_tokens, C)` flattened encoder positional embeddings.
            self_attn_mask: optional `(Q, Q)` bool mask, used by denoising queries.
        """
        batch_size, num_queries, _ = tgt.shape
        memory_padded, memory_padding_mask = pad_flattened_sequence(memory, memory_lens)
        pos_padded, _ = pad_flattened_sequence(pos_embed, memory_lens)

        output = tgt
        reference_points = refpoint_embed.sigmoid().reshape(batch_size * num_queries, self.query_dim)
        intermediate = []
        references = [reference_points.view(batch_size, num_queries, self.query_dim)]

        for layer_idx, layer in enumerate(self.layers):
            obj_center = reference_points[:, : self.query_dim]
            query_sine_embed = reference_box_sine_embed(obj_center, self.hidden_dim)
            query_pos = self.ref_point_head(query_sine_embed).view(batch_size, num_queries, self.hidden_dim)

            if self.query_scale_type != "fix_elewise":
                pos_transformation = 1 if layer_idx == 0 else self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_idx].view(1, 1, -1)

            query_sine_for_cross = query_sine_embed[:, : self.hidden_dim].view(batch_size, num_queries, self.hidden_dim)
            query_sine_for_cross = query_sine_for_cross * pos_transformation

            if self.modulate_hw_attn:
                ref_hw_cond = self.ref_anchor_head(output).sigmoid()
                ref_boxes = obj_center.view(batch_size, num_queries, self.query_dim)
                query_sine_for_cross[..., self.hidden_dim // 2 :] *= (
                    ref_hw_cond[..., 0:1] / (ref_boxes[..., 2:3] + 1e-6)
                )
                query_sine_for_cross[..., : self.hidden_dim // 2] *= (
                    ref_hw_cond[..., 1:2] / (ref_boxes[..., 3:4] + 1e-6)
                )

            output = layer(
                tgt=output,
                memory=memory_padded,
                memory_pos=pos_padded,
                memory_padding_mask=memory_padding_mask,
                query_pos=query_pos,
                query_sine_embed=query_sine_for_cross,
                is_first=(layer_idx == 0),
                self_attn_mask=self_attn_mask,
            )

            normalized_output = self.norm(output)
            if self.bbox_embed is not None:
                flat_output = normalized_output.reshape(batch_size * num_queries, self.hidden_dim)
                reference_points = iterative_reference_update(
                    hidden=flat_output,
                    reference_points=reference_points,
                    bbox_head=self._bbox_head_for_layer(layer_idx),
                    query_dim=self.query_dim,
                )
                if layer_idx != self.num_layers - 1:
                    references.append(reference_points.view(batch_size, num_queries, self.query_dim))
                reference_points = reference_points.detach()

            intermediate.append(normalized_output)

        return torch.stack(intermediate), torch.stack(references)
