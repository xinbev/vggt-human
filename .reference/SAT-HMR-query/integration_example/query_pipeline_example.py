"""End-to-end query mechanism example.

This file shows how to combine query initialization, decoder, reference updates,
and bbox predictions. The resulting `hidden_states` can feed SMPL heads or any
other per-query prediction heads.

SAT-HMR source reference:
- models/sat_model.py: query init, decoder call, bbox prediction
- models/decoder.py: query decoder and reference update
"""

from typing import Dict, List, Optional

import torch
from torch import nn

try:
    from reference_query_mechanism.bbox_update.bbox_refinement import BBoxRefinementHeads
    from reference_query_mechanism.decoder.query_decoder import QueryDecoder
    from reference_query_mechanism.denoising_optional.denoising_queries import prepare_denoising_queries
    from reference_query_mechanism.query_init.query_initializer import QueryInitializer
except ImportError:
    from bbox_update.bbox_refinement import BBoxRefinementHeads
    from decoder.query_decoder import QueryDecoder
    from denoising_optional.denoising_queries import prepare_denoising_queries
    from query_init.query_initializer import QueryInitializer


class QueryPipeline(nn.Module):
    """Portable SAT-HMR-style query pipeline."""

    def __init__(
        self,
        hidden_dim: int,
        num_queries: int,
        num_decoder_layers: int,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        use_denoising: bool = False,
        dn_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.use_denoising = use_denoising
        self.dn_cfg = dn_cfg or {"use_dn": False}

        self.query_initializer = QueryInitializer(
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            query_dim=4,
        )
        self.bbox_heads = BBoxRefinementHeads(
            hidden_dim=hidden_dim,
            num_decoder_layers=num_decoder_layers,
            query_dim=4,
            diff_each_layer=True,
        )
        self.decoder = QueryDecoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            query_dim=4,
            modulate_hw_attn=True,
        )
        self.decoder.attach_bbox_heads(self.bbox_heads.bbox_embed, diff_each_layer=True)

        if self.use_denoising:
            tgt_embed_type = self.dn_cfg.get("tgt_embed_type", "labels")
            if tgt_embed_type == "labels":
                self.dn_encoder = nn.Embedding(self.dn_cfg["dn_labelbook_size"], hidden_dim)
            elif tgt_embed_type == "params":
                self.dn_encoder = nn.Linear(24 * 3 + 10, hidden_dim)
            else:
                raise ValueError("dn_cfg['tgt_embed_type'] must be 'labels' or 'params'.")
        else:
            self.dn_encoder = None

    def forward(
        self,
        memory: torch.Tensor,
        memory_lens: List[int],
        pos_embed: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the query pipeline.

        Args:
            memory: `(sum_tokens, hidden_dim)` flattened encoder features.
            memory_lens: token count per image.
            pos_embed: `(sum_tokens, hidden_dim)` flattened positional encodings.
            targets: optional training targets for denoising queries.
        """
        batch_size = len(memory_lens)
        tgt, refpoint_embed = self.query_initializer(batch_size)
        attn_mask = None
        dn_meta = None

        if self.training and self.use_denoising:
            if targets is None:
                raise ValueError("targets are required when denoising is enabled during training.")
            input_query_tgt, input_query_bbox, attn_mask, dn_meta = prepare_denoising_queries(
                targets=targets,
                dn_cfg=self.dn_cfg,
                num_queries=self.num_queries,
                hidden_dim=self.hidden_dim,
                dn_encoder=self.dn_encoder,
            )
            tgt = torch.cat([input_query_tgt, tgt], dim=1)
            refpoint_embed = torch.cat([input_query_bbox, refpoint_embed], dim=1)

        hidden_states, references = self.decoder(
            memory=memory,
            memory_lens=memory_lens,
            tgt=tgt,
            refpoint_embed=refpoint_embed,
            pos_embed=pos_embed,
            self_attn_mask=attn_mask,
        )
        pred_boxes = self.bbox_heads.refine_all_layers(hidden_states, references)

        output = {
            "hidden_states": hidden_states,
            "references": references,
            "pred_boxes": pred_boxes[-1],
            "aux_pred_boxes": pred_boxes[:-1],
        }
        if dn_meta is not None:
            output["dn_meta"] = dn_meta
        return output


def example_connect_to_smpl_head(query_outputs: Dict[str, torch.Tensor], smpl_head: nn.Module) -> Dict[str, torch.Tensor]:
    """Show how to connect this query pipeline to a SMPL regression head."""
    hidden_states = query_outputs["hidden_states"]
    return smpl_head(hidden_states, return_aux=True)
