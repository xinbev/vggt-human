"""Learnable query initialization extracted from SAT-HMR.

SAT-HMR source reference:
- models/sat_model.py: refpoint_embed, tgt_embed, random_refpoints_xy
"""

from typing import Tuple

import torch
from torch import nn


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Numerically stable inverse of sigmoid for values in [0, 1]."""
    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1.0 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class QueryInitializer(nn.Module):
    """Create regular DETR/DAB-DETR-style content and reference queries.

    Args:
        num_queries: maximum number of object/person slots.
        hidden_dim: decoder hidden dimension.
        query_dim: reference dimension. SAT-HMR uses 4 for `(cx, cy, w, h)`.
        random_refpoints_xy: initialize xy references randomly and freeze xy.
    """

    def __init__(
        self,
        num_queries: int,
        hidden_dim: int,
        query_dim: int = 4,
        random_refpoints_xy: bool = False,
    ) -> None:
        super().__init__()
        if query_dim != 4:
            raise ValueError("This reference follows SAT-HMR and expects query_dim=4.")

        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.query_dim = query_dim
        self.refpoint_embed = nn.Embedding(num_queries, query_dim)
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        if random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0.0, 1.0)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

    def forward(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return batched content queries and reference queries.

        Returns:
            tgt: `(B, Q, C)` content query embeddings.
            refpoint_embed: `(B, Q, 4)` reference boxes in inverse-sigmoid space.
        """
        tgt = self.tgt_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        refpoint_embed = self.refpoint_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        return tgt, refpoint_embed
