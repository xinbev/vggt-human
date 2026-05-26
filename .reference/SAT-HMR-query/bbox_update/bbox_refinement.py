"""BBox/reference refinement heads extracted from SAT-HMR.

SAT-HMR source reference:
- models/sat_model.py: bbox_embed and final bbox prediction
- models/decoder.py: iterative reference update inside decoder
"""

import copy
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

try:
    from reference_query_mechanism.query_init.query_initializer import inverse_sigmoid
except ImportError:
    from query_init.query_initializer import inverse_sigmoid


class MLP(nn.Module):
    """Simple multi-layer perceptron used for bbox heads."""

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


def get_clones(module: nn.Module, num_copies: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_copies)])


class BBoxRefinementHeads(nn.Module):
    """One bbox head per decoder layer."""

    def __init__(
        self,
        hidden_dim: int,
        num_decoder_layers: int,
        query_dim: int = 4,
        diff_each_layer: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_decoder_layers = num_decoder_layers
        self.query_dim = query_dim
        self.diff_each_layer = diff_each_layer

        if diff_each_layer:
            self.bbox_embed = nn.ModuleList(
                [MLP(hidden_dim, hidden_dim, query_dim, 3) for _ in range(num_decoder_layers)]
            )
            for head in self.bbox_embed:
                nn.init.constant_(head.layers[-1].weight.data, 0.0)
                nn.init.constant_(head.layers[-1].bias.data, 0.0)
        else:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, query_dim, 3)
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0.0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0.0)

    def head_for_layer(self, layer_idx: int) -> nn.Module:
        if self.diff_each_layer:
            return self.bbox_embed[layer_idx]
        return self.bbox_embed

    def refine_all_layers(
        self,
        hidden_states: torch.Tensor,
        references: torch.Tensor,
    ) -> torch.Tensor:
        """Produce bbox predictions from decoder states and references.

        Args:
            hidden_states: `(L, B, Q, C)` decoder outputs.
            references: `(L, B, Q, 4)` reference boxes before each layer.

        Returns:
            `(L, B, Q, 4)` normalized box predictions.
        """
        outputs = []
        reference_before_sigmoid = inverse_sigmoid(references)
        for layer_idx in range(hidden_states.shape[0]):
            tmp = self.head_for_layer(layer_idx)(hidden_states[layer_idx])
            tmp[..., : self.query_dim] += reference_before_sigmoid[layer_idx]
            outputs.append(tmp[..., : self.query_dim].sigmoid())
        return torch.stack(outputs)


def iterative_reference_update(
    hidden: torch.Tensor,
    reference_points: torch.Tensor,
    bbox_head: nn.Module,
    query_dim: int = 4,
) -> torch.Tensor:
    """Update reference boxes inside a decoder layer.

    Args:
        hidden: `(B * Q, C)` normalized decoder output.
        reference_points: `(B * Q, 4)` current normalized reference boxes.
        bbox_head: bbox MLP for this decoder layer.
        query_dim: number of reference dimensions.

    Returns:
        `(B * Q, 4)` updated normalized reference boxes, detached by caller if needed.
    """
    tmp = bbox_head(hidden)
    tmp[..., :query_dim] += inverse_sigmoid(reference_points)
    return tmp[..., :query_dim].sigmoid()
