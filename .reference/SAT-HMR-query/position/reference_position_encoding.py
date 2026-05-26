"""Reference-box sine position encoding.

SAT-HMR source reference:
- models/position_encoding.py: position_encoding_xy
- models/decoder.py: query_sine_embed from obj_center
"""

import math

import torch


def position_encoding_xy(
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    embedding_dim: int,
    temperature: float = 20.0,
    scale: float = 2.0 * math.pi,
) -> torch.Tensor:
    """Encode normalized xy coordinates into sine/cosine embeddings.

    Args:
        pos_x: `(N,)` normalized x coordinate.
        pos_y: `(N,)` normalized y coordinate.
        embedding_dim: output dimension. Must be even.

    Returns:
        `(N, embedding_dim)` positional embedding.
    """
    if embedding_dim % 2 != 0:
        raise ValueError("embedding_dim must be even.")
    if pos_x.ndim != 1 or pos_y.ndim != 1:
        raise ValueError("pos_x and pos_y must be 1D tensors.")

    dim_t = torch.arange(embedding_dim // 2, dtype=torch.float32, device=pos_x.device)
    dim_t = temperature ** (2 * (dim_t // 2) / (embedding_dim // 2))

    x_embed = pos_x * scale
    y_embed = pos_y * scale
    pos_x = x_embed[:, None] / dim_t
    pos_y = y_embed[:, None] / dim_t

    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
    pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
    return torch.cat([pos_y, pos_x], dim=1)


def reference_box_sine_embed(reference_boxes: torch.Tensor, hidden_dim: int) -> torch.Tensor:
    """Build SAT-HMR-style sine embeddings from reference boxes.

    Args:
        reference_boxes: `(B * Q, 4)` or `(N, 4)` normalized `(cx, cy, w, h)`.
        hidden_dim: decoder hidden dimension.

    Returns:
        `(N, hidden_dim * 2)`, concatenating xy and wh embeddings.
    """
    if reference_boxes.shape[-1] != 4:
        raise ValueError("reference_boxes must end with 4 values: cx, cy, w, h.")

    xy_embed = position_encoding_xy(reference_boxes[:, 0], reference_boxes[:, 1], hidden_dim)
    wh_embed = position_encoding_xy(reference_boxes[:, 2], reference_boxes[:, 3], hidden_dim)
    return torch.cat([xy_embed, wh_embed], dim=1)
