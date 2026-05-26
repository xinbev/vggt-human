from __future__ import annotations

import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def normalize_image_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize an RGB image tensor whose channel dimension is last or first."""
    if tensor.ndim < 3:
        raise ValueError(f"Expected an image tensor with at least 3 dims, got {tuple(tensor.shape)}")
    if tensor.shape[-1] == 3:
        return (tensor - IMAGENET_MEAN.to(tensor)) / IMAGENET_STD.to(tensor)
    if tensor.shape[-3] == 3:
        shape = [1] * tensor.ndim
        shape[-3] = 3
        mean = IMAGENET_MEAN.to(tensor).view(*shape)
        std = IMAGENET_STD.to(tensor).view(*shape)
        return (tensor - mean) / std
    raise ValueError(f"Could not infer RGB channel dimension from shape {tuple(tensor.shape)}")
