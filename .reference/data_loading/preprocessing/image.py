from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .imagenet import normalize_image_tensor
from .letterbox import letterbox_pil_image


def load_image_tensor(path: str | Path, size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load RGB image, direct-resize to a square, and ImageNet-normalize it."""
    img = Image.open(path).convert("RGB")
    orig_hw = (img.height, img.width)
    img = img.resize((int(size), int(size)), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = normalize_image_tensor(torch.from_numpy(arr)).permute(2, 0, 1)
    return tensor, orig_hw


def load_multihmr_letterbox_tensor(path: str | Path, size: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load RGB image with the same contain+pad letterbox used by Multi-HMR."""
    img = Image.open(path).convert("RGB")
    padded, meta = letterbox_pil_image(img, int(size))
    arr = np.asarray(padded, dtype=np.float32) / 255.0
    tensor = normalize_image_tensor(torch.from_numpy(arr)).permute(2, 0, 1)
    return tensor, meta
