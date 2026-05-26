from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .imagenet import normalize_image_tensor


def compute_multihmr_letterbox_meta(orig_w: int, orig_h: int, img_size: int) -> dict[str, Any]:
    """Return explicit Multi-HMR contain+pad geometry metadata."""
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid original image size: {(orig_w, orig_h)}")
    scale = min(float(img_size) / float(orig_w), float(img_size) / float(orig_h))
    new_w = max(1, int(round(float(orig_w) * scale)))
    new_h = max(1, int(round(float(orig_h) * scale)))
    new_w = min(int(img_size), new_w)
    new_h = min(int(img_size), new_h)
    pad_x = (int(img_size) - new_w) // 2
    pad_y = (int(img_size) - new_h) // 2
    scale_x = float(new_w) / float(orig_w)
    scale_y = float(new_h) / float(orig_h)
    return {
        "orig_w": int(orig_w),
        "orig_h": int(orig_h),
        "img_size": int(img_size),
        "new_w": int(new_w),
        "new_h": int(new_h),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "scale": float(scale),
    }


def letterbox_pil_image(
    img: Image.Image,
    img_size: int,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> tuple[Image.Image, dict[str, Any]]:
    """Resize with aspect ratio preserved and paste on a square canvas."""
    orig_w, orig_h = img.size
    meta = compute_multihmr_letterbox_meta(orig_w, orig_h, img_size)
    resized = img.resize((int(meta["new_w"]), int(meta["new_h"])), Image.BILINEAR)
    canvas = Image.new("RGB", (int(img_size), int(img_size)), fill)
    canvas.paste(resized, (int(meta["pad_x"]), int(meta["pad_y"])))
    return canvas, meta


def letterbox_intrinsics(K: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    """Map camera intrinsics from original pixels to padded Multi-HMR pixels."""
    K_l = np.asarray(K, dtype=np.float32).copy()
    K_l[0, 0] *= float(meta["scale_x"])
    K_l[0, 2] = K_l[0, 2] * float(meta["scale_x"]) + float(meta["pad_x"])
    K_l[1, 1] *= float(meta["scale_y"])
    K_l[1, 2] = K_l[1, 2] * float(meta["scale_y"]) + float(meta["pad_y"])
    return K_l


def multihmr_meta_to_tensors(
    meta: dict[str, Any],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Convert one letterbox metadata dict to the Scal3RHuman tensor API."""
    return {
        "mhmr_letterbox_scale": torch.tensor(
            [[float(meta["scale_x"]), float(meta["scale_y"])]], device=device, dtype=dtype
        ),
        "mhmr_letterbox_pad": torch.tensor(
            [[float(meta["pad_x"]), float(meta["pad_y"])]], device=device, dtype=dtype
        ),
        "mhmr_orig_hw": torch.tensor(
            [[float(meta["orig_h"]), float(meta["orig_w"])]], device=device, dtype=dtype
        ),
    }


def preprocess_multihmr_image(image_path: str | Path, img_size: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """Preprocess one image with Multi-HMR contain+pad letterbox semantics."""
    path = Path(image_path).expanduser().resolve()
    img = Image.open(path).convert("RGB")
    padded, meta = letterbox_pil_image(img, img_size)

    arr = np.asarray(padded, dtype=np.float32) / 255.0
    x = normalize_image_tensor(torch.from_numpy(arr)).permute(2, 0, 1).unsqueeze(0)
    meta.update(
        {
            "image_path": str(path),
            "normalization": "Multi-HMR contain+pad letterbox + ImageNet mean/std",
        }
    )
    return x, meta
