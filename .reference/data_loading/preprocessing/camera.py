from __future__ import annotations

from typing import Sequence

import numpy as np


def scale_intrinsics_for_resize(
    K: np.ndarray,
    src_hw: Sequence[int],
    dst_hw: Sequence[int] | int,
) -> np.ndarray:
    """Scale a 3x3 camera intrinsics matrix for direct image resize."""
    if isinstance(dst_hw, int):
        dst_h = dst_w = int(dst_hw)
    else:
        dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Invalid source image size: {(src_h, src_w)}")

    K_scaled = np.asarray(K, dtype=np.float32).copy()
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    K_scaled[0, 0] *= sx
    K_scaled[0, 2] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[1, 2] *= sy
    return K_scaled


def make_default_intrinsics(size: int, focal: float | None = None) -> np.ndarray:
    """Create a square-image pinhole intrinsics matrix."""
    f = float(focal if focal is not None else size)
    c = (int(size) - 1) * 0.5
    return np.asarray([[f, 0.0, c], [0.0, f, c], [0.0, 0.0, 1.0]], dtype=np.float32)
