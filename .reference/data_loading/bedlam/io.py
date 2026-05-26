from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_depth_tensor(path: str | Path, size: int) -> torch.Tensor:
    """Load metric BEDLAM depth from ``.npy`` and resize to a square tensor."""
    depth = np.load(path).astype(np.float32)
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D BEDLAM depth map, got shape {depth.shape} from {path}")

    depth_img = Image.fromarray(depth, mode="F")
    depth_img = depth_img.resize((int(size), int(size)), Image.BILINEAR)
    depth_arr = np.asarray(depth_img, dtype=np.float32)
    return torch.from_numpy(depth_arr).unsqueeze(0)


def load_intrinsics(path: str | Path) -> np.ndarray:
    """Load a 3x3 intrinsics matrix from a BEDLAM camera ``.npz`` file."""
    data = np.load(path)
    return data["intrinsics"].astype(np.float32)


def load_persons(path: str | Path) -> list[dict]:
    """Load per-frame BEDLAM person dictionaries from a pickle file."""
    with open(path, "rb") as file:
        return pickle.load(file)
