from __future__ import annotations

import os
from pathlib import Path


def build_sequence_index(root: str | os.PathLike[str], split: str) -> list[tuple[str, list[str]]]:
    """Scan a preprocessed BEDLAM split and return ``(seq_dir, frame_ids)`` pairs."""
    split_dir = os.path.join(str(root), split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"BEDLAM split directory not found: {split_dir}")

    sequences: list[tuple[str, list[str]]] = []
    for seq_name in sorted(os.listdir(split_dir)):
        seq_dir = os.path.join(split_dir, seq_name)
        rgb_dir = os.path.join(seq_dir, "rgb")
        if not os.path.isdir(rgb_dir):
            continue
        frames = sorted(Path(path).stem for path in os.listdir(rgb_dir) if path.endswith(".png"))
        if len(frames) >= 2:
            sequences.append((seq_dir, frames))

    if not sequences:
        raise RuntimeError(f"No valid sequences found under {split_dir}")
    return sequences
