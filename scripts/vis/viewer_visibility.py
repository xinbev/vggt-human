"""Dependency-light visibility rules shared by Viser viewer tests."""

from __future__ import annotations

from collections.abc import Set


def human_frame_visible(
    mode: str,
    frame_index: int,
    current_index: int,
    pinned_frame_indices: Set[int] | set[int],
) -> bool:
    """Return frame-level SMPL visibility before person/source filtering."""
    if mode == "3D accumulate":
        return int(frame_index) <= int(current_index)
    if mode == "Hybrid":
        return int(frame_index) == int(current_index) or int(frame_index) in pinned_frame_indices
    return int(frame_index) == int(current_index)
