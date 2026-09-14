"""Pure NumPy helpers for interactive cached-viewer editing."""

from __future__ import annotations

from typing import Any

import numpy as np


def points_outside_spheres(points: np.ndarray, strokes: list[dict[str, Any]]) -> np.ndarray:
    points_np = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    keep = np.ones(points_np.shape[0], dtype=bool)
    for stroke in strokes:
        center = np.asarray(stroke["center_source_xyz"], dtype=np.float32).reshape(3)
        radius = max(0.0, float(stroke["radius_source"]))
        delta = points_np - center[None, :]
        keep &= np.einsum("ij,ij->i", delta, delta) > np.float32(radius * radius)
    return keep


def smpl_recolor_targets(
    entries: list[dict[str, Any]],
    clicked: dict[str, Any],
    pinned_frame_indices: set[int],
    scope: str,
) -> list[dict[str, Any]]:
    if scope == "clicked mesh":
        return [clicked]
    if scope == "clicked frame":
        return [entry for entry in entries if int(entry["frame_index"]) == int(clicked["frame_index"])]
    if scope == "same track in pinned frames":
        return [
            entry
            for entry in entries
            if int(entry["frame_index"]) in pinned_frame_indices
            and int(entry["track_id"]) == int(clicked["track_id"])
            and str(entry["kind"]) == str(clicked["kind"])
        ]
    if scope == "all pinned frames":
        return [entry for entry in entries if int(entry["frame_index"]) in pinned_frame_indices]
    raise ValueError(f"Unknown SMPL recolor scope: {scope}")


def smpl_id_reassignment_targets(
    entries: list[dict[str, Any]],
    selected: dict[str, Any],
    scope: str,
) -> list[dict[str, Any]]:
    selected_frame = int(selected["frame_index"])
    selected_query = int(selected["query_index"])
    selected_id = int(selected["track_id"])
    if scope == "selected frame person":
        return [
            entry
            for entry in entries
            if int(entry["frame_index"]) == selected_frame and int(entry["query_index"]) == selected_query
        ]
    if scope == "same ID from selected frame onward":
        return [
            entry
            for entry in entries
            if int(entry["frame_index"]) >= selected_frame and int(entry["track_id"]) == selected_id
        ]
    if scope == "same ID all frames":
        return [entry for entry in entries if int(entry["track_id"]) == selected_id]
    raise ValueError(f"Unknown SMPL ID reassignment scope: {scope}")


def set_scene_node_click_button(handle: Any, button: str) -> bool:
    """Switch Viser 1.1+ node click input while keeping its registered callback."""
    if button not in {"left", "right"}:
        raise ValueError(f"Unsupported scene-node click button: {button}")
    try:
        from viser import _messages as viser_messages  # noqa: PLC0415

        impl = handle._impl
        binding = viser_messages.DragBinding(button=button, modifier=None)
        bindings = (binding,)
        message = viser_messages.SetSceneNodeClickBindingsMessage(impl.name, bindings)
        impl.api._queue_scene_message(message)
        impl._last_published_click_bindings = bindings
        return True
    except (AttributeError, ImportError, TypeError):
        return False
