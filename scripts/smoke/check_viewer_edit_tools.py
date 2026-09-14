#!/usr/bin/env python
"""Static smoke checks for cached-viewer recolor and point eraser helpers."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vis.viewer_edit_tools import (  # noqa: E402
    points_outside_spheres,
    set_scene_node_click_button,
    smpl_recolor_targets,
)


def check_point_eraser() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    strokes = [
        {
            "center_source_xyz": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            "radius_source": 0.11,
        },
        {
            "center_source_xyz": np.asarray([0.2, 0.0, 0.0], dtype=np.float32),
            "radius_source": 0.01,
        },
    ]
    keep = points_outside_spheres(points, strokes)
    assert keep.tolist() == [False, False, False, True]
    assert points_outside_spheres(points, []).all()


def check_recolor_scopes() -> None:
    entries = [
        {"key": "f0-base-t1", "frame_index": 0, "kind": "base", "track_id": 1},
        {"key": "f0-hsi-t1", "frame_index": 0, "kind": "hsi", "track_id": 1},
        {"key": "f2-base-t1", "frame_index": 2, "kind": "base", "track_id": 1},
        {"key": "f2-base-t2", "frame_index": 2, "kind": "base", "track_id": 2},
        {"key": "f3-base-t1", "frame_index": 3, "kind": "base", "track_id": 1},
    ]
    clicked = entries[0]
    pinned = {0, 2}
    assert [entry["key"] for entry in smpl_recolor_targets(entries, clicked, pinned, "clicked mesh")] == [
        "f0-base-t1"
    ]
    assert {entry["key"] for entry in smpl_recolor_targets(entries, clicked, pinned, "clicked frame")} == {
        "f0-base-t1",
        "f0-hsi-t1",
    }
    assert {
        entry["key"] for entry in smpl_recolor_targets(entries, clicked, pinned, "same track in pinned frames")
    } == {"f0-base-t1", "f2-base-t1"}
    assert {entry["key"] for entry in smpl_recolor_targets(entries, clicked, pinned, "all pinned frames")} == {
        "f0-base-t1",
        "f0-hsi-t1",
        "f2-base-t1",
        "f2-base-t2",
    }


def check_right_click_binding() -> None:
    @dataclass(frozen=True)
    class DragBinding:
        button: str
        modifier: str | None

    @dataclass
    class SetSceneNodeClickBindingsMessage:
        name: str
        bindings: tuple[DragBinding, ...]

    fake_viser = ModuleType("viser")
    fake_viser._messages = SimpleNamespace(  # type: ignore[attr-defined]
        DragBinding=DragBinding,
        SetSceneNodeClickBindingsMessage=SetSceneNodeClickBindingsMessage,
    )
    previous = sys.modules.get("viser")
    sys.modules["viser"] = fake_viser
    try:
        queued = []
        impl = SimpleNamespace(name="/mesh", api=SimpleNamespace(_queue_scene_message=queued.append))
        handle = SimpleNamespace(_impl=impl)
        assert set_scene_node_click_button(handle, "right")
        assert queued[0].name == "/mesh"
        assert queued[0].bindings[0].button == "right"
        assert impl._last_published_click_bindings[0].button == "right"
    finally:
        if previous is None:
            sys.modules.pop("viser", None)
        else:
            sys.modules["viser"] = previous


def main() -> None:
    check_point_eraser()
    check_recolor_scopes()
    check_right_click_binding()
    print("[ok] viewer recolor and point eraser helper checks passed")


if __name__ == "__main__":
    main()
