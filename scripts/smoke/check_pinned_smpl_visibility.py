#!/usr/bin/env python
"""Check Hybrid pinned-SMPL frame visibility without Viser or model dependencies."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vis.viewer_visibility import human_frame_visible


def main() -> None:
    pinned = {2, 8}
    assert human_frame_visible("Hybrid", 2, 5, pinned)
    assert human_frame_visible("Hybrid", 5, 5, pinned)
    assert human_frame_visible("Hybrid", 8, 5, pinned)
    assert not human_frame_visible("Hybrid", 4, 5, pinned)
    assert human_frame_visible("4D current frame", 5, 5, pinned)
    assert not human_frame_visible("4D current frame", 2, 5, pinned)
    assert human_frame_visible("3D accumulate", 2, 5, pinned)
    assert not human_frame_visible("3D accumulate", 8, 5, pinned)
    print("[ok] pinned SMPL Hybrid visibility checks passed")


if __name__ == "__main__":
    main()
