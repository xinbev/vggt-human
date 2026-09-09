#!/usr/bin/env python
"""Round-trip the dependency-light sequence viewer cache format."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vis.viewer_cache_io import export_sequence_viewer_cache, load_sequence_viewer_manifest


def main() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 2.0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    scene = {
        "image_hw": [2, 2],
        "frames": [
            {
                "frame_index": 0,
                "frame_id": "frame_0000",
                "image": "/dataset/frame_0000.jpg",
                "hsi_points": np.asarray([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]], dtype=np.float32),
                "hsi_colors": np.asarray([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
                "people": [
                    {
                        "hsi_vertices": vertices,
                        "faces": faces,
                        "track_id": 7,
                        "query_index": 1,
                        "confidence": 0.9,
                        "track_quality": 0.8,
                        "color": (41, 98, 255),
                    }
                ],
            }
        ],
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        manifest_path = export_sequence_viewer_cache(scene, Path(temp_dir) / "viewer_cache")
        root, manifest = load_sequence_viewer_manifest(manifest_path.parent)
        assert manifest["num_frames"] == 1
        assert manifest["total_points"] == 2
        assert manifest["total_people"] == 1
        with np.load(root / manifest["frames"][0]["file"], allow_pickle=False) as frame:
            assert frame["points"].shape == (2, 3)
            assert frame["smpl_vertices"].shape == (1, 4, 3)
            assert int(frame["smpl_track_ids"][0]) == 7
    print("[ok] sequence viewer cache round-trip passed")


if __name__ == "__main__":
    main()
