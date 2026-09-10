#!/usr/bin/env python
"""Round-trip the complete SequenceViewer scene cache without model dependencies."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vis.full_viewer_cache_io import export_full_sequence_viewer_cache, load_full_sequence_viewer_cache


def main() -> None:
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    vertices = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    frame = {
        "frame_index": 0,
        "frame_id": "frame_0000",
        "hsi_points": vertices.copy(),
        "hsi_colors": np.full((3, 3), 127, dtype=np.uint8),
        "raw_depth_map": np.ones((2, 2), dtype=np.float32),
        "hsi_human_exclusion_mask": np.asarray([[True, False], [False, False]]),
        "people": [{"hsi_vertices": vertices, "faces": faces, "track_id": 3}],
    }
    scene = {
        "frames": [frame],
        "image_hw": [2, 2],
        "camera_trajectory_hsi": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        "trstr_active": True,
    }
    args = SimpleNamespace(
        point_size=0.006,
        viewer_mode="Hybrid",
        filter_human_points=True,
        start_index=100,
        end_index=299,
        frame_stride=2,
        max_frames=-1,
        display_people=1,
        cascade_effective_affine_mode="clip_median",
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        manifest = export_full_sequence_viewer_cache(scene, args, Path(temp_dir) / "full_cache")
        restored, restored_args, restored_manifest = load_full_sequence_viewer_cache(manifest.parent)
        assert restored_manifest == manifest
        assert restored["trstr_active"] is True
        assert np.array_equal(restored["frames"][0]["raw_depth_map"], frame["raw_depth_map"])
        assert np.array_equal(restored["frames"][0]["people"][0]["faces"], faces)
        assert restored_args.viewer_mode == "Hybrid"
        assert restored_args.filter_human_points is True
        assert restored_args.start_index == 100
        assert restored_args.end_index == 299
        assert restored_args.frame_stride == 2
        assert restored_args.max_frames == -1
        assert restored_args.display_people == 1
        assert restored_args.cascade_effective_affine_mode == "clip_median"
    print("[ok] full sequence viewer cache round-trip passed")


if __name__ == "__main__":
    main()
