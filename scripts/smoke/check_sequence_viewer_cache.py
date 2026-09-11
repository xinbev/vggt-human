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
    intrinsic = np.asarray([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    raw_extrinsic = np.concatenate(
        [np.eye(3, dtype=np.float32), np.asarray([[0.0], [0.0], [1.0]], dtype=np.float32)],
        axis=1,
    )
    hsi_extrinsic = raw_extrinsic.copy()
    hsi_extrinsic[:, 3] *= 2.0
    raw_camera = {
        "rotation_c2w": np.eye(3, dtype=np.float32),
        "position": np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        "fov": 0.8,
        "aspect": 1.0,
    }
    hsi_camera = {
        "rotation_c2w": np.eye(3, dtype=np.float32),
        "position": np.asarray([0.0, 0.0, -2.0], dtype=np.float32),
        "fov": 0.8,
        "aspect": 1.0,
    }
    scene = {
        "image_hw": [2, 2],
        "camera_trajectory_raw": np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32),
        "camera_trajectory_hsi": np.asarray([[0.0, 0.0, -2.0]], dtype=np.float32),
        "frames": [
            {
                "frame_index": 0,
                "frame_id": "frame_0000",
                "image": "/dataset/frame_0000.jpg",
                "hsi_points": np.asarray([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]], dtype=np.float32),
                "hsi_colors": np.asarray([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
                "intrinsic": intrinsic,
                "raw_extrinsic": raw_extrinsic,
                "hsi_extrinsic": hsi_extrinsic,
                "raw_camera": raw_camera,
                "hsi_camera": hsi_camera,
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
        assert manifest["camera_parameters"]["raw_trajectory_file"] == "camera_trajectory_raw.npy"
        assert np.load(root / "camera_trajectory_raw.npy", allow_pickle=False).shape == (1, 3)
        assert np.load(root / "camera_trajectory_hsi.npy", allow_pickle=False).shape == (1, 3)
        with np.load(root / manifest["frames"][0]["file"], allow_pickle=False) as frame:
            assert frame["points"].shape == (2, 3)
            assert frame["smpl_vertices"].shape == (1, 4, 3)
            assert int(frame["smpl_track_ids"][0]) == 7
            assert frame["intrinsic"].shape == (3, 3)
            assert frame["raw_extrinsic"].shape == (3, 4)
            assert frame["hsi_extrinsic"].shape == (3, 4)
            assert np.allclose(frame["raw_camera_position"], [0.0, 0.0, -1.0])
            assert np.allclose(frame["hsi_camera_position"], [0.0, 0.0, -2.0])
    print("[ok] sequence viewer cache round-trip passed")


if __name__ == "__main__":
    main()
