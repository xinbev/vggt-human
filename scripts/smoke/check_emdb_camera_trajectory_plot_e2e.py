#!/usr/bin/env python
"""End-to-end synthetic cache/GT test for the EMDB camera plot exporter."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vis.full_viewer_cache_io import export_full_sequence_viewer_cache


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        cache_dir = root / "full_viewer_cache"
        output_dir = root / "output"
        gt_path = root / "sequence_data.pkl"
        source_indices = [2, 4, 6, 8, 10]
        gt_centers = np.asarray(
            [[float(i), 0.1 * float(i), 0.25 * float(i * i)] for i in range(12)],
            dtype=np.float32,
        )
        gt_w2c = np.repeat(np.eye(4, dtype=np.float32)[None], len(gt_centers), axis=0)
        gt_w2c[:, :3, 3] = -gt_centers
        with gt_path.open("wb") as file:
            pickle.dump(
                {
                    "name": "synthetic_emdb",
                    "n_frames": len(gt_centers),
                    "good_frames_mask": np.ones(len(gt_centers), dtype=bool),
                    "camera": {"extrinsics": gt_w2c},
                },
                file,
            )

        predicted = 1.8 * gt_centers[source_indices] + np.asarray([4.0, -2.0, 3.0], dtype=np.float32)
        faces = np.asarray([[0, 1, 2]], dtype=np.int64)
        vertices = np.asarray([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]], dtype=np.float32)
        frames = []
        for position, source_index in enumerate(source_indices):
            camera = {
                "rotation_c2w": np.eye(3, dtype=np.float32),
                "position": predicted[position],
                "fov": 0.8,
                "aspect": 1.0,
            }
            frames.append(
                {
                    "frame_index": position,
                    "source_frame_index": source_index,
                    "frame_id": f"image_{source_index:05d}",
                    "image": f"/dataset/image_{source_index:05d}.jpg",
                    "raw_camera": camera,
                    "hsi_camera": camera,
                    "camera": camera,
                    "people": [{"hsi_vertices": vertices, "faces": faces, "track_id": 0}],
                }
            )
        scene = {
            "frames": frames,
            "camera_trajectory_raw": predicted,
            "camera_trajectory_hsi": predicted,
            "camera_trajectory": predicted,
        }
        export_full_sequence_viewer_cache(
            scene,
            SimpleNamespace(frames_dir="/dataset/images", output_dir=str(root / "inference")),
            cache_dir,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/vis/plot_emdb_camera_trajectory_comparison.py"),
                "--cache-dir",
                str(cache_dir),
                "--gt-pkl",
                str(gt_path),
                "--output-dir",
                str(output_dir),
                "--plot-axes",
                "xz",
            ],
            cwd=ROOT,
            check=True,
        )
        expected = (
            "camera_trajectory_sim3_paper.png",
            "camera_trajectory_sim3_paper.pdf",
            "camera_trajectory_alignment_diagnostic.png",
            "camera_trajectory_metrics.json",
            "camera_trajectory_aligned.csv",
        )
        for name in expected:
            path = output_dir / name
            assert path.is_file() and path.stat().st_size > 0, path
        metrics = json.loads((output_dir / "camera_trajectory_metrics.json").read_text(encoding="utf-8"))
        assert metrics["matched_frames"] == len(source_indices)
        assert metrics["first_frame_index"] == source_indices[0]
        assert metrics["last_frame_index"] == source_indices[-1]
        assert metrics["hsi_metric"]["sim3_ate_rmse_m"] < 1e-5
    print("[ok] EMDB camera trajectory end-to-end plot export passed")


if __name__ == "__main__":
    main()
