#!/usr/bin/env python
"""Verify two standalone frames and their sequence manifest without a checkpoint."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vis.sequence_scene_export import (  # noqa: E402
    export_single_frame_scene,
    finalize_sequence_scene_export,
)


def main() -> None:
    points = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    colors = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    camera = {"rotation_c2w": np.eye(3), "position": [1, 2, 3], "fov": 0.8, "aspect": 1.5}
    frame = {"frame_index": 7, "source_frame_index": 42, "frame_id": "test", "intrinsic": np.eye(3)}
    mesh = {"name": "person_hsi_t3_q0", "kind": "hsi", "track_id": 3, "query_index": 0,
            "vertices": np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=np.float32),
            "faces": np.array([[0, 1, 2]]), "color": (255, 0, 0), "opacity": 0.65}
    debug_root = ROOT / "outputs" / "debug"
    debug_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=debug_root) as temporary:
        batch = Path(temporary) / "sequence_example"
        batch.mkdir()
        (batch / ".incomplete").write_text("in progress", encoding="utf-8")
        target = export_single_frame_scene(batch, frame, {"hsi": (points, colors)},
                                           [mesh], {"hsi": camera}, {"hsi_visual_scale": 1.2},
                                           environment_meshes={"hsi": (mesh["vertices"],
                                                                        np.full((3, 3), 127, dtype=np.uint8),
                                                                        mesh["faces"])},
                                           folder_name="frame_0000")
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        assert not (target / ".incomplete").exists()
        assert manifest["frame_index"] == 7 and manifest["source_frame_index"] == 42
        assert manifest["files"]["environment"]["hsi"]["point_count"] == 2
        ply = (target / "environment_hsi.ply").read_bytes()
        header, payload = ply.split(b"end_header\n", 1)
        assert b"element vertex 2" in header and len(payload) == 2 * 15
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                          ("r", "u1"), ("g", "u1"), ("b", "u1")])
        assert np.allclose(np.frombuffer(payload, dtype=dtype)["x"], [1, 4])
        mesh_ply = (target / "environment_hsi_mesh.ply").read_bytes()
        mesh_header, mesh_payload = mesh_ply.split(b"end_header\n", 1)
        assert b"element vertex 3" in mesh_header and b"element face 1" in mesh_header
        assert len(mesh_payload) == 3 * 15 + 13
        obj = (target / "person_hsi_t3_q0.obj").read_text(encoding="ascii")
        assert "f 1 2 3\n" in obj and obj.count("\nv ") == 3
        assert "d 0.650000" in (target / "person_hsi_t3_q0.mtl").read_text(encoding="ascii")
        cameras = json.loads((target / "cameras.json").read_text(encoding="utf-8"))
        c2w = np.asarray(cameras["cameras"]["hsi"]["camera_to_world_4x4"])
        w2c = np.asarray(cameras["cameras"]["hsi"]["world_to_camera_4x4"])
        assert np.allclose(c2w @ w2c, np.eye(4)) and np.allclose(c2w[:3, 3], [1, 2, 3])
        next_frame = {**frame, "frame_index": 8}
        second = export_single_frame_scene(batch, next_frame, {"hsi": (points, colors)},
                                           [mesh], {"hsi": camera}, {}, folder_name="frame_0001")
        records = [
            {"position": 0, "frame_index": 7, "directory": target.name, "manifest": f"{target.name}/manifest.json"},
            {"position": 1, "frame_index": 8, "directory": second.name, "manifest": f"{second.name}/manifest.json"},
        ]
        sequence_manifest = finalize_sequence_scene_export(batch, records, None)
        sequence = json.loads(sequence_manifest.read_text(encoding="utf-8"))
        assert sequence["num_frames"] == 2 and len(sequence["frames"]) == 2
        assert not (batch / ".incomplete").exists()
    print("[ok] sequence scene export formats and index")


if __name__ == "__main__":
    main()
