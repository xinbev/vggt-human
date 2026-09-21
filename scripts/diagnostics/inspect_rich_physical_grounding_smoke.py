#!/usr/bin/env python
"""Inspect a trusted full-viewer cache produced by the RICH pipeline smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vis.full_viewer_cache_io import load_full_sequence_viewer_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a RICH physical-grounding smoke cache.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def finite_points(value: Any, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"{name} must have shape (N, 3), got {points.shape}")
    points = points[np.isfinite(points).all(axis=1)]
    if points.size == 0:
        raise RuntimeError(f"{name} has no finite points")
    return points


def axis_summary(points: np.ndarray) -> dict[str, Any]:
    percentiles = np.percentile(points, [1, 5, 50, 95, 99], axis=0)
    return {
        "count": int(points.shape[0]),
        "min_xyz": points.min(axis=0).tolist(),
        "max_xyz": points.max(axis=0).tolist(),
        "percentiles_xyz": {
            key: value.tolist()
            for key, value in zip(("p01", "p05", "p50", "p95", "p99"), percentiles)
        },
    }


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    array = np.asarray(value).reshape(-1)
    return int(array[0]) if array.size else None


def main() -> None:
    args = parse_args()
    scene, viewer_args, manifest_path = load_full_sequence_viewer_cache(args.cache_dir)
    frames = scene.get("frames", [])
    if not frames:
        raise RuntimeError("Smoke cache contains no frames")

    frame_records = []
    total_people = 0
    for position, frame in enumerate(frames):
        scene_points = finite_points(frame.get("hsi_points", []), f"frame[{position}].hsi_points")
        people_records = []
        for person_index, person in enumerate(frame.get("people", [])):
            vertices = finite_points(
                person.get("hsi_vertices", []),
                f"frame[{position}].people[{person_index}].hsi_vertices",
            )
            people_records.append(
                {
                    "person_index": person_index,
                    "track_id": optional_int(person.get("track_id")),
                    "vertices": axis_summary(vertices),
                    "lowest_y_for_up_minus_y": float(vertices[:, 1].max()),
                    "lowest_z_for_up_plus_z": float(vertices[:, 2].min()),
                }
            )
        total_people += len(people_records)
        frame_records.append(
            {
                "position": position,
                "frame_index": int(frame.get("frame_index", position)),
                "source_frame_index": int(frame.get("source_frame_index", -1)),
                "frame_id": str(frame.get("frame_id", position)),
                "scene_points": axis_summary(scene_points),
                "people": people_records,
            }
        )

    if total_people == 0:
        raise RuntimeError("Smoke cache contains no decoded people")

    report = {
        "cache_manifest": str(manifest_path),
        "num_frames": len(frames),
        "total_people": total_people,
        "viewer_up_direction": "-y",
        "coordinate_note": (
            "The current viewer treats -Y as up. Do not apply the HuMoS Z-up formula "
            "directly until the robust ground estimator and sign convention are validated."
        ),
        "cascade_effective_affine_mode": getattr(viewer_args, "cascade_effective_affine_mode", None),
        "frames": frame_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Smoke frames: {len(frames)}")
    print(f"Decoded people: {total_people}")
    print("Viewer up direction: -y")
    print(f"Geometry report: {args.output}")
    print("[ok] RICH physical-grounding smoke geometry passed")


if __name__ == "__main__":
    main()
