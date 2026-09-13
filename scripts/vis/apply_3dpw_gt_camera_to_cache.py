#!/usr/bin/env python
"""Place one predicted full-sequence cache in the native 3DPW GT camera world.

This is intentionally a post-processing display conversion.  VGGT, HSI and
NLF keep their original predicted camera/intrinsic contract during inference;
only the cached camera-space reconstruction is re-expressed with 3DPW
``cam_poses`` for visualization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vis.full_viewer_cache_io import (  # noqa: E402
    FULL_CACHE_FORMAT,
    INCOMPLETE_NAME,
    MANIFEST_NAME,
    load_full_sequence_viewer_cache,
)
from scripts.vis.merge_3dpw_gt_camera_full_caches import (  # noqa: E402
    load_3dpw_camera,
    place_frame_in_gt_camera_world,
    write_merged_cache,
)


def main() -> None:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    gt_path = resolve_path(args.gt_pkl)
    manifest_path = input_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing input cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FULL_CACHE_FORMAT:
        raise ValueError(f"Unsupported input cache format: {manifest.get('format')!r}")
    if (input_dir / INCOMPLETE_NAME).exists():
        raise RuntimeError(f"Input cache is incomplete: {input_dir}")

    gt = load_3dpw_camera(gt_path)
    # Reuse the canonical cache loader for validation and SMPL-face recovery.
    scene, _, _ = load_full_sequence_viewer_cache(input_dir)
    records = manifest.get("frames", [])
    if not isinstance(records, list) or not records:
        raise ValueError(f"Input cache has no frames: {manifest_path}")

    selected: dict[int, tuple[Path, dict]] = {}
    for record in records:
        source_index = resolve_gt_frame_index_from_record(record, gt, int(args.frame_index_offset))
        if source_index < 0 or source_index >= len(gt["cam_poses"]):
            raise IndexError(
                f"Frame {source_index} is outside GT camera range [0,{len(gt['cam_poses']) - 1}]"
            )
        if source_index in selected:
            continue
        selected[source_index] = (input_dir, record)

    first_manifest = manifest
    output_manifest = write_merged_cache(
        selected=selected,
        gt=gt,
        gt_path=gt_path,
        output_dir=output_dir,
        first_cache_dir=input_dir,
        first_manifest=first_manifest,
        expected_frames=0,
        missing_frames=[],
    )
    print(
        json.dumps(
            {
                "output_manifest": str(output_manifest),
                "frames": len(selected),
                "camera_source": "native_3dpw_cam_poses_T_w2c",
                "gt_camera_oracle": True,
                "input_scene_frames": len(scene["frames"]),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="One full_viewer_cache directory")
    parser.add_argument("--gt-pkl", required=True, help="Native 3DPW sequence pickle")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-index-offset", type=int, default=0)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def resolve_gt_frame_index_from_record(record: dict, gt: dict, offset: int) -> int:
    # Prefer the actual image filename/frame id, then fall back to the source
    # index recorded by the sampler.  This also handles imageFiles sequences
    # whose numeric names are sparse (e.g. 000000, 000002, ...).
    from scripts.vis.merge_3dpw_gt_camera_full_caches import resolve_gt_frame_index

    return resolve_gt_frame_index(
        frame_id=str(record.get("frame_id", "")),
        source_index=int(record.get("source_frame_index", -1)),
        gt=gt,
        frame_index_offset=offset,
    )


if __name__ == "__main__":
    main()
