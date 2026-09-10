#!/usr/bin/env python
"""Lightweight checks for long-sequence and accumulated-SMPL sampling."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vis.sequence_sampling import (
    sample_sequence,
    sample_with_max_frames,
    select_frame_candidates,
    select_inclusive_range,
    uniform_sample_indices,
)


def main() -> None:
    inference_indices = uniform_sample_indices(2200, 500)
    assert len(inference_indices) == 500
    assert inference_indices[0] == 0 and inference_indices[-1] == 2199
    assert set(right - left for left, right in zip(inference_indices, inference_indices[1:])) == {4, 5}

    smpl_indices = uniform_sample_indices(500, 50)
    assert len(smpl_indices) == 50
    assert smpl_indices[0] == 0 and smpl_indices[-1] == 499
    assert all(left < right for left, right in zip(smpl_indices, smpl_indices[1:]))

    frames = [f"frame_{index:04d}.jpg" for index in range(22)]
    assert select_inclusive_range(frames, 0, -1) == frames
    assert select_inclusive_range(frames, 5, 9) == frames[5:10]
    assert select_inclusive_range(frames, -3, 2) == frames[:3]
    assert select_inclusive_range(frames, 20, 99) == frames[20:]
    ranged_stride = select_inclusive_range(frames, 4, 14)[::3]
    assert ranged_stride == [frames[4], frames[7], frames[10], frames[13]]
    assert select_frame_candidates(frames, 4, 14, 3, 0, "head") == ranged_stride
    assert select_frame_candidates(frames, 4, 14, 3, 2, "head") == [frames[4], frames[7]]
    assert sample_with_max_frames(frames, 0, "head") == frames
    assert sample_with_max_frames(frames, 5, "head") == frames[:5]
    assert sample_with_max_frames(frames, -1, "head") == frames
    assert sample_sequence(frames, 5, "head") == frames[:5]
    assert sample_sequence(frames, 5, "uniform") == [
        "frame_0000.jpg",
        "frame_0005.jpg",
        "frame_0010.jpg",
        "frame_0015.jpg",
        "frame_0021.jpg",
    ]

    long_frames = list(range(2200))
    long_selected = sample_with_max_frames(long_frames, -1, "head")
    assert len(long_selected) == 500
    assert long_selected[0] == 0 and long_selected[-1] == 2199
    assert sample_with_max_frames(list(range(300)), -1, "head") == list(range(300))
    ranged_long = select_frame_candidates(
        long_frames,
        start_index=100,
        end_index=2100,
        frame_stride=2,
        max_frames=-1,
        strategy="head",
    )
    assert len(ranged_long) == 500
    assert ranged_long[0] == 100 and ranged_long[-1] == 2100
    try:
        sample_with_max_frames(frames, -2, "head")
    except ValueError:
        pass
    else:
        raise AssertionError("MAX_FRAMES values below -1 must be rejected")
    for invalid_end in (-2, 3):
        try:
            select_inclusive_range(frames, 4, invalid_end)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid end index must be rejected: {invalid_end}")

    print("[ok] sequence viewer sampling checks passed")


if __name__ == "__main__":
    main()
