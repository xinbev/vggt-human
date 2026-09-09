#!/usr/bin/env python
"""Lightweight checks for long-sequence and accumulated-SMPL sampling."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vis.sequence_sampling import sample_sequence, sample_with_max_frames, uniform_sample_indices


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
    try:
        sample_with_max_frames(frames, -2, "head")
    except ValueError:
        pass
    else:
        raise AssertionError("MAX_FRAMES values below -1 must be rejected")

    print("[ok] sequence viewer sampling checks passed")


if __name__ == "__main__":
    main()
