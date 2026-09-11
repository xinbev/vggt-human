#!/usr/bin/env python
"""Lightweight math checks for EMDB camera trajectory comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vis.plot_emdb_camera_trajectory_comparison import (
    align_trajectory,
    path_length,
    rmse,
    set_centered_limits,
)


class _AxisStub:
    def __init__(self) -> None:
        self.xlim = (0.0, 0.0)
        self.ylim = (0.0, 0.0)

    def set_xlim(self, left: float, right: float) -> None:
        self.xlim = (float(left), float(right))

    def set_ylim(self, bottom: float, top: float) -> None:
        self.ylim = (float(bottom), float(top))


def main() -> None:
    gt = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.5], [2.0, 0.0, 1.5], [3.0, 0.0, 3.0]])
    rotation = np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    pred = 2.5 * (gt @ rotation) + np.asarray([7.0, -3.0, 5.0])
    sim3, scale = align_trajectory(pred, gt, fixed_scale=False)
    assert rmse(sim3, gt) < 1e-8
    assert abs(scale - 0.4) < 1e-8
    se3, _ = align_trajectory(pred, gt, fixed_scale=True)
    assert rmse(se3, gt) > 0.1
    assert path_length(gt) > 0.0
    axis = _AxisStub()
    first = np.asarray([[10.0, -2.0], [14.0, 6.0]])
    second = np.asarray([[11.0, 0.0], [13.0, 4.0]])
    set_centered_limits(axis, (first, second), padding_ratio=0.10)
    assert np.allclose(axis.xlim, [7.2, 16.8])
    assert np.allclose(axis.ylim, [-2.8, 6.8])
    print("[ok] EMDB camera trajectory alignment checks passed")


if __name__ == "__main__":
    main()
