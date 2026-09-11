#!/usr/bin/env python
"""Lightweight math checks for EMDB camera trajectory comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vis.plot_emdb_camera_trajectory_comparison import align_trajectory, path_length, rmse


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
    print("[ok] EMDB camera trajectory alignment checks passed")


if __name__ == "__main__":
    main()
