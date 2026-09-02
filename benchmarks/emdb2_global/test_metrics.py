"""Synthetic protocol tests runnable without EMDB or model checkpoints."""

from __future__ import annotations

import unittest

import numpy as np

from benchmarks.emdb2_global.metrics import (
    evaluate_global_metrics,
    root_translation_error,
)


class GlobalMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        frames, joints = 220, 24
        local = rng.normal(0.0, 0.2, size=(frames, joints, 3))
        root = np.stack(
            [np.linspace(0.0, 8.0, frames), np.sin(np.linspace(0.0, 4.0, frames)), np.zeros(frames)],
            axis=-1,
        )
        self.target = local + root[:, None]

    def test_perfect_prediction_is_zero(self) -> None:
        result = evaluate_global_metrics(self.target, self.target.copy())
        self.assertLess(float(np.abs(result.w_mpjpe_mm).max()), 1e-7)
        self.assertLess(float(np.abs(result.wa_mpjpe_mm).max()), 1e-7)
        self.assertLess(float(np.abs(result.rte_percent).max()), 1e-7)

    def test_global_similarity_is_removed_by_w_and_wa(self) -> None:
        theta = 0.4
        rotation = np.asarray(
            [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]
        )
        pred = 1.7 * np.einsum("ij,fnj->fni", rotation, self.target) + np.asarray([2.0, -1.0, 0.5])
        result = evaluate_global_metrics(self.target, pred)
        self.assertLess(float(result.w_mpjpe_mm.mean()), 1e-6)
        self.assertLess(float(result.wa_mpjpe_mm.mean()), 1e-6)
        # RTE deliberately forbids scale correction.
        self.assertGreater(float(result.rte_percent.mean()), 0.0)

    def test_drift_is_visible_to_first_frame_alignment(self) -> None:
        pred = self.target.copy()
        pred[:, :, 0] += np.linspace(0.0, 1.0, pred.shape[0])[:, None]
        result = evaluate_global_metrics(self.target, pred, chunk_length=100)
        self.assertGreater(float(result.w_mpjpe_mm.mean()), float(result.wa_mpjpe_mm.mean()))
        self.assertGreater(float(result.rte_percent.mean()), 0.0)

    def test_rte_rejects_static_gt_trajectory(self) -> None:
        roots = np.zeros((10, 3), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "displacement is zero"):
            root_translation_error(roots, roots)


if __name__ == "__main__":
    unittest.main()

