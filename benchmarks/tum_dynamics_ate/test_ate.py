#!/usr/bin/env python3
"""Fast, data-free tests for the TUM-Dynamics ATE implementation."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from benchmarks.tum_dynamics_ate.evaluate_ate import (
    associate_by_timestamp,
    compute_ate,
    evaluate_prediction_root,
)


class AteTest(unittest.TestCase):
    def test_sim3_alignment_recovers_known_transform(self) -> None:
        rng = np.random.default_rng(7)
        reference = rng.normal(size=(32, 3))
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        q[:, 0] *= np.linalg.det(q)
        estimate = 2.5 * (reference @ q) + np.array([3.0, -1.0, 4.0])
        ate, scale = compute_ate(reference, estimate)
        self.assertLess(ate, 1e-10)
        self.assertAlmostEqual(scale, 0.4, places=10)

    def test_timestamp_association_is_one_to_one(self) -> None:
        gt, pred = associate_by_timestamp(
            np.asarray([0.0, 1.0, 2.0]),
            np.asarray([0.01, 1.01, 5.0]),
            max_difference=0.02,
        )
        np.testing.assert_array_equal(gt, np.asarray([0, 1]))
        np.testing.assert_array_equal(pred, np.asarray([0, 1]))

    def test_human3r_index_protocol_for_equal_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset" / "sequence_a"
            predictions = root / "predictions" / "sequence_a"
            dataset.mkdir(parents=True)
            predictions.mkdir(parents=True)
            reference = np.stack([np.arange(8), np.zeros(8), np.ones(8)], axis=1)
            estimate = reference * 3.0 + np.array([5.0, -2.0, 1.0])

            def write(path: Path, positions: np.ndarray, timestamps: np.ndarray) -> None:
                path.write_text(
                    "\n".join(
                        f"{stamp} {x} {y} {z} 1 0 0 0"
                        for stamp, (x, y, z) in zip(timestamps, positions)
                    )
                    + "\n",
                    encoding="utf-8",
                )

            # Prediction timestamps deliberately use Human3R's synthetic 0..N-1;
            # equal lengths must still pair by index.
            write(dataset / "groundtruth_8.txt", reference, np.arange(8) + 100.0)
            write(predictions / "pred_traj.txt", estimate, np.arange(8))
            summary, rows = evaluate_prediction_root(
                root / "dataset",
                root / "predictions",
                8,
                "wxyz",
                "auto",
                0.02,
            )
            self.assertEqual(rows[0]["association"], "index")
            self.assertLess(summary["ate_rmse_m_mean_over_sequences"], 1e-10)


if __name__ == "__main__":
    unittest.main()

