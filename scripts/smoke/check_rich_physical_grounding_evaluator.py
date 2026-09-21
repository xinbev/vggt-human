from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vggt_omega.data.rich_physical_grounding import RichPhysicalGroundingDataset
from vggt_omega.evaluation.rich_physical_grounding import (
    GroundEstimatorConfig,
    compute_physical_metrics,
    select_local_support_y,
)


def main() -> None:
    metrics = compute_physical_metrics([-0.010, -0.005, 0.004, 0.005, 0.020], tolerance_m=0.005)
    assert metrics["collision_frames"] == 1
    assert metrics["floating_frames"] == 2
    assert abs(metrics["collision_ratio_pct"] - 20.0) < 1e-8
    assert abs(metrics["penetrate_cm"] - 1.0) < 1e-8
    assert abs(metrics["float_cm"] - 1.25) < 1e-8
    assert abs(metrics["penetration_max_cm"] - 1.0) < 1e-8

    body = torch.tensor([[-0.1, 0.0, 1.0], [0.1, 1.0, 1.2]], dtype=torch.float32)
    scene = torch.tensor([[0.0, 1.02, 1.1], [0.0, 1.03, 1.15], [2.0, 1.02, 1.1]], dtype=torch.float32)
    support = select_local_support_y(
        scene,
        body,
        GroundEstimatorConfig(footprint_margin_m=0.2, vertical_window_m=0.2, min_support_points=1),
    )
    assert support.shape == (2,)
    assert float(torch.quantile(support, 0.5)) > float(body[:, 1].max())

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        official = root / "official"
        support_root = root / "support"
        image_dir = official / "test" / "Gym_010_cooking1" / "cam_01"
        image_dir.mkdir(parents=True)
        support_root.mkdir(parents=True)
        for index in range(3):
            Image.fromarray(np.full((32, 48, 3), index * 50, dtype=np.uint8)).save(image_dir / f"{index:05d}.jpg")
        key = "test/Gym_010_cooking1/cam_01"
        torch.save({key: {"frame_id": torch.tensor([2, 0])}}, support_root / "rich_test_labels.pt")
        torch.save(
            {key: {"bbx_xys": torch.tensor([[24.0, 16.0, 20.0], [24.0, 16.0, 20.0]]), "img_wh": (48, 32)}},
            support_root / "rich_test_preproc.pt",
        )
        dataset = RichPhysicalGroundingDataset(
            official_root=official,
            support_root=support_root,
            image_resolution=32,
            patch_size=16,
            max_humans=2,
            sequence_filters=["Gym_010_cooking1/cam_01"],
        )
        record = dataset.records[0]
        batch = dataset.load_batch(record, [0, 1])
        assert batch.source_frame_ids == (2, 0)
        assert [path.name for path in batch.image_paths] == ["00002.jpg", "00000.jpg"]
        assert batch.images.shape == (2, 3, 32, 32)
        assert bool(batch.query_mask[:, 0].all())
        assert not bool(batch.query_mask[:, 1].any())

    print("RICH physical-grounding evaluator smoke checks passed.")


if __name__ == "__main__":
    main()
