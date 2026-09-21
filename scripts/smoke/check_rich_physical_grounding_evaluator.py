from __future__ import annotations

import tempfile
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from vggt_omega.data.rich_physical_grounding import (
    RICH_PROTOCOL_HMR4D_VIEWS,
    RICH_PROTOCOL_MOVING_TEST9,
    RichPhysicalGroundingDataset,
)
from vggt_omega.evaluation.rich_physical_grounding import (
    GroundEstimatorConfig,
    compute_physical_metrics,
    configure_reference_cascade_model,
    normalized_cxcywh_iou,
    select_local_support_y,
)
from vggt_omega.evaluation.rich_scale_oracle import (
    CACHE_FORMAT,
    load_scale_window_cache,
    reconstruct_scale_frame,
    save_scale_window_cache,
)


def main() -> None:
    metrics = compute_physical_metrics([-0.010, -0.005, 0.004, 0.005, 0.020], tolerance_m=0.005)
    assert metrics["collision_frames"] == 1
    assert metrics["floating_frames"] == 2
    assert abs(metrics["collision_ratio_pct"] - 20.0) < 1e-8
    assert abs(metrics["penetrate_cm"] - 1.0) < 1e-8
    assert abs(metrics["float_cm"] - 1.25) < 1e-8
    assert abs(metrics["penetration_max_cm"] - 1.0) < 1e-8

    ious = normalized_cxcywh_iou(
        torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.1, 0.1, 0.1, 0.1]]),
        torch.tensor([0.5, 0.5, 0.4, 0.4]),
    )
    assert int(torch.argmax(ious)) == 0
    assert abs(float(ious[0]) - 1.0) < 1e-6

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
        checkpoint_path = root / "legacy_stage2.pt"
        torch.save(
            {
                "model": {
                    "hsi_human_scene_align_head.mlp.0.weight": torch.zeros(256, 25),
                },
                "config": {"model": {"enable_hsi_human_scene_align": True}},
            },
            checkpoint_path,
        )
        restored, report = configure_reference_cascade_model(
            {"model": {"hsi_align_hidden_dim": 256}}, checkpoint_path, max_humans=8
        )
        assert restored["model"]["hsi_align_feature_version"] == "legacy_scale_bias_v0"
        assert restored["model"]["hsi_scene_affine_mode"] == "per_frame"
        assert restored["model"]["smpl_use_aggregator_queries"] is False
        assert restored["model"]["num_smpl_queries"] == 8
        assert report["checkpoint_align_input_dim"] == 25

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
            protocol=RICH_PROTOCOL_HMR4D_VIEWS,
        )
        record = dataset.records[0]
        batch = dataset.load_batch(record, [0, 1])
        assert batch.source_frame_ids == (2, 0)
        assert [path.name for path in batch.image_paths] == ["00002.jpg", "00000.jpg"]
        assert batch.images.shape == (2, 3, 32, 32)
        assert bool(batch.query_mask[:, 0].all())
        assert not bool(batch.query_mask[:, 1].any())

        moving_dir = official / "test" / "ParkingLot2_017_burpeejump2" / "cam_10"
        moving_dir.mkdir(parents=True)
        for index in range(3):
            Image.fromarray(np.full((24, 40, 3), index * 60, dtype=np.uint8)).save(
                moving_dir / f"{index:05d}.jpg"
            )
        moving_dataset = RichPhysicalGroundingDataset(
            official_root=official,
            support_root=support_root,
            image_resolution=32,
            patch_size=16,
            max_humans=2,
            sequence_filters=["ParkingLot2_017_burpeejump2/cam_10"],
            protocol=RICH_PROTOCOL_MOVING_TEST9,
        )
        moving_record = moving_dataset.records[0]
        moving_batch = moving_dataset.load_batch(moving_record, [0, 2])
        assert moving_record.vid == "test/ParkingLot2_017_burpeejump2/cam_10"
        assert moving_record.frame_ids == (0, 1, 2)
        assert moving_batch.source_frame_ids == (0, 2)
        assert not bool(moving_batch.query_mask.any())
        assert moving_batch.images.shape[:2] == (2, 3)
        assert moving_batch.images.shape[-2] % 16 == 0
        assert moving_batch.images.shape[-1] % 16 == 0

        scale_cache = {
            "format": np.asarray(CACHE_FORMAT),
            "depth": np.full((1, 2, 2), 2.0, dtype=np.float32),
            "rgb": np.full((1, 2, 2, 3), 127, dtype=np.uint8),
            "intrinsics": np.asarray([[[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]]], dtype=np.float32),
            "extrinsics": np.asarray(
                [[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]]],
                dtype=np.float32,
            ),
            "vertices_cam": np.asarray([[[0.0, 0.0, 2.0], [0.0, 1.0, 2.0]]], dtype=np.float32),
            "selection_boxes": np.zeros((1, 4), dtype=np.float32),
            "selected_valid": np.ones((1,), dtype=np.uint8),
            "query_indices": np.zeros((1,), dtype=np.int16),
            "confidences": np.ones((1,), dtype=np.float32),
            "model_scale": np.ones((1,), dtype=np.float32),
            "image_hw": np.asarray((2, 2), dtype=np.int32),
            "scene_point_stride": np.asarray(1, dtype=np.int32),
        }
        cache_path = root / "manual_scale_window.npz"
        save_scale_window_cache(cache_path, scale_cache)
        restored_cache = load_scale_window_cache(cache_path)
        base_points, _, base_vertices = reconstruct_scale_frame(
            restored_cache, 0, 1.0, GroundEstimatorConfig(max_scene_depth_m=0.0), exclude_person=False
        )
        scaled_points, _, scaled_vertices = reconstruct_scale_frame(
            restored_cache, 0, 2.0, GroundEstimatorConfig(max_scene_depth_m=0.0), exclude_person=False
        )
        assert torch.allclose(scaled_points, base_points * 2.0)
        assert torch.allclose(scaled_vertices - base_vertices, torch.tensor([[-1.0, -2.0, -3.0]]).expand_as(base_vertices))

    print("RICH physical-grounding evaluator smoke checks passed.")


if __name__ == "__main__":
    main()
