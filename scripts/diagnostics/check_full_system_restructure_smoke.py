from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch

from vggt_omega.tracking.query_builder import (
    build_detection_query_tensors_from_sidecar,
    build_external_track_prior_from_sidecar,
)
from vggt_omega.tracking.smpl_track_assigner import BaseSMPLTrackAssigner


def main() -> None:
    check_query_builder()
    check_track_assigner_gap_reconnect()
    print("[ok] full-system restructure smoke checks passed")


def check_query_builder() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sidecar = root / "smpl_boxes"
        masks_dir = root / "masks"
        sidecar.mkdir(parents=True)
        masks_dir.mkdir(parents=True)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 20:44] = 1
        mask_path = masks_dir / "000000.npz"
        np.savez_compressed(mask_path, det_000000=mask)
        frame = {
            "frame_id": "000000",
            "frame_index": 0,
            "image_hw": [64, 64],
            "detections": [
                {
                    "det_id": 0,
                    "bbox_xyxy_pixels": [16, 12, 48, 52],
                    "bbox_cxcywh_norm": [0.5, 0.5, 0.5, 0.625],
                    "det_score": 0.9,
                    "mask": {
                        "format": "npz_bool",
                        "path": str(mask_path),
                        "array_key": "det_000000",
                    },
                }
            ],
            "persons": [
                {
                    "frame_id": "000000",
                    "frame_index": 0,
                    "person_id": 7,
                    "person_id_valid": True,
                    "bbox_valid": True,
                    "bbox_xyxy_pixels": [16, 12, 48, 52],
                    "bbox_cxcywh_norm": [0.5, 0.5, 0.5, 0.625],
                    "track_confidence": 0.8,
                    "valid": True,
                }
            ],
        }
        with (sidecar / "000000.pkl").open("wb") as file:
            pickle.dump(frame, file)

        query = build_detection_query_tensors_from_sidecar(root, max_humans=2, image_size=64, patch_size=16)
        assert tuple(query["smpl_query_boxes"].shape) == (1, 1, 2, 4)
        assert tuple(query["smpl_query_patch_masks"].shape) == (1, 1, 2, 16)
        assert bool(query["smpl_query_boxes_mask"][0, 0, 0])
        assert int(query["smpl_query_det_ids"][0, 0, 0]) == 0
        assert int(query["smpl_query_patch_masks"][0, 0, 0].sum()) >= 4

        prior = build_external_track_prior_from_sidecar(root, query)
        assert int(prior["external_track_ids"][0, 0, 0]) == 7
        assert bool(prior["external_track_mask"][0, 0, 0])


def check_track_assigner_gap_reconnect() -> None:
    assigner = BaseSMPLTrackAssigner(max_age=4, min_track_quality=0.25)
    boxes = torch.zeros(1, 5, 1, 4)
    boxes[:, :, 0] = torch.tensor(
        [
            [0.5, 0.5, 0.2, 0.4],
            [0.52, 0.5, 0.2, 0.4],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.56, 0.5, 0.2, 0.4],
        ]
    )
    mask = torch.tensor([[[True], [True], [False], [False], [True]]])
    betas = torch.zeros(1, 5, 1, 10)
    transl = torch.zeros(1, 5, 1, 3)
    transl[0, :, 0, 0] = torch.tensor([0.0, 0.1, 0.0, 0.0, 0.3])
    conf = torch.ones(1, 5, 1, 1) * 0.9
    out = assigner.assign(boxes, betas, transl, conf, query_mask=mask)
    ids = out["assigned_track_ids"][0, :, 0]
    assert int(ids[0]) == int(ids[1])
    assert int(ids[4]) == int(ids[0])
    assert int(out["assigned_track_gap"][0, 4, 0]) == 3


if __name__ == "__main__":
    main()
