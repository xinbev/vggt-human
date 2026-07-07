import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_smpl import apply_overrides
from vggt_omega.data import BedlamDataset, bedlam_collate_fn
from vggt_omega.training.config import deep_update, load_yaml_config, require_path


def main() -> None:
    args = parse_args()
    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.train_config))
    config = apply_overrides(config, args.override)
    data_cfg = config["data"]
    split = args.split or data_cfg.get("train_split", "Training")
    boxes_root = require_path(config, data_cfg["boxes_root_key"], allow_empty=False)
    dataset = BedlamDataset(
        root=require_path(config, data_cfg.get("root_key", "datasets.bedlam_root")),
        split=split,
        sequence_length=int(args.sequence_length or data_cfg["sequence_length"]),
        stride=int(args.stride or data_cfg["stride"]),
        image_size=int(args.image_size or data_cfg.get("image_size", data_cfg.get("image_resolution", 512))),
        image_resolution=int(data_cfg.get("image_resolution", args.image_size or data_cfg.get("image_size", 512))),
        resize_mode=str(data_cfg.get("resize_mode", "balanced")),
        max_humans=int(args.max_humans or data_cfg["max_humans"]),
        require_smpl=True,
        require_depth=bool(data_cfg.get("require_depth", False)),
        boxes_root=boxes_root,
        require_boxes=True,
        query_source="detections",
        patch_size=int(config.get("model", {}).get("patch_size", 16)),
        mask_patch_threshold=float(data_cfg.get("mask_patch_threshold", 0.10)),
        min_mask_patches=int(data_cfg.get("min_mask_patches", 4)),
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=bedlam_collate_fn)
    batch = next(iter(loader))
    required = [
        "images",
        "K_scal3r",
        "gt_depth",
        "gt_pose_6d",
        "gt_betas",
        "gt_transl_cam",
        "gt_boxes",
        "boxes_mask",
        "smpl_query_boxes",
        "smpl_query_boxes_mask",
        "smpl_query_scores",
        "smpl_query_det_ids",
        "smpl_query_patch_masks",
        "smpl_query_patch_masks_valid",
        "external_track_ids",
        "external_track_mask",
        "external_track_confidence",
    ]
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"BEDLAM full-system batch missing fields: {missing}")

    summary = {
        "split": split,
        "num_windows": len(dataset),
        "batch_size": int(args.batch_size),
        "boxes_root": str(boxes_root),
        "tensor_shapes": {key: list(batch[key].shape) for key in required},
        "query_valid_count": int(batch["smpl_query_boxes_mask"].sum().item()),
        "gt_valid_count": int(batch["smpl_mask"].sum().item()),
        "sam2_patch_mask_valid_count": int(batch["smpl_query_patch_masks_valid"].sum().item()),
        "external_track_prior_valid_count": int(batch["external_track_mask"].sum().item()),
        "query_score_mean": tensor_mean(batch["smpl_query_scores"], batch["smpl_query_boxes_mask"]),
        "external_track_confidence_mean": tensor_mean(batch["external_track_confidence"], batch["external_track_mask"]),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check full-system BEDLAM training data tensors")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_hsi_full_system_restructure.yaml")
    parser.add_argument("--split", default="")
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--max-humans", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def tensor_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.bool()
    if not bool(mask.any()):
        return 0.0
    return float(value[mask].float().mean().item())


if __name__ == "__main__":
    main()
