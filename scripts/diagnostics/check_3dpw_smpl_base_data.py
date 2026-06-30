from __future__ import annotations

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

from vggt_omega.data import ThreeDPWDataset, threedpw_collate_fn
from vggt_omega.training.config import deep_update, load_yaml_config, require_path


def main() -> None:
    args = parse_args()
    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.train_config))
    data_cfg = config["data"]
    split = args.split or data_cfg.get("train_split", "train")
    dataset = ThreeDPWDataset(
        root=require_path(config, data_cfg.get("root_key", "datasets.threedpw_root")),
        annotation_root=require_path(config, data_cfg.get("annotation_root_key", "datasets.threedpw_smpl_base_root")),
        split=split,
        sequence_length=int(args.sequence_length or data_cfg["sequence_length"]),
        stride=int(args.stride or data_cfg["stride"]),
        image_size=int(args.image_size or data_cfg["image_size"]),
        max_humans=int(args.max_humans or data_cfg["max_humans"]),
        require_boxes=True,
        require_smpl=True,
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=threedpw_collate_fn)
    batch = next(iter(loader))
    fields = [
        "images",
        "K_scal3r",
        "gt_pose_6d",
        "gt_betas",
        "gt_transl_cam",
        "gt_boxes",
        "boxes_mask",
        "smpl_mask",
        "gt_track_ids",
    ]
    summary: dict[str, Any] = {
        "split": split,
        "num_windows": len(dataset),
        "tensor_shapes": {key: list(batch[key].shape) for key in fields},
        "valid_people": int(batch["smpl_mask"].sum().item()),
        "valid_boxes": int(batch["boxes_mask"].sum().item()),
        "transl_z_mean": tensor_mean(batch["gt_transl_cam"][..., 2], batch["smpl_mask"]),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check compact 3DPW SMPL-base data")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_base_3dpw.yaml")
    parser.add_argument("--split", default="")
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--max-humans", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def tensor_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.bool()
    if not bool(mask.any()):
        return 0.0
    return float(value[mask].float().mean().item())


if __name__ == "__main__":
    main()
