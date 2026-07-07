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

from scripts.train.train_smpl import apply_overrides
from vggt_omega.data import HFBedlamDataset, hf_bedlam_collate_fn
from vggt_omega.data.geometry import resolve_image_size_config
from vggt_omega.training.config import deep_update, load_yaml_config, require_path


def main() -> None:
    args = parse_args()
    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.train_config))
    config = apply_overrides(config, args.override)
    data_cfg = config["data"]
    image_size, image_resolution = resolve_image_size_config(data_cfg, args.image_size)
    dataset = HFBedlamDataset(
        images_root=require_path(config, data_cfg.get("images_root_key", "datasets.hf_bedlam_images_root")),
        npz_root=require_path(config, data_cfg.get("npz_root_key", "datasets.hf_bedlam_npz_root")),
        sequence_length=int(args.sequence_length or data_cfg.get("sequence_length", 1)),
        stride=int(args.stride or data_cfg.get("stride", 1)),
        image_size=image_size,
        image_resolution=image_resolution,
        resize_mode=str(data_cfg.get("resize_mode", "balanced")),
        max_humans=int(args.max_humans or data_cfg.get("max_humans", 20)),
        require_boxes=bool(data_cfg.get("require_boxes", True)),
        require_smpl=bool(data_cfg.get("require_smpl", True)),
        bbox_expand=float(data_cfg.get("bbox_expand", 0.15)),
        transl_add_cam_ext=bool(data_cfg.get("transl_add_cam_ext", True)),
        skip_missing_images=bool(data_cfg.get("skip_missing_images", True)),
        max_npz_files=int(args.max_npz_files or data_cfg.get("max_npz_files", 0) or 0),
        max_frames=int(args.max_frames or data_cfg.get("max_frames", 0) or 0),
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=hf_bedlam_collate_fn)
    batch = next(iter(loader))
    required = [
        "images",
        "K_scal3r",
        "gt_intrinsics",
        "gt_pose_6d",
        "gt_betas",
        "gt_transl_cam",
        "gt_boxes",
        "boxes_mask",
        "smpl_mask",
        "gt_track_ids",
    ]
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"HF BEDLAM batch missing fields: {missing}")
    summary: dict[str, Any] = {
        "dataset": "hf_bedlam",
        "num_windows": len(dataset),
        "batch_size": int(args.batch_size),
        "tensor_shapes": {key: list(batch[key].shape) for key in required},
        "geometry": tensor_previews(batch, ("image_hw", "valid_hw", "orig_hw", "pad_xyxy")),
        "smpl_valid_count": int(batch["smpl_mask"].sum().item()),
        "box_valid_count": int(batch["boxes_mask"].sum().item()),
        "transl_mean": tensor_mean(batch["gt_transl_cam"], batch["smpl_mask"]),
        "transl_min_xyz": tensor_reduce_xyz(batch["gt_transl_cam"], batch["smpl_mask"], "min"),
        "transl_max_xyz": tensor_reduce_xyz(batch["gt_transl_cam"], batch["smpl_mask"], "max"),
        "box_mean": tensor_mean(batch["gt_boxes"], batch["boxes_mask"]),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check HF BEDLAM raw NPZ/image tensors")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_base_hf_bedlam_ray_refine.yaml")
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--max-humans", type=int, default=0)
    parser.add_argument("--max-npz-files", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def tensor_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.bool()
    if value.ndim == mask.ndim + 1:
        mask = mask.unsqueeze(-1).expand_as(value)
    if not bool(mask.any()):
        return 0.0
    return float(value[mask].float().mean().item())


def tensor_reduce_xyz(value: torch.Tensor, mask: torch.Tensor, mode: str) -> list[float]:
    mask = mask.bool()
    if value.ndim != mask.ndim + 1:
        raise ValueError(f"Expected value shape ending in xyz and mask one dim fewer, got value={tuple(value.shape)} mask={tuple(mask.shape)}")
    if not bool(mask.any()):
        return [0.0, 0.0, 0.0]
    selected = value[mask].float()
    if mode == "min":
        out = selected.min(dim=0).values
    elif mode == "max":
        out = selected.max(dim=0).values
    else:
        raise ValueError(f"Unsupported reduce mode: {mode}")
    return [float(x) for x in out.detach().cpu().tolist()]


def tensor_previews(batch: dict[str, torch.Tensor], keys: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().reshape(-1, value.shape[-1]).tolist()[:4]
    return out


if __name__ == "__main__":
    main()
