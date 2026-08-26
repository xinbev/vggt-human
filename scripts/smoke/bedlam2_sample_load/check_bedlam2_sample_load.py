#!/usr/bin/env python
"""Check one processed BEDLAM2 sequence through the training data path.

This test is intentionally independent from training configuration and does not
write into the dataset. It validates the on-disk modalities, one dataset item,
and one collated batch using the project's BedlamDataset implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.data import BedlamDataset, bedlam_collate_fn  # noqa: E402


DEFAULT_SEQUENCE = Path(
    "/home/zhw/xyb_space/bedlam2/bedlam2_processed/Training/"
    "20241213_1_250_rome_tracking_seq_000002"
)
REQUIRED_BATCH_KEYS = {
    "images",
    "gt_depth",
    "K_scal3r",
    "gt_pose_6d",
    "gt_betas",
    "gt_transl_cam",
    "smpl_mask",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--sequence-length", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--max-humans", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=0, help="Limit pre-check frames; 0 checks all frames")
    parser.add_argument("--output", type=Path, default=Path("outputs/debug/bedlam2_sample_load/report.json"))
    return parser.parse_args()


def check_files(sequence_dir: Path, max_frames: int) -> dict[str, object]:
    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"Sequence directory not found: {sequence_dir}")
    frame_ids = sorted(path.stem for path in (sequence_dir / "rgb").glob("*.png"))
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]
    if not frame_ids:
        raise RuntimeError(f"No PNG frames found under {sequence_dir / 'rgb'}")

    missing: dict[str, list[str]] = {key: [] for key in ("depth", "cam", "smpl")}
    image_shapes: set[tuple[int, int]] = set()
    depth_shapes: set[tuple[int, ...]] = set()
    depth_finite = 0
    depth_positive = 0
    person_counts: list[int] = []
    for frame_id in frame_ids:
        image_path = sequence_dir / "rgb" / f"{frame_id}.png"
        image_shapes.add(Image.open(image_path).size[::-1])
        depth_path = sequence_dir / "depth" / f"{frame_id}.npy"
        cam_path = sequence_dir / "cam" / f"{frame_id}.npz"
        smpl_path = sequence_dir / "smpl" / f"{frame_id}.pkl"
        for name, path in (("depth", depth_path), ("cam", cam_path), ("smpl", smpl_path)):
            if not path.is_file():
                missing[name].append(frame_id)
        if depth_path.is_file():
            depth = np.asarray(np.load(depth_path), dtype=np.float32).squeeze()
            depth_shapes.add(tuple(depth.shape))
            depth_finite += int(np.isfinite(depth).sum())
            depth_positive += int((np.isfinite(depth) & (depth > 0)).sum())
        if cam_path.is_file():
            with np.load(cam_path) as camera:
                if "intrinsics" not in camera:
                    raise ValueError(f"Camera file missing intrinsics: {cam_path}")
                intrinsics = np.asarray(camera["intrinsics"])
                if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
                    raise ValueError(f"Invalid intrinsics shape/values in {cam_path}: {intrinsics.shape}")
        if smpl_path.is_file():
            import pickle

            with smpl_path.open("rb") as file:
                persons = pickle.load(file)
            if not isinstance(persons, list):
                raise TypeError(f"SMPL frame must contain a list: {smpl_path}")
            person_counts.append(len(persons))

    missing = {key: values for key, values in missing.items() if values}
    if missing:
        raise FileNotFoundError(f"Missing modality files: {missing}")
    if len(image_shapes) != 1 or len(depth_shapes) != 1:
        raise ValueError(f"Inconsistent frame shapes: images={image_shapes}, depth={depth_shapes}")

    return {
        "frame_count_checked": len(frame_ids),
        "first_frame": frame_ids[0],
        "last_frame": frame_ids[-1],
        "image_shapes_hw": [list(shape) for shape in sorted(image_shapes)],
        "depth_shapes": [list(shape) for shape in sorted(depth_shapes)],
        "depth_finite_values": depth_finite,
        "depth_positive_values": depth_positive,
        "person_count_min": min(person_counts),
        "person_count_max": max(person_counts),
    }


def tensor_summary(batch: dict[str, torch.Tensor]) -> dict[str, dict[str, list[int] | str]]:
    return {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    sequence_dir = args.sequence_dir.expanduser().resolve()
    if args.sequence_length <= 0 or args.stride <= 0:
        raise ValueError("sequence-length and stride must be positive")
    root = sequence_dir.parents[1]
    split = sequence_dir.parents[0].name
    file_report = check_files(sequence_dir, args.max_frames)

    dataset = BedlamDataset(
        root=root,
        split=split,
        sequence_length=args.sequence_length,
        stride=args.stride,
        image_resolution=args.image_resolution,
        max_humans=args.max_humans,
        require_smpl=True,
        require_depth=True,
    )
    sequence_names = {path.name for path, _ in dataset._sequences}  # noqa: SLF001
    if sequence_dir.name not in sequence_names:
        raise AssertionError(f"Target sequence was not indexed: {sequence_dir.name}")
    sample = dataset[0]
    missing_sample_keys = sorted(REQUIRED_BATCH_KEYS - sample.keys())
    if missing_sample_keys:
        raise AssertionError(f"Dataset sample missing training keys: {missing_sample_keys}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=bedlam_collate_fn)
    batch = next(iter(loader))
    missing_batch_keys = sorted(REQUIRED_BATCH_KEYS - batch.keys())
    if missing_batch_keys:
        raise AssertionError(f"Collated batch missing training keys: {missing_batch_keys}")
    if batch["images"].shape[1] != args.sequence_length:
        raise AssertionError(f"Unexpected sequence dimension: {batch['images'].shape}")
    if not torch.isfinite(batch["images"]).all() or not torch.isfinite(batch["gt_depth"]).all():
        raise AssertionError("Images or depth contain non-finite values after loading")

    report = {
        "status": "passed",
        "sequence_dir": str(sequence_dir),
        "dataset_root": str(root),
        "split": split,
        "dataset_windows": len(dataset),
        "file_check": file_report,
        "sample": tensor_summary(sample),
        "batch": tensor_summary(batch),
        "batch_smpl_valid_count": int(batch["smpl_mask"].sum().item()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
