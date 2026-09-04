#!/usr/bin/env python3
"""Evaluate Bonn video-depth predictions using the UniSH/Pi3 protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SEQUENCES = ("balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous")


def read_bonn_depth(path: Path) -> np.ndarray:
    raw = np.asarray(Image.open(path))
    if raw.ndim != 2 or raw.dtype not in (np.uint16, np.int32, np.uint32):
        raise ValueError(f"Expected a 16-bit depth PNG, got {path}: {raw.dtype}, {raw.shape}")
    if int(raw.max()) <= 255:
        raise ValueError(f"Depth PNG does not look like 16-bit Bonn depth: {path}")
    depth = raw.astype(np.float32) / 5000.0
    depth[raw == 0] = -1.0
    return depth


def read_prediction(path: Path, shape: tuple[int, int]) -> np.ndarray:
    pred = np.asarray(np.load(path), dtype=np.float32)
    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred[..., 0]
    if pred.ndim != 2:
        raise ValueError(f"Prediction must be HxW or HxWx1: {path}, got {pred.shape}")
    if pred.shape != shape:
        pred = cv2.resize(pred, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return pred


def fit_scale(pred: np.ndarray, gt: np.ndarray) -> float:
    """Pi3/CUT3R scale-only alignment (10 IRLS/Weiszfeld updates)."""
    scale = float(np.mean(gt) / max(np.mean(pred), 1e-8))
    for _ in range(10):
        residual = scale * pred - gt
        weights = 1.0 / (np.abs(residual) + 1e-8)
        numerator = np.sum(weights * pred * gt)
        denominator = np.sum(weights * pred * pred)
        if denominator <= 1e-12 or not np.isfinite(denominator):
            break
        scale = float(numerator / denominator)
    return max(scale, 1e-3)


def evaluate_sequence(dataset_root: Path, pred_root: Path, sequence: str, start: int, count: int, max_depth: float, alignment: str = "scale") -> dict:
    seq_root = dataset_root / f"rgbd_bonn_{sequence}"
    gt_paths = sorted((seq_root / "depth").glob("*.png"))[start : start + count]
    pred_paths = sorted((pred_root / sequence).glob("*.npy"))
    if len(gt_paths) != count:
        raise ValueError(f"{sequence}: expected {count} GT frames after slicing, found {len(gt_paths)}")
    if len(pred_paths) == count:
        pass
    elif len(pred_paths) >= start + count:
        pred_paths = pred_paths[start : start + count]
    else:
        raise ValueError(f"{sequence}: expected {count} predictions (or a full sequence) in {pred_root / sequence}, found {len(pred_paths)}")
    gt = np.stack([read_bonn_depth(p) for p in gt_paths])
    pred = np.stack([read_prediction(p, gt.shape[1:]) for p in pred_paths])
    valid = np.isfinite(gt) & (gt > 0) & (gt < max_depth) & np.isfinite(pred) & (pred > 0)
    if not np.any(valid):
        raise ValueError(f"{sequence}: no valid depth pixels")
    scale = fit_scale(pred[valid], gt[valid]) if alignment == "scale" else 1.0
    if alignment not in {"scale", "metric"}:
        raise ValueError(f"Unknown alignment mode: {alignment}")
    aligned = scale * pred[valid]
    target = gt[valid]
    abs_rel = float(np.mean(np.abs(aligned - target) / target))
    ratio = np.maximum(aligned / target, target / np.maximum(aligned, 1e-8))
    return {"sequence": sequence, "alignment": alignment, "frames": count, "valid_pixels": int(valid.sum()), "scale": scale, "Abs Rel": abs_rel, "delta<1.25": float(np.mean(ratio < 1.25))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--pred-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval/bonn_depth"))
    parser.add_argument("--start-frame", type=int, default=30)
    parser.add_argument("--num-frames", type=int, default=110)
    parser.add_argument("--max-depth", type=float, default=70.0)
    parser.add_argument("--alignment", choices=["scale", "metric"], default="scale")
    args = parser.parse_args()
    rows = [evaluate_sequence(args.dataset_root, args.pred_root, seq, args.start_frame, args.num_frames, args.max_depth, args.alignment) for seq in SEQUENCES]
    weights = np.asarray([row["valid_pixels"] for row in rows], dtype=np.float64)
    summary = {"protocol": f"UniSH/Pi3 Bonn video depth; frames [{args.start_frame}:{args.start_frame + args.num_frames}), alignment={args.alignment}", "sequences": list(SEQUENCES), "Abs Rel": float(np.average([row["Abs Rel"] for row in rows], weights=weights)), "delta<1.25": float(np.average([row["delta<1.25"] for row in rows], weights=weights)), "valid_pixels": int(weights.sum()), "per_sequence": rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "bonn_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "bonn_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "alignment", "frames", "valid_pixels", "scale", "Abs Rel", "delta<1.25"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"sequence": "AVERAGE", "alignment": args.alignment, "frames": args.num_frames * len(rows), "valid_pixels": summary["valid_pixels"], "scale": "", "Abs Rel": summary["Abs Rel"], "delta<1.25": summary["delta<1.25"]})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
