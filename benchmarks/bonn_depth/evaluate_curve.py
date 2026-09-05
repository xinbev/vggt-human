#!/usr/bin/env python3
"""Build absolute-depth prefix curves from Bonn prediction folders."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from evaluate import SEQUENCES, evaluate_sequence, find_prediction_paths


T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
}


def confidence_interval_95(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) < 2:
        return mean, mean, 0.0
    sem = float(array.std(ddof=1) / math.sqrt(len(array)))
    margin = T_975.get(len(array) - 1, 1.96) * sem
    return mean, mean - margin, margin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--prediction-root",
        required=True,
        type=Path,
        help="Root containing prefix_100/<sequence>/*.npy, etc.",
    )
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
    parser.add_argument("--start-frame", type=int, default=30)
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="Treat each requested prefix as a maximum and evaluate all available predictions (Human3R Fig. 9 protocol).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval/bonn_depth_curve"))
    parser.add_argument("--max-depth", type=float, default=70.0)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    points: list[dict[str, object]] = []
    for length in args.prefix_lengths:
        prefix_root = args.prediction_root / f"prefix_{length}"
        per_sequence = []
        for sequence in SEQUENCES:
            actual_count = len(find_prediction_paths(prefix_root, sequence))
            if actual_count <= 0:
                raise ValueError(f"prefix={length}, sequence={sequence}: no prediction files found under {prefix_root}")
            if actual_count > length:
                raise ValueError(f"prefix={length}, sequence={sequence}: found {actual_count} predictions, expected at most {length}")
            if actual_count != length and not args.allow_short:
                raise ValueError(
                    f"prefix={length}, sequence={sequence}: found only {actual_count} predictions. "
                    "Use --allow-short only for the Human3R Figure 9 truncation protocol."
                )
            item = evaluate_sequence(
                dataset_root=args.dataset_root,
                pred_root=prefix_root,
                sequence=sequence,
                start=args.start_frame,
                count=actual_count,
                max_depth=args.max_depth,
                alignment="metric",
            )
            item["requested_frames"] = length
            item["actual_frames"] = actual_count
            per_sequence.append(item)
        abs_values = [float(item["Abs Rel"]) for item in per_sequence]
        delta_values = [float(item["delta<1.25"]) for item in per_sequence]
        abs_mean, abs_low, abs_margin = confidence_interval_95(abs_values)
        delta_mean, delta_low, delta_margin = confidence_interval_95(delta_values)
        valid_weights = np.asarray([float(item["valid_pixels"]) for item in per_sequence], dtype=np.float64)
        point = {
            "stage": args.stage_name,
            "requested_frames": length,
            "actual_frames_min": min(int(item["actual_frames"]) for item in per_sequence),
            "actual_frames_max": max(int(item["actual_frames"]) for item in per_sequence),
            "num_sequences": len(per_sequence),
            "Abs Rel sequence mean": abs_mean,
            "Abs Rel ci95 low": abs_low,
            "Abs Rel ci95 high": abs_mean + abs_margin,
            "delta<1.25 sequence mean": delta_mean,
            "delta<1.25 ci95 low": delta_low,
            "delta<1.25 ci95 high": delta_mean + delta_margin,
            "Abs Rel Human3R pixel-weighted": float(np.average(abs_values, weights=valid_weights)),
            "delta<1.25 Human3R pixel-weighted": float(np.average(delta_values, weights=valid_weights)),
            "per_sequence": per_sequence,
        }
        points.append(point)
        for item in per_sequence:
            rows.append(
                {
                    "stage": args.stage_name,
                    "requested_frames": length,
                    "actual_frames": item["actual_frames"],
                    "sequence": item["sequence"],
                    "Abs Rel": item["Abs Rel"],
                    "delta<1.25": item["delta<1.25"],
                    "valid_pixels": item["valid_pixels"],
                }
            )

    report = {
        "protocol": (
            f"Bonn video depth; requested windows [{args.start_frame}:{args.start_frame}+N); "
            f"absolute metric depth; no GT scale/shift alignment; allow_short={args.allow_short}"
        ),
        "aggregation": (
            "Both unweighted sequence mean with two-sided 95% Student-t CI and Human3R-style "
            "valid-pixel-weighted aggregate are reported."
        ),
        "stage": args.stage_name,
        "prediction_root": str(args.prediction_root),
        "sequences": list(SEQUENCES),
        "points": points,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.stage_name}_curve.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / f"{args.stage_name}_curve_points.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [key for key in points[0] if key != "per_sequence"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: value for key, value in point.items() if key != "per_sequence"} for point in points)
    with (args.output_dir / f"{args.stage_name}_per_sequence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "requested_frames", "actual_frames", "sequence", "Abs Rel", "delta<1.25", "valid_pixels"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
