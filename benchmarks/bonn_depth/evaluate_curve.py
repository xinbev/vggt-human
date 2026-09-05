#!/usr/bin/env python3
"""Build absolute-depth prefix curves from Bonn prediction folders."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from evaluate import SEQUENCES, evaluate_sequence


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
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[100, 200, 300, 400, 500])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval/bonn_depth_curve"))
    parser.add_argument("--max-depth", type=float, default=70.0)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    points: list[dict[str, object]] = []
    for length in args.prefix_lengths:
        prefix_root = args.prediction_root / f"prefix_{length}"
        per_sequence = [
            evaluate_sequence(
                dataset_root=args.dataset_root,
                pred_root=prefix_root,
                sequence=sequence,
                start=0,
                count=length,
                max_depth=args.max_depth,
                alignment="metric",
            )
            for sequence in SEQUENCES
        ]
        abs_values = [float(item["Abs Rel"]) for item in per_sequence]
        delta_values = [float(item["delta<1.25"]) for item in per_sequence]
        abs_mean, abs_low, abs_margin = confidence_interval_95(abs_values)
        delta_mean, delta_low, delta_margin = confidence_interval_95(delta_values)
        point = {
            "stage": args.stage_name,
            "frames": length,
            "num_sequences": len(per_sequence),
            "Abs Rel mean": abs_mean,
            "Abs Rel ci95 low": abs_low,
            "Abs Rel ci95 high": abs_mean + abs_margin,
            "delta<1.25 mean": delta_mean,
            "delta<1.25 ci95 low": delta_low,
            "delta<1.25 ci95 high": delta_mean + delta_margin,
            "per_sequence": per_sequence,
        }
        points.append(point)
        for item in per_sequence:
            rows.append(
                {
                    "stage": args.stage_name,
                    "frames": length,
                    "sequence": item["sequence"],
                    "Abs Rel": item["Abs Rel"],
                    "delta<1.25": item["delta<1.25"],
                    "valid_pixels": item["valid_pixels"],
                }
            )

    report = {
        "protocol": "Bonn prefix video depth; frames [0:N); absolute metric depth; no GT scale/shift alignment",
        "aggregation": "unweighted sequence mean; two-sided 95% Student-t confidence interval across five sequences",
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
        writer = csv.DictWriter(handle, fieldnames=["stage", "frames", "sequence", "Abs Rel", "delta<1.25", "valid_pixels"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
