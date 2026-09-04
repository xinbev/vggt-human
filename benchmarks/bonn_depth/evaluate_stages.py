#!/usr/bin/env python3
"""Compare multiple Bonn depth prediction stages with one shared protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate import SEQUENCES, evaluate_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--stage", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval/bonn_depth_stages"))
    parser.add_argument("--start-frame", type=int, default=30)
    parser.add_argument("--num-frames", type=int, default=110)
    parser.add_argument("--max-depth", type=float, default=70.0)
    parser.add_argument("--alignment", choices=["scale", "metric"], default="scale")
    args = parser.parse_args()
    stages: dict[str, Path] = {}
    for item in args.stage:
        if "=" not in item:
            raise ValueError(f"Invalid --stage {item!r}; expected NAME=PATH")
        name, path = item.split("=", 1)
        if not name or name in stages:
            raise ValueError(f"Duplicate or empty stage name: {name!r}")
        stages[name] = Path(path).expanduser()
    report: dict[str, object] = {"protocol": f"UniSH/Pi3 Bonn video depth; frames [{args.start_frame}:{args.start_frame + args.num_frames}), alignment={args.alignment}", "stages": {}}
    rows: list[dict[str, object]] = []
    for name, pred_root in stages.items():
        per_sequence = [evaluate_sequence(args.dataset_root, pred_root, seq, args.start_frame, args.num_frames, args.max_depth, args.alignment) for seq in SEQUENCES]
        weights = np.asarray([row["valid_pixels"] for row in per_sequence], dtype=np.float64)
        summary = {"Abs Rel": float(np.average([row["Abs Rel"] for row in per_sequence], weights=weights)), "delta<1.25": float(np.average([row["delta<1.25"] for row in per_sequence], weights=weights)), "valid_pixels": int(weights.sum()), "per_sequence": per_sequence, "prediction_root": str(pred_root)}
        report["stages"][name] = summary  # type: ignore[index]
        rows.append({"stage": name, "Abs Rel": summary["Abs Rel"], "delta<1.25": summary["delta<1.25"], "valid_pixels": summary["valid_pixels"]})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "bonn_stage_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "bonn_stage_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "Abs Rel", "delta<1.25", "valid_pixels"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
