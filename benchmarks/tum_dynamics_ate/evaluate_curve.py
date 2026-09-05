#!/usr/bin/env python3
"""Evaluate all Human3R TUM-Dynamics prefix lengths and write curve data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.tum_dynamics_ate.evaluate_ate import evaluate_prediction_root, write_outputs  # noqa: E402


DEFAULT_LENGTHS = (50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000)


def parse_lengths(raw: str) -> tuple[int, ...]:
    lengths = tuple(dict.fromkeys(int(token.strip()) for token in raw.split(",") if token.strip()))
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("--lengths must contain positive comma-separated integers")
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--pred-parent", required=True, type=Path, help="Parent containing directories named by --pred-dir-pattern")
    parser.add_argument("--pred-dir-pattern", default="tum_{length}_human3r", help="Python format pattern with {length}; may include {model}")
    parser.add_argument("--model", default="human3r")
    parser.add_argument("--lengths", default=",".join(map(str, DEFAULT_LENGTHS)))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prediction-quaternion-order", choices=("wxyz", "xyzw"), default="wxyz")
    parser.add_argument("--association", choices=("auto", "index", "timestamp"), default="auto")
    parser.add_argument("--max-time-difference", type=float, default=0.02)
    parser.add_argument("--sequence", default=None)
    args = parser.parse_args()
    lengths = parse_lengths(args.lengths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve_rows: list[dict[str, Any]] = []
    for length in lengths:
        try:
            dirname = args.pred_dir_pattern.format(length=length, model=args.model)
        except KeyError as exc:
            raise ValueError(f"Unknown placeholder in --pred-dir-pattern: {exc}; use {{length}} and/or {{model}}") from exc
        pred_root = args.pred_parent / dirname
        summary, rows = evaluate_prediction_root(
            args.dataset_root,
            pred_root,
            length,
            args.prediction_quaternion_order,
            args.association,
            args.max_time_difference,
            args.sequence,
        )
        length_output = args.output_dir / f"length_{length}"
        write_outputs(length_output, summary, rows)
        curve_rows.append(
            {
                "length": length,
                "sequence_count": summary["sequence_count"],
                "ate_rmse_m_mean_over_sequences": summary["ate_rmse_m_mean_over_sequences"],
                "ate_rmse_m_median_over_sequences": summary["ate_rmse_m_median_over_sequences"],
                "prediction_root": str(pred_root),
            }
        )
        print(f"[ATE] length={length}: {summary['ate_rmse_m_mean_over_sequences']:.6f} m over {summary['sequence_count']} sequences", flush=True)
    with (args.output_dir / "curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0].keys()))
        writer.writeheader()
        writer.writerows(curve_rows)
    curve_summary = {
        "benchmark": "human3r_tum_dynamics_ate_curve_v1",
        "protocol": "Human3R/evo translation APE RMSE with Sim(3) alignment",
        "dataset_root": str(args.dataset_root),
        "pred_parent": str(args.pred_parent),
        "lengths": list(lengths),
        "model": args.model,
        "prediction_quaternion_order": args.prediction_quaternion_order,
        "rows": curve_rows,
    }
    (args.output_dir / "curve_summary.json").write_text(json.dumps(curve_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(curve_summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

