from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    args = parse_args()
    data = json.loads((Path(args.output_dir) / "metrics_latest.json").read_text(encoding="utf-8"))
    metrics = data.get("val") or data.get("train") or {}
    required = {
        "stage2": [
            "metric_stage2_selection",
            "metric_hsi_base_transl_l2_median",
            "metric_hsi_transl_l2_median",
            "metric_hsi_transl_improvement_rate",
        ],
        "stage3": [
            "metric_stage3_selection",
            "metric_hsi_contact_float_p95_m",
            "metric_hsi_contact_penetration_p95_m",
            "metric_hsi_contact_base_abs_p95_m",
            "metric_hsi_contact_refined_abs_p95_m",
            "metric_hsi_contact_false_pull_rate",
        ],
    }[args.stage]
    missing = [key for key in required if key not in metrics or not math.isfinite(float(metrics[key]))]
    if missing:
        raise SystemExit(f"Missing or non-finite gate metrics: {missing}")
    if args.mode == "overfit":
        if args.stage == "stage2" and float(metrics["metric_hsi_transl_improvement_rate"]) < 0.90:
            raise SystemExit("Stage2 overfit gate failed: improvement_rate < 0.90")
        if args.stage == "stage2":
            base = float(metrics["metric_hsi_base_transl_l2_median"])
            refined = float(metrics["metric_hsi_transl_l2_median"])
            if base <= 0.0 or refined > 0.30 * base:
                raise SystemExit("Stage2 overfit gate failed: refined median is above 30% of base")
        if args.stage == "stage3":
            if float(metrics["metric_hsi_contact_false_pull_rate"]) > 0.05:
                raise SystemExit("Stage3 overfit gate failed: false_pull_rate > 0.05")
            base = float(metrics["metric_hsi_contact_base_abs_p95_m"])
            refined = float(metrics["metric_hsi_contact_refined_abs_p95_m"])
            if base <= 0.0 or refined > 0.30 * base:
                raise SystemExit("Stage3 overfit gate failed: contact p95 reduction is below 70%")
    if args.mode == "distribution" and args.stage == "stage2":
        if float(metrics["metric_hsi_transl_improvement_rate"]) <= 0.50:
            raise SystemExit("Stage2 distribution gate failed: improvement_rate <= 0.50")
    if args.mode == "distribution" and args.stage == "stage3":
        base = float(metrics["metric_hsi_contact_base_abs_p95_m"])
        refined = float(metrics["metric_hsi_contact_refined_abs_p95_m"])
        if base <= 0.0 or refined >= base:
            raise SystemExit("Stage3 distribution gate failed: contact p95 did not improve")
        if float(metrics["metric_hsi_contact_false_pull_rate"]) > 0.10:
            raise SystemExit("Stage3 distribution gate failed: false_pull_rate > 0.10")
    print(json.dumps({"gate": "pass", "stage": args.stage, "mode": args.mode, "metrics": {key: metrics[key] for key in required}}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=("stage2", "stage3"), required=True)
    parser.add_argument("--mode", choices=("smoke", "overfit", "distribution"), default="smoke")
    return parser.parse_args()


if __name__ == "__main__":
    main()
