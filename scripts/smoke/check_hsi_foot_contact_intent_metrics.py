from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    payload = json.loads((output_dir / "metrics_latest.json").read_text(encoding="utf-8"))
    metrics = payload.get("val") or payload.get("train") or {}
    values = {
        "loss": float(metrics.get("loss_hsi_foot_contact_intent", float("nan"))),
        "valid_coverage": float(metrics.get("metric_hsi_foot_contact_intent_valid_coverage", 0.0)),
        "temporal_coverage": float(metrics.get("metric_hsi_foot_contact_intent_temporal_coverage", 0.0)),
        "target_rate": float(metrics.get("metric_hsi_foot_contact_intent_target_rate", 0.0)),
        "accuracy": float(metrics.get("metric_hsi_foot_contact_intent_accuracy", 0.0)),
        "recall": float(metrics.get("metric_hsi_foot_contact_intent_recall", 0.0)),
        "precision": float(metrics.get("metric_hsi_foot_contact_intent_precision", 0.0)),
        "false_positive_rate": float(
            metrics.get("metric_hsi_foot_contact_intent_false_positive_rate", 1.0)
        ),
        "airborne_false_positive_rate": float(
            metrics.get("metric_hsi_foot_contact_intent_airborne_false_positive_rate", 1.0)
        ),
        "positive_probability": float(
            metrics.get("metric_hsi_foot_contact_intent_positive_probability", 0.0)
        ),
        "negative_probability": float(
            metrics.get("metric_hsi_foot_contact_intent_negative_probability", 1.0)
        ),
    }
    checks = {
        "finite": all(math.isfinite(value) for value in values.values()),
        "teacher_coverage": values["valid_coverage"] > 0.05,
        "temporal_features_present": values["temporal_coverage"] > 0.25,
        "positive_and_negative_targets": 0.0 < values["target_rate"] < 1.0,
    }
    limits: dict[str, float] = {}
    if args.mode == "overfit":
        limits = {"recall": 0.98, "precision": 0.95, "fpr": 0.02, "airborne_fpr": 0.02}
    elif args.mode == "distribution":
        limits = {"recall": 0.90, "precision": 0.80, "fpr": 0.05, "airborne_fpr": 0.03}
    if limits:
        checks.update(
            {
                "contact_recall": values["recall"] >= limits["recall"],
                "contact_precision": values["precision"] >= limits["precision"],
                "negative_false_positive_rate": values["false_positive_rate"] <= limits["fpr"],
                "airborne_false_positive_rate": (
                    values["airborne_false_positive_rate"] <= limits["airborne_fpr"]
                ),
                "probability_separation": values["positive_probability"] > values["negative_probability"],
            }
        )
    report = {
        "gate": "pass" if all(checks.values()) else "fail",
        "mode": args.mode,
        "metrics": values,
        "limits": limits,
        "checks": checks,
    }
    report_path = output_dir / f"contact_intent_gate_{args.mode}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["gate"] != "pass":
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("smoke", "overfit", "distribution"), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
