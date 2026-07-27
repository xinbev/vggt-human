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
    raw_recall = float(metrics.get("metric_hsi_foot_contact_intent_recall", 0.0))
    raw_precision = float(metrics.get("metric_hsi_foot_contact_intent_precision", 0.0))
    accuracy = float(metrics.get("metric_hsi_foot_contact_intent_accuracy", 0.0))
    target_rate = float(metrics.get("metric_hsi_foot_contact_intent_target_rate", 0.0))
    empty_denominator_adjusted = accuracy >= 1.0 - 1e-7 and 0.0 < target_rate < 1.0
    values = {
        "loss": float(metrics.get("loss_hsi_foot_contact_intent", float("nan"))),
        "valid_coverage": float(metrics.get("metric_hsi_foot_contact_intent_valid_coverage", 0.0)),
        "temporal_coverage": float(metrics.get("metric_hsi_foot_contact_intent_temporal_coverage", 0.0)),
        "target_rate": target_rate,
        "accuracy": accuracy,
        "recall": 1.0 if empty_denominator_adjusted else raw_recall,
        "precision": 1.0 if empty_denominator_adjusted else raw_precision,
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
        "speed_error_median_m": float(
            metrics.get("metric_hsi_foot_contact_intent_speed_error_median_m", float("inf"))
        ),
        "speed_error_p95_m": float(
            metrics.get("metric_hsi_foot_contact_intent_speed_error_p95_m", float("inf"))
        ),
        "static_negative_false_positive_rate": float(
            metrics.get("metric_hsi_foot_contact_intent_static_negative_false_positive_rate", 1.0)
        ),
        "fast_path_active": float(
            metrics.get("metric_hsi_foot_contact_intent_fast_path_active", 0.0)
        ),
    }
    checks = {
        "finite": all(math.isfinite(value) for value in values.values()),
        "teacher_coverage": values["valid_coverage"] > 0.05,
        "temporal_features_present": values["temporal_coverage"] > 0.25,
        "positive_and_negative_targets": 0.0 < values["target_rate"] < 1.0,
        "teacher_speed_contract": values["speed_error_p95_m"] <= 0.005,
    }
    limits: dict[str, float] = {}
    if args.mode == "overfit":
        limits = {
            "recall": 0.98,
            "precision": 0.95,
            "fpr": 0.02,
            "airborne_fpr": 0.02,
            "static_negative_fpr": 0.05,
        }
    elif args.mode == "distribution":
        limits = {
            "recall": 0.90,
            "precision": 0.80,
            "fpr": 0.05,
            "airborne_fpr": 0.03,
            "static_negative_fpr": 0.10,
        }
    if limits:
        checks.update(
            {
                "contact_recall": values["recall"] >= limits["recall"],
                "contact_precision": values["precision"] >= limits["precision"],
                "negative_false_positive_rate": values["false_positive_rate"] <= limits["fpr"],
                "airborne_false_positive_rate": (
                    values["airborne_false_positive_rate"] <= limits["airborne_fpr"]
                ),
                "static_negative_false_positive_rate": (
                    values["static_negative_false_positive_rate"] <= limits["static_negative_fpr"]
                ),
                "probability_separation": values["positive_probability"] > values["negative_probability"],
            }
        )
    if args.mode in {"fast", "distribution"}:
        checks["fast_gt_path_active"] = values["fast_path_active"] >= 0.99
    report = {
        "gate": "pass" if all(checks.values()) else "fail",
        "mode": args.mode,
        "metrics": values,
        "aggregation_audit": {
            "empty_denominator_adjusted": empty_denominator_adjusted,
            "raw_recall": raw_recall,
            "raw_precision": raw_precision,
        },
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
    parser.add_argument("--mode", choices=("smoke", "fast", "overfit", "distribution"), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
