from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    args = parse_args()
    path = Path(args.output_dir) / "metrics_latest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("val") or payload.get("train") or {}
    if args.mode.startswith("severe_float_"):
        report = check_severe_float(args, metrics)
        report_path = Path(args.output_dir) / f"grounding_gate_{args.mode}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        if report["gate"] != "pass":
            raise SystemExit(2)
        return
    base = float(metrics.get("metric_hsi_grounding_base_l2_p95_m", 0.0))
    refined = float(metrics.get("metric_hsi_grounding_refined_l2_p95_m", float("inf")))
    coverage = float(metrics.get("metric_hsi_grounding_valid_coverage", 0.0))
    accuracy = float(metrics.get("metric_hsi_grounding_gate_accuracy", 0.0))
    improvement = float(metrics.get("metric_hsi_grounding_improvement_rate", 0.0))
    clean = float(metrics.get("metric_hsi_grounding_clean_displacement_p95_m", float("inf")))
    if args.mode == "overfit":
        limits = {"ratio": 0.35, "coverage": 0.50, "accuracy": 0.90, "improvement": 0.90, "clean": 0.005}
    elif args.mode == "distribution":
        limits = {"ratio": 0.60, "coverage": 0.50, "accuracy": 0.75, "improvement": 0.70, "clean": 0.005}
    else:
        limits = {"ratio": 0.90, "coverage": 0.40, "accuracy": 0.65, "improvement": 0.55, "clean": 0.010}
    ratio = refined / max(base, 1e-8)
    checks = {
        "refined_over_base_p95": ratio <= limits["ratio"],
        "geometry_coverage": coverage >= limits["coverage"],
        "gate_accuracy": accuracy >= limits["accuracy"],
        "improvement_rate": improvement >= limits["improvement"],
        "clean_displacement": clean <= limits["clean"],
    }
    report = {
        "gate": "pass" if all(checks.values()) else "fail",
        "mode": args.mode,
        "metrics": {
            "base_p95_m": base,
            "refined_p95_m": refined,
            "refined_over_base_p95": ratio,
            "geometry_coverage": coverage,
            "gate_accuracy": accuracy,
            "improvement_rate": improvement,
            "clean_displacement_p95_m": clean,
        },
        "limits": limits,
        "checks": checks,
    }
    report_path = Path(args.output_dir) / f"grounding_gate_{args.mode}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["gate"] != "pass":
        raise SystemExit(2)


def check_severe_float(args: argparse.Namespace, metrics: dict) -> dict:
    resolved_path = Path(args.output_dir) / "resolved_config.json"
    if not resolved_path.is_file():
        raise FileNotFoundError(resolved_path)
    config = json.loads(resolved_path.read_text(encoding="utf-8"))
    model = config.get("model", {})
    loss = config.get("loss", {})
    model_threshold = float(model.get("hsi_grounding_gate_threshold", -1.0))
    loss_threshold = float(loss.get("hsi_grounding_gate_decision_threshold", -2.0))
    allowed_nonzero_weights = {
        "hsi_grounding_gate_weight",
        "hsi_grounding_gate_positive_weight",
        "hsi_grounding_gate_negative_weight",
    }
    conflicting_weights = {
        key: float(value)
        for key, value in loss.items()
        if key.endswith("_weight")
        and key not in allowed_nonzero_weights
        and isinstance(value, (int, float))
        and abs(float(value)) > 0.0
    }
    config_checks = {
        "target_mode": loss.get("hsi_grounding_gate_target_mode") == "severe_float",
        "hard_gate_train": model.get("hsi_grounding_hard_gate_train") is True,
        "hard_gate_eval": model.get("hsi_grounding_hard_gate_eval") is True,
        "matching_thresholds": abs(model_threshold - loss_threshold) <= 1e-8,
        "gate_only_loss": not conflicting_weights,
    }
    values = {
        "loss": float(metrics.get("loss_hsi_grounding_gate", float("nan"))),
        "target_rate": float(metrics.get("metric_hsi_grounding_apply_target_rate", 0.0)),
        "coverage": float(metrics.get("metric_hsi_grounding_valid_coverage", 0.0)),
        "recall": float(metrics.get("metric_hsi_grounding_severe_float_recall", 0.0)),
        "clean_false_apply": float(metrics.get("metric_hsi_grounding_clean_false_apply_rate", 1.0)),
        "negative_false_apply": float(metrics.get("metric_hsi_grounding_negative_false_apply_rate", 1.0)),
        "positive_probability": float(metrics.get("metric_hsi_grounding_positive_gate_mean", 0.0)),
        "negative_probability": float(metrics.get("metric_hsi_grounding_negative_gate_mean", 1.0)),
        "base_p95_m": float(metrics.get("metric_hsi_grounding_severe_base_p95_m", 0.0)),
        "candidate_p95_m": float(metrics.get("metric_hsi_grounding_severe_candidate_p95_m", float("inf"))),
        "refined_p95_m": float(metrics.get("metric_hsi_grounding_severe_refined_p95_m", float("inf"))),
        "improvement_rate": float(metrics.get("metric_hsi_grounding_severe_improvement_rate", 0.0)),
        "clean_displacement_p95_m": float(
            metrics.get("metric_hsi_grounding_clean_displacement_p95_m", float("inf"))
        ),
    }
    common_checks = {
        "finite": all(math.isfinite(value) for value in values.values()),
        "positive_targets_present": values["target_rate"] > 0.0,
        "geometry_coverage": values["coverage"] >= 0.50,
    }
    if args.mode == "severe_float_smoke":
        metric_checks = common_checks
        limits = {"coverage": 0.50, "target_rate": ">0"}
    else:
        metric_checks = {
            **common_checks,
            "severe_float_recall": values["recall"] >= 0.95,
            "clean_false_apply": values["clean_false_apply"] <= 0.01,
            "negative_false_apply": values["negative_false_apply"] <= 0.02,
            "probability_separation": values["positive_probability"] > values["negative_probability"],
            "candidate_reduces_tail": values["candidate_p95_m"] <= 0.50 * max(values["base_p95_m"], 1e-8),
            "refined_tracks_candidate": values["refined_p95_m"] <= values["candidate_p95_m"] + 0.005,
            "improvement_rate": values["improvement_rate"] >= 0.95,
            "clean_displacement": values["clean_displacement_p95_m"] <= 0.001,
        }
        limits = {
            "coverage": 0.50,
            "recall": 0.95,
            "clean_false_apply": 0.01,
            "negative_false_apply": 0.02,
            "candidate_over_base": 0.50,
            "refined_candidate_margin_m": 0.005,
            "improvement_rate": 0.95,
            "clean_displacement_m": 0.001,
        }
    checks = {**config_checks, **metric_checks}
    return {
        "gate": "pass" if all(checks.values()) else "fail",
        "mode": args.mode,
        "metrics": values,
        "limits": limits,
        "checks": checks,
        "conflicting_loss_weights": conflicting_weights,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check HSI grounding training gate")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "overfit",
            "distribution",
            "real",
            "severe_float_smoke",
            "severe_float_overfit",
            "severe_float_distribution",
        ),
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
