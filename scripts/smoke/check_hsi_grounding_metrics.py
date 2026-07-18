from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    args = parse_args()
    path = Path(args.output_dir) / "metrics_latest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("val") or payload.get("train") or {}
    base = float(metrics.get("metric_hsi_grounding_base_l2_p95_m", 0.0))
    refined = float(metrics.get("metric_hsi_grounding_refined_l2_p95_m", float("inf")))
    coverage = float(metrics.get("metric_hsi_grounding_valid_coverage", 0.0))
    accuracy = float(metrics.get("metric_hsi_grounding_gate_accuracy", 0.0))
    improvement = float(metrics.get("metric_hsi_grounding_improvement_rate", 0.0))
    clean = float(metrics.get("metric_hsi_grounding_clean_displacement_p95_m", float("inf")))
    if args.mode == "overfit":
        limits = {"ratio": 0.35, "coverage": 0.70, "accuracy": 0.90, "improvement": 0.90, "clean": 0.005}
    elif args.mode == "distribution":
        limits = {"ratio": 0.60, "coverage": 0.70, "accuracy": 0.75, "improvement": 0.70, "clean": 0.005}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check HSI grounding training gate")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("overfit", "distribution", "real"), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
