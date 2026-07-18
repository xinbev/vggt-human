from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Require a passing G0 analytic grounding audit")
    parser.add_argument("--metrics", required=True)
    args = parser.parse_args()
    path = Path(args.metrics)
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    passed, checks = _focused_grounding_gate(report)
    if not passed:
        print(json.dumps(report, indent=2))
        print(json.dumps({"focused_checks": checks}, indent=2))
        raise SystemExit(2)
    print(
        "[g0] pass "
        f"float8_reduction={checks['float_8cm_reduction']:.3f} "
        f"float12_reduction={checks['float_12cm_reduction']:.3f} "
        f"clean_p95={float(report['clean_candidate_displacement_p95_m']):.6f}m"
    )


def _focused_grounding_gate(report: dict) -> tuple[bool, dict[str, float | bool]]:
    groups = report.get("by_noise_level", {})
    float_5 = groups.get("+5cm", {})
    float_8 = groups.get("+8cm", {})
    float_12 = groups.get("+12cm", {})
    checks: dict[str, float | bool] = {
        "clean_p95_m": float(report.get("clean_candidate_displacement_p95_m", float("inf"))),
        "plane_signed_error_p95_m": float(
            report.get("online_vs_teacher_signed_error_p95_m", float("inf"))
        ),
        "plane_normal_cosine_p10": float(
            report.get("online_vs_teacher_normal_cosine_p10", float("-inf"))
        ),
        "float_5cm_improvement": float(float_5.get("improvement_rate", 0.0)),
        "float_8cm_improvement": float(float_8.get("improvement_rate", 0.0)),
        "float_8cm_reduction": float(float_8.get("p95_reduction", float("-inf"))),
        "float_12cm_improvement": float(float_12.get("improvement_rate", 0.0)),
        "float_12cm_reduction": float(float_12.get("p95_reduction", float("-inf"))),
    }
    passed = (
        checks["clean_p95_m"] <= 0.001
        and checks["plane_signed_error_p95_m"] <= 0.005
        and checks["plane_normal_cosine_p10"] >= 0.98
        and checks["float_5cm_improvement"] >= 0.90
        and checks["float_8cm_improvement"] >= 0.95
        and checks["float_8cm_reduction"] >= 0.50
        and checks["float_12cm_improvement"] >= 0.95
        and checks["float_12cm_reduction"] >= 0.60
    )
    checks["pass"] = passed
    return passed, checks


if __name__ == "__main__":
    main()
