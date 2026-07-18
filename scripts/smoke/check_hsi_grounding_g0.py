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
    if report.get("gate") != "pass":
        print(json.dumps(report, indent=2))
        raise SystemExit(2)
    print(
        "[g0] pass "
        f"reduction={float(report['p95_reduction']):.3f} "
        f"coverage={float(report['geometry_valid_coverage']):.3f} "
        f"clean_p95={float(report['clean_candidate_displacement_p95_m']):.6f}m"
    )


if __name__ == "__main__":
    main()
