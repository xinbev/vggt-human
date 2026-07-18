from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def main() -> None:
    args = parse_args()
    records_path = Path(args.records_jsonl).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    active = [record for record in records if bool(record.get("active", False))]
    if not active:
        raise SystemExit(f"No active V4 records in {records_path}")

    groupers: dict[str, Callable[[dict[str, Any]], str]] = {
        "noise_category": lambda row: str(row["noise_category"]),
        "base_error_bin": lambda row: metric_bin(
            float(row["base_l2_m"]), (0.05, 0.10, 0.25, 0.50, 1.00), "m"
        ),
        "base_depth_bin": lambda row: metric_bin(float(row["base_depth_m"]), (0.0, 5.0, 10.0, 20.0), "m"),
        "ray_target": lambda row: ray_target_group(float(row["expected_ray_m"])),
    }
    groups = {
        name: summarize_groups(active, grouper)
        for name, grouper in groupers.items()
    }
    failures = [record for record in active if not bool(record["improved"])]
    failures.sort(key=lambda row: float(row["candidate_l2_m"]) - float(row["base_l2_m"]), reverse=True)
    summary = {
        "records_jsonl": str(records_path),
        "num_records": len(records),
        "num_active": len(active),
        "num_failures": len(failures),
        "overall": summarize(active),
        "groups": groups,
    }
    (output_dir / "failure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(output_dir / "failed_active_people.csv", failures)
    write_csv(output_dir / "worst_50_active_people.csv", failures[:50])
    print(json.dumps(summary, indent=2))


def summarize_groups(
    rows: list[dict[str, Any]],
    grouper: Callable[[dict[str, Any]], str],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[grouper(row)].append(row)
    return {key: summarize(value) for key, value in sorted(grouped.items())}


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    base = [float(row["base_l2_m"]) for row in rows]
    candidate = [float(row["candidate_l2_m"]) for row in rows]
    failures = [row for row in rows if not bool(row["improved"])]
    expected_ray = [abs(float(row["expected_ray_m"])) for row in rows]
    candidate_ray = [abs(float(row["candidate_ray_m"])) for row in rows]
    return {
        "count": len(rows),
        "improvement_rate": sum(bool(row["improved"]) for row in rows) / max(len(rows), 1),
        "failure_count": len(failures),
        "eligibility_coverage": sum(bool(row["eligible"]) for row in rows) / max(len(rows), 1),
        "base_median_m": percentile(base, 0.50),
        "candidate_median_m": percentile(candidate, 0.50),
        "candidate_p90_m": percentile(candidate, 0.90),
        "candidate_to_base_median_ratio": percentile(candidate, 0.50) / max(percentile(base, 0.50), 1e-8),
        "expected_ray_abs_median_m": percentile(expected_ray, 0.50),
        "candidate_ray_abs_median_m": percentile(candidate_ray, 0.50),
        "tangent_delta_mean_m": sum(
            float(row["tangent_refined_l1_m"]) - float(row["tangent_base_l1_m"])
            for row in rows
        )
        / max(len(rows), 1),
    }


def metric_bin(value: float, boundaries: tuple[float, ...], unit: str) -> str:
    if not boundaries:
        return "all"
    if value < boundaries[0]:
        return f"<{boundaries[0]:g}{unit}"
    for lower, upper in zip(boundaries, boundaries[1:]):
        if lower <= value < upper:
            return f"{lower:g}-{upper:g}{unit}"
    return f">={boundaries[-1]:g}{unit}"


def ray_target_group(value: float) -> str:
    if abs(value) < 0.005:
        return "near_zero"
    return "positive" if value > 0.0 else "negative"


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(max(quantile, 0.0), 1.0) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
