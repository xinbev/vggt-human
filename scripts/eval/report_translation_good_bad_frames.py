#!/usr/bin/env python
"""Summarize good/bad SMPL translation frames from per-person scan CSV.

This script is intentionally lightweight: it does not load the model.  Run a
dataset scan first to create ``all_frame_person_translation_rows.csv``, then use
this reporter to count good/bad frames, long-tail percentiles, and focused frame
per-person errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


SOURCE_KEY = {
    "base": "base_transl_l2_m",
    "hsi": "hsi_transl_l2_m",
}


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser()
    if not input_csv.is_file():
        raise FileNotFoundError(f"Per-person translation CSV not found: {input_csv}")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_csv)
    if args.dedupe_frame_person:
        rows = dedupe_frame_person_rows(rows)
    focus_frames = parse_focus_frames(args.focus_frames)
    sources = parse_sources(args.sources)

    frame_rows = build_frame_rows(rows, sources, float(args.bad_transl_m), float(args.severe_transl_m))
    focused_rows = select_focused_rows(rows, focus_frames)
    summary = build_summary(
        rows=rows,
        frame_rows=frame_rows,
        sources=sources,
        bad_threshold=float(args.bad_transl_m),
        severe_threshold=float(args.severe_transl_m),
        focus_frames=focus_frames,
        input_csv=input_csv,
    )
    summary["top_bad_person_rows"] = {
        source: compact_person_rows(top_rows(rows, SOURCE_KEY[source], int(args.top_k)), source)
        for source in sources
    }
    summary["focused_frame_person_rows"] = compact_focus_rows(focused_rows, sources)

    summary_json = output_dir / "translation_good_bad_summary.json"
    frame_csv = output_dir / "translation_good_bad_frame_summary.csv"
    focus_csv = output_dir / "focused_frame_person_translation_errors.csv"
    report_md = output_dir / "translation_good_bad_report.md"

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(frame_csv, frame_rows, frame_fieldnames(sources))
    write_csv(focus_csv, focused_rows, focus_fieldnames(focused_rows, sources))
    report_md.write_text(render_markdown(summary), encoding="utf-8")

    print_human_summary(summary)
    print(
        json.dumps(
            {
                "summary_json": str(summary_json),
                "frame_csv": str(frame_csv),
                "focus_csv": str(focus_csv),
                "report_md": str(report_md),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report good/bad SMPL translation frames from scan CSV")
    parser.add_argument(
        "--input-csv",
        default="outputs/eval/hsi_bad_translation_scan_after_translation_ray_refine/all_frame_person_translation_rows.csv",
        help="CSV from scan_hsi_bad_translation_frames.py",
    )
    parser.add_argument("--output-dir", default="outputs/eval/translation_good_bad_report")
    parser.add_argument("--bad-transl-m", type=float, default=0.50, help="Frame/person is bad when transl L2 is above this threshold")
    parser.add_argument("--severe-transl-m", type=float, default=0.80, help="Frame/person is severe when transl L2 is above this threshold")
    parser.add_argument("--sources", default="base,hsi", help="Comma-separated sources: base,hsi")
    parser.add_argument(
        "--focus-frames",
        default="seq_000000_0085,seq_000000_0100",
        help="Comma-separated frame stems to list per-person errors for",
    )
    parser.add_argument("--dedupe-frame-person", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def parse_sources(value: str) -> list[str]:
    sources = []
    for item in str(value).split(","):
        source = item.strip().lower()
        if not source:
            continue
        if source not in SOURCE_KEY:
            raise ValueError(f"Unsupported source {source!r}; choose from {sorted(SOURCE_KEY)}")
        sources.append(source)
    if not sources:
        raise ValueError("--sources cannot be empty")
    return sources


def parse_focus_frames(value: str) -> set[str]:
    return {item.strip() for item in str(value).split(",") if item.strip()}


def dedupe_frame_person_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("sequence_name", "")), str(row.get("frame_name", "")), str(row.get("track_id", "")))
        current = by_key.get(key)
        if current is None or row_worst_translation(row) > row_worst_translation(current):
            by_key[key] = row
    return list(by_key.values())


def row_worst_translation(row: dict[str, Any]) -> float:
    values = [as_float(row.get("base_transl_l2_m")), as_float(row.get("hsi_transl_l2_m"))]
    finite = [value for value in values if value is not None]
    return max(finite) if finite else -1.0


def build_frame_rows(
    rows: list[dict[str, Any]],
    sources: list[str],
    bad_threshold: float,
    severe_threshold: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("sequence_name", "")), str(row.get("frame_name", "")))].append(row)

    frame_rows = []
    for (sequence_name, frame_name), group in grouped.items():
        out: dict[str, Any] = {
            "sequence_name": sequence_name,
            "frame_name": frame_name,
            "image_path": first_nonempty(group, "image_path"),
            "num_people": len(group),
        }
        for source in sources:
            key = SOURCE_KEY[source]
            values = values_for(group, key)
            max_value = max(values) if values else None
            mean_value = sum(values) / len(values) if values else None
            bad_people = sum(1 for value in values if value > bad_threshold)
            severe_people = sum(1 for value in values if value > severe_threshold)
            worst = max(group, key=lambda row: as_sort(row.get(key)))
            out[f"{source}_max_transl_l2_m"] = max_value
            out[f"{source}_mean_transl_l2_m"] = mean_value
            out[f"{source}_bad_people"] = bad_people
            out[f"{source}_severe_people"] = severe_people
            out[f"{source}_is_good_frame"] = int(max_value is not None and max_value <= bad_threshold)
            out[f"{source}_is_bad_frame"] = int(max_value is not None and max_value > bad_threshold)
            out[f"{source}_is_severe_frame"] = int(max_value is not None and max_value > severe_threshold)
            out[f"{source}_worst_track_id"] = worst.get("track_id", "")
            out[f"{source}_worst_query_idx"] = worst.get("query_idx", "")
        frame_rows.append(out)
    return sorted(frame_rows, key=lambda row: max(as_sort(row.get(f"{source}_max_transl_l2_m")) for source in sources), reverse=True)


def build_summary(
    rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    sources: list[str],
    bad_threshold: float,
    severe_threshold: float,
    focus_frames: set[str],
    input_csv: Path,
) -> dict[str, Any]:
    sequence_names = {str(row.get("sequence_name", "")) for row in rows}
    out: dict[str, Any] = {
        "input_csv": str(input_csv),
        "thresholds": {
            "bad_transl_m": bad_threshold,
            "severe_transl_m": severe_threshold,
        },
        "deduped_frame_person_rows": len(rows),
        "num_frames": len(frame_rows),
        "num_sequences": len(sequence_names),
        "focus_frames": sorted(focus_frames),
        "sources": {},
    }
    for source in sources:
        key = SOURCE_KEY[source]
        person_values = values_for(rows, key)
        frame_values = values_for(frame_rows, f"{source}_max_transl_l2_m")
        bad_people = sum(1 for value in person_values if value > bad_threshold)
        severe_people = sum(1 for value in person_values if value > severe_threshold)
        bad_frames = sum(1 for row in frame_rows if int(row.get(f"{source}_is_bad_frame", 0)) == 1)
        severe_frames = sum(1 for row in frame_rows if int(row.get(f"{source}_is_severe_frame", 0)) == 1)
        good_frames = sum(1 for row in frame_rows if int(row.get(f"{source}_is_good_frame", 0)) == 1)
        out["sources"][source] = {
            "person": {
                **describe_values(person_values),
                f"count_gt_{bad_threshold:.2f}m": bad_people,
                f"count_gt_{severe_threshold:.2f}m": severe_people,
                f"ratio_gt_{bad_threshold:.2f}m": ratio(bad_people, len(person_values)),
                f"ratio_gt_{severe_threshold:.2f}m": ratio(severe_people, len(person_values)),
            },
            "frame_max": {
                **describe_values(frame_values),
                "good_frames": good_frames,
                "bad_frames": bad_frames,
                "severe_frames": severe_frames,
                "good_frame_ratio": ratio(good_frames, len(frame_rows)),
                "bad_frame_ratio": ratio(bad_frames, len(frame_rows)),
                "severe_frame_ratio": ratio(severe_frames, len(frame_rows)),
            },
        }
    if "base" in sources and "hsi" in sources:
        out["base_vs_hsi"] = compare_base_hsi(rows, frame_rows, bad_threshold, severe_threshold)
    return out


def compare_base_hsi(rows: list[dict[str, Any]], frame_rows: list[dict[str, Any]], bad_threshold: float, severe_threshold: float) -> dict[str, Any]:
    base_values = values_for(rows, "base_transl_l2_m")
    hsi_values = values_for(rows, "hsi_transl_l2_m")
    paired = [
        (as_float(row.get("base_transl_l2_m")), as_float(row.get("hsi_transl_l2_m")))
        for row in rows
    ]
    paired = [(base, hsi) for base, hsi in paired if base is not None and hsi is not None]
    hsi_better = sum(1 for base, hsi in paired if hsi < base)
    hsi_worse_5cm = sum(1 for base, hsi in paired if hsi > base + 0.05)
    rescued = sum(1 for base, hsi in paired if base > bad_threshold and hsi <= bad_threshold)
    newly_bad = sum(1 for base, hsi in paired if base <= bad_threshold and hsi > bad_threshold)
    both_bad = sum(1 for base, hsi in paired if base > bad_threshold and hsi > bad_threshold)
    frame_rescued = sum(
        1
        for row in frame_rows
        if as_sort(row.get("base_max_transl_l2_m")) > bad_threshold and as_sort(row.get("hsi_max_transl_l2_m")) <= bad_threshold
    )
    frame_newly_bad = sum(
        1
        for row in frame_rows
        if as_sort(row.get("base_max_transl_l2_m")) <= bad_threshold and as_sort(row.get("hsi_max_transl_l2_m")) > bad_threshold
    )
    return {
        "paired_person_count": len(paired),
        "hsi_better_person_count": hsi_better,
        "hsi_better_person_ratio": ratio(hsi_better, len(paired)),
        "hsi_worse_by_more_than_5cm_person_count": hsi_worse_5cm,
        "hsi_worse_by_more_than_5cm_person_ratio": ratio(hsi_worse_5cm, len(paired)),
        "person_rescued_bad_to_good_count": rescued,
        "person_newly_bad_count": newly_bad,
        "person_both_bad_count": both_bad,
        "frame_rescued_bad_to_good_count": frame_rescued,
        "frame_newly_bad_count": frame_newly_bad,
        "base_mean_minus_hsi_mean_m": (sum(base_values) / len(base_values) - sum(hsi_values) / len(hsi_values)) if base_values and hsi_values else None,
        "bad_threshold_m": bad_threshold,
        "severe_threshold_m": severe_threshold,
    }


def select_focused_rows(rows: list[dict[str, Any]], focus_frames: set[str]) -> list[dict[str, Any]]:
    if not focus_frames:
        return []
    selected = [row for row in rows if str(row.get("frame_name", "")) in focus_frames]
    return sorted(selected, key=lambda row: (str(row.get("sequence_name", "")), str(row.get("frame_name", "")), as_sort(row.get("base_transl_l2_m"))), reverse=True)


def top_rows(rows: list[dict[str, Any]], key: str, limit: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: as_sort(row.get(key)), reverse=True)[: max(limit, 0)]


def compact_person_rows(rows: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    key = SOURCE_KEY[source]
    return [
        {
            "sequence_name": row.get("sequence_name", ""),
            "frame_name": row.get("frame_name", ""),
            "image_path": row.get("image_path", ""),
            "track_id": row.get("track_id", ""),
            "query_idx": row.get("query_idx", ""),
            f"{source}_transl_l2_m": as_float(row.get(key)),
            "base_transl_l2_m": as_float(row.get("base_transl_l2_m")),
            "hsi_transl_l2_m": as_float(row.get("hsi_transl_l2_m")),
            "base_mpjpe_m": as_float(row.get("base_mpjpe_m")),
            "hsi_mpjpe_m": as_float(row.get("hsi_mpjpe_m")),
        }
        for row in rows
    ]


def compact_focus_rows(rows: list[dict[str, Any]], sources: list[str]) -> list[dict[str, Any]]:
    keys = [
        "sequence_name",
        "frame_name",
        "image_path",
        "track_id",
        "query_idx",
        "pred_conf",
        "base_transl_l2_m",
        "hsi_transl_l2_m",
        "base_mpjpe_m",
        "hsi_mpjpe_m",
    ]
    return [{key: row.get(key) for key in keys if key in row or key in {"base_transl_l2_m", "hsi_transl_l2_m"}} for row in rows]


def describe_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "mean": sum(sorted_values) / len(sorted_values),
        "min": sorted_values[0],
        "p50": percentile(sorted_values, 50.0),
        "p90": percentile(sorted_values, 90.0),
        "p95": percentile(sorted_values, 95.0),
        "p99": percentile(sorted_values, 99.0),
        "max": sorted_values[-1],
    }


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * (q / 100.0)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return sorted_values[low]
    weight = pos - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def values_for(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = [as_float(row.get(key)) for row in rows]
    return [value for value in values if value is not None]


def first_nonempty(rows: list[dict[str, Any]], key: str) -> str:
    for row in rows:
        value = str(row.get(key, ""))
        if value:
            return value
    return ""


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def as_sort(value: Any) -> float:
    parsed = as_float(value)
    return parsed if parsed is not None else -1.0


def ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator) / float(denominator) if denominator > 0 else None


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def frame_fieldnames(sources: list[str]) -> list[str]:
    fields = ["sequence_name", "frame_name", "image_path", "num_people"]
    for source in sources:
        fields.extend(
            [
                f"{source}_max_transl_l2_m",
                f"{source}_mean_transl_l2_m",
                f"{source}_bad_people",
                f"{source}_severe_people",
                f"{source}_is_good_frame",
                f"{source}_is_bad_frame",
                f"{source}_is_severe_frame",
                f"{source}_worst_track_id",
                f"{source}_worst_query_idx",
            ]
        )
    return fields


def focus_fieldnames(rows: list[dict[str, Any]], sources: list[str]) -> list[str]:
    preferred = [
        "sequence_name",
        "frame_name",
        "image_path",
        "track_id",
        "query_idx",
        "pred_conf",
        "base_transl_l2_m",
        "hsi_transl_l2_m",
        "base_mpjpe_m",
        "hsi_mpjpe_m",
        "base_pve_m",
        "hsi_pve_m",
        "hsi_transl_delta_m",
        "hsi_mpjpe_delta_m",
    ]
    present = set().union(*(row.keys() for row in rows)) if rows else set(preferred)
    return [field for field in preferred if field in present]


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Translation Good/Bad Frame Report",
        "",
        f"- Input CSV: `{summary['input_csv']}`",
        f"- Bad threshold: `{summary['thresholds']['bad_transl_m']:.3f}m`",
        f"- Severe threshold: `{summary['thresholds']['severe_transl_m']:.3f}m`",
        f"- Frames: `{summary['num_frames']}`",
        f"- Frame-person rows: `{summary['deduped_frame_person_rows']}`",
        "",
    ]
    for source, source_summary in summary["sources"].items():
        person = source_summary["person"]
        frame = source_summary["frame_max"]
        lines.extend(
            [
                f"## {source.upper()}",
                "",
                "| level | count | mean | p50 | p90 | p95 | p99 | max | >bad | >severe |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                (
                    f"| person | {person['count']} | {fmt(person['mean'])} | {fmt(person['p50'])} | "
                    f"{fmt(person['p90'])} | {fmt(person['p95'])} | {fmt(person['p99'])} | {fmt(person['max'])} | "
                    f"{person.get('count_gt_' + format_threshold(summary['thresholds']['bad_transl_m']) + 'm', 0)} | "
                    f"{person.get('count_gt_' + format_threshold(summary['thresholds']['severe_transl_m']) + 'm', 0)} |"
                ),
                (
                    f"| frame max | {frame['count']} | {fmt(frame['mean'])} | {fmt(frame['p50'])} | "
                    f"{fmt(frame['p90'])} | {fmt(frame['p95'])} | {fmt(frame['p99'])} | {fmt(frame['max'])} | "
                    f"{frame['bad_frames']} | {frame['severe_frames']} |"
                ),
                "",
                f"Good frames: `{frame['good_frames']}` / `{frame['count']}`",
                "",
            ]
        )
    if "base_vs_hsi" in summary:
        compare = summary["base_vs_hsi"]
        lines.extend(
            [
                "## Base vs HSI",
                "",
                f"- HSI better persons: `{compare['hsi_better_person_count']}` / `{compare['paired_person_count']}`",
                f"- HSI worse by >5cm persons: `{compare['hsi_worse_by_more_than_5cm_person_count']}`",
                f"- Bad-to-good rescued persons: `{compare['person_rescued_bad_to_good_count']}`",
                f"- Newly bad persons: `{compare['person_newly_bad_count']}`",
                f"- Both bad persons: `{compare['person_both_bad_count']}`",
                f"- Bad-to-good rescued frames: `{compare['frame_rescued_bad_to_good_count']}`",
                f"- Newly bad frames: `{compare['frame_newly_bad_count']}`",
                "",
            ]
        )
    return "\n".join(lines)


def format_threshold(value: float) -> str:
    return f"{value:.2f}"


def fmt(value: Any) -> str:
    parsed = as_float(value)
    return "NA" if parsed is None else f"{parsed:.6f}"


def print_human_summary(summary: dict[str, Any]) -> None:
    print("========== Translation good/bad frame report ==========")
    print(f"Input rows : {summary['deduped_frame_person_rows']}")
    print(f"Frames     : {summary['num_frames']}")
    print(f"Thresholds : bad>{summary['thresholds']['bad_transl_m']:.3f}m severe>{summary['thresholds']['severe_transl_m']:.3f}m")
    for source, source_summary in summary["sources"].items():
        person = source_summary["person"]
        frame = source_summary["frame_max"]
        bad_key = "count_gt_" + format_threshold(summary["thresholds"]["bad_transl_m"]) + "m"
        severe_key = "count_gt_" + format_threshold(summary["thresholds"]["severe_transl_m"]) + "m"
        print(
            f"{source.upper()} person L2     "
            f"p90={fmt(person['p90'])} p95={fmt(person['p95'])} p99={fmt(person['p99'])} "
            f">bad={person.get(bad_key, 0)} >severe={person.get(severe_key, 0)} max={fmt(person['max'])}"
        )
        print(
            f"{source.upper()} frame max L2  "
            f"good={frame['good_frames']} bad={frame['bad_frames']} severe={frame['severe_frames']} "
            f"p90={fmt(frame['p90'])} p95={fmt(frame['p95'])} p99={fmt(frame['p99'])} max={fmt(frame['max'])}"
        )
    if summary.get("focused_frame_person_rows"):
        print("Focused frame rows:")
        for row in summary["focused_frame_person_rows"][:30]:
            print(
                f"  {row.get('sequence_name')}/{row.get('frame_name')} "
                f"track={row.get('track_id')} q={row.get('query_idx')} "
                f"base={row.get('base_transl_l2_m')} hsi={row.get('hsi_transl_l2_m')}"
            )


if __name__ == "__main__":
    main()
