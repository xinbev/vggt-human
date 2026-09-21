#!/usr/bin/env python
"""Evaluate cached RICH windows after manual or oracle scale calibration."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.evaluation.rich_physical_grounding import (  # noqa: E402
    GroundEstimatorConfig,
    compute_physical_metrics,
)
from vggt_omega.evaluation.rich_scale_oracle import (  # noqa: E402
    CACHE_FORMAT,
    evaluate_scale_window_cache,
    load_scale_window_cache,
)


SCALE_FILE_FORMAT = "vggt_omega_rich_manual_scale_selections_v1"
METRIC_KEYS = ("collision_ratio_pct", "penetrate_cm", "float_cm", "penetration_max_cm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--scale-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("manual", "oracle_grid", "base"), default="manual")
    parser.add_argument("--allow-missing-manual-scales", action="store_true")
    parser.add_argument("--oracle-scale-min", type=float, default=0.25)
    parser.add_argument("--oracle-scale-max", type=float, default=4.0)
    parser.add_argument("--oracle-steps", type=int, default=41)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != CACHE_FORMAT:
        raise ValueError(f"Unsupported cache manifest: {manifest_path}")
    windows = list(manifest.get("windows", []))
    if not windows:
        raise RuntimeError(f"No cached windows in {manifest_path}")
    ground_config = GroundEstimatorConfig(**manifest["ground_config"])

    scale_path = (
        args.scale_file.expanduser().resolve()
        if args.scale_file is not None
        else cache_dir / "manual_scales.json"
    )
    selections = load_scale_file(scale_path) if args.mode == "manual" else {"windows": {}}
    missing = [row["window_id"] for row in windows if row["window_id"] not in selections.get("windows", {})]
    if args.mode == "manual" and missing and not args.allow_missing_manual_scales:
        raise RuntimeError(
            f"Manual scale file is missing {len(missing)}/{len(windows)} windows. "
            f"First missing: {missing[0]}. Save every window in Viser or pass "
            "--allow-missing-manual-scales to use x1.0 for missing windows."
        )

    adjusted_frames: list[dict[str, Any]] = []
    baseline_frames: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    chosen_scales: dict[str, Any] = {}
    oracle_factors = None
    if args.mode == "oracle_grid":
        if args.oracle_scale_min <= 0 or args.oracle_scale_max <= args.oracle_scale_min or args.oracle_steps < 2:
            raise ValueError("Oracle grid requires 0 < min < max and at least two steps")
        oracle_factors = np.geomspace(args.oracle_scale_min, args.oracle_scale_max, args.oracle_steps)

    for index, window in enumerate(windows, start=1):
        window_id = str(window["window_id"])
        cache = load_scale_window_cache(cache_dir / str(window["cache_file"]))
        baseline_rows, baseline_report = evaluate_scale_window_cache(cache, 1.0, ground_config)
        if args.mode == "base":
            scale = 1.0
            rows, report = baseline_rows, baseline_report
        elif args.mode == "manual":
            scale = float(
                selections.get("windows", {}).get(window_id, {}).get("scale_multiplier", 1.0)
            )
            rows, report = evaluate_scale_window_cache(cache, scale, ground_config)
        else:
            assert oracle_factors is not None
            scale, rows, report = choose_oracle_scale(cache, oracle_factors, ground_config)

        chosen_scales[window_id] = {
            "scale_multiplier": scale,
            "vid": window["vid"],
            "window_index": int(window["window_index"]),
        }
        attach_frame_metadata(baseline_rows, window, 1.0)
        attach_frame_metadata(rows, window, scale)
        baseline_frames.extend(baseline_rows)
        adjusted_frames.extend(rows)
        metrics = report.get("metrics", compute_physical_metrics([], ground_config.tolerance_m))
        baseline_metrics = baseline_report.get("metrics", compute_physical_metrics([], ground_config.tolerance_m))
        window_rows.append(
            {
                "window_id": window_id,
                "vid": window["vid"],
                "window_index": int(window["window_index"]),
                "scale_multiplier": scale,
                **metrics,
                **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
            }
        )
        print(
            f"[{index}/{len(windows)}] {window_id} x{scale:.6g} "
            f"collision={metrics['collision_ratio_pct']:.3f}% "
            f"penetrate={metrics['penetrate_cm']:.3f}cm float={metrics['float_cm']:.3f}cm",
            flush=True,
        )

    adjusted_summary = summarize_frames(adjusted_frames, ground_config.tolerance_m)
    baseline_summary = summarize_frames(baseline_frames, ground_config.tolerance_m)
    summary = {
        "experiment": "RICH test-9 cached scale upper-bound diagnostic",
        "official_model_result": False,
        "mode": args.mode,
        "cache_manifest": str(manifest_path),
        "manual_scale_file": str(scale_path) if args.mode == "manual" else None,
        "window_count": len(windows),
        "saved_manual_windows": len(windows) - len(missing) if args.mode == "manual" else None,
        "adjusted": adjusted_summary,
        "baseline_from_same_cache": baseline_summary,
    }
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "per_window.csv", window_rows)
    write_csv(output_dir / "per_frame.csv", adjusted_frames)
    write_json(
        output_dir / ("oracle_scales.json" if args.mode == "oracle_grid" else "applied_scales.json"),
        {
            "format": SCALE_FILE_FORMAT,
            "mode": args.mode,
            "cache_manifest": str(manifest_path),
            "windows": chosen_scales,
        },
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[output] {output_dir}", flush=True)


def choose_oracle_scale(
    cache: dict[str, np.ndarray],
    factors: np.ndarray,
    config: GroundEstimatorConfig,
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    best: tuple[float, float, list[dict[str, Any]], dict[str, Any]] | None = None
    frame_count = int(cache["depth"].shape[0])
    for factor in factors:
        rows, report = evaluate_scale_window_cache(cache, float(factor), config)
        metrics = report.get("metrics")
        if metrics is None:
            score = float("inf")
        else:
            invalid_ratio = 1.0 - float(metrics["valid_frames"]) / max(frame_count, 1)
            score = float(metrics["mean_abs_clearance_cm"]) + 100.0 * invalid_ratio
        candidate = (score, float(factor), rows, report)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return best[1], best[2], best[3]


def attach_frame_metadata(rows: list[dict[str, Any]], window: dict[str, Any], scale: float) -> None:
    for index, row in enumerate(rows):
        row.update(
            {
                "window_id": window["window_id"],
                "vid": window["vid"],
                "window_index": int(window["window_index"]),
                "window_frame_index": index,
                "source_frame_id": int(window["source_frame_ids"][index]),
                "image": window["image_paths"][index],
                "scale_multiplier": float(scale),
            }
        )


def summarize_frames(rows: list[dict[str, Any]], tolerance_m: float) -> dict[str, Any]:
    clearances = [float(row["clearance_m"]) for row in rows if row.get("valid") and "clearance_m" in row]
    pooled = compute_physical_metrics(clearances, tolerance_m=tolerance_m)
    vids = sorted({str(row["vid"]) for row in rows})
    per_sequence = []
    for vid in vids:
        values = [
            float(row["clearance_m"])
            for row in rows
            if row["vid"] == vid and row.get("valid") and "clearance_m" in row
        ]
        per_sequence.append({"vid": vid, **compute_physical_metrics(values, tolerance_m=tolerance_m)})
    mean_sequences = {
        key: float(np.mean([float(row[key]) for row in per_sequence])) if per_sequence else 0.0
        for key in METRIC_KEYS
    }
    return {
        "selected_frames": len(rows),
        "invalid_frames": sum(not bool(row.get("valid")) for row in rows),
        "pooled_frames": pooled,
        "mean_of_sequences": mean_sequences,
        "per_sequence": per_sequence,
    }


def load_scale_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Manual scale file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != SCALE_FILE_FORMAT:
        raise ValueError(f"Unsupported manual scale file: {path}")
    return payload


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
