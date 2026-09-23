#!/usr/bin/env python3
"""Re-evaluate cached 3DPW SHOW samples after manual per-sequence scaling."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.evaluate_show_human_scene_3dpw import resolve_output_dir  # noqa: E402
from vggt_omega.evaluation import HumanSceneConsistencyConfig, compute_human_scene_consistency, render_mesh_silhouette  # noqa: E402


CACHE_FORMAT = "vggt_omega_show_3dpw_manual_scale_cache_v3"
SCALE_FILE_FORMAT = "vggt_omega_show_3dpw_manual_scale_selections_v3"


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != CACHE_FORMAT:
        raise ValueError(f"Unsupported cache manifest: {manifest_path}")
    windows = list(manifest.get("sequences", []))
    if not windows:
        raise ValueError("Cache contains no sequences")
    scale_file = Path(args.scale_file).expanduser().resolve() if args.scale_file else cache_dir / "manual_scales.json"
    scales = load_scales(scale_file, windows, mode=args.scale_mode, allow_missing=bool(args.allow_missing_manual_scales))
    metric_config = HumanSceneConsistencyConfig(**manifest["metric_config"])
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    faces = torch.from_numpy(np.load(cache_dir / str(manifest["faces_file"]), allow_pickle=False).astype(np.int64)).to(device)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for window in windows:
        with (cache_dir / str(window["cache_file"])).open("rb") as file:
            frames = pickle.load(file)
        scale = float(scales[str(window["sequence_id"])])
        values: list[dict[str, Any]] = []
        evaluated_count = sum(bool(frame.get("eval_valid", True)) for frame in frames)
        match_valid_count = sum(bool(frame.get("match_valid", True)) for frame in frames)
        gt_eval_count = sum(bool(frame.get("gt_eval_valid", True)) for frame in frames)
        original_count = int(window.get("original_frame_count", len(frames)))
        expansion_weight = float(original_count) / float(max(evaluated_count, 1))
        for frame in frames:
            if not bool(frame.get("eval_valid", True)):
                continue
            pred = torch.from_numpy(np.asarray(frame["pred_vertices_cam"], dtype=np.float32)).to(device)
            gt = torch.from_numpy(np.asarray(frame["gt_vertices_cam"], dtype=np.float32)).to(device)
            gt_k = torch.from_numpy(np.asarray(frame["gt_intrinsics"], dtype=np.float32)).to(device)
            pred_k = torch.from_numpy(np.asarray(frame["pred_intrinsics"], dtype=np.float32)).to(device)
            depth = torch.from_numpy(np.asarray(frame["scene_depth"], dtype=np.float32)).to(device) * scale
            valid = torch.from_numpy(np.asarray(frame["scene_valid_mask"], dtype=np.uint8)).bool().to(device)
            human_mask = render_mesh_silhouette(gt, gt_k, image_hw=tuple(depth.shape), faces=faces, backend=metric_config.visibility_backend)
            metric = compute_human_scene_consistency(pred, depth, pred_k, human_mask, faces=faces, scene_valid_mask=valid, config=metric_config)
            values.append(metric)
            frame_rows.append({
                "sequence_id": window["sequence_id"], "vid": window["vid"], "sequence_index": window["sequence_index"],
                "source_frame_id": frame["source_frame_id"], "scale_multiplier": scale,
                "original_frame_count": original_count, "sampled_frame_count": len(frames),
                "sample_expansion_weight": expansion_weight, **metric,
            })
        means = mean_metrics(values)
        window_rows.append({
            "sequence_id": window["sequence_id"], "vid": window["vid"], "sequence_index": window["sequence_index"],
            "sampled_frame_count": len(frames), "evaluated_sampled_frame_count": evaluated_count,
            "match_valid_frame_count": match_valid_count,
            "association_invalid_frame_count": len(frames) - match_valid_count,
            "gt_eval_valid_frame_count": gt_eval_count,
            "original_frame_count": original_count, "sequence_weight": original_count,
            "sample_expansion_weight": expansion_weight, "scale_multiplier": scale, **means,
        })

    write_csv(output_dir / "per_frame.csv", frame_rows)
    write_csv(output_dir / "per_sequence.csv", window_rows)
    (output_dir / "applied_scales.json").write_text(json.dumps({"format": SCALE_FILE_FORMAT, "mode": args.scale_mode, "sequences": scales}, indent=2), encoding="utf-8")
    summary = {
        "dataset": "3dpw", "cache_manifest": str(manifest_path), "scale_file": str(scale_file),
        "scale_mode": args.scale_mode, "scale_scope": "one shared manual multiplier per sequence",
        "sampling_protocol": manifest.get("sampling_protocol"),
        "aggregation_protocol": "sequence sampled means weighted by original sequence length N",
        "aggregation_formula": "sum_i(original_frame_count_i * sampled_sequence_mean_i) / sum_i(original_frame_count_i)",
        "sample_frames_per_sequence": manifest.get("sample_frames_per_sequence", 200), "num_sequences": len(windows),
        "num_sampled_frames": sum(int(row["sampled_frame_count"]) for row in window_rows),
        "num_evaluated_sampled_frames": len(frame_rows),
        "effective_original_frames": sum(int(row["original_frame_count"]) for row in window_rows),
        "valid_sampled_frames": sum(bool(row.get("valid", False)) for row in frame_rows),
        "association_invalid_sampled_frames": sum(int(row["association_invalid_frame_count"]) for row in window_rows),
        "mask_source": manifest.get("mask_source"), "branch": manifest.get("branch"),
        "visibility_backend": metric_config.visibility_backend,
        "table3": {key: weighted_sequence_metric(window_rows, key) for key in ("hs_v5", "hs_v10", "hs_cf5", "hs_cf10")},
        "weighted_means": {key: weighted_sequence_metric(window_rows, key) for key in ("hs_v5", "hs_v10", "hs_cf5", "hs_cf10")},
        "unweighted_sampled_frame_means": {key: mean_metric(frame_rows, key) for key in ("hs_v5", "hs_v10", "hs_cf5", "hs_cf10")},
        "files": {"per_frame": str(output_dir / "per_frame.csv"), "per_sequence": str(output_dir / "per_sequence.csv"), "applied_scales": str(output_dir / "applied_scales.json")},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def load_scales(path: Path, windows: list[dict[str, Any]], *, mode: str, allow_missing: bool) -> dict[str, float]:
    if mode == "base":
        return {str(row["sequence_id"]): 1.0 for row in windows}
    if not path.is_file():
        raise FileNotFoundError(f"Manual scale file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != SCALE_FILE_FORMAT:
        raise ValueError(f"Unsupported scale file: {path}")
    records = payload.get("sequences", {})
    out: dict[str, float] = {}
    missing = []
    for row in windows:
        key = str(row["sequence_id"])
        record = records.get(key)
        if record is None:
            missing.append(key)
            out[key] = 1.0
        else:
            value = float(record.get("scale_multiplier", 1.0))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid scale for {key}: {value}")
            out[key] = value
    if missing and not allow_missing:
        raise ValueError(f"Manual scales are missing for {len(missing)} sequences; first={missing[:3]}. Finish Viser annotation or pass --allow-missing-manual-scales for a diagnostic run.")
    return out


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    return float(np.mean(vals)) if vals else 0.0


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {key: mean_metric(rows, key) for key in ("hs_v5", "hs_v10", "hs_cf5", "hs_cf10")}


def weighted_sequence_metric(rows: list[dict[str, Any]], key: str) -> float:
    pairs = [
        (float(row[key]), float(row["original_frame_count"]))
        for row in rows
        if key in row and np.isfinite(float(row[key])) and float(row["original_frame_count"]) > 0
    ]
    if not pairs:
        return 0.0
    numerator = sum(value * weight for value, weight in pairs)
    denominator = sum(weight for _, weight in pairs)
    return float(numerator / denominator)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["sequence_id"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--scale-file", default="")
    parser.add_argument("--output-dir", default="outputs/eval/show_3dpw_manual_scale/metrics")
    parser.add_argument("--device", default="")
    parser.add_argument("--scale-mode", choices=("manual", "base"), default="manual")
    parser.add_argument("--allow-missing-manual-scales", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    main()
