#!/usr/bin/env python
"""Scan BEDLAM training data for long-tail bad SMPL translation frames.

This script is intentionally a thin dataset-wide wrapper around
``evaluate_hsi_sequence_person_diagnostics.py``.  It reuses the same model
loading, GT-box-prior matching, SMPL decoding, and per-person metrics, then
adds bad-frame flags and aggregate CSV/JSON summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.evaluate_hsi_refine_metrics import (  # noqa: E402
    describe_array,
    fmt,
    load_config,
    load_training_checkpoint,
    load_vggt_baseline,
    move_to_device,
)
from scripts.eval.evaluate_hsi_sequence_person_diagnostics import (  # noqa: E402
    FRAME_FIELDS,
    evaluate_batch,
    frame_paths_for_dataset_index,
    make_bedlam_dataset,
    write_csv,
)
from scripts.train.train_smpl import build_model  # noqa: E402
from vggt_omega.data import bedlam_collate_fn  # noqa: E402
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.training.config import require_path  # noqa: E402


def main() -> None:
    args = parse_args()
    if args.image:
        raise ValueError("scan_hsi_bad_translation_frames.py is a dataset-wide scanner; leave --image empty")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args)
    config.setdefault("data", {})
    config["data"]["sequence_length"] = int(args.num_frames)
    config["data"]["stride"] = int(args.frame_stride)
    config["data"]["require_depth"] = True
    config["data"]["require_boxes"] = True
    config.setdefault("model", {})
    config["model"]["enable_camera"] = True
    config["model"]["enable_depth"] = True
    config["model"]["enable_hsi_refine"] = True

    model = build_model(config).to(device)
    load_vggt_baseline(model, config, device)
    load_training_checkpoint(model, Path(args.checkpoint), device)
    model.eval()

    dataset = make_bedlam_dataset(config, args)
    selected_indices = select_dataset_indices(len(dataset), args)
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(config["data"].get("pin_memory", True)),
        collate_fn=bedlam_collate_fn,
        drop_last=False,
    )
    smpl = SMPLLayer(require_path(config, "assets.smpl_model_dir", allow_empty=False)).to(device).eval()

    frame_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    scales: list[float] = []
    biases: list[float] = []

    processed = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = int(batch["images"].shape[0])
            batch_indices = selected_indices[processed : processed + batch_size]
            batch_frame_paths = [frame_paths_for_dataset_index(dataset, index) for index in batch_indices]
            batch = move_to_device(batch, device)
            predictions = model(
                batch["images"],
                smpl_query_boxes=batch["gt_boxes"] if args.use_gt_box_prior else None,
                smpl_query_boxes_mask=batch["boxes_mask"] if args.use_gt_box_prior else None,
                smpl_track_ids=batch.get("gt_track_ids", batch.get("person_ids")),
                smpl_track_mask=batch.get("gt_track_mask", batch.get("person_id_mask")),
            )
            batch_frame_rows, batch_depth_rows, _ = evaluate_batch(
                predictions,
                batch,
                smpl,
                config,
                args,
                batch_indices,
                batch_frame_paths,
            )
            frame_rows.extend(batch_frame_rows)
            depth_rows.extend(batch_depth_rows)
            if "hsi_scene_scale" in predictions:
                scales.extend(predictions["hsi_scene_scale"].detach().float().cpu().reshape(-1).tolist())
            if "hsi_scene_depth_bias" in predictions:
                biases.extend(predictions["hsi_scene_depth_bias"].detach().float().cpu().reshape(-1).tolist())
            processed += batch_size
            if args.log_interval > 0 and (processed % args.log_interval == 0 or processed >= len(selected_indices)):
                print(f"[bad-translation-scan] processed={processed}/{len(selected_indices)} rows={len(frame_rows)}")

    if args.dedupe_frame_person:
        frame_rows = dedupe_frame_person_rows(frame_rows)
    flagged_rows = [add_bad_flags(row, args) for row in frame_rows]
    bad_rows = [row for row in flagged_rows if is_interesting_bad_row(row)]
    frame_summary = build_frame_summary(flagged_rows)
    bad_frame_summary = [row for row in frame_summary if int(row["base_bad_count"]) > 0 or int(row["hsi_bad_count"]) > 0]
    sequence_summary = build_sequence_summary(flagged_rows)
    bad_sequence_summary = [
        row for row in sequence_summary if int(row["base_bad_count"]) > 0 or int(row["hsi_bad_count"]) > 0
    ]
    global_summary = build_global_summary(
        flagged_rows,
        bad_rows,
        frame_summary,
        bad_frame_summary,
        sequence_summary,
        bad_sequence_summary,
        depth_rows,
        scales,
        biases,
        args,
        selected_indices,
    )

    all_rows_csv = output_dir / "all_frame_person_translation_rows.csv"
    bad_rows_csv = output_dir / "bad_frame_person_rows.csv"
    frame_csv = output_dir / "bad_frame_summary.csv"
    sequence_csv = output_dir / "bad_sequence_summary.csv"
    write_csv(all_rows_csv, flagged_rows, BAD_FRAME_PERSON_FIELDS)
    write_csv(bad_rows_csv, bad_rows, BAD_FRAME_PERSON_FIELDS)
    write_csv(frame_csv, bad_frame_summary, BAD_FRAME_FIELDS)
    write_csv(sequence_csv, bad_sequence_summary, BAD_SEQUENCE_FIELDS)

    out_json = output_dir / "hsi_bad_translation_scan_summary.json"
    payload = {
        "checkpoint": str(args.checkpoint),
        "train_config": str(args.train_config),
        "split": str(args.split),
        "num_frames": int(args.num_frames),
        "frame_stride": int(args.frame_stride),
        "use_gt_box_prior": bool(args.use_gt_box_prior),
        "match_source": str(args.match_source),
        "intrinsics_source": str(args.intrinsics_source),
        "selected_dataset_indices": [int(index) for index in selected_indices],
        "global_summary": global_summary,
        "top_bad_base_translation_rows": top_rows(bad_rows, "base_transl_l2_m", int(args.top_k)),
        "top_bad_hsi_translation_rows": top_rows(bad_rows, "hsi_transl_l2_m", int(args.top_k)),
        "top_bad_base_mpjpe_rows": top_rows(bad_rows, "base_mpjpe_m", int(args.top_k)),
        "outputs": {
            "all_frame_person_rows_csv": str(all_rows_csv),
            "bad_frame_person_rows_csv": str(bad_rows_csv),
            "bad_frame_summary_csv": str(frame_csv),
            "bad_sequence_summary_csv": str(sequence_csv),
            "json": str(out_json),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print_summary(payload)
    print(json.dumps({"output_json": str(out_json), "num_bad_frame_person_rows": len(bad_rows)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan training data for bad SMPL translation frames")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_hsi_refine.yaml")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--output-dir", default="outputs/eval/hsi_bad_translation_scan")
    parser.add_argument("--device", default="")
    parser.add_argument("--split", default="Training")
    parser.add_argument("--image", default="", help="Optional start RGB image; normally unused for full scan")
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means scan all dataset windows")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--conf-threshold", type=float, default=0.10)
    parser.add_argument("--use-gt-box-prior", action="store_true")
    parser.add_argument("--match-source", choices=["base", "hsi", "best"], default="base")
    parser.add_argument("--intrinsics-source", choices=["gt", "vggt"], default="gt")
    parser.add_argument("--depth-max-m", type=float, default=30.0)
    parser.add_argument("--roi-expand", type=float, default=0.75)
    parser.add_argument("--foot-contact-threshold-m", type=float, default=0.12)
    parser.add_argument("--foot-sole-contact-threshold-m", type=float, default=0.08)
    parser.add_argument("--foot-float-margin-m", type=float, default=0.04)
    parser.add_argument("--foot-penetration-margin-m", type=float, default=0.02)
    parser.add_argument("--foot-sole-num-vertices", type=int, default=80)
    parser.add_argument("--support-plane-window", type=int, default=9)
    parser.add_argument("--support-plane-min-points", type=int, default=6)
    parser.add_argument("--smpl-bad-mpjpe-m", type=float, default=0.08)
    parser.add_argument("--smpl-bad-transl-m", type=float, default=0.08)
    parser.add_argument("--depth-bad-human-roi-m", type=float, default=0.35)
    parser.add_argument("--contact-bad-m", type=float, default=0.05)
    parser.add_argument("--bad-base-transl-m", type=float, default=0.50)
    parser.add_argument("--severe-base-transl-m", type=float, default=0.80)
    parser.add_argument("--bad-hsi-transl-m", type=float, default=0.50)
    parser.add_argument("--severe-hsi-transl-m", type=float, default=0.80)
    parser.add_argument("--bad-base-mpjpe-m", type=float, default=0.50)
    parser.add_argument("--severe-base-mpjpe-m", type=float, default=0.80)
    parser.add_argument("--bad-hsi-mpjpe-m", type=float, default=0.50)
    parser.add_argument("--severe-hsi-mpjpe-m", type=float, default=0.80)
    parser.add_argument("--hsi-worse-margin-m", type=float, default=0.05)
    parser.add_argument("--dedupe-frame-person", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def select_dataset_indices(dataset_len: int, args: argparse.Namespace) -> list[int]:
    start = max(int(args.start_index), 0)
    if start >= dataset_len:
        raise ValueError(f"start-index={start} is outside dataset length {dataset_len}")
    max_samples = int(args.max_samples)
    end = dataset_len if max_samples <= 0 else min(dataset_len, start + max_samples)
    return list(range(start, end))


def dedupe_frame_person_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("sequence_name")), str(row.get("frame_name")), int(row.get("track_id", -1)))
        current = by_key.get(key)
        if current is None or row_bad_score(row) > row_bad_score(current):
            by_key[key] = row
    return list(by_key.values())


def row_bad_score(row: dict[str, Any]) -> float:
    values = [
        as_float(row.get("base_transl_l2_m")),
        as_float(row.get("hsi_transl_l2_m")),
        as_float(row.get("base_mpjpe_m")),
        as_float(row.get("hsi_mpjpe_m")),
    ]
    finite = [value for value in values if value is not None]
    return max(finite) if finite else -1.0


def add_bad_flags(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(row)
    base_transl = as_float(out.get("base_transl_l2_m"))
    hsi_transl = as_float(out.get("hsi_transl_l2_m"))
    base_mpjpe = as_float(out.get("base_mpjpe_m"))
    hsi_mpjpe = as_float(out.get("hsi_mpjpe_m"))

    out["is_base_transl_bad"] = int(is_at_least(base_transl, args.bad_base_transl_m))
    out["is_base_transl_severe"] = int(is_at_least(base_transl, args.severe_base_transl_m))
    out["is_hsi_transl_bad"] = int(is_at_least(hsi_transl, args.bad_hsi_transl_m))
    out["is_hsi_transl_severe"] = int(is_at_least(hsi_transl, args.severe_hsi_transl_m))
    out["is_base_mpjpe_bad"] = int(is_at_least(base_mpjpe, args.bad_base_mpjpe_m))
    out["is_base_mpjpe_severe"] = int(is_at_least(base_mpjpe, args.severe_base_mpjpe_m))
    out["is_hsi_mpjpe_bad"] = int(is_at_least(hsi_mpjpe, args.bad_hsi_mpjpe_m))
    out["is_hsi_mpjpe_severe"] = int(is_at_least(hsi_mpjpe, args.severe_hsi_mpjpe_m))
    out["is_base_bad"] = int(bool(out["is_base_transl_bad"] or out["is_base_mpjpe_bad"]))
    out["is_base_severe"] = int(bool(out["is_base_transl_severe"] or out["is_base_mpjpe_severe"]))
    out["is_hsi_bad"] = int(bool(out["is_hsi_transl_bad"] or out["is_hsi_mpjpe_bad"]))
    out["is_hsi_severe"] = int(bool(out["is_hsi_transl_severe"] or out["is_hsi_mpjpe_severe"]))

    transl_worse = (
        base_transl is not None
        and hsi_transl is not None
        and hsi_transl > base_transl + float(args.hsi_worse_margin_m)
    )
    mpjpe_worse = base_mpjpe is not None and hsi_mpjpe is not None and hsi_mpjpe > base_mpjpe + float(args.hsi_worse_margin_m)
    out["is_hsi_transl_worse"] = int(transl_worse)
    out["is_hsi_mpjpe_worse"] = int(mpjpe_worse)
    out["is_hsi_worse"] = int(transl_worse or mpjpe_worse)
    out["is_hsi_rescued"] = int(bool(out["is_base_bad"] and not out["is_hsi_bad"]))
    out["is_both_bad"] = int(bool(out["is_base_bad"] and out["is_hsi_bad"]))
    out["scan_reason"] = build_scan_reason(out)
    return out


def build_scan_reason(row: dict[str, Any]) -> str:
    reasons = []
    if row.get("is_base_transl_severe"):
        reasons.append("base_transl_severe")
    elif row.get("is_base_transl_bad"):
        reasons.append("base_transl_bad")
    if row.get("is_base_mpjpe_severe"):
        reasons.append("base_mpjpe_severe")
    elif row.get("is_base_mpjpe_bad"):
        reasons.append("base_mpjpe_bad")
    if row.get("is_hsi_transl_severe"):
        reasons.append("hsi_transl_severe")
    elif row.get("is_hsi_transl_bad"):
        reasons.append("hsi_transl_bad")
    if row.get("is_hsi_worse"):
        reasons.append("hsi_worse")
    return "+".join(reasons) if reasons else "ok"


def is_interesting_bad_row(row: dict[str, Any]) -> bool:
    return bool(row.get("is_base_bad") or row.get("is_hsi_bad") or row.get("is_hsi_worse"))


def build_frame_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("sequence_name")), str(row.get("frame_name")))].append(row)
    out = []
    for (sequence_name, frame_name), group in grouped.items():
        worst_base = max(group, key=lambda row: row_bad_value(row, "base_transl_l2_m"))
        worst_hsi = max(group, key=lambda row: row_bad_value(row, "hsi_transl_l2_m"))
        out.append(
            {
                "sequence_name": sequence_name,
                "frame_name": frame_name,
                "image_path": str(worst_base.get("image_path", "")),
                "num_people": len(group),
                "base_bad_count": count_flag(group, "is_base_bad"),
                "base_severe_count": count_flag(group, "is_base_severe"),
                "hsi_bad_count": count_flag(group, "is_hsi_bad"),
                "hsi_severe_count": count_flag(group, "is_hsi_severe"),
                "hsi_rescued_count": count_flag(group, "is_hsi_rescued"),
                "hsi_worse_count": count_flag(group, "is_hsi_worse"),
                "both_bad_count": count_flag(group, "is_both_bad"),
                "max_base_transl_l2_m": max_value(group, "base_transl_l2_m"),
                "max_hsi_transl_l2_m": max_value(group, "hsi_transl_l2_m"),
                "max_base_mpjpe_m": max_value(group, "base_mpjpe_m"),
                "max_hsi_mpjpe_m": max_value(group, "hsi_mpjpe_m"),
                "worst_base_track_id": int(worst_base.get("track_id", -1)),
                "worst_hsi_track_id": int(worst_hsi.get("track_id", -1)),
                "worst_base_reason": str(worst_base.get("scan_reason", "")),
                "worst_hsi_reason": str(worst_hsi.get("scan_reason", "")),
            }
        )
    return sorted(out, key=lambda row: (int(row["base_bad_count"]) + int(row["hsi_bad_count"]), as_sort(row["max_base_transl_l2_m"])), reverse=True)


def build_sequence_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("sequence_name"))].append(row)
    out = []
    for sequence_name, group in grouped.items():
        frames = {(str(row.get("sequence_name")), str(row.get("frame_name"))) for row in group}
        worst_base = max(group, key=lambda row: row_bad_value(row, "base_transl_l2_m"))
        worst_hsi = max(group, key=lambda row: row_bad_value(row, "hsi_transl_l2_m"))
        out.append(
            {
                "sequence_name": sequence_name,
                "num_frames": len(frames),
                "num_frame_person_rows": len(group),
                "base_bad_count": count_flag(group, "is_base_bad"),
                "base_severe_count": count_flag(group, "is_base_severe"),
                "hsi_bad_count": count_flag(group, "is_hsi_bad"),
                "hsi_severe_count": count_flag(group, "is_hsi_severe"),
                "hsi_rescued_count": count_flag(group, "is_hsi_rescued"),
                "hsi_worse_count": count_flag(group, "is_hsi_worse"),
                "both_bad_count": count_flag(group, "is_both_bad"),
                "max_base_transl_l2_m": max_value(group, "base_transl_l2_m"),
                "max_hsi_transl_l2_m": max_value(group, "hsi_transl_l2_m"),
                "max_base_mpjpe_m": max_value(group, "base_mpjpe_m"),
                "max_hsi_mpjpe_m": max_value(group, "hsi_mpjpe_m"),
                "worst_base_frame_name": str(worst_base.get("frame_name", "")),
                "worst_base_image_path": str(worst_base.get("image_path", "")),
                "worst_base_track_id": int(worst_base.get("track_id", -1)),
                "worst_hsi_frame_name": str(worst_hsi.get("frame_name", "")),
                "worst_hsi_image_path": str(worst_hsi.get("image_path", "")),
                "worst_hsi_track_id": int(worst_hsi.get("track_id", -1)),
            }
        )
    return sorted(out, key=lambda row: (int(row["base_bad_count"]) + int(row["hsi_bad_count"]), as_sort(row["max_base_transl_l2_m"])), reverse=True)


def build_global_summary(
    rows: list[dict[str, Any]],
    bad_rows: list[dict[str, Any]],
    frame_summary: list[dict[str, Any]],
    bad_frame_summary: list[dict[str, Any]],
    sequence_summary: list[dict[str, Any]],
    bad_sequence_summary: list[dict[str, Any]],
    depth_rows: list[dict[str, Any]],
    scales: list[float],
    biases: list[float],
    args: argparse.Namespace,
    selected_indices: list[int],
) -> dict[str, Any]:
    num_rows = len(rows)
    num_frames = len(frame_summary)
    num_sequences = len(sequence_summary)
    return {
        "num_dataset_windows": len(selected_indices),
        "num_frame_person_rows": num_rows,
        "num_bad_frame_person_rows": len(bad_rows),
        "num_frames": num_frames,
        "num_bad_frames": len(bad_frame_summary),
        "num_sequences": num_sequences,
        "num_bad_sequences": len(bad_sequence_summary),
        "base_bad_count": count_flag(rows, "is_base_bad"),
        "base_severe_count": count_flag(rows, "is_base_severe"),
        "hsi_bad_count": count_flag(rows, "is_hsi_bad"),
        "hsi_severe_count": count_flag(rows, "is_hsi_severe"),
        "hsi_rescued_count": count_flag(rows, "is_hsi_rescued"),
        "hsi_worse_count": count_flag(rows, "is_hsi_worse"),
        "both_bad_count": count_flag(rows, "is_both_bad"),
        "base_bad_ratio": ratio(count_flag(rows, "is_base_bad"), num_rows),
        "hsi_bad_ratio": ratio(count_flag(rows, "is_hsi_bad"), num_rows),
        "bad_frame_ratio": ratio(len(bad_frame_summary), num_frames),
        "bad_sequence_ratio": ratio(len(bad_sequence_summary), num_sequences),
        "base_transl_l2_m": describe_metric(rows, "base_transl_l2_m"),
        "hsi_transl_l2_m": describe_metric(rows, "hsi_transl_l2_m"),
        "base_mpjpe_m": describe_metric(rows, "base_mpjpe_m"),
        "hsi_mpjpe_m": describe_metric(rows, "hsi_mpjpe_m"),
        "hsi_scene_scale": describe_array(np.asarray(scales, dtype=np.float64)),
        "hsi_scene_depth_bias": describe_array(np.asarray(biases, dtype=np.float64)),
        "num_frame_depth_rows": len(depth_rows),
        "thresholds": {
            "bad_base_transl_m": float(args.bad_base_transl_m),
            "severe_base_transl_m": float(args.severe_base_transl_m),
            "bad_hsi_transl_m": float(args.bad_hsi_transl_m),
            "severe_hsi_transl_m": float(args.severe_hsi_transl_m),
            "bad_base_mpjpe_m": float(args.bad_base_mpjpe_m),
            "severe_base_mpjpe_m": float(args.severe_base_mpjpe_m),
            "bad_hsi_mpjpe_m": float(args.bad_hsi_mpjpe_m),
            "severe_hsi_mpjpe_m": float(args.severe_hsi_mpjpe_m),
            "hsi_worse_margin_m": float(args.hsi_worse_margin_m),
        },
    }


def describe_metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [as_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return describe_array(np.asarray(values, dtype=np.float64))


def top_rows(rows: list[dict[str, Any]], key: str, limit: int) -> list[dict[str, Any]]:
    selected = sorted(rows, key=lambda row: row_bad_value(row, key), reverse=True)[: max(limit, 0)]
    keep = [
        "sequence_name",
        "frame_name",
        "image_path",
        "track_id",
        "scan_reason",
        "base_transl_l2_m",
        "hsi_transl_l2_m",
        "base_mpjpe_m",
        "hsi_mpjpe_m",
        "pred_conf",
        "query_idx",
        "hsi_scene_scale",
        "hsi_scene_depth_bias_m",
    ]
    return [{name: row.get(name) for name in keep} for row in selected]


def count_flag(rows: list[dict[str, Any]], key: str) -> int:
    return int(sum(1 for row in rows if bool(row.get(key))))


def max_value(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [as_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def row_bad_value(row: dict[str, Any], key: str) -> float:
    value = as_float(row.get(key))
    return value if value is not None else -1.0


def as_sort(value: Any) -> float:
    parsed = as_float(value)
    return parsed if parsed is not None else -1.0


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def is_at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value >= float(threshold)


def ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator) / float(denominator) if denominator > 0 else None


def print_summary(payload: dict[str, Any]) -> None:
    summary = payload["global_summary"]
    print("========== HSI bad translation scan ==========")
    print(
        "Rows                    "
        f"windows={summary['num_dataset_windows']} "
        f"frame_person={summary['num_frame_person_rows']} "
        f"bad_rows={summary['num_bad_frame_person_rows']}"
    )
    print(
        "Bad counts              "
        f"base={summary['base_bad_count']} severe={summary['base_severe_count']} "
        f"hsi={summary['hsi_bad_count']} severe={summary['hsi_severe_count']} "
        f"rescued={summary['hsi_rescued_count']} worse={summary['hsi_worse_count']}"
    )
    print(
        "Bad ratios              "
        f"base={fmt(summary['base_bad_ratio'])} "
        f"hsi={fmt(summary['hsi_bad_ratio'])} "
        f"frames={fmt(summary['bad_frame_ratio'])} "
        f"sequences={fmt(summary['bad_sequence_ratio'])}"
    )
    print(
        "Transl L2 median/max    "
        f"base={fmt(summary['base_transl_l2_m'].get('median'))}/{fmt(summary['base_transl_l2_m'].get('max'))}m "
        f"hsi={fmt(summary['hsi_transl_l2_m'].get('median'))}/{fmt(summary['hsi_transl_l2_m'].get('max'))}m"
    )
    print(
        "MPJPE median/max        "
        f"base={fmt(summary['base_mpjpe_m'].get('median'))}/{fmt(summary['base_mpjpe_m'].get('max'))}m "
        f"hsi={fmt(summary['hsi_mpjpe_m'].get('median'))}/{fmt(summary['hsi_mpjpe_m'].get('max'))}m"
    )
    print("Top bad base translation rows:")
    for row in payload["top_bad_base_translation_rows"][:10]:
        print(
            f"  {row.get('sequence_name')}/{row.get('frame_name')} "
            f"track={row.get('track_id')} "
            f"base={fmt(as_float(row.get('base_transl_l2_m')))}m "
            f"hsi={fmt(as_float(row.get('hsi_transl_l2_m')))}m "
            f"reason={row.get('scan_reason')}"
        )


BAD_FLAG_FIELDS = [
    "is_base_transl_bad",
    "is_base_transl_severe",
    "is_hsi_transl_bad",
    "is_hsi_transl_severe",
    "is_base_mpjpe_bad",
    "is_base_mpjpe_severe",
    "is_hsi_mpjpe_bad",
    "is_hsi_mpjpe_severe",
    "is_base_bad",
    "is_base_severe",
    "is_hsi_bad",
    "is_hsi_severe",
    "is_hsi_transl_worse",
    "is_hsi_mpjpe_worse",
    "is_hsi_worse",
    "is_hsi_rescued",
    "is_both_bad",
    "scan_reason",
]

BAD_FRAME_PERSON_FIELDS = [*FRAME_FIELDS, *BAD_FLAG_FIELDS]

BAD_FRAME_FIELDS = [
    "sequence_name",
    "frame_name",
    "image_path",
    "num_people",
    "base_bad_count",
    "base_severe_count",
    "hsi_bad_count",
    "hsi_severe_count",
    "hsi_rescued_count",
    "hsi_worse_count",
    "both_bad_count",
    "max_base_transl_l2_m",
    "max_hsi_transl_l2_m",
    "max_base_mpjpe_m",
    "max_hsi_mpjpe_m",
    "worst_base_track_id",
    "worst_hsi_track_id",
    "worst_base_reason",
    "worst_hsi_reason",
]

BAD_SEQUENCE_FIELDS = [
    "sequence_name",
    "num_frames",
    "num_frame_person_rows",
    "base_bad_count",
    "base_severe_count",
    "hsi_bad_count",
    "hsi_severe_count",
    "hsi_rescued_count",
    "hsi_worse_count",
    "both_bad_count",
    "max_base_transl_l2_m",
    "max_hsi_transl_l2_m",
    "max_base_mpjpe_m",
    "max_hsi_mpjpe_m",
    "worst_base_frame_name",
    "worst_base_image_path",
    "worst_base_track_id",
    "worst_hsi_frame_name",
    "worst_hsi_image_path",
    "worst_hsi_track_id",
]


if __name__ == "__main__":
    main()
