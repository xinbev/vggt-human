from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_smpl import (
    apply_overrides,
    build_loader,
    build_model,
    forward_model,
    load_initial_checkpoint,
    load_overlay_checkpoint,
    move_to_device,
    set_seed,
)
from vggt_omega.training.config import deep_update, load_yaml_config


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.train_config))
    config = apply_overrides(config, args.override)
    set_seed(int(config.get("experiment", {}).get("seed", 42)))
    device = torch.device(args.device)
    loader = build_loader(config, split=config["data"]["val_split"], shuffle=False, role="val")
    model = build_model(config).to(device)
    load_initial_checkpoint(model, config, device)
    checkpoint_cfg = config.setdefault("checkpoint", {})
    checkpoint_cfg["overlay_prefixes"] = ["hsi_foot_contact_intent_head."]
    checkpoint_cfg["overlay_required_prefixes"] = ["hsi_foot_contact_intent_head."]
    load_overlay_checkpoint(model, args.checkpoint, device, config)
    model.eval()

    rows: list[dict[str, Any]] = []
    for batch_idx, batch in enumerate(loader):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        batch = move_to_device(batch, device, config=config)
        predictions = forward_model(model, batch, config, epoch=0)
        rows.extend(collect_rows(loader.dataset, batch, predictions, args.static_speed_m))
        if (batch_idx + 1) % 50 == 0:
            print(f"[intent-audit] batches={batch_idx + 1}/{len(loader)} rows={len(rows)}", flush=True)

    thresholds = sorted(set(parse_thresholds(args.thresholds)))
    sweep = [threshold_metrics(rows, threshold) for threshold in thresholds]
    exact_operating_points = exact_threshold_operating_points(rows)
    feasible = [
        item
        for item in exact_operating_points
        if item["recall"] >= args.min_recall
        and item["precision"] >= args.min_precision
        and item["false_positive_rate"] <= args.max_fpr
        and item["static_negative_false_positive_rate"] <= args.max_static_fpr
    ]
    selected = max(
        feasible,
        key=lambda item: (
            float(item["recall"]),
            float(item["precision"]),
            -float(item["false_positive_rate"]),
            float(item["threshold"]),
        ),
        default=None,
    )
    diagnostic_threshold = float(selected["threshold"]) if selected is not None else 0.5
    report = {
        "gate": "pass" if selected is not None else "fail",
        "implementation": "person_support_intent_threshold_audit_v1",
        "checkpoint": args.checkpoint,
        "num_rows": len(rows),
        "num_positive": sum(bool(row["label"]) for row in rows),
        "num_negative": sum(not bool(row["label"]) for row in rows),
        "temporal_coverage": sum(bool(row["temporal_valid"]) for row in rows) / max(len(rows), 1),
        "threshold_sweep": sweep,
        "num_exact_operating_points": len(exact_operating_points),
        "num_feasible_operating_points": len(feasible),
        "selected_operating_point": selected,
        "feasibility_limits": {
            "min_recall": args.min_recall,
            "min_precision": args.min_precision,
            "max_fpr": args.max_fpr,
            "max_static_fpr": args.max_static_fpr,
            "temporal_context_required": True,
        },
        "by_sequence": grouped_sequence_metrics(rows, threshold=diagnostic_threshold),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, output_dir / "intent_people.jsonl")
    (output_dir / "intent_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    false_positive_rows = sorted(
        [
            row
            for row in rows
            if row["temporal_valid"]
            and not row["label"]
            and float(row["probability"]) >= diagnostic_threshold
        ],
        key=lambda row: float(row["probability"]),
        reverse=True,
    )
    false_negative_rows = sorted(
        [
            row
            for row in rows
            if row["label"]
            and (
                not row["temporal_valid"]
                or float(row["probability"]) < diagnostic_threshold
            )
        ],
        key=lambda row: float(row["probability"]),
    )
    (output_dir / "highest_confidence_negatives.json").write_text(
        json.dumps(false_positive_rows[: args.max_failure_rows], indent=2), encoding="utf-8"
    )
    (output_dir / "lowest_confidence_positives.json").write_text(
        json.dumps(false_negative_rows[: args.max_failure_rows], indent=2), encoding="utf-8"
    )
    print(
        f"[intent-audit] gate={report['gate']} "
        f"selected_operating_point={report['selected_operating_point']}"
    )
    print(f"[intent-audit] report={output_dir / 'intent_audit.json'}")


def collect_rows(
    dataset: Any,
    batch: dict[str, torch.Tensor],
    predictions: dict[str, torch.Tensor],
    static_speed_m: float,
) -> list[dict[str, Any]]:
    probability = predictions["hsi_person_support_intent_probability"].detach().cpu().float()
    temporal_valid = predictions["hsi_person_support_intent_temporal_valid"].detach().cpu().bool()
    teacher_valid = batch["contact_teacher_valid"].detach().cpu().bool()
    contact = batch["contact_label"].detach().cpu().bool()
    speed = batch["contact_foot_velocity_m"].detach().cpu().float()
    provider_valid = predictions["gt_smpl_provider_mask"].detach().cpu().bool()
    dataset_indices = batch["dataset_index"].detach().cpu().long()
    track_ids = batch["gt_track_ids"].detach().cpu().long()
    center = probability.shape[1] // 2
    positive = (teacher_valid & contact).any(dim=-1)
    reliable_negative = teacher_valid.all(dim=-1) & ~contact.any(dim=-1)
    valid = provider_valid & (positive | reliable_negative)
    rows: list[dict[str, Any]] = []
    for batch_idx, person_idx in valid[:, center].nonzero(as_tuple=False).tolist():
        label = bool(positive[batch_idx, center, person_idx])
        sequence, sample_path = resolve_sequence_and_frame(dataset, int(dataset_indices[batch_idx]), center)
        foot_speed = speed[batch_idx, center, person_idx]
        rows.append(
            {
                "dataset_index": int(dataset_indices[batch_idx]),
                "person_slot": int(person_idx),
                "track_id": int(track_ids[batch_idx, center, person_idx]),
                "sequence": sequence,
                "sample_path": sample_path,
                "probability": float(probability[batch_idx, center, person_idx]),
                "label": label,
                "temporal_valid": bool(temporal_valid[batch_idx, center, person_idx]),
                "static_negative": (not label) and float(foot_speed.max()) <= static_speed_m,
                "foot_speed_m": [float(value) for value in foot_speed.tolist()],
                "teacher_valid": [bool(value) for value in teacher_valid[batch_idx, center, person_idx].tolist()],
                "teacher_contact": [bool(value) for value in contact[batch_idx, center, person_idx].tolist()],
            }
        )
    return rows


def threshold_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, float | int]:
    tp = fp = fn = tn = static_fp = static_count = 0
    for row in rows:
        predicted = bool(row["temporal_valid"]) and float(row["probability"]) >= threshold
        label = bool(row["label"])
        tp += int(predicted and label)
        fp += int(predicted and not label)
        fn += int(not predicted and label)
        tn += int(not predicted and not label)
        if bool(row["static_negative"]):
            static_count += 1
            static_fp += int(predicted)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "static_negative_false_positive_rate": static_fp / max(static_count, 1),
    }


def exact_threshold_operating_points(rows: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    """Evaluate every distinct executable probability in O(N log N)."""
    executable = sorted(
        [row for row in rows if bool(row["temporal_valid"])],
        key=lambda row: float(row["probability"]),
        reverse=True,
    )
    total_positive = sum(bool(row["label"]) for row in rows)
    total_negative = len(rows) - total_positive
    static_count = sum(bool(row["static_negative"]) for row in rows)
    tp = fp = static_fp = 0
    output: list[dict[str, float | int]] = []
    index = 0
    while index < len(executable):
        threshold = float(executable[index]["probability"])
        end = index
        while end < len(executable) and float(executable[end]["probability"]) == threshold:
            row = executable[end]
            if bool(row["label"]):
                tp += 1
            else:
                fp += 1
                static_fp += int(bool(row["static_negative"]))
            end += 1
        fn = total_positive - tp
        tn = total_negative - fp
        output.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "recall": tp / max(total_positive, 1),
                "precision": tp / max(tp + fp, 1),
                "false_positive_rate": fp / max(total_negative, 1),
                "static_negative_false_positive_rate": static_fp / max(static_count, 1),
            }
        )
        index = end
    return output


def grouped_sequence_metrics(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sequence"])].append(row)
    output = []
    for sequence, items in grouped.items():
        metrics = threshold_metrics(items, threshold)
        metrics.update({"sequence": sequence, "count": len(items)})
        output.append(metrics)
    output.sort(key=lambda item: (float(item["false_positive_rate"]), int(item["count"])), reverse=True)
    return output[:100]


def resolve_sequence_and_frame(dataset: Any, dataset_index: int, frame_offset: int) -> tuple[str, str]:
    base = getattr(dataset, "dataset", dataset)
    index = getattr(base, "_index", None)
    sequences = getattr(base, "_sequences", None)
    stride = int(getattr(base, "stride", 1))
    if not isinstance(index, list) or not isinstance(sequences, list) or not (0 <= dataset_index < len(index)):
        return "unknown", f"dataset_index={dataset_index}"
    seq_idx, start_idx = index[dataset_index]
    seq_dir, frame_ids = sequences[seq_idx]
    physical_index = start_idx + frame_offset * stride
    frame_id = frame_ids[physical_index]
    sequence = Path(seq_dir).name
    return sequence, f"{getattr(base, 'split', '')}/{sequence}/rgb/{frame_id}.png"


def parse_thresholds(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--thresholds", default="0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.925,0.95,0.975,0.99")
    parser.add_argument("--static-speed-m", type=float, default=0.04)
    parser.add_argument("--min-recall", type=float, default=0.20)
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--max-fpr", type=float, default=0.03)
    parser.add_argument("--max-static-fpr", type=float, default=0.05)
    parser.add_argument("--max-failure-rows", type=int, default=200)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    main()
