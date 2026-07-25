from __future__ import annotations

import argparse
import json
import math
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
    seed = int(config.get("experiment", {}).get("seed", 42))
    set_seed(seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = build_loader(config, split=config["data"]["val_split"], shuffle=False, role="val")
    model = build_model(config).to(device)
    load_initial_checkpoint(model, config, device)
    checkpoint_cfg = config.setdefault("checkpoint", {})
    checkpoint_cfg["overlay_prefixes"] = ["hsi_grounding_head."]
    checkpoint_cfg["overlay_required_prefixes"] = ["hsi_grounding_head."]
    load_overlay_checkpoint(model, args.checkpoint, device, config)
    model.eval()

    loss_cfg = config.get("loss", {})
    severe_threshold = float(loss_cfg.get("hsi_grounding_severe_float_threshold_m", 0.04))
    candidate_margin = float(loss_cfg.get("hsi_grounding_gate_margin_m", 0.002))
    deadzone = float(config.get("model", {}).get("hsi_grounding_clearance_deadzone_m", 0.025))
    configured_threshold = float(loss_cfg.get("hsi_grounding_gate_decision_threshold", 0.70))
    rows: list[dict[str, Any]] = []

    for batch_step, batch in enumerate(loader):
        if args.max_batches > 0 and batch_step >= args.max_batches:
            break
        batch = move_to_device(batch, device)
        predictions = forward_model(model, batch, config, epoch=0)
        rows.extend(
            collect_rows(
                loader.dataset,
                batch,
                predictions,
                severe_threshold=severe_threshold,
                candidate_margin=candidate_margin,
                deadzone=deadzone,
            )
        )
        if (batch_step + 1) % 20 == 0:
            print(f"[gate-audit] batches={batch_step + 1}/{len(loader)} rows={len(rows)}", flush=True)

    thresholds = sorted(set(parse_thresholds(args.thresholds) + [configured_threshold]))
    sweep = [threshold_metrics(rows, threshold) for threshold in thresholds]
    configured = min(sweep, key=lambda item: abs(float(item["threshold"]) - configured_threshold))
    feasible = [
        item
        for item in sweep
        if float(item["recall"]) >= args.min_recall
        and float(item["clean_false_apply_rate"]) <= args.max_clean_false_apply
        and float(item["negative_false_apply_rate"]) <= args.max_negative_false_apply
    ]
    eligible = [row for row in rows if bool(row["candidate_valid"])]
    report = {
        "implementation": "grounding_gate_person_audit_v1",
        "checkpoint": args.checkpoint,
        "seed": seed,
        "max_batches": args.max_batches,
        "num_provider_people": len(rows),
        "num_candidate_valid": len(eligible),
        "candidate_valid_coverage": len(eligible) / max(len(rows), 1),
        "num_positive_targets": sum(bool(row["apply_target"]) for row in eligible),
        "configured_threshold": configured_threshold,
        "configured_metrics": configured,
        "threshold_sweep": sweep,
        "feasible_thresholds": [float(item["threshold"]) for item in feasible],
        "feasibility_limits": {
            "min_recall": args.min_recall,
            "max_clean_false_apply": args.max_clean_false_apply,
            "max_negative_false_apply": args.max_negative_false_apply,
        },
        "by_valid_foot_count": grouped_metrics(rows, "valid_foot_count", configured_threshold),
        "by_geometry_pattern": grouped_metrics(rows, "geometry_pattern", configured_threshold),
        "by_noise_level": grouped_metrics(rows, "noise_level", configured_threshold),
        "by_negative_reason": grouped_metrics(rows, "negative_reason", configured_threshold),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, output_dir / "gate_people.jsonl")
    (output_dir / "gate_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    false_positives = sorted(
        [row for row in eligible if not row["apply_target"] and row["gate_probability"] >= configured_threshold],
        key=lambda row: float(row["gate_probability"]),
        reverse=True,
    )
    false_negatives = sorted(
        [row for row in eligible if row["apply_target"] and row["gate_probability"] < configured_threshold],
        key=lambda row: float(row["gate_probability"]),
    )
    (output_dir / "false_positives_top.json").write_text(
        json.dumps(false_positives[: args.max_failure_rows], indent=2), encoding="utf-8"
    )
    (output_dir / "false_negatives_top.json").write_text(
        json.dumps(false_negatives[: args.max_failure_rows], indent=2), encoding="utf-8"
    )
    print(
        "[gate-audit] configured "
        f"threshold={configured_threshold:.2f} recall={configured['recall']:.4f} "
        f"clean_far={configured['clean_false_apply_rate']:.4f} "
        f"negative_far={configured['negative_false_apply_rate']:.4f}"
    )
    print(f"[gate-audit] feasible_thresholds={report['feasible_thresholds']}")
    print(f"[gate-audit] report={output_dir / 'gate_audit.json'}")


def collect_rows(
    dataset: Any,
    batch: dict[str, torch.Tensor],
    predictions: dict[str, torch.Tensor],
    severe_threshold: float,
    candidate_margin: float,
    deadzone: float,
) -> list[dict[str, Any]]:
    required_predictions = {
        "base": "hsi_grounding_base_pred_transl_cam",
        "candidate": "hsi_grounding_candidate_pred_transl_cam",
        "probability": "hsi_grounding_gate_probability",
        "candidate_valid": "hsi_grounding_candidate_valid",
        "delta_scalar": "hsi_grounding_delta_scalar",
        "support_valid": "hsi_grounding_support_valid",
        "signed": "hsi_grounding_support_signed_m",
        "rmse": "hsi_grounding_support_rmse",
        "normal": "hsi_grounding_support_normal",
        "point_count": "hsi_grounding_support_point_count",
        "provider_valid": "gt_smpl_provider_mask",
        "noise": "contact_noise_signed_m",
        "clean": "transl_noise_is_clean",
    }
    missing = [key for key in required_predictions.values() if not isinstance(predictions.get(key), torch.Tensor)]
    if missing:
        raise RuntimeError(f"Grounding Gate audit missing prediction tensors: {missing}")
    values = {name: predictions[key].detach().cpu() for name, key in required_predictions.items()}
    target = batch["gt_transl_cam"].detach().cpu().float()
    dataset_indices = batch["dataset_index"].detach().cpu().long()
    track_ids = batch["gt_track_ids"].detach().cpu().long()
    teacher_valid = batch["contact_teacher_valid"].detach().cpu().bool()
    contact_label = batch["contact_label"].detach().cpu().bool()
    teacher_normal = batch["contact_plane_normal_cam"].detach().cpu().float()
    teacher_signed = batch["contact_signed_distance_m"].detach().cpu().float()
    base_error = torch.linalg.norm(values["base"].float() - target, dim=-1)
    candidate_error = torch.linalg.norm(values["candidate"].float() - target, dim=-1)
    candidate_delta = torch.linalg.norm(values["candidate"].float() - values["base"].float(), dim=-1)
    provider_valid = values["provider_valid"].bool()

    rows: list[dict[str, Any]] = []
    for batch_idx, frame_idx, person_idx in provider_valid.nonzero(as_tuple=False).tolist():
        foot_valid = values["support_valid"][batch_idx, frame_idx, person_idx].bool()
        signed = values["signed"][batch_idx, frame_idx, person_idx].float()
        rmse = values["rmse"][batch_idx, frame_idx, person_idx].float()
        normals = values["normal"][batch_idx, frame_idx, person_idx].float()
        point_count = values["point_count"][batch_idx, frame_idx, person_idx].float()
        valid_count = int(foot_valid.sum())
        candidate_valid = bool(values["candidate_valid"][batch_idx, frame_idx, person_idx, 0] > 0.5)
        delta_scalar = float(values["delta_scalar"][batch_idx, frame_idx, person_idx, 0])
        probability = float(values["probability"][batch_idx, frame_idx, person_idx, 0])
        clean = bool(values["clean"][batch_idx, frame_idx, person_idx, 0] > 0.5)
        noise = float(values["noise"][batch_idx, frame_idx, person_idx, 0])
        base_l2 = float(base_error[batch_idx, frame_idx, person_idx])
        candidate_l2 = float(candidate_error[batch_idx, frame_idx, person_idx])
        candidate_better = candidate_l2 + candidate_margin < base_l2
        severe_float = delta_scalar <= -severe_threshold
        apply_target = candidate_valid and not clean and severe_float and candidate_better
        common_positive = valid_count > 0 and bool((signed[foot_valid] > deadzone).all())
        common_negative = valid_count > 0 and bool((signed[foot_valid] < -deadzone).all())
        geometry_pattern = geometry_pattern_name(valid_count, common_positive, common_negative)
        normal_cosine = (
            float(torch.dot(normals[0], normals[1]).clamp(-1.0, 1.0)) if valid_count == 2 else None
        )
        online_teacher_valid = foot_valid & teacher_valid[batch_idx, frame_idx, person_idx]
        online_teacher_normal_cosine = [
            float(torch.dot(normals[foot], teacher_normal[batch_idx, frame_idx, person_idx, foot]).clamp(-1.0, 1.0))
            if bool(online_teacher_valid[foot])
            else None
            for foot in range(2)
        ]
        signed_teacher_error = [
            float(
                abs(
                    signed[foot]
                    - (
                        teacher_signed[batch_idx, frame_idx, person_idx, foot]
                        + values["noise"][batch_idx, frame_idx, person_idx, 0]
                    )
                )
            )
            if bool(online_teacher_valid[foot])
            else None
            for foot in range(2)
        ]
        negative_reason = classify_negative_reason(
            candidate_valid, clean, severe_float, candidate_better, apply_target
        )
        dataset_index = int(dataset_indices[batch_idx])
        rows.append(
            {
                "dataset_index": dataset_index,
                "frame_offset": int(frame_idx),
                "person_slot": int(person_idx),
                "track_id": int(track_ids[batch_idx, frame_idx, person_idx]),
                "sample_path": resolve_frame_path(dataset, dataset_index, int(frame_idx)),
                "clean": clean,
                "noise_signed_m": noise,
                "noise_level": "clean" if clean else f"{round(noise * 100):+d}cm",
                "candidate_valid": candidate_valid,
                "valid_foot_count": valid_count,
                "geometry_pattern": geometry_pattern,
                "common_positive": common_positive,
                "common_negative": common_negative,
                "support_valid": [bool(value) for value in foot_valid.tolist()],
                "support_signed_m": [float(value) for value in signed.tolist()],
                "support_rmse_m": [float(value) for value in rmse.tolist()],
                "support_point_count": [int(value) for value in point_count.tolist()],
                "support_normal_cosine": normal_cosine,
                "online_teacher_normal_cosine": online_teacher_normal_cosine,
                "online_teacher_signed_error_m": signed_teacher_error,
                "teacher_valid": [bool(value) for value in teacher_valid[batch_idx, frame_idx, person_idx].tolist()],
                "teacher_contact": [bool(value) for value in contact_label[batch_idx, frame_idx, person_idx].tolist()],
                "delta_scalar_m": delta_scalar,
                "gate_probability": probability,
                "base_l2_m": base_l2,
                "candidate_l2_m": candidate_l2,
                "candidate_delta_m": float(candidate_delta[batch_idx, frame_idx, person_idx]),
                "candidate_better": candidate_better,
                "severe_float": severe_float,
                "apply_target": apply_target,
                "negative_reason": negative_reason,
            }
        )
    return rows


def threshold_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    eligible = [row for row in rows if bool(row["candidate_valid"])]
    positive = [row for row in eligible if bool(row["apply_target"])]
    negative = [row for row in eligible if not bool(row["apply_target"])]
    clean = [row for row in eligible if bool(row["clean"])]
    tp = sum(float(row["gate_probability"]) >= threshold for row in positive)
    fp = sum(float(row["gate_probability"]) >= threshold for row in negative)
    clean_fp = sum(float(row["gate_probability"]) >= threshold for row in clean)
    predicted_positive = tp + fp
    refined_errors = [
        float(row["candidate_l2_m"])
        if float(row["gate_probability"]) >= threshold
        else float(row["base_l2_m"])
        for row in positive
    ]
    clean_displacements = [
        float(row["candidate_delta_m"]) if float(row["gate_probability"]) >= threshold else 0.0
        for row in clean
    ]
    recall = tp / max(len(positive), 1)
    negative_far = fp / max(len(negative), 1)
    clean_far = clean_fp / max(len(clean), 1)
    refined_p95 = percentile(refined_errors, 0.95)
    return {
        "threshold": float(threshold),
        "num_positive": len(positive),
        "num_negative": len(negative),
        "num_clean": len(clean),
        "true_positive": tp,
        "false_positive": fp,
        "recall": recall,
        "precision": tp / max(predicted_positive, 1),
        "negative_false_apply_rate": negative_far,
        "clean_false_apply_rate": clean_far,
        "severe_refined_p95_m": refined_p95,
        "clean_displacement_p95_m": percentile(clean_displacements, 0.95),
        "selection": refined_p95 + 0.10 * (1.0 - recall) + 0.20 * clean_far + 0.10 * negative_far,
    }


def grouped_metrics(rows: list[dict[str, Any]], key: str, threshold: float) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key)), []).append(row)
    return {
        name: {"num_provider_people": len(group_rows), **threshold_metrics(group_rows, threshold)}
        for name, group_rows in sorted(groups.items())
    }


def geometry_pattern_name(valid_count: int, common_positive: bool, common_negative: bool) -> str:
    if valid_count <= 0:
        return "invalid"
    if valid_count == 1:
        return "one_foot"
    if common_positive:
        return "two_common_float"
    if common_negative:
        return "two_common_penetration"
    return "two_mixed_or_deadzone"


def classify_negative_reason(
    candidate_valid: bool,
    clean: bool,
    severe_float: bool,
    candidate_better: bool,
    apply_target: bool,
) -> str:
    if apply_target:
        return "positive"
    if not candidate_valid:
        return "invalid_candidate"
    if clean:
        return "clean"
    if not severe_float:
        return "not_severe_float"
    if not candidate_better:
        return "candidate_not_better"
    return "other"


def resolve_frame_path(dataset: Any, dataset_index: int, frame_offset: int) -> str:
    base = getattr(dataset, "dataset", dataset)
    sequences = getattr(base, "_sequences", None)
    index = getattr(base, "_index", None)
    if not isinstance(sequences, list) or not isinstance(index, list) or not (0 <= dataset_index < len(index)):
        return f"dataset_index={dataset_index}/frame_offset={frame_offset}"
    seq_idx, start_idx = index[dataset_index]
    seq_dir, frame_ids = sequences[seq_idx]
    stride = int(getattr(base, "stride", 1))
    frame_id = frame_ids[start_idx + frame_offset * stride]
    split = str(getattr(base, "split", ""))
    return f"{split}/{Path(seq_dir).name}/rgb/{frame_id}.png"


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = min(max(float(quantile), 0.0), 1.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def parse_thresholds(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"Invalid Gate thresholds: {raw!r}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit HSI Grounding Gate decisions per person")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--max-failure-rows", type=int, default=100)
    parser.add_argument("--thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--max-clean-false-apply", type=float, default=0.01)
    parser.add_argument("--max-negative-false-apply", type=float, default=0.02)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    main()
