#!/usr/bin/env python
"""Evaluate HSI temporal stability on BEDLAM sequence windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Compatibility patch for old chumpy on Python 3.11+.
import inspect
from collections import namedtuple

if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = getargspec

if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "complex"):
    np.complex = complex

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.evaluate_hsi_refine_metrics import (  # noqa: E402
    Meter,
    decode_smpl_batch,
    describe_array,
    extract_state_dict,
    fmt,
    greedy_match,
    move_to_device,
)
from scripts.train.train_smpl import apply_overrides, build_model, load_yaml_config  # noqa: E402
from vggt_omega.data import BedlamDataset, bedlam_collate_fn  # noqa: E402
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.training.config import deep_update, require_path  # noqa: E402
from vggt_omega.utils.rotation import rot6d_to_axis_angle  # noqa: E402


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args)
    model = build_model(config).to(device)
    load_vggt_baseline(model, config, device)
    load_training_checkpoint(model, Path(args.checkpoint), device)
    model.eval()

    loader = build_eval_loader(config, args)
    smpl = SMPLLayer(require_path(config, "assets.smpl_model_dir", allow_empty=False)).to(device).eval()
    metrics = init_metrics()
    sequence_scales: list[float] = []
    sequence_biases: list[float] = []
    examples: list[dict[str, Any]] = []

    processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = move_to_device(batch, device)
            predictions = model(
                batch["images"],
                smpl_query_boxes=batch["gt_boxes"] if args.use_gt_box_prior else None,
                smpl_query_boxes_mask=batch["boxes_mask"] if args.use_gt_box_prior else None,
                smpl_track_ids=batch.get("gt_track_ids", batch.get("person_ids")),
                smpl_track_mask=batch.get("gt_track_mask", batch.get("person_id_mask")),
            )
            summary = evaluate_temporal_batch(predictions, batch, smpl, args, metrics, sequence_scales, sequence_biases)
            if len(examples) < 8:
                examples.append({"batch_idx": int(batch_idx), **summary})
            processed += int(batch["images"].shape[0])
            if processed >= args.max_samples:
                break
            if args.log_interval > 0 and processed % args.log_interval == 0:
                print(f"[temporal-eval] processed={processed}")

    out = build_summary(metrics, sequence_scales, sequence_biases, processed, args, examples)
    out_json = output_dir / "hsi_temporal_metrics.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print_human_summary(out)
    print(json.dumps({"output_json": str(out_json), "num_sequences": processed}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HSI temporal stability on sequence windows")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_hsi_refine.yaml")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--output-dir", default="outputs/eval/hsi_temporal_metrics")
    parser.add_argument("--device", default="")
    parser.add_argument("--split", default="Training")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--conf-threshold", type=float, default=0.10)
    parser.add_argument("--use-gt-box-prior", action="store_true")
    parser.add_argument("--temporal-no-worse-margin-m", type=float, default=0.002)
    parser.add_argument("--temporal-no-worse-accel-margin-m", type=float, default=0.003)
    parser.add_argument("--log-interval", type=int, default=8)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.train_config))
    config = apply_overrides(config, args.override)
    if args.baseline_checkpoint:
        config.setdefault("checkpoints", {})["vggt_baseline"] = args.baseline_checkpoint
    config.setdefault("model", {})["enable_camera"] = True
    config.setdefault("model", {})["enable_depth"] = True
    config.setdefault("model", {})["enable_hsi_refine"] = True
    config.setdefault("data", {})["require_depth"] = True
    config.setdefault("data", {})["require_boxes"] = True
    return config


def build_eval_loader(config: dict[str, Any], args: argparse.Namespace) -> DataLoader:
    data_cfg = config["data"]
    dataset = BedlamDataset(
        root=require_path(config, data_cfg.get("root_key", "datasets.bedlam_root")),
        split=args.split,
        sequence_length=int(data_cfg["sequence_length"]),
        stride=int(data_cfg["stride"]),
        image_size=int(data_cfg.get("image_size", data_cfg.get("image_resolution", 512))),
        image_resolution=int(data_cfg.get("image_resolution", data_cfg.get("image_size", 512))),
        resize_mode=str(data_cfg.get("resize_mode", "balanced")),
        max_humans=int(data_cfg["max_humans"]),
        require_smpl=True,
        require_depth=bool(data_cfg.get("require_depth", True)),
        boxes_root=require_path(config, data_cfg["boxes_root_key"], allow_empty=False),
        require_boxes=True,
    )
    end = min(len(dataset), int(args.start_index) + int(args.max_samples))
    subset = Subset(dataset, list(range(int(args.start_index), end)))
    return DataLoader(
        subset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        collate_fn=bedlam_collate_fn,
        drop_last=False,
    )


def load_vggt_baseline(model: torch.nn.Module, config: dict[str, Any], device: torch.device) -> None:
    checkpoint_path = require_path(config, "checkpoints.vggt_baseline", allow_empty=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(extract_state_dict(checkpoint), strict=False)
    print(f"[ckpt] loaded VGGT baseline: {checkpoint_path}")
    print(f"[ckpt] baseline missing={len(missing)} unexpected={len(unexpected)}")


def load_training_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(extract_state_dict(checkpoint), strict=False)
    print(f"[ckpt] loaded training checkpoint: {checkpoint_path}")
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")


def init_metrics() -> dict[str, Meter]:
    names = [
        "num_gt_per_frame",
        "num_matched_per_frame",
        "track_valid_per_frame",
        "track_explicit_per_frame",
        "track_person_index_per_frame",
        "track_slot_per_frame",
        "track_quality_mean",
        "temporal_pair_count",
        "temporal_triple_count",
        "temporal_quad_count",
        "temporal_pair_explicit_count",
        "temporal_pair_person_index_count",
        "temporal_pair_slot_count",
        "query_switch_rate",
        "base_transl_velocity_l1_m",
        "hsi_transl_velocity_l1_m",
        "base_joints_velocity_l1_m",
        "hsi_joints_velocity_l1_m",
        "base_joints_acceleration_l1_m",
        "hsi_joints_acceleration_l1_m",
        "base_joints_jerk_l1_m",
        "hsi_joints_jerk_l1_m",
        "hsi_transl_velocity_worse_margin_ratio",
        "hsi_joints_velocity_worse_margin_ratio",
        "hsi_joints_acceleration_worse_margin_ratio",
        "hsi_temporal_no_worse_margin_ratio",
        "hsi_temporal_no_worse_excess_m",
        "hsi_scene_log_scale_abs_delta",
        "hsi_scene_log_scale_seq_abs",
        "hsi_scene_scale_range",
        "hsi_scene_scale_saturation_rate",
        "hsi_scene_bias_abs_delta_m",
        "hsi_scene_bias_seq_abs_m",
        "hsi_scene_bias_range_m",
    ]
    return {name: Meter() for name in names}


def evaluate_temporal_batch(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    smpl: SMPLLayer,
    args: argparse.Namespace,
    metrics: dict[str, Meter],
    sequence_scales: list[float],
    sequence_biases: list[float],
) -> dict[str, Any]:
    base = decode_smpl_batch(predictions["pred_poses"], predictions["pred_betas"], predictions["pred_transl_cam"], smpl)
    hsi = decode_smpl_batch(
        predictions["hsi_refined_pred_poses"],
        predictions["hsi_refined_pred_betas"],
        predictions["hsi_refined_pred_transl_cam"],
        smpl,
    )
    gt_poses = rot6d_to_axis_angle(batch["gt_pose_6d"].reshape(-1, 24, 6)).reshape(*batch["gt_pose_6d"].shape[:3], 72)
    gt = decode_smpl_batch(gt_poses, batch["gt_betas"], batch["gt_transl_cam"], smpl)

    confs = predictions["pred_confs"].detach()
    if confs.ndim == 4 and confs.shape[-1] == 1:
        confs = confs[..., 0]
    smpl_mask = batch["smpl_mask"].bool()
    track_ids = batch.get("gt_track_ids", batch.get("person_ids"))
    track_mask = batch.get("gt_track_mask", batch.get("person_id_mask"))
    track_source = batch.get("gt_track_source")
    track_quality = batch.get("gt_track_quality")
    if track_ids is None or track_mask is None:
        return {"temporal_pairs": 0, "reason": "missing_gt_track_ids"}

    batch_size, num_frames, _ = smpl_mask.shape
    records: dict[tuple[int, int], dict[int, dict[str, int]]] = {}
    frame_match_counts = []
    for b in range(batch_size):
        for s in range(num_frames):
            gt_idx = torch.nonzero(smpl_mask[b, s], as_tuple=False).flatten()
            pred_idx = torch.nonzero(confs[b, s] >= float(args.conf_threshold), as_tuple=False).flatten()
            metrics["num_gt_per_frame"].add(float(gt_idx.numel()))
            add_track_source_metrics(metrics, track_mask, track_source, track_quality, b, s, gt_idx)
            if gt_idx.numel() == 0 or pred_idx.numel() == 0:
                frame_match_counts.append(0)
                continue
            matches = greedy_match(base["joints"][b, s, pred_idx, :24], gt["joints"][b, s, gt_idx, :24])
            metrics["num_matched_per_frame"].add(float(len(matches)))
            frame_match_counts.append(len(matches))
            for pred_local, gt_local in matches:
                q = int(pred_idx[pred_local].item())
                g = int(gt_idx[gt_local].item())
                if not bool(track_mask[b, s, g].detach().cpu()):
                    continue
                pid = int(track_ids[b, s, g].detach().cpu())
                if pid < 0:
                    continue
                source = int(track_source[b, s, g].detach().cpu()) if track_source is not None else -1
                quality = float(track_quality[b, s, g].detach().cpu()) if track_quality is not None else 0.0
                records.setdefault((b, pid), {}).setdefault(s, {"query": q, "gt_index": g, "source": source, "quality": quality})

    pair_count, triple_count, quad_count, query_switches = add_person_temporal_metrics(metrics, records, base, hsi, gt, num_frames, args)
    scale_summary = add_scene_temporal_metrics(metrics, predictions, sequence_scales, sequence_biases)
    return {
        "num_frames": int(num_frames),
        "frame_match_counts": frame_match_counts,
        "temporal_pairs": int(pair_count),
        "temporal_triples": int(triple_count),
        "temporal_quads": int(quad_count),
        "query_switches": int(query_switches),
        "scene": scale_summary,
    }


def add_track_source_metrics(
    metrics: dict[str, Meter],
    track_mask: torch.Tensor,
    track_source: torch.Tensor | None,
    track_quality: torch.Tensor | None,
    batch_idx: int,
    seq_idx: int,
    gt_idx: torch.Tensor,
) -> None:
    if gt_idx.numel() == 0:
        metrics["track_valid_per_frame"].add(0.0)
        return
    valid = track_mask[batch_idx, seq_idx, gt_idx].bool()
    metrics["track_valid_per_frame"].add(float(valid.sum().detach().cpu()))
    if not valid.any():
        return
    if track_quality is not None:
        quality = track_quality[batch_idx, seq_idx, gt_idx][valid].float()
        metrics["track_quality_mean"].add(float(quality.mean().detach().cpu()), int(valid.sum().detach().cpu()))
    if track_source is None:
        return
    source = track_source[batch_idx, seq_idx, gt_idx][valid].long()
    metrics["track_explicit_per_frame"].add(float((source == 2).sum().detach().cpu()))
    metrics["track_person_index_per_frame"].add(float((source == 1).sum().detach().cpu()))
    metrics["track_slot_per_frame"].add(float((source == 0).sum().detach().cpu()))


def add_person_temporal_metrics(
    metrics: dict[str, Meter],
    records: dict[tuple[int, int], dict[int, dict[str, int]]],
    base: dict[str, torch.Tensor],
    hsi: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    num_frames: int,
    args: argparse.Namespace,
) -> tuple[int, int, int, int]:
    pair_count = 0
    triple_count = 0
    quad_count = 0
    query_switches = 0
    pair_source_counts = {"explicit": 0, "person_index": 0, "slot": 0}
    for (batch_idx, _), seq_map in records.items():
        for seq in range(1, num_frames):
            if seq - 1 not in seq_map or seq not in seq_map:
                continue
            prev = seq_map[seq - 1]
            curr = seq_map[seq]
            pair_count += 1
            pair_source = pair_source_name(prev, curr)
            if pair_source in pair_source_counts:
                pair_source_counts[pair_source] += 1
            query_switches += int(prev["query"] != curr["query"])
            add_velocity_metric(metrics, "base", base, gt, batch_idx, seq - 1, seq, prev, curr)
            add_velocity_metric(metrics, "hsi", hsi, gt, batch_idx, seq - 1, seq, prev, curr)
            add_velocity_no_worse_metrics(
                metrics,
                base,
                hsi,
                gt,
                batch_idx,
                seq - 1,
                seq,
                prev,
                curr,
                float(args.temporal_no_worse_margin_m),
            )
        for seq in range(1, num_frames - 1):
            if seq - 1 not in seq_map or seq not in seq_map or seq + 1 not in seq_map:
                continue
            triple_count += 1
            add_acceleration_metric(metrics, "base", base, gt, batch_idx, seq - 1, seq, seq + 1, seq_map)
            add_acceleration_metric(metrics, "hsi", hsi, gt, batch_idx, seq - 1, seq, seq + 1, seq_map)
            add_acceleration_no_worse_metrics(
                metrics,
                base,
                hsi,
                gt,
                batch_idx,
                seq - 1,
                seq,
                seq + 1,
                seq_map,
                float(args.temporal_no_worse_accel_margin_m),
            )
        for seq in range(3, num_frames):
            if any(item not in seq_map for item in (seq - 3, seq - 2, seq - 1, seq)):
                continue
            quad_count += 1
            add_jerk_metric(metrics, "base", base, gt, batch_idx, seq - 3, seq - 2, seq - 1, seq, seq_map)
            add_jerk_metric(metrics, "hsi", hsi, gt, batch_idx, seq - 3, seq - 2, seq - 1, seq, seq_map)

    metrics["temporal_pair_count"].add(float(pair_count))
    metrics["temporal_triple_count"].add(float(triple_count))
    metrics["temporal_quad_count"].add(float(quad_count))
    metrics["temporal_pair_explicit_count"].add(float(pair_source_counts["explicit"]))
    metrics["temporal_pair_person_index_count"].add(float(pair_source_counts["person_index"]))
    metrics["temporal_pair_slot_count"].add(float(pair_source_counts["slot"]))
    if pair_count > 0:
        metrics["query_switch_rate"].add(float(query_switches) / float(pair_count), pair_count)
    return pair_count, triple_count, quad_count, query_switches


def pair_source_name(prev: dict[str, int], curr: dict[str, int]) -> str:
    source = min(int(prev.get("source", -1)), int(curr.get("source", -1)))
    if source == 2:
        return "explicit"
    if source == 1:
        return "person_index"
    if source == 0:
        return "slot"
    return "unknown"


def add_velocity_metric(
    metrics: dict[str, Meter],
    prefix: str,
    pred: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    batch_idx: int,
    prev_seq: int,
    curr_seq: int,
    prev: dict[str, int],
    curr: dict[str, int],
) -> None:
    q0, q1 = prev["query"], curr["query"]
    g0, g1 = prev["gt_index"], curr["gt_index"]
    pred_transl_vel = pred["transl"][batch_idx, curr_seq, q1] - pred["transl"][batch_idx, prev_seq, q0]
    gt_transl_vel = gt["transl"][batch_idx, curr_seq, g1] - gt["transl"][batch_idx, prev_seq, g0]
    pred_joint_vel = pred["joints"][batch_idx, curr_seq, q1, :24] - pred["joints"][batch_idx, prev_seq, q0, :24]
    gt_joint_vel = gt["joints"][batch_idx, curr_seq, g1, :24] - gt["joints"][batch_idx, prev_seq, g0, :24]
    metrics[f"{prefix}_transl_velocity_l1_m"].add(float(torch.abs(pred_transl_vel - gt_transl_vel).mean().detach().cpu()))
    metrics[f"{prefix}_joints_velocity_l1_m"].add(float(torch.abs(pred_joint_vel - gt_joint_vel).mean().detach().cpu()))


def add_velocity_no_worse_metrics(
    metrics: dict[str, Meter],
    base: dict[str, torch.Tensor],
    hsi: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    batch_idx: int,
    prev_seq: int,
    curr_seq: int,
    prev: dict[str, int],
    curr: dict[str, int],
    margin_m: float,
) -> None:
    q0, q1 = prev["query"], curr["query"]
    g0, g1 = prev["gt_index"], curr["gt_index"]
    gt_transl_vel = gt["transl"][batch_idx, curr_seq, g1] - gt["transl"][batch_idx, prev_seq, g0]
    base_transl_vel = base["transl"][batch_idx, curr_seq, q1] - base["transl"][batch_idx, prev_seq, q0]
    hsi_transl_vel = hsi["transl"][batch_idx, curr_seq, q1] - hsi["transl"][batch_idx, prev_seq, q0]
    add_no_worse_metric(
        metrics,
        "hsi_transl_velocity_worse_margin_ratio",
        torch.abs(base_transl_vel - gt_transl_vel).mean(),
        torch.abs(hsi_transl_vel - gt_transl_vel).mean(),
        margin_m,
    )

    gt_joint_vel = gt["joints"][batch_idx, curr_seq, g1, :24] - gt["joints"][batch_idx, prev_seq, g0, :24]
    base_joint_vel = base["joints"][batch_idx, curr_seq, q1, :24] - base["joints"][batch_idx, prev_seq, q0, :24]
    hsi_joint_vel = hsi["joints"][batch_idx, curr_seq, q1, :24] - hsi["joints"][batch_idx, prev_seq, q0, :24]
    add_no_worse_metric(
        metrics,
        "hsi_joints_velocity_worse_margin_ratio",
        torch.abs(base_joint_vel - gt_joint_vel).mean(),
        torch.abs(hsi_joint_vel - gt_joint_vel).mean(),
        margin_m,
    )


def add_acceleration_metric(
    metrics: dict[str, Meter],
    prefix: str,
    pred: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    batch_idx: int,
    seq0: int,
    seq1: int,
    seq2: int,
    seq_map: dict[int, dict[str, int]],
) -> None:
    q0, q1, q2 = seq_map[seq0]["query"], seq_map[seq1]["query"], seq_map[seq2]["query"]
    g0, g1, g2 = seq_map[seq0]["gt_index"], seq_map[seq1]["gt_index"], seq_map[seq2]["gt_index"]
    pred_acc = pred["joints"][batch_idx, seq2, q2, :24] - 2.0 * pred["joints"][batch_idx, seq1, q1, :24] + pred["joints"][batch_idx, seq0, q0, :24]
    gt_acc = gt["joints"][batch_idx, seq2, g2, :24] - 2.0 * gt["joints"][batch_idx, seq1, g1, :24] + gt["joints"][batch_idx, seq0, g0, :24]
    metrics[f"{prefix}_joints_acceleration_l1_m"].add(float(torch.abs(pred_acc - gt_acc).mean().detach().cpu()))


def add_acceleration_no_worse_metrics(
    metrics: dict[str, Meter],
    base: dict[str, torch.Tensor],
    hsi: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    batch_idx: int,
    seq0: int,
    seq1: int,
    seq2: int,
    seq_map: dict[int, dict[str, int]],
    margin_m: float,
) -> None:
    q0, q1, q2 = seq_map[seq0]["query"], seq_map[seq1]["query"], seq_map[seq2]["query"]
    g0, g1, g2 = seq_map[seq0]["gt_index"], seq_map[seq1]["gt_index"], seq_map[seq2]["gt_index"]
    gt_acc = gt["joints"][batch_idx, seq2, g2, :24] - 2.0 * gt["joints"][batch_idx, seq1, g1, :24] + gt["joints"][batch_idx, seq0, g0, :24]
    base_acc = base["joints"][batch_idx, seq2, q2, :24] - 2.0 * base["joints"][batch_idx, seq1, q1, :24] + base["joints"][batch_idx, seq0, q0, :24]
    hsi_acc = hsi["joints"][batch_idx, seq2, q2, :24] - 2.0 * hsi["joints"][batch_idx, seq1, q1, :24] + hsi["joints"][batch_idx, seq0, q0, :24]
    add_no_worse_metric(
        metrics,
        "hsi_joints_acceleration_worse_margin_ratio",
        torch.abs(base_acc - gt_acc).mean(),
        torch.abs(hsi_acc - gt_acc).mean(),
        margin_m,
    )


def add_no_worse_metric(
    metrics: dict[str, Meter],
    component_key: str,
    base_err: torch.Tensor,
    hsi_err: torch.Tensor,
    margin_m: float,
) -> None:
    excess = torch.relu(hsi_err - base_err.detach() - float(margin_m))
    worse = float((excess > 0).to(dtype=torch.float32).detach().cpu())
    metrics[component_key].add(worse)
    metrics["hsi_temporal_no_worse_margin_ratio"].add(worse)
    metrics["hsi_temporal_no_worse_excess_m"].add(float(excess.detach().cpu()))


def add_jerk_metric(
    metrics: dict[str, Meter],
    prefix: str,
    pred: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    batch_idx: int,
    seq0: int,
    seq1: int,
    seq2: int,
    seq3: int,
    seq_map: dict[int, dict[str, int]],
) -> None:
    qs = [seq_map[seq]["query"] for seq in (seq0, seq1, seq2, seq3)]
    gs = [seq_map[seq]["gt_index"] for seq in (seq0, seq1, seq2, seq3)]
    pred_j = (
        pred["joints"][batch_idx, seq3, qs[3], :24]
        - 3.0 * pred["joints"][batch_idx, seq2, qs[2], :24]
        + 3.0 * pred["joints"][batch_idx, seq1, qs[1], :24]
        - pred["joints"][batch_idx, seq0, qs[0], :24]
    )
    gt_j = (
        gt["joints"][batch_idx, seq3, gs[3], :24]
        - 3.0 * gt["joints"][batch_idx, seq2, gs[2], :24]
        + 3.0 * gt["joints"][batch_idx, seq1, gs[1], :24]
        - gt["joints"][batch_idx, seq0, gs[0], :24]
    )
    metrics[f"{prefix}_joints_jerk_l1_m"].add(float(torch.abs(pred_j - gt_j).mean().detach().cpu()))


def add_scene_temporal_metrics(
    metrics: dict[str, Meter],
    predictions: dict[str, torch.Tensor],
    sequence_scales: list[float],
    sequence_biases: list[float],
) -> dict[str, Any]:
    scale = predictions.get("hsi_scene_scale")
    bias = predictions.get("hsi_scene_depth_bias")
    summary: dict[str, Any] = {}
    if scale is not None and scale.ndim >= 3:
        scale_f = scale.detach().float().clamp(min=1e-6)
        log_scale = torch.log(scale_f)
        seq_log_scale = log_scale.median(dim=1, keepdim=True).values
        seq_scale = torch.exp(seq_log_scale).reshape(-1)
        sequence_scales.extend(seq_scale.cpu().tolist())
        seq_abs = torch.abs(log_scale - seq_log_scale)
        scale_range = scale_f.amax(dim=1) - scale_f.amin(dim=1)
        saturation = (scale_f >= 19.5).to(dtype=scale_f.dtype)
        metrics["hsi_scene_log_scale_seq_abs"].add(float(seq_abs.mean().cpu()))
        metrics["hsi_scene_scale_range"].add(float(scale_range.mean().cpu()))
        metrics["hsi_scene_scale_saturation_rate"].add(float(saturation.mean().cpu()))
        summary["sequence_scale"] = seq_scale.cpu().tolist()
        if scale_f.shape[1] > 1:
            delta = torch.abs(log_scale[:, 1:] - log_scale[:, :-1])
            metrics["hsi_scene_log_scale_abs_delta"].add(float(delta.mean().cpu()))
            summary["mean_log_scale_delta"] = float(delta.mean().cpu())
    if bias is not None and bias.ndim >= 3:
        bias_f = bias.detach().float()
        seq_bias = bias_f.median(dim=1, keepdim=True).values
        sequence_biases.extend(seq_bias.reshape(-1).cpu().tolist())
        seq_abs = torch.abs(bias_f - seq_bias)
        bias_range = bias_f.amax(dim=1) - bias_f.amin(dim=1)
        metrics["hsi_scene_bias_seq_abs_m"].add(float(seq_abs.mean().cpu()))
        metrics["hsi_scene_bias_range_m"].add(float(bias_range.mean().cpu()))
        summary["sequence_bias"] = seq_bias.reshape(-1).cpu().tolist()
        if bias_f.shape[1] > 1:
            delta = torch.abs(bias_f[:, 1:] - bias_f[:, :-1])
            metrics["hsi_scene_bias_abs_delta_m"].add(float(delta.mean().cpu()))
            summary["mean_bias_delta_m"] = float(delta.mean().cpu())
    return summary


def build_summary(
    metrics: dict[str, Meter],
    sequence_scales: list[float],
    sequence_biases: list[float],
    processed: int,
    args: argparse.Namespace,
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    values = {name: meter.mean for name, meter in metrics.items()}
    pairs = {
        "transl_velocity_l1_m": ("base_transl_velocity_l1_m", "hsi_transl_velocity_l1_m"),
        "joints_velocity_l1_m": ("base_joints_velocity_l1_m", "hsi_joints_velocity_l1_m"),
        "joints_acceleration_l1_m": ("base_joints_acceleration_l1_m", "hsi_joints_acceleration_l1_m"),
        "joints_jerk_l1_m": ("base_joints_jerk_l1_m", "hsi_joints_jerk_l1_m"),
    }
    improvements = {name: improvement(values[base], values[hsi]) for name, (base, hsi) in pairs.items()}
    return {
        "checkpoint": args.checkpoint,
        "num_sequences": processed,
        "conf_threshold": float(args.conf_threshold),
        "use_gt_box_prior": bool(args.use_gt_box_prior),
        "metrics": values,
        "improvement_percent_lower_is_better": improvements,
        "sequence_robust_scale": describe_array(np.asarray(sequence_scales, dtype=np.float64)),
        "sequence_robust_bias": describe_array(np.asarray(sequence_biases, dtype=np.float64)),
        "examples": examples,
    }


def improvement(base: float | None, hsi: float | None) -> float | None:
    if base is None or hsi is None or abs(base) < 1e-12:
        return None
    return (base - hsi) / base * 100.0


def print_human_summary(summary: dict[str, Any]) -> None:
    print("========== HSI temporal metrics ==========")
    metrics = summary["metrics"]
    improvements = summary["improvement_percent_lower_is_better"]
    for label, base_key, hsi_key, imp_key in [
        ("Transl velocity L1 (m)", "base_transl_velocity_l1_m", "hsi_transl_velocity_l1_m", "transl_velocity_l1_m"),
        ("Joints velocity L1 (m)", "base_joints_velocity_l1_m", "hsi_joints_velocity_l1_m", "joints_velocity_l1_m"),
        ("Joints accel L1 (m)", "base_joints_acceleration_l1_m", "hsi_joints_acceleration_l1_m", "joints_acceleration_l1_m"),
        ("Joints jerk L1 (m)", "base_joints_jerk_l1_m", "hsi_joints_jerk_l1_m", "joints_jerk_l1_m"),
    ]:
        print(f"{label:24s} base={fmt(metrics.get(base_key))} hsi={fmt(metrics.get(hsi_key))} improvement={fmt(improvements.get(imp_key))}%")
    print(
        "Temporal counts        "
        f"pairs={fmt(metrics.get('temporal_pair_count'))} "
        f"triples={fmt(metrics.get('temporal_triple_count'))} "
        f"quads={fmt(metrics.get('temporal_quad_count'))} "
        f"query_switch={fmt(metrics.get('query_switch_rate'))}"
    )
    print(
        "Track coverage         "
        f"valid/frame={fmt(metrics.get('track_valid_per_frame'))} "
        f"explicit/frame={fmt(metrics.get('track_explicit_per_frame'))} "
        f"person_index/frame={fmt(metrics.get('track_person_index_per_frame'))} "
        f"slot/frame={fmt(metrics.get('track_slot_per_frame'))} "
        f"quality={fmt(metrics.get('track_quality_mean'))}"
    )
    print(
        "Temporal pair source   "
        f"explicit={fmt(metrics.get('temporal_pair_explicit_count'))} "
        f"person_index={fmt(metrics.get('temporal_pair_person_index_count'))} "
        f"slot={fmt(metrics.get('temporal_pair_slot_count'))}"
    )
    print(
        "Temporal no-worse      "
        f"worse={fmt(metrics.get('hsi_temporal_no_worse_margin_ratio'))} "
        f"excess={fmt(metrics.get('hsi_temporal_no_worse_excess_m'))}m "
        f"transl={fmt(metrics.get('hsi_transl_velocity_worse_margin_ratio'))} "
        f"joints={fmt(metrics.get('hsi_joints_velocity_worse_margin_ratio'))} "
        f"accel={fmt(metrics.get('hsi_joints_acceleration_worse_margin_ratio'))}"
    )
    print(
        "Scene temporal         "
        f"log_scale_delta={fmt(metrics.get('hsi_scene_log_scale_abs_delta'))} "
        f"log_scale_seq_abs={fmt(metrics.get('hsi_scene_log_scale_seq_abs'))} "
        f"scale_range={fmt(metrics.get('hsi_scene_scale_range'))} "
        f"saturation={fmt(metrics.get('hsi_scene_scale_saturation_rate'))}"
    )
    print(
        "Scene bias temporal    "
        f"bias_delta={fmt(metrics.get('hsi_scene_bias_abs_delta_m'))}m "
        f"bias_seq_abs={fmt(metrics.get('hsi_scene_bias_seq_abs_m'))}m "
        f"bias_range={fmt(metrics.get('hsi_scene_bias_range_m'))}m"
    )
    print(f"robust seq scale: {summary['sequence_robust_scale']}")
    print(f"robust seq bias : {summary['sequence_robust_bias']}")


if __name__ == "__main__":
    main()
