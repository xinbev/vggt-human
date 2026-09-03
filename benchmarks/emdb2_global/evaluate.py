#!/usr/bin/env python3
"""Evaluate world-space EMDB-2 predictions with Human3R global metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.emdb2_global.data import (  # noqa: E402
    EMDB2Sequence,
    decode_gt_world_joints,
    load_emdb2_sequences,
)
from benchmarks.emdb2_global.metrics import evaluate_global_metrics, transform_points  # noqa: E402
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.training.config import deep_update, load_yaml_config, require_path  # noqa: E402


STAGE_ORDER = (
    "vggt_nlf",
    "vggt_nlf_hsi_scale",
    "vggt_nlf_hsi_scale_trstr",
)
STAGE_LABELS = {
    "vggt_nlf": "RGB-VGGT-NLF (analytic coarse gauge)",
    "vggt_nlf_hsi_scale": "RGB-VGGT-NLF-HSI Scale",
    "vggt_nlf_hsi_scale_trstr": "RGB-VGGT-NLF-HSI Scale-TRSTR",
}


def main() -> None:
    args = parse_args()
    import torch

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({args.device}) but unavailable")
    device = torch.device(args.device)
    cfg = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.config))
    protocol_cfg = cfg.get("protocol", {})
    subsample_stride = max(
        int(args.subsample_stride or protocol_cfg.get("subsample_stride", 1)), 1
    )
    chunk_length = int(
        args.chunk_length
        or protocol_cfg.get("chunk_length", 0)
        or max(int(100 / subsample_stride), 1)
    )
    root_index = int(
        args.root_index if args.root_index >= 0 else protocol_cfg.get("root_index", 0)
    )
    emdb_root = Path(
        args.emdb_root
        or require_path(cfg, str(cfg.get("data", {}).get("root_key", "datasets.emdb_root")))
    ).expanduser()
    predictions_root = Path(args.predictions_root).expanduser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences = load_emdb2_sequences(emdb_root, args.sequence_filter)
    if args.max_sequences > 0:
        sequences = sequences[: args.max_sequences]
    smpl_root = require_path(cfg, "assets.smpl_model_dir", allow_empty=False)
    smpl_layers = {
        gender: SMPLLayer(smpl_root, gender=gender).to(device).eval()
        for gender in ("neutral", "male", "female")
    }
    neutral_joint_regressor = smpl_layers["neutral"].layer.J_regressor.detach()

    all_metrics = {
        stage: {"w": [], "wa": [], "rte": []}
        for stage in STAGE_ORDER
    }
    sequence_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    missing_sequences: list[str] = []
    total_good_frames = 0
    total_sampled_frames = 0
    total_matched_frames = 0
    inference_protocol: dict[str, Any] | None = None

    for sequence in sequences:
        total_good_frames += int(sequence.good_frame_indices.size)
        sequence_sampled_frames = int(sequence.good_frame_indices[::subsample_stride].size)
        total_sampled_frames += sequence_sampled_frames
        archive_path = find_prediction_archive(predictions_root, sequence)
        if archive_path is None:
            missing_sequences.append(sequence.name)
            if args.require_all_sequences:
                raise FileNotFoundError(f"Missing prediction archive for {sequence.name}")
            continue
        prediction = load_prediction_archive(archive_path, sequence)
        current_protocol = {
            "inference_chunk_size": int(prediction["inference_chunk_size"]),
            "inference_chunk_overlap": int(prediction["inference_chunk_overlap"]),
            "stitch_protocol": str(prediction["stitch_protocol"]),
        }
        if inference_protocol is None:
            inference_protocol = current_protocol
        elif inference_protocol != current_protocol:
            raise ValueError(
                "Prediction archives use inconsistent inference protocols: "
                f"{inference_protocol} vs {current_protocol} for {sequence.name}"
            )
        frame_indices, pred_joints_world_by_stage = select_protocol_frames(
            sequence, prediction, subsample_stride=subsample_stride
        )
        if frame_indices.size < 2:
            if args.require_all_sequences:
                raise RuntimeError(f"{sequence.name} has fewer than two matched good frames")
            continue
        target_joints_world = decode_gt_world_joints(
            sequence,
            frame_indices,
            smpl_layers[sequence.gender],
            device,
            chunk_size=int(args.smpl_batch_size),
            joint_regressor=neutral_joint_regressor,
        )
        total_matched_frames += int(frame_indices.size)
        prediction_coverage = frame_indices.size / max(sequence_sampled_frames, 1)
        sampling_rate = sequence_sampled_frames / max(sequence.good_frame_indices.size, 1)
        root_displacement = float(
            np.linalg.norm(np.diff(target_joints_world[:, root_index], axis=0), axis=-1).sum()
        )
        max_frame_gap = int(np.diff(frame_indices).max()) if frame_indices.size > 1 else 0
        sequence_stage_summaries: dict[str, dict[str, float]] = {}
        for stage in STAGE_ORDER:
            result = evaluate_global_metrics(
                target_joints_world,
                pred_joints_world_by_stage[stage],
                chunk_length=chunk_length,
                root_index=root_index,
            )
            all_metrics[stage]["w"].append(result.w_mpjpe_mm)
            all_metrics[stage]["wa"].append(result.wa_mpjpe_mm)
            all_metrics[stage]["rte"].append(result.rte_percent)
            stage_summary = result.summary()
            sequence_stage_summaries[stage] = stage_summary
            sequence_rows.append(
                {
                    "sequence": sequence.name,
                    "stage": stage,
                    "stage_label": STAGE_LABELS[stage],
                    "prediction_archive": str(archive_path),
                    "good_frames": int(sequence.good_frame_indices.size),
                    "matched_frames": int(frame_indices.size),
                    "sampling_rate": float(sampling_rate),
                    "prediction_coverage": float(prediction_coverage),
                    "gt_root_displacement_m": root_displacement,
                    "max_matched_frame_gap": max_frame_gap,
                    **current_protocol,
                    **stage_summary,
                }
            )
            for local, frame_index in enumerate(frame_indices.tolist()):
                frame_rows.append(
                    {
                        "sequence": sequence.name,
                        "stage": stage,
                        "stage_label": STAGE_LABELS[stage],
                        "frame_index": int(frame_index),
                        "W-MPJPE_mm": float(result.w_mpjpe_mm[local]),
                        "WA-MPJPE_mm": float(result.wa_mpjpe_mm[local]),
                        "RTE_percent": float(result.rte_percent[local]),
                    }
                )
        base_summary = sequence_stage_summaries[STAGE_ORDER[0]]
        scale_summary = sequence_stage_summaries[STAGE_ORDER[1]]
        final_summary = sequence_stage_summaries[STAGE_ORDER[2]]
        if not args.metrics_only_output:
            print(
                f"[sequence] {sequence.name} frames={frame_indices.size}/{sequence.good_frame_indices.size} "
                f"base=({base_summary['W-MPJPE_mm']:.2f},"
                f"{base_summary['WA-MPJPE_mm']:.2f},{base_summary['RTE_percent']:.3f}%) "
                f"hsi=({scale_summary['W-MPJPE_mm']:.2f},"
                f"{scale_summary['WA-MPJPE_mm']:.2f},{scale_summary['RTE_percent']:.3f}%) "
                f"trstr=({final_summary['W-MPJPE_mm']:.2f},"
                f"{final_summary['WA-MPJPE_mm']:.2f},{final_summary['RTE_percent']:.3f}%)",
                flush=True,
            )

    if not all_metrics[STAGE_ORDER[-1]]["w"]:
        raise RuntimeError("No EMDB-2 sequence produced metrics")
    stage_summaries: dict[str, dict[str, Any]] = {}
    for stage in STAGE_ORDER:
        w = np.concatenate(all_metrics[stage]["w"])
        wa = np.concatenate(all_metrics[stage]["wa"])
        rte = np.concatenate(all_metrics[stage]["rte"])
        stage_rows = [row for row in sequence_rows if row["stage"] == stage]
        stage_summaries[stage] = {
            "label": STAGE_LABELS[stage],
            "paper_metrics_frame_weighted": {
                "W-MPJPE_mm": float(w.mean()),
                "WA-MPJPE_mm": float(wa.mean()),
                "RTE_percent": float(rte.mean()),
            },
            "sequence_macro_diagnostics": {
                key: float(np.mean([float(row[key]) for row in stage_rows]))
                for key in ("W-MPJPE_mm", "WA-MPJPE_mm", "RTE_percent")
            },
        }
    contributions = build_contribution_summary(stage_summaries)
    stage_rows = [
        {
            "stage": stage,
            "stage_label": STAGE_LABELS[stage],
            **stage_summaries[stage]["paper_metrics_frame_weighted"],
        }
        for stage in STAGE_ORDER
    ]
    inference_protocol = inference_protocol or {
        "inference_chunk_size": 0,
        "inference_chunk_overlap": 0,
        "stitch_protocol": "none",
    }
    is_chunk100_protocol = (
        subsample_stride == 1
        and inference_protocol["inference_chunk_size"] == 100
        and inference_protocol["stitch_protocol"] != "none"
    )
    summary = {
        "benchmark": (
            "emdb2_global_chunk100_prediction_stitch_v1"
            if is_chunk100_protocol
            else "emdb2_global_human3r_protocol_v1"
            if subsample_stride == 1
            else f"emdb2_s{subsample_stride}_unchunked_two_pass_protocol_v1"
        ),
        "dataset": "EMDB-2",
        "protocol_sequence_count": len(sequences),
        "evaluated_sequence_count": len(sequence_rows) // len(STAGE_ORDER),
        "missing_sequences": missing_sequences,
        "good_frames": int(total_good_frames),
        "sampled_frames": int(total_sampled_frames),
        "matched_frames": int(total_matched_frames),
        "sampling_rate": float(total_sampled_frames / max(total_good_frames, 1)),
        "prediction_coverage": float(total_matched_frames / max(total_sampled_frames, 1)),
        "stages": stage_summaries,
        "contributions_error_reduction": contributions,
        "metric_mapping": {
            "W-MPJPE": "Human3R wa2_mpjpe; chunk Sim(3) from first two frames; mm",
            "WA-MPJPE": "Human3R waa_mpjpe; chunk Sim(3) from all chunk joints; mm",
            "RTE": "Rigid-aligned root error / total GT root displacement * 100; percent",
        },
        "stage_protocol": {
            "vggt_nlf": "NLF base SMPL + shared analytic-coarse-scaled predicted VGGT camera",
            "vggt_nlf_hsi_scale": "same NLF base SMPL + shared analytic coarse * v3 HSI residual camera scale",
            "vggt_nlf_hsi_scale_trstr": "same metric camera as HSI stage + TRSTR refined translation",
        },
        "matching_protocol": "human3r_gt_smpl2d_iou_v1",
        "chunk_length": chunk_length,
        "subsample_stride": subsample_stride,
        "official_emdb2_subsample": bool(subsample_stride == 1),
        "inference_protocol": inference_protocol,
        "root_index": root_index,
        "joint_format": "project SMPL-24; root index 0",
        "prediction_protocol": (
            "100-frame local VGGT worlds joined by prediction-only overlapping-camera SE(3)"
            if is_chunk100_protocol
            else "continuous unchunked world joints, or camera joints with predicted per-frame T_c2w"
        ),
        "gt_protocol": "native EMDB-2 good_frames_mask; gender-specific SMPL; world root pose/transl",
        "predictions_root": str(predictions_root),
        "emdb_root": str(emdb_root),
        "sequence_metrics_csv": str(output_dir / "sequence_metrics.csv"),
        "frame_metrics_csv": str(output_dir / "frame_metrics.csv"),
        "stage_metrics_csv": str(output_dir / "stage_metrics.csv"),
    }
    write_csv(output_dir / "stage_metrics.csv", stage_rows)
    write_csv(output_dir / "sequence_metrics.csv", sequence_rows)
    write_csv(output_dir / "frame_metrics.csv", frame_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.metrics_only_output:
        print_metric_tables(sequence_rows, stage_rows)
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emdb-root", default="")
    parser.add_argument("--predictions-root", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--config", default="benchmarks/emdb2_global/config.yaml")
    parser.add_argument("--output-dir", default="outputs/eval/emdb2_global")
    parser.add_argument("--sequence-filter", default="")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--chunk-length", type=int, default=0)
    parser.add_argument("--subsample-stride", type=int, default=0)
    parser.add_argument("--root-index", type=int, default=-1)
    parser.add_argument("--smpl-batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--require-all-sequences",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--metrics-only-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print only compact per-sequence and aggregate metric tables.",
    )
    return parser.parse_args()


def print_metric_tables(
    sequence_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
) -> None:
    print("sequence,stage,W-MPJPE_mm,WA-MPJPE_mm,RTE_percent", flush=True)
    for row in sequence_rows:
        print(
            f"{row['sequence']},{row['stage']},"
            f"{float(row['W-MPJPE_mm']):.3f},"
            f"{float(row['WA-MPJPE_mm']):.3f},"
            f"{float(row['RTE_percent']):.5f}",
            flush=True,
        )
    print("summary_stage,W-MPJPE_mm,WA-MPJPE_mm,RTE_percent", flush=True)
    for row in stage_rows:
        print(
            f"{row['stage']},"
            f"{float(row['W-MPJPE_mm']):.3f},"
            f"{float(row['WA-MPJPE_mm']):.3f},"
            f"{float(row['RTE_percent']):.5f}",
            flush=True,
        )


def find_prediction_archive(root: Path, sequence: EMDB2Sequence) -> Path | None:
    participant, action = sequence.name.split("/", 1)
    candidates = (
        root / f"{sequence.safe_name}.npz",
        root / participant / f"{action}.npz",
        root / participant / action / "prediction.npz",
        root / participant / action / "predictions.npz",
    )
    return next((path for path in candidates if path.is_file()), None)


def load_prediction_archive(path: Path, sequence: EMDB2Sequence) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        required = {"frame_indices"}
        if not required.issubset(keys):
            raise KeyError(f"{path} missing keys: {sorted(required - keys)}")
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64).reshape(-1)
        valid = (
            np.asarray(archive["valid"], dtype=bool).reshape(-1)
            if "valid" in keys
            else np.ones(frame_indices.shape[0], dtype=bool)
        )
        if valid.shape[0] != frame_indices.shape[0]:
            raise ValueError(f"{path}: valid/frame_indices length mismatch")
        if "stage_names" in keys:
            archive_stages = tuple(str(value) for value in np.asarray(archive["stage_names"]).reshape(-1))
            if archive_stages != STAGE_ORDER:
                raise ValueError(
                    f"{path}: stage_names={archive_stages} does not match required {STAGE_ORDER}"
                )
        joints_world_by_stage: dict[str, np.ndarray] = {}
        for stage in STAGE_ORDER:
            world_key = f"pred_joints_world__{stage}"
            cam_key = f"pred_joints_cam__{stage}"
            camera_key = f"pred_T_c2w__{stage}"
            if world_key in keys:
                joints_world = np.asarray(archive[world_key], dtype=np.float32)
            elif {cam_key, camera_key}.issubset(keys):
                joints_world = transform_points(
                    np.asarray(archive[camera_key], dtype=np.float32),
                    np.asarray(archive[cam_key], dtype=np.float32),
                ).astype(np.float32, copy=False)
            else:
                raise KeyError(
                    f"{path} missing stage {stage}: require {world_key} or {cam_key}+{camera_key}"
                )
            if joints_world.shape != (frame_indices.shape[0], 24, 3):
                raise ValueError(
                    f"{path} stage={stage}: expected [F,24,3], got {joints_world.shape}"
                )
            if not np.isfinite(joints_world[valid]).all():
                raise ValueError(f"{path}: valid stage={stage} joints contain NaN/Inf")
            joints_world_by_stage[stage] = joints_world
        if np.unique(frame_indices).size != frame_indices.size:
            raise ValueError(f"{path}: duplicate frame_indices")
        units = _archive_scalar_string(archive, "units", "m")
        if units not in {"m", "meter", "meters"}:
            raise ValueError(f"{path}: predictions must use meters, got units={units!r}")
        joint_format = _archive_scalar_string(archive, "joint_format", "smpl24")
        if joint_format.lower() not in {"smpl24", "smpl-24"}:
            raise ValueError(f"{path}: expected joint_format=smpl24, got {joint_format!r}")
        matching_protocol = _archive_scalar_string(archive, "matching_protocol", "")
        if matching_protocol != "human3r_gt_smpl2d_iou_v1":
            raise ValueError(
                f"{path}: unsupported or stale matching_protocol={matching_protocol!r}; "
                "re-export predictions with Human3R 2D matching"
            )
        sequence_name = _archive_scalar_string(archive, "sequence_name", sequence.name)
        if sequence_name not in {sequence.name, sequence.safe_name}:
            raise ValueError(
                f"{path}: sequence_name={sequence_name!r} does not match {sequence.name!r}"
            )
        archive_stride = int(np.asarray(archive["subsample_stride"]).reshape(-1)[0]) if "subsample_stride" in keys else 1
        inference_chunk_size = (
            int(np.asarray(archive["inference_chunk_size"]).reshape(-1)[0])
            if "inference_chunk_size" in keys
            else 0
        )
        inference_chunk_overlap = (
            int(np.asarray(archive["inference_chunk_overlap"]).reshape(-1)[0])
            if "inference_chunk_overlap" in keys
            else 0
        )
        stitch_protocol = _archive_scalar_string(archive, "stitch_protocol", "none")
    return {
        "frame_indices": frame_indices,
        "valid": valid,
        "pred_joints_world_by_stage": joints_world_by_stage,
        "subsample_stride": np.asarray(archive_stride, dtype=np.int64),
        "inference_chunk_size": np.asarray(inference_chunk_size, dtype=np.int64),
        "inference_chunk_overlap": np.asarray(inference_chunk_overlap, dtype=np.int64),
        "stitch_protocol": stitch_protocol,
    }


def select_protocol_frames(
    sequence: EMDB2Sequence,
    prediction: dict[str, np.ndarray],
    subsample_stride: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    frame_indices = prediction["frame_indices"]
    valid = prediction["valid"]
    if frame_indices.size and (frame_indices.min() < 0 or frame_indices.max() >= sequence.frame_count):
        raise IndexError(
            f"{sequence.name}: prediction frame range {frame_indices.min()}..{frame_indices.max()} "
            f"outside 0..{sequence.frame_count - 1}"
        )
    archive_stride = int(np.asarray(prediction["subsample_stride"]).reshape(-1)[0])
    if archive_stride != int(subsample_stride):
        raise ValueError(
            f"{sequence.name}: archive stride={archive_stride}, evaluation stride={subsample_stride}"
        )
    expected_indices = sequence.good_frame_indices[:: max(int(subsample_stride), 1)]
    if not np.array_equal(frame_indices, expected_indices):
        raise ValueError(
            f"{sequence.name}: archive frame_indices do not equal good_frames[::{subsample_stride}] "
            f"(archive={frame_indices.size}, expected={expected_indices.size})"
        )
    good = sequence.good_frame_mask[frame_indices]
    keep = valid & good
    selected_indices = frame_indices[keep]
    order = np.argsort(selected_indices)
    selected_by_stage = {
        stage: prediction["pred_joints_world_by_stage"][stage][keep][order]
        for stage in STAGE_ORDER
    }
    return selected_indices[order], selected_by_stage


def build_contribution_summary(stages: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comparisons = {
        "hsi_scale": ("vggt_nlf", "vggt_nlf_hsi_scale"),
        "trstr": ("vggt_nlf_hsi_scale", "vggt_nlf_hsi_scale_trstr"),
        "total": ("vggt_nlf", "vggt_nlf_hsi_scale_trstr"),
    }
    output: dict[str, Any] = {}
    for name, (before, after) in comparisons.items():
        before_metrics = stages[before]["paper_metrics_frame_weighted"]
        after_metrics = stages[after]["paper_metrics_frame_weighted"]
        output[name] = {
            "from": STAGE_LABELS[before],
            "to": STAGE_LABELS[after],
            "positive_means_improvement": True,
            **{
                key: float(before_metrics[key] - after_metrics[key])
                for key in ("W-MPJPE_mm", "WA-MPJPE_mm", "RTE_percent")
            },
        }
    return output


def _archive_scalar_string(archive: Any, key: str, default: str) -> str:
    if key not in archive.files:
        return default
    value = np.asarray(archive[key])
    if value.size != 1:
        raise ValueError(f"Archive metadata {key!r} must be scalar")
    return str(value.reshape(-1)[0])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
