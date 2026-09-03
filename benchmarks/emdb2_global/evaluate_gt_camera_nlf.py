#!/usr/bin/env python3
"""Evaluate the EMDB-2 NLF plus GT-camera oracle ablation."""

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

from benchmarks.emdb2_global.data import decode_gt_world_joints, load_emdb2_sequences  # noqa: E402
from benchmarks.emdb2_global.export_gt_camera_nlf import (  # noqa: E402
    MATCHING_PROTOCOL,
    ORACLE_PROTOCOL,
)
from benchmarks.emdb2_global.metrics import evaluate_global_metrics  # noqa: E402
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.training.config import deep_update, load_yaml_config, require_path  # noqa: E402


def main() -> None:
    args = parse_args()
    import torch

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({args.device}) but unavailable")
    device = torch.device(args.device)
    cfg = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.config))
    stride = max(int(args.subsample_stride), 1)
    chunk_length = int(args.chunk_length or max(int(100 / stride), 1))
    emdb_root = Path(args.emdb_root or require_path(cfg, "datasets.emdb_root")).expanduser()
    predictions_root = Path(args.predictions_root).expanduser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences = load_emdb2_sequences(emdb_root, args.sequence_filter)
    if args.max_sequences > 0:
        sequences = sequences[: args.max_sequences]

    smpl_root = require_path(cfg, "assets.smpl_model_dir", allow_empty=False)
    neutral_smpl = SMPLLayer(smpl_root).to(device).eval()
    joint_regressor = neutral_smpl.layer.J_regressor.detach()
    smpl_layers = {
        gender: SMPLLayer(smpl_root, gender=gender).to(device).eval()
        for gender in ("male", "female")
    }
    all_w: list[np.ndarray] = []
    all_wa: list[np.ndarray] = []
    all_rte: list[np.ndarray] = []
    sequence_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    total_good = total_sampled = total_matched = 0

    for sequence in sequences:
        expected_indices = sequence.good_frame_indices[::stride]
        total_good += int(sequence.good_frame_indices.size)
        total_sampled += int(expected_indices.size)
        path = predictions_root / f"{sequence.safe_name}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing GT-camera NLF archive: {path}")
        with np.load(path, allow_pickle=False) as archive:
            require_metadata(archive, path)
            frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64).reshape(-1)
            valid = np.asarray(archive["valid"], dtype=bool).reshape(-1)
            pred_world = np.asarray(archive["pred_joints_world"], dtype=np.float32)
            archive_stride = int(np.asarray(archive["subsample_stride"]).reshape(-1)[0])
        if archive_stride != stride or not np.array_equal(frame_indices, expected_indices):
            raise ValueError(f"{path}: frame protocol does not match good_frames[::{stride}]")
        if valid.shape != frame_indices.shape or pred_world.shape != (frame_indices.size, 24, 3):
            raise ValueError(f"{path}: invalid prediction shapes")
        keep_indices = frame_indices[valid]
        pred_world = pred_world[valid]
        if keep_indices.size < 2:
            raise RuntimeError(f"{sequence.name}: fewer than two matched predictions")
        target_world = decode_gt_world_joints(
            sequence,
            keep_indices,
            smpl_layers[sequence.gender],
            device,
            chunk_size=int(args.smpl_batch_size),
            joint_regressor=joint_regressor,
        )
        result = evaluate_global_metrics(
            target_world,
            pred_world,
            chunk_length=chunk_length,
            root_index=int(args.root_index),
        )
        all_w.append(result.w_mpjpe_mm)
        all_wa.append(result.wa_mpjpe_mm)
        all_rte.append(result.rte_percent)
        total_matched += int(keep_indices.size)
        summary = result.summary()
        sequence_rows.append(
            {
                "sequence": sequence.name,
                "good_frames": int(sequence.good_frame_indices.size),
                "sampled_frames": int(expected_indices.size),
                "matched_frames": int(keep_indices.size),
                "prediction_coverage": float(keep_indices.size / max(expected_indices.size, 1)),
                "gt_root_displacement_m": float(np.linalg.norm(np.diff(target_world[:, int(args.root_index)], axis=0), axis=-1).sum()),
                "max_matched_frame_gap": int(np.diff(keep_indices).max()),
                **summary,
            }
        )
        for local, frame_index in enumerate(keep_indices.tolist()):
            frame_rows.append(
                {
                    "sequence": sequence.name,
                    "frame_index": int(frame_index),
                    "W-MPJPE_mm": float(result.w_mpjpe_mm[local]),
                    "WA-MPJPE_mm": float(result.wa_mpjpe_mm[local]),
                    "RTE_percent": float(result.rte_percent[local]),
                }
            )
        print(
            f"[gt-camera] {sequence.name} frames={keep_indices.size}/{expected_indices.size} "
            f"W={summary['W-MPJPE_mm']:.2f}mm WA={summary['WA-MPJPE_mm']:.2f}mm "
            f"RTE={summary['RTE_percent']:.3f}%",
            flush=True,
        )

    metrics = {
        "W-MPJPE_mm": float(np.concatenate(all_w).mean()),
        "WA-MPJPE_mm": float(np.concatenate(all_wa).mean()),
        "RTE_percent": float(np.concatenate(all_rte).mean()),
    }
    summary = {
        "benchmark": f"emdb2_s{stride}_nlf_gt_camera_oracle_v1",
        "dataset": "EMDB-2",
        "method": "RGB-NLF + EMDB GT intrinsics + EMDB GT extrinsics",
        "oracle_warning": "GT camera is privileged test-time information; do not report as a pure RGB prediction result.",
        "paper_metrics_frame_weighted": metrics,
        "sequence_macro_diagnostics": {
            key: float(np.mean([float(row[key]) for row in sequence_rows]))
            for key in metrics
        },
        "protocol_sequence_count": len(sequences),
        "good_frames": total_good,
        "sampled_frames": total_sampled,
        "matched_frames": total_matched,
        "sampling_rate": float(total_sampled / max(total_good, 1)),
        "prediction_coverage": float(total_matched / max(total_sampled, 1)),
        "subsample_stride": stride,
        "chunk_length": chunk_length,
        "matching_protocol": MATCHING_PROTOCOL,
        "oracle_camera_protocol": ORACLE_PROTOCOL,
        "predictions_root": str(predictions_root),
    }
    write_csv(output_dir / "sequence_metrics.csv", sequence_rows)
    write_csv(output_dir / "frame_metrics.csv", frame_rows)
    write_csv(output_dir / "stage_metrics.csv", [{"stage": "nlf_gt_camera", **metrics}])
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--config", default="benchmarks/emdb2_global/config.yaml")
    parser.add_argument("--emdb-root", default="")
    parser.add_argument("--predictions-root", required=True)
    parser.add_argument("--output-dir", default="outputs/eval/emdb2_s7_nlf_gt_camera/metrics")
    parser.add_argument("--subsample-stride", type=int, default=7)
    parser.add_argument("--chunk-length", type=int, default=14)
    parser.add_argument("--root-index", type=int, default=0)
    parser.add_argument("--smpl-batch-size", type=int, default=512)
    parser.add_argument("--sequence-filter", default="")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def require_metadata(archive: Any, path: Path) -> None:
    required = {
        "frame_indices", "pred_joints_world", "valid", "subsample_stride",
        "matching_protocol", "oracle_camera_protocol",
    }
    missing = required - set(archive.files)
    if missing:
        raise KeyError(f"{path} missing keys: {sorted(missing)}")
    matching = str(np.asarray(archive["matching_protocol"]).reshape(-1)[0])
    oracle = str(np.asarray(archive["oracle_camera_protocol"]).reshape(-1)[0])
    if matching != MATCHING_PROTOCOL or oracle != ORACLE_PROTOCOL:
        raise ValueError(f"{path}: stale or incompatible protocol metadata")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
