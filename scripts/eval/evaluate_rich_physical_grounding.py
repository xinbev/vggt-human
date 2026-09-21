from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_smpl import build_model  # noqa: E402
from vggt_omega.data.rich_physical_grounding import RichPhysicalGroundingDataset  # noqa: E402
from vggt_omega.evaluation.rich_physical_grounding import (  # noqa: E402
    CascadeConfig,
    GroundEstimatorConfig,
    compute_physical_metrics,
    evaluate_prediction_chunk,
    load_evaluation_weights,
    configure_reference_cascade_model,
    run_metric_cascade,
)
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.training.config import deep_update, load_yaml_config, require_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RICH physical-grounding metrics without visualization code")
    parser.add_argument("--official-root", type=Path, default=Path("/home/zhw/xyb_space/RICH/official"))
    parser.add_argument("--support-root", type=Path, default=Path("/home/zhw/xyb_space/RICH/hmr4d_support"))
    parser.add_argument("--path-config", type=Path, default=ROOT / "configs/path.yaml")
    parser.add_argument(
        "--train-config",
        type=Path,
        default=ROOT / "configs/train_smpl_hsi_nlf_stage2_human_scene_align.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt",
    )
    parser.add_argument(
        "--scale-checkpoint",
        type=Path,
        default=ROOT
        / "outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt",
    )
    parser.add_argument("--baseline-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/eval/rich_physical_grounding")
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--sequence-manifest", type=Path, default=None)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-frames-per-sequence", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--window-size",
        "--chunk-size",
        dest="window_size",
        type=int,
        default=100,
        help="Non-overlapping evaluation window length; the final shorter window is retained.",
    )
    parser.add_argument("--max-humans", type=int, default=8)
    parser.add_argument("--image-resolution", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--coarse-min-anchor-pixels", type=int, default=32)
    parser.add_argument("--coarse-scale-min", type=float, default=0.10)
    parser.add_argument("--coarse-scale-max", type=float, default=10.0)
    parser.add_argument("--coarse-anchor-stride", type=int, default=8)
    parser.add_argument("--scene-point-stride", type=int, default=2)
    parser.add_argument("--max-scene-depth-m", type=float, default=80.0)
    parser.add_argument("--bbox-expand-ratio", type=float, default=0.10)
    parser.add_argument("--footprint-margin-m", type=float, default=0.35)
    parser.add_argument("--vertical-window-m", type=float, default=0.75)
    parser.add_argument("--ground-quantile", type=float, default=0.90)
    parser.add_argument("--min-support-points", type=int, default=64)
    parser.add_argument("--tolerance-m", type=float, default=0.005)
    parser.add_argument("--target-min-iou", type=float, default=0.30)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("RICH model evaluation requires a CUDA GPU")
    device = torch.device("cuda")
    output_dir = args.output_dir.expanduser().resolve()
    sequence_dir = output_dir / "sequences"
    sequence_dir.mkdir(parents=True, exist_ok=True)

    path_config = load_yaml_config(args.path_config)
    train_config = load_yaml_config(args.train_config)
    config = deep_update(path_config, train_config)
    config, checkpoint_config_report = configure_reference_cascade_model(
        config, args.checkpoint, max_humans=args.max_humans
    )
    model_config = config.setdefault("model", {})
    if not bool(model_config.get("nlf_use_detector", False)):
        raise ValueError("Reference-aligned RICH evaluation requires model.nlf_use_detector=true")
    if str(model_config.get("smpl_provider", "")) != "nlf":
        raise ValueError("Dedicated RICH evaluation requires model.smpl_provider=nlf")
    image_resolution = int(args.image_resolution or config.get("data", {}).get("image_resolution", 512))
    max_humans = int(model_config["num_smpl_queries"])
    baseline_checkpoint = args.baseline_checkpoint or Path(require_path(config, "checkpoints.vggt_baseline"))
    smpl_model_dir = Path(require_path(config, "assets.smpl_model_dir"))

    sequence_filters = list(args.sequence)
    if args.sequence_manifest is not None:
        sequence_filters.extend(load_manifest(args.sequence_manifest))
    dataset = RichPhysicalGroundingDataset(
        official_root=args.official_root,
        support_root=args.support_root,
        image_resolution=image_resolution,
        patch_size=int(model_config.get("patch_size", 16)),
        resize_mode=str(config.get("data", {}).get("resize_mode", "balanced")),
        max_humans=max_humans,
        sequence_filters=sequence_filters,
        max_sequences=args.max_sequences,
        frame_stride=args.frame_stride,
        max_frames_per_sequence=args.max_frames_per_sequence,
    )
    cascade_config = CascadeConfig(
        confidence_threshold=args.confidence_threshold,
        coarse_min_anchor_pixels=args.coarse_min_anchor_pixels,
        coarse_scale_min=args.coarse_scale_min,
        coarse_scale_max=args.coarse_scale_max,
        coarse_anchor_stride=args.coarse_anchor_stride,
    )
    ground_config = GroundEstimatorConfig(
        scene_point_stride=args.scene_point_stride,
        max_scene_depth_m=args.max_scene_depth_m,
        bbox_expand_ratio=args.bbox_expand_ratio,
        footprint_margin_m=args.footprint_margin_m,
        vertical_window_m=args.vertical_window_m,
        ground_quantile=args.ground_quantile,
        min_support_points=args.min_support_points,
        tolerance_m=args.tolerance_m,
        target_min_iou=args.target_min_iou,
        target_min_confidence=args.confidence_threshold,
    )

    print(
        f"[setup] camera views={len(dataset)} image_resolution={image_resolution} "
        f"window_size={args.window_size}",
        flush=True,
    )
    print(
        "[setup] Stage-2 align schema="
        f"{checkpoint_config_report['configured_align_feature_version']} "
        f"(input_dim={checkpoint_config_report['checkpoint_align_input_dim']})",
        flush=True,
    )
    model = build_model(config).to(device).eval()
    weight_report = load_evaluation_weights(
        model=model,
        baseline_checkpoint=baseline_checkpoint,
        stage2_checkpoint=args.checkpoint,
        scale_checkpoint=args.scale_checkpoint,
        device=device,
    )
    smpl = SMPLLayer(smpl_model_dir).to(device).eval()
    run_metadata = {
        "protocol": "UniCon3R Table 3 physical-grounding reproduction hypothesis",
        "official_identity_claim": False,
        "target_person_selection": "NLF detector prediction matched to RICH bbx_xys by maximum IoU",
        "ground_source": "model-reconstructed metric depth, not the RICH reference scan",
        "aggregation": ["pooled_frames", "mean_of_camera_view_metrics"],
        "evaluation_window_frames": args.window_size,
        "windowing": "non-overlapping within each camera view; final shorter window retained",
        "selection_warning": (
            "UniCon3R's exact 40-sequence moving-camera list is not published in the provided materials. "
            "This run evaluates exactly the camera-view keys listed below."
        ),
        "selected_camera_views": [record.vid for record in dataset],
        "cascade": asdict(cascade_config),
        "ground_estimator": asdict(ground_config),
        "weights": weight_report,
        "checkpoint_model_config": checkpoint_config_report,
        "paths": {
            "official_root": str(args.official_root),
            "support_root": str(args.support_root),
            "baseline_checkpoint": str(baseline_checkpoint),
            "stage2_checkpoint": str(args.checkpoint),
            "scale_checkpoint": str(args.scale_checkpoint),
            "smpl_model_dir": str(smpl_model_dir),
        },
    }
    write_json(output_dir / "run_config.json", run_metadata)

    all_frames: list[dict[str, Any]] = []
    all_windows: list[dict[str, Any]] = []
    sequence_results: list[dict[str, Any]] = []
    for sequence_index, record in enumerate(dataset, start=1):
        result_path = sequence_dir / f"{safe_name(record.vid)}.json"
        if args.resume and result_path.is_file():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if int(payload.get("evaluation_window_frames", -1)) == args.window_size:
                sequence_results.append(payload["sequence"])
                all_frames.extend(payload["frames"])
                all_windows.extend(payload["windows"])
                print(f"[resume {sequence_index}/{len(dataset)}] {record.vid}", flush=True)
                continue
            print(
                f"[recompute {sequence_index}/{len(dataset)}] {record.vid}: "
                "saved result uses a different evaluation window",
                flush=True,
            )

        positions = dataset.selected_label_positions(record)
        print(f"[sequence {sequence_index}/{len(dataset)}] {record.vid} frames={len(positions)}", flush=True)
        frame_rows: list[dict[str, Any]] = []
        window_reports = []
        for window_index, start in enumerate(range(0, len(positions), args.window_size)):
            window_positions = positions[start : start + args.window_size]
            batch = dataset.load_batch(record, window_positions)
            images = batch.images.unsqueeze(0).to(device=device, non_blocking=True)
            boxes = batch.query_boxes.unsqueeze(0).to(device=device, non_blocking=True)
            mask = batch.query_mask.unsqueeze(0).to(device=device, non_blocking=True)
            predictions, cascade_report = run_metric_cascade(
                model, smpl, images, cascade_config
            )
            evaluated, ground_report = evaluate_prediction_chunk(predictions, smpl, boxes, mask, ground_config)
            for local_index, row in enumerate(evaluated):
                row.update(
                    {
                        "vid": record.vid,
                        "recording": record.recording,
                        "camera": record.camera,
                        "window_index": window_index,
                        "label_position": int(batch.label_positions[local_index]),
                        "source_frame_id": int(batch.source_frame_ids[local_index]),
                        "image": str(batch.image_paths[local_index]),
                    }
                )
                frame_rows.append(row)
            window_clearances = [
                float(row["clearance_m"])
                for row in evaluated
                if row.get("valid") and "clearance_m" in row
            ]
            window_metrics = compute_physical_metrics(
                window_clearances, tolerance_m=ground_config.tolerance_m
            )
            window_summary = {
                "vid": record.vid,
                "recording": record.recording,
                "camera": record.camera,
                "window_index": window_index,
                "selected_frames": len(evaluated),
                "invalid_frames": len(evaluated) - window_metrics["valid_frames"],
                "label_position_start": int(batch.label_positions[0]),
                "label_position_end": int(batch.label_positions[-1]),
                "source_frame_id_start": int(batch.source_frame_ids[0]),
                "source_frame_id_end": int(batch.source_frame_ids[-1]),
                **window_metrics,
            }
            all_windows.append(window_summary)
            window_reports.append(
                {
                    "window_index": window_index,
                    "label_positions": list(batch.label_positions),
                    "metrics": window_summary,
                    "ground": ground_report,
                    "cascade": cascade_report,
                }
            )
            print(
                f"  [window {window_index + 1}] labels={batch.label_positions[0]}..{batch.label_positions[-1]} "
                f"valid={sum(bool(row['valid']) for row in evaluated)}/{len(evaluated)}",
                flush=True,
            )
            del predictions, images, boxes, mask
            torch.cuda.empty_cache()

        valid_clearances = [float(row["clearance_m"]) for row in frame_rows if row.get("valid") and "clearance_m" in row]
        metrics = compute_physical_metrics(valid_clearances, tolerance_m=ground_config.tolerance_m)
        sequence_summary = {
            "vid": record.vid,
            "recording": record.recording,
            "camera": record.camera,
            "selected_frames": len(positions),
            "invalid_frames": len(positions) - metrics["valid_frames"],
            **metrics,
        }
        payload = {
            "evaluation_window_frames": args.window_size,
            "sequence": sequence_summary,
            "assets": {
                "scan_path": str(record.scan_path),
                "calibration_path": str(record.calibration_path),
                "multicam2world_path": str(record.multicam2world_path),
                "reference_assets_used_for_ground": False,
            },
            "frames": frame_rows,
            "windows": window_reports,
        }
        write_json(result_path, payload)
        sequence_results.append(sequence_summary)
        all_frames.extend(frame_rows)

    valid_clearances = [float(row["clearance_m"]) for row in all_frames if row.get("valid") and "clearance_m" in row]
    pooled = compute_physical_metrics(valid_clearances, tolerance_m=ground_config.tolerance_m)
    metric_names = ("collision_ratio_pct", "penetrate_cm", "float_cm", "penetration_max_cm")
    sequence_mean = {
        key: float(np.mean([float(result[key]) for result in sequence_results])) if sequence_results else 0.0
        for key in metric_names
    }
    summary = {
        **run_metadata,
        "camera_view_count": len(sequence_results),
        "selected_frames": len(all_frames),
        "invalid_frames": sum(not bool(row.get("valid")) for row in all_frames),
        "pooled_frames": pooled,
        "mean_of_camera_views": sequence_mean,
    }
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "per_frame.csv", all_frames)
    write_csv(output_dir / "per_window.csv", all_windows)
    write_csv(output_dir / "per_sequence.csv", sequence_results)
    write_csv(output_dir / "summary.csv", [{"aggregation": "pooled_frames", **pooled}, {"aggregation": "mean_of_camera_views", **sequence_mean}])
    print("[done] RICH physical-grounding evaluation", flush=True)
    print(json.dumps({"pooled_frames": pooled, "mean_of_camera_views": sequence_mean}, indent=2), flush=True)
    print(f"[output] {output_dir}", flush=True)


def load_manifest(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Sequence manifest not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]


def safe_name(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True)
    return value


if __name__ == "__main__":
    main()
