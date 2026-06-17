#!/usr/bin/env python
"""Per-person long-sequence diagnostics for HSI SMPL/depth/contact results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Any

import inspect
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

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
    canonical_depth,
    decode_smpl_batch,
    depth_triplet,
    describe_array,
    fmt,
    get_foot_sole_indices,
    greedy_match,
    load_config,
    load_training_checkpoint,
    load_vggt_baseline,
    move_to_device,
    project_points,
    sample_depth_at_points,
    sample_local_support_plane_signed_delta,
    scale_points_to_depth,
)
from scripts.train.train_smpl import build_model  # noqa: E402
from vggt_omega.data import BedlamDataset, bedlam_collate_fn  # noqa: E402
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.training.config import require_path  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402
from vggt_omega.utils.rotation import rot6d_to_axis_angle  # noqa: E402


TRACK_SOURCE_NAMES = {
    2: "explicit",
    1: "person_index",
    0: "slot",
    -1: "unknown",
}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args)
    requested_num_frames = int(args.num_frames)
    args.num_frames = resolve_effective_num_frames(args)
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

    loader, selected_indices, frame_paths_by_index = build_sequence_loader(config, args)
    smpl = SMPLLayer(require_path(config, "assets.smpl_model_dir", allow_empty=False)).to(device).eval()

    frame_rows: list[dict[str, Any]] = []
    global_depth_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    scales: list[float] = []
    biases: list[float] = []

    processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch_size = int(batch["images"].shape[0])
            batch_indices = selected_indices[processed : processed + batch_size]
            batch_frame_paths = [frame_paths_by_index[index] for index in batch_indices]
            batch = move_to_device(batch, device)
            predictions = model(
                batch["images"],
                smpl_query_boxes=batch["gt_boxes"] if args.use_gt_box_prior else None,
                smpl_query_boxes_mask=batch["boxes_mask"] if args.use_gt_box_prior else None,
            )
            batch_frame_rows, batch_depth_rows, batch_temporal_rows = evaluate_batch(
                predictions,
                batch,
                smpl,
                config,
                args,
                batch_indices,
                batch_frame_paths,
            )
            frame_rows.extend(batch_frame_rows)
            global_depth_rows.extend(batch_depth_rows)
            temporal_rows.extend(batch_temporal_rows)
            if "hsi_scene_scale" in predictions:
                scales.extend(predictions["hsi_scene_scale"].detach().float().cpu().reshape(-1).tolist())
            if "hsi_scene_depth_bias" in predictions:
                biases.extend(predictions["hsi_scene_depth_bias"].detach().float().cpu().reshape(-1).tolist())
            processed += batch_size
            if args.log_interval > 0 and processed % args.log_interval == 0:
                print(f"[sequence-person-diagnostics] processed={processed}")

    person_summary = summarize_person_rows(frame_rows, temporal_rows, args)
    global_summary = build_global_summary(frame_rows, global_depth_rows, temporal_rows, scales, biases, args, selected_indices)

    frame_csv = output_dir / "hsi_sequence_frame_person_metrics.csv"
    person_csv = output_dir / "hsi_sequence_person_summary.csv"
    depth_csv = output_dir / "hsi_sequence_frame_depth_metrics.csv"
    temporal_csv = output_dir / "hsi_sequence_person_temporal_metrics.csv"
    write_csv(frame_csv, frame_rows, FRAME_FIELDS)
    write_csv(person_csv, person_summary, PERSON_FIELDS)
    write_csv(depth_csv, global_depth_rows, DEPTH_FIELDS)
    write_csv(temporal_csv, temporal_rows, TEMPORAL_FIELDS)

    out_json = output_dir / "hsi_sequence_person_diagnostics.json"
    payload = {
        "checkpoint": str(args.checkpoint),
        "train_config": str(args.train_config),
        "num_sequences": int(processed),
        "num_frames_requested": int(requested_num_frames),
        "num_frames_effective": int(args.num_frames),
        "frame_stride": int(args.frame_stride),
        "intrinsics_source": str(args.intrinsics_source),
        "use_gt_box_prior": bool(args.use_gt_box_prior),
        "selected_dataset_indices": [int(index) for index in selected_indices],
        "global_summary": global_summary,
        "per_person_summary": person_summary,
        "outputs": {
            "frame_person_csv": str(frame_csv),
            "person_summary_csv": str(person_csv),
            "frame_depth_csv": str(depth_csv),
            "temporal_csv": str(temporal_csv),
            "json": str(out_json),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_human_summary(payload)
    print(json.dumps({"output_json": str(out_json), "num_frame_person_rows": len(frame_rows), "num_person_tracks": len(person_summary)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate per-person HSI sequence diagnostics")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_hsi_refine.yaml")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--output-dir", default="outputs/eval/hsi_sequence_person_diagnostics")
    parser.add_argument("--device", default="")
    parser.add_argument("--split", default="Training")
    parser.add_argument("--image", default="", help="Optional start RGB image path for one BEDLAM clip")
    parser.add_argument("--num-frames", type=int, default=27)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=1)
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
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def resolve_effective_num_frames(args: argparse.Namespace) -> int:
    requested = max(int(args.num_frames), 1)
    stride = max(int(args.frame_stride), 1)
    if not args.image:
        return requested
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Start image not found: {image_path}")
    if image_path.parent.name != "rgb":
        raise ValueError(f"--image must point to a BEDLAM rgb frame, got: {image_path}")
    frames = sorted(path.resolve() for path in image_path.parent.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    try:
        start_idx = frames.index(image_path)
    except ValueError as exc:
        raise ValueError(f"Start image is not in its rgb directory listing: {image_path}") from exc
    available = 0
    for offset in range(requested):
        frame_idx = start_idx + offset * stride
        if frame_idx >= len(frames):
            break
        available += 1
    if available <= 0:
        raise RuntimeError(f"No frames available from start image: {image_path}")
    if available < requested:
        print(
            "[sequence-person-diagnostics] "
            f"requested NUM_FRAMES={requested}, but only {available} frames are available "
            f"from {image_path.name} with stride={stride}; using {available}."
        )
    return available


def build_sequence_loader(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[DataLoader, list[int], dict[int, list[Path]]]:
    data_cfg = config["data"]
    dataset = make_bedlam_dataset(config, args)
    selected_indices = select_dataset_indices(dataset, args)
    frame_paths_by_index = {index: frame_paths_for_dataset_index(dataset, index) for index in selected_indices}
    subset = Subset(dataset, selected_indices)
    loader = DataLoader(
        subset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        collate_fn=bedlam_collate_fn,
        drop_last=False,
    )
    return loader, selected_indices, frame_paths_by_index


def make_bedlam_dataset(config: dict[str, Any], args: argparse.Namespace) -> BedlamDataset:
    data_cfg = config["data"]
    root = require_path(config, data_cfg.get("root_key", "datasets.bedlam_root"))
    boxes_root = require_path(config, data_cfg["boxes_root_key"], allow_empty=False)
    try:
        return BedlamDataset(
            root=root,
            split=args.split,
            sequence_length=int(args.num_frames),
            stride=int(args.frame_stride),
            image_size=int(data_cfg["image_size"]),
            max_humans=int(data_cfg["max_humans"]),
            require_smpl=True,
            require_depth=True,
            boxes_root=boxes_root,
            require_boxes=True,
        )
    except RuntimeError as exc:
        message = str(exc)
        if args.image or "No trainable frame windows found" not in message:
            raise
        fallback_frames = infer_max_sequence_length(Path(root), str(args.split), int(args.frame_stride))
        if fallback_frames >= int(args.num_frames):
            raise
        args.num_frames = fallback_frames
        config["data"]["sequence_length"] = fallback_frames
        print(
            "[sequence-person-diagnostics] "
            f"requested NUM_FRAMES is unavailable for split={args.split}; "
            f"falling back to max sequence_length={fallback_frames}."
        )
        return BedlamDataset(
            root=root,
            split=args.split,
            sequence_length=fallback_frames,
            stride=int(args.frame_stride),
            image_size=int(data_cfg["image_size"]),
            max_humans=int(data_cfg["max_humans"]),
            require_smpl=True,
            require_depth=True,
            boxes_root=boxes_root,
            require_boxes=True,
        )


def infer_max_sequence_length(root: Path, split: str, stride: int) -> int:
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"BEDLAM split directory not found: {split_dir}")
    stride = max(int(stride), 1)
    max_frames = 0
    for seq_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        rgb_dir = seq_dir / "rgb"
        if not rgb_dir.is_dir():
            continue
        count = sum(1 for path in rgb_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if count > 0:
            max_frames = max(max_frames, 1 + (count - 1) // stride)
    if max_frames <= 0:
        raise RuntimeError(f"No valid RGB frames found under {split_dir}")
    return max_frames


def select_dataset_indices(dataset: BedlamDataset, args: argparse.Namespace) -> list[int]:
    max_samples = max(int(args.max_samples), 1)
    if args.image:
        first = find_dataset_index_for_image(dataset, Path(args.image).expanduser())
        seq_idx, start_idx = dataset._index[first]  # noqa: SLF001
        selected = []
        for index, (candidate_seq_idx, candidate_start_idx) in enumerate(dataset._index):  # noqa: SLF001
            if candidate_seq_idx == seq_idx and candidate_start_idx >= start_idx:
                selected.append(index)
                if len(selected) >= max_samples:
                    break
        return selected
    start = int(args.start_index)
    end = min(len(dataset), start + max_samples)
    return list(range(start, end))


def find_dataset_index_for_image(dataset: BedlamDataset, image_path: Path) -> int:
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Start image not found: {image_path}")
    if image_path.parent.name != "rgb":
        raise ValueError(f"--image must point to a BEDLAM rgb frame, got: {image_path}")
    seq_dir = image_path.parent.parent.resolve()
    frame_id = image_path.stem
    for index, (seq_idx, start_idx) in enumerate(dataset._index):  # noqa: SLF001
        candidate_seq_dir, frame_ids = dataset._sequences[seq_idx]  # noqa: SLF001
        if candidate_seq_dir.resolve() == seq_dir and frame_ids[start_idx] == frame_id:
            return index
    raise ValueError(f"Could not find a dataset window starting at {image_path}")


def frame_paths_for_dataset_index(dataset: BedlamDataset, dataset_index: int) -> list[Path]:
    seq_idx, start_idx = dataset._index[int(dataset_index)]  # noqa: SLF001
    seq_dir, frame_ids = dataset._sequences[seq_idx]  # noqa: SLF001
    return [seq_dir / "rgb" / f"{frame_ids[start_idx + step * dataset.stride]}.png" for step in range(dataset.sequence_length)]


def evaluate_batch(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    smpl: SMPLLayer,
    config: dict[str, Any],
    args: argparse.Namespace,
    dataset_indices: list[int],
    frame_paths: list[list[Path]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    required_hsi = ("hsi_refined_pred_poses", "hsi_refined_pred_betas", "hsi_refined_pred_transl_cam")
    missing_hsi = [key for key in required_hsi if key not in predictions]
    if missing_hsi:
        raise ValueError(f"Model did not produce HSI refined SMPL outputs: missing {missing_hsi}")

    base = decode_smpl_batch(predictions["pred_poses"], predictions["pred_betas"], predictions["pred_transl_cam"], smpl)
    hsi = decode_smpl_batch(
        predictions["hsi_refined_pred_poses"],
        predictions["hsi_refined_pred_betas"],
        predictions["hsi_refined_pred_transl_cam"],
        smpl,
    )
    gt_poses = rot6d_to_axis_angle(batch["gt_pose_6d"].reshape(-1, 24, 6)).reshape(*batch["gt_pose_6d"].shape[:3], 72)
    gt = decode_smpl_batch(gt_poses, batch["gt_betas"], batch["gt_transl_cam"], smpl)
    raw_depth, hsi_depth, gt_depth = depth_triplet(predictions, batch)
    intrinsics = resolve_intrinsics(predictions, batch, config, args)
    confs = predictions["pred_confs"].detach()
    if confs.ndim == 4 and confs.shape[-1] == 1:
        confs = confs[..., 0]
    smpl_mask = batch["smpl_mask"].bool()
    track_ids = batch.get("gt_track_ids", batch.get("person_ids"))
    track_mask = batch.get("gt_track_mask", batch.get("person_id_mask"))
    track_source = batch.get("gt_track_source")
    track_quality = batch.get("gt_track_quality")
    if track_ids is None or track_mask is None:
        raise ValueError("Batch is missing gt_track_ids/gt_track_mask; rebuild BEDLAM boxes or update dataset")

    sole_indices = get_foot_sole_indices(smpl, int(args.foot_sole_num_vertices), device=gt["vertices"].device)
    frame_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    temporal_records: dict[tuple[int, int], dict[int, dict[str, int]]] = {}
    batch_size, num_frames, _ = smpl_mask.shape

    for b in range(batch_size):
        sequence_name = frame_paths[b][0].parent.parent.name if frame_paths[b] else f"sample_{dataset_indices[b]}"
        for s in range(num_frames):
            frame_depth_row = frame_depth_metrics(
                raw_depth[b, s],
                hsi_depth[b, s],
                gt_depth[b, s],
                predictions,
                b,
                s,
                dataset_indices[b],
                sequence_name,
                frame_paths[b][s],
                args,
            )
            depth_rows.append(frame_depth_row)
            gt_idx = torch.nonzero(smpl_mask[b, s], as_tuple=False).flatten()
            pred_idx = torch.nonzero(confs[b, s] >= float(args.conf_threshold), as_tuple=False).flatten()
            if gt_idx.numel() == 0 or pred_idx.numel() == 0:
                continue
            matches = match_frame_predictions(base, hsi, gt, b, s, pred_idx, gt_idx, args.match_source)
            for pred_local, gt_local in matches:
                q = int(pred_idx[pred_local].item())
                g = int(gt_idx[gt_local].item())
                if not bool(track_mask[b, s, g].detach().cpu()):
                    continue
                track_id = int(track_ids[b, s, g].detach().cpu())
                if track_id < 0:
                    continue
                row = build_frame_person_row(
                    base,
                    hsi,
                    gt,
                    raw_depth[b, s],
                    hsi_depth[b, s],
                    gt_depth[b, s],
                    intrinsics[b, s],
                    batch,
                    predictions,
                    confs,
                    sole_indices,
                    args,
                    dataset_indices[b],
                    sequence_name,
                    frame_paths[b][s],
                    b,
                    s,
                    q,
                    g,
                    track_id,
                    int(track_source[b, s, g].detach().cpu()) if track_source is not None else -1,
                    float(track_quality[b, s, g].detach().cpu()) if track_quality is not None else np.nan,
                )
                frame_rows.append(row)
                temporal_records.setdefault((int(dataset_indices[b]), track_id), {})[s] = {
                    "batch": b,
                    "query": q,
                    "gt_index": g,
                    "track_source": int(row["track_source_code"]),
                    "sequence_name": sequence_name,
                }

    temporal_rows = build_temporal_rows(temporal_records, base, hsi, gt, frame_paths, dataset_indices)
    return frame_rows, depth_rows, temporal_rows


def resolve_intrinsics(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.intrinsics_source == "gt":
        return batch["K_scal3r"].to(device=batch["images"].device, dtype=batch["images"].dtype)
    return encoding_to_camera(
        predictions["pose_enc"],
        image_size_hw=(int(config["data"]["image_size"]), int(config["data"]["image_size"])),
        build_intrinsics=True,
    )[1]


def match_frame_predictions(
    base: dict[str, torch.Tensor],
    hsi: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    batch_idx: int,
    frame_idx: int,
    pred_idx: torch.Tensor,
    gt_idx: torch.Tensor,
    match_source: str,
) -> list[tuple[int, int]]:
    if match_source == "hsi":
        return greedy_match(hsi["joints"][batch_idx, frame_idx, pred_idx, :24], gt["joints"][batch_idx, frame_idx, gt_idx, :24])
    if match_source == "best":
        base_matches = greedy_match(base["joints"][batch_idx, frame_idx, pred_idx, :24], gt["joints"][batch_idx, frame_idx, gt_idx, :24])
        hsi_matches = greedy_match(hsi["joints"][batch_idx, frame_idx, pred_idx, :24], gt["joints"][batch_idx, frame_idx, gt_idx, :24])
        return hsi_matches if match_cost(hsi_matches, hsi, gt, batch_idx, frame_idx, pred_idx, gt_idx) < match_cost(base_matches, base, gt, batch_idx, frame_idx, pred_idx, gt_idx) else base_matches
    return greedy_match(base["joints"][batch_idx, frame_idx, pred_idx, :24], gt["joints"][batch_idx, frame_idx, gt_idx, :24])


def match_cost(
    matches: list[tuple[int, int]],
    pred: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    batch_idx: int,
    frame_idx: int,
    pred_idx: torch.Tensor,
    gt_idx: torch.Tensor,
) -> float:
    if not matches:
        return float("inf")
    values = []
    for pred_local, gt_local in matches:
        q = int(pred_idx[pred_local].item())
        g = int(gt_idx[gt_local].item())
        values.append(float(torch.linalg.norm(pred["joints"][batch_idx, frame_idx, q, :24] - gt["joints"][batch_idx, frame_idx, g, :24], dim=-1).mean().detach().cpu()))
    return float(np.mean(values))


def build_frame_person_row(
    base: dict[str, torch.Tensor],
    hsi: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    raw_depth: torch.Tensor,
    hsi_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    intrinsics: torch.Tensor,
    batch: dict[str, torch.Tensor],
    predictions: dict[str, torch.Tensor],
    confs: torch.Tensor,
    sole_indices: torch.Tensor,
    args: argparse.Namespace,
    dataset_index: int,
    sequence_name: str,
    frame_path: Path,
    batch_idx: int,
    frame_idx: int,
    query_idx: int,
    gt_idx: int,
    track_id: int,
    track_source: int,
    track_quality: float,
) -> dict[str, Any]:
    base_j = base["joints"][batch_idx, frame_idx, query_idx, :24]
    hsi_j = hsi["joints"][batch_idx, frame_idx, query_idx, :24]
    gt_j = gt["joints"][batch_idx, frame_idx, gt_idx, :24]
    base_v = base["vertices"][batch_idx, frame_idx, query_idx]
    hsi_v = hsi["vertices"][batch_idx, frame_idx, query_idx]
    gt_v = gt["vertices"][batch_idx, frame_idx, gt_idx]

    row: dict[str, Any] = {
        "dataset_index": int(dataset_index),
        "sequence_name": sequence_name,
        "frame_idx": int(frame_idx),
        "frame_name": frame_path.stem,
        "image_path": str(frame_path),
        "track_id": int(track_id),
        "track_source": TRACK_SOURCE_NAMES.get(int(track_source), "unknown"),
        "track_source_code": int(track_source),
        "track_quality": safe_float(track_quality),
        "gt_slot": int(gt_idx),
        "query_idx": int(query_idx),
        "pred_conf": safe_float(confs[batch_idx, frame_idx, query_idx]),
        "hsi_scene_scale": prediction_scalar(predictions, "hsi_scene_scale", batch_idx, frame_idx),
        "hsi_scene_depth_bias_m": prediction_scalar(predictions, "hsi_scene_depth_bias", batch_idx, frame_idx),
    }
    row.update(smpl_error_values(base_j, hsi_j, gt_j, base_v, hsi_v, gt_v, base, hsi, gt, batch_idx, frame_idx, query_idx, gt_idx))
    row.update(person_depth_values(raw_depth, hsi_depth, gt_depth, batch["gt_boxes"][batch_idx, frame_idx, gt_idx], batch["boxes_mask"][batch_idx, frame_idx, gt_idx], args))
    row.update(contact_values(base_j, hsi_j, gt_j, base_v, hsi_v, gt_v, sole_indices, intrinsics, hsi_depth, gt_depth, int(batch["images"].shape[-1]), args))
    row["likely_source"] = classify_row(row, args)
    return row


def smpl_error_values(
    base_j: torch.Tensor,
    hsi_j: torch.Tensor,
    gt_j: torch.Tensor,
    base_v: torch.Tensor,
    hsi_v: torch.Tensor,
    gt_v: torch.Tensor,
    base: dict[str, torch.Tensor],
    hsi: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    batch_idx: int,
    frame_idx: int,
    query_idx: int,
    gt_idx: int,
) -> dict[str, float]:
    base_mpjpe = torch.linalg.norm(base_j - gt_j, dim=-1).mean()
    hsi_mpjpe = torch.linalg.norm(hsi_j - gt_j, dim=-1).mean()
    base_pve = torch.linalg.norm(base_v - gt_v, dim=-1).mean()
    hsi_pve = torch.linalg.norm(hsi_v - gt_v, dim=-1).mean()
    base_transl = torch.linalg.norm(base["transl"][batch_idx, frame_idx, query_idx] - gt["transl"][batch_idx, frame_idx, gt_idx])
    hsi_transl = torch.linalg.norm(hsi["transl"][batch_idx, frame_idx, query_idx] - gt["transl"][batch_idx, frame_idx, gt_idx])
    base_pelvis = torch.linalg.norm(base_j[0] - gt_j[0])
    hsi_pelvis = torch.linalg.norm(hsi_j[0] - gt_j[0])
    return {
        "base_mpjpe_m": safe_float(base_mpjpe),
        "hsi_mpjpe_m": safe_float(hsi_mpjpe),
        "hsi_mpjpe_delta_m": safe_float(hsi_mpjpe - base_mpjpe),
        "base_pve_m": safe_float(base_pve),
        "hsi_pve_m": safe_float(hsi_pve),
        "hsi_pve_delta_m": safe_float(hsi_pve - base_pve),
        "base_transl_l2_m": safe_float(base_transl),
        "hsi_transl_l2_m": safe_float(hsi_transl),
        "hsi_transl_delta_m": safe_float(hsi_transl - base_transl),
        "base_pelvis_l2_m": safe_float(base_pelvis),
        "hsi_pelvis_l2_m": safe_float(hsi_pelvis),
        "hsi_pelvis_delta_m": safe_float(hsi_pelvis - base_pelvis),
    }


def person_depth_values(
    raw_depth: torch.Tensor,
    hsi_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    box: torch.Tensor,
    box_mask: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    valid = torch.isfinite(raw_depth) & torch.isfinite(hsi_depth) & torch.isfinite(gt_depth) & (gt_depth > 1e-6)
    if float(args.depth_max_m) > 0:
        valid = valid & (gt_depth <= float(args.depth_max_m))
    roi = bbox_depth_mask(box, bool(box_mask.detach().cpu()), raw_depth.shape[-2], raw_depth.shape[-1], float(args.roi_expand))
    roi_valid = valid & roi
    return depth_error_values(raw_depth, hsi_depth, gt_depth, roi_valid, "human_roi_depth")


def bbox_depth_mask(box: torch.Tensor, valid: bool, height: int, width: int, expand: float) -> torch.Tensor:
    mask = torch.zeros(height, width, dtype=torch.bool, device=box.device)
    if not valid:
        return mask
    cx, cy, box_w, box_h = box.to(dtype=torch.float32).unbind(dim=-1)
    box_w = box_w * (1.0 + max(float(expand), 0.0))
    box_h = box_h * (1.0 + max(float(expand), 0.0))
    x1 = int(torch.floor((cx - 0.5 * box_w).clamp(0.0, 1.0) * width).item())
    x2 = int(torch.ceil((cx + 0.5 * box_w).clamp(0.0, 1.0) * width).item())
    y1 = int(torch.floor((cy - 0.5 * box_h).clamp(0.0, 1.0) * height).item())
    y2 = int(torch.ceil((cy + 0.5 * box_h).clamp(0.0, 1.0) * height).item())
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def frame_depth_metrics(
    raw_depth: torch.Tensor,
    hsi_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    batch_idx: int,
    frame_idx: int,
    dataset_index: int,
    sequence_name: str,
    frame_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    valid = torch.isfinite(raw_depth) & torch.isfinite(hsi_depth) & torch.isfinite(gt_depth) & (gt_depth > 1e-6)
    near_valid = valid & (gt_depth <= float(args.depth_max_m)) if float(args.depth_max_m) > 0 else valid
    row = {
        "dataset_index": int(dataset_index),
        "sequence_name": sequence_name,
        "frame_idx": int(frame_idx),
        "frame_name": frame_path.stem,
        "image_path": str(frame_path),
        "hsi_scene_scale": prediction_scalar(predictions, "hsi_scene_scale", batch_idx, frame_idx),
        "hsi_scene_depth_bias_m": prediction_scalar(predictions, "hsi_scene_depth_bias", batch_idx, frame_idx),
    }
    row.update(depth_error_values(raw_depth, hsi_depth, gt_depth, valid, "full_depth"))
    row.update(depth_error_values(raw_depth, hsi_depth, gt_depth, near_valid, "near_depth"))
    return row


def depth_error_values(
    raw_depth: torch.Tensor,
    hsi_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid: torch.Tensor,
    prefix: str,
) -> dict[str, Any]:
    count = int(valid.sum().detach().cpu())
    out: dict[str, Any] = {f"{prefix}_valid_pixels": count}
    if count <= 0:
        out.update(
            {
                f"raw_{prefix}_l1_mean_m": None,
                f"hsi_{prefix}_l1_mean_m": None,
                f"raw_{prefix}_l1_median_m": None,
                f"hsi_{prefix}_l1_median_m": None,
            }
        )
        return out
    raw_abs = torch.abs(raw_depth[valid] - gt_depth[valid])
    hsi_abs = torch.abs(hsi_depth[valid] - gt_depth[valid])
    out.update(
        {
            f"raw_{prefix}_l1_mean_m": safe_float(raw_abs.mean()),
            f"hsi_{prefix}_l1_mean_m": safe_float(hsi_abs.mean()),
            f"raw_{prefix}_l1_median_m": safe_float(raw_abs.median()),
            f"hsi_{prefix}_l1_median_m": safe_float(hsi_abs.median()),
        }
    )
    return out


def contact_values(
    base_joints: torch.Tensor,
    hsi_joints: torch.Tensor,
    gt_joints: torch.Tensor,
    base_vertices: torch.Tensor,
    hsi_vertices: torch.Tensor,
    gt_vertices: torch.Tensor,
    sole_indices: torch.Tensor,
    intrinsics: torch.Tensor,
    hsi_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    image_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    foot_idx = torch.tensor([7, 8, 10, 11], dtype=torch.long, device=gt_joints.device)
    out.update(contact_group_values(
        "foot",
        base_joints[foot_idx],
        hsi_joints[foot_idx],
        gt_joints[foot_idx],
        intrinsics,
        hsi_depth,
        gt_depth,
        image_size,
        float(args.foot_contact_threshold_m),
        args,
        use_support_plane=False,
    ))
    out.update(contact_group_values(
        "sole",
        base_vertices[sole_indices],
        hsi_vertices[sole_indices],
        gt_vertices[sole_indices],
        intrinsics,
        hsi_depth,
        gt_depth,
        image_size,
        float(args.foot_sole_contact_threshold_m),
        args,
        use_support_plane=True,
    ))
    return out


def contact_group_values(
    group: str,
    base_points: torch.Tensor,
    hsi_points: torch.Tensor,
    gt_points: torch.Tensor,
    intrinsics: torch.Tensor,
    hsi_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    image_size: int,
    contact_threshold: float,
    args: argparse.Namespace,
    use_support_plane: bool,
) -> dict[str, Any]:
    gt_projected = scale_points_to_depth(project_points(gt_points, intrinsics), image_size, gt_depth.shape[-2], gt_depth.shape[-1])
    sampled_gt, gt_valid = sample_depth_at_points(gt_depth, gt_projected)
    contact = (torch.abs(sampled_gt - gt_points[:, 2].to(dtype=sampled_gt.dtype)) < float(contact_threshold)) & gt_valid
    out: dict[str, Any] = {f"{group}_contact_count": int(contact.sum().detach().cpu())}
    out.update(point_depth_delta_values(f"base_{group}", base_points, contact, intrinsics, hsi_depth, image_size, args))
    out.update(point_depth_delta_values(f"hsi_{group}", hsi_points, contact, intrinsics, hsi_depth, image_size, args))
    if use_support_plane:
        out.update(point_plane_delta_values(f"base_{group}_plane", base_points, contact, intrinsics, hsi_depth, image_size, args))
        out.update(point_plane_delta_values(f"hsi_{group}_plane", hsi_points, contact, intrinsics, hsi_depth, image_size, args))
    return out


def point_depth_delta_values(
    prefix: str,
    points: torch.Tensor,
    contact: torch.Tensor,
    intrinsics: torch.Tensor,
    hsi_depth: torch.Tensor,
    image_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    projected = scale_points_to_depth(project_points(points, intrinsics), image_size, hsi_depth.shape[-2], hsi_depth.shape[-1])
    sampled, valid = sample_depth_at_points(hsi_depth, projected)
    use = contact & valid & torch.isfinite(sampled) & torch.isfinite(points[:, 2])
    if not use.any():
        return {
            f"{prefix}_valid_count": 0,
            f"{prefix}_abs_delta_m": None,
            f"{prefix}_float_m": None,
            f"{prefix}_penetration_m": None,
        }
    delta = sampled - points[:, 2].to(dtype=sampled.dtype)
    floating = torch.relu(delta - float(args.foot_float_margin_m))
    penetration = torch.relu(-delta - float(args.foot_penetration_margin_m))
    return {
        f"{prefix}_valid_count": int(use.sum().detach().cpu()),
        f"{prefix}_abs_delta_m": safe_float(torch.abs(delta[use]).mean()),
        f"{prefix}_float_m": safe_float(floating[use].mean()),
        f"{prefix}_penetration_m": safe_float(penetration[use].mean()),
    }


def point_plane_delta_values(
    prefix: str,
    points: torch.Tensor,
    contact: torch.Tensor,
    intrinsics: torch.Tensor,
    hsi_depth: torch.Tensor,
    image_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    projected = scale_points_to_depth(project_points(points, intrinsics), image_size, hsi_depth.shape[-2], hsi_depth.shape[-1])
    signed, plane_valid = sample_local_support_plane_signed_delta(
        hsi_depth,
        projected,
        points,
        intrinsics,
        image_size=image_size,
        window_size=int(args.support_plane_window),
        min_points=int(args.support_plane_min_points),
    )
    use = contact & plane_valid & torch.isfinite(signed)
    if not use.any():
        return {
            f"{prefix}_valid_count": 0,
            f"{prefix}_abs_signed_m": None,
            f"{prefix}_float_m": None,
            f"{prefix}_penetration_m": None,
        }
    floating = torch.relu(signed - float(args.foot_float_margin_m))
    penetration = torch.relu(-signed - float(args.foot_penetration_margin_m))
    return {
        f"{prefix}_valid_count": int(use.sum().detach().cpu()),
        f"{prefix}_abs_signed_m": safe_float(torch.abs(signed[use]).mean()),
        f"{prefix}_float_m": safe_float(floating[use].mean()),
        f"{prefix}_penetration_m": safe_float(penetration[use].mean()),
    }


def build_temporal_rows(
    records: dict[tuple[int, int], dict[int, dict[str, int]]],
    base: dict[str, torch.Tensor],
    hsi: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    frame_paths: list[list[Path]],
    dataset_indices: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    batch_by_dataset_index = {int(dataset_index): batch_idx for batch_idx, dataset_index in enumerate(dataset_indices)}
    for (dataset_index, track_id), seq_map in sorted(records.items()):
        batch_idx = batch_by_dataset_index[int(dataset_index)]
        sequence_name = next(iter(seq_map.values())).get("sequence_name", "")
        for frame_idx in sorted(seq_map):
            if frame_idx - 1 not in seq_map:
                continue
            prev = seq_map[frame_idx - 1]
            curr = seq_map[frame_idx]
            row = temporal_pair_row(base, hsi, gt, batch_idx, dataset_index, sequence_name, track_id, frame_idx - 1, frame_idx, prev, curr)
            row["frame_name"] = frame_paths[batch_idx][frame_idx].stem
            rows.append(row)
        for frame_idx in sorted(seq_map):
            if frame_idx - 1 not in seq_map or frame_idx + 1 not in seq_map:
                continue
            row = temporal_accel_row(base, hsi, gt, batch_idx, dataset_index, sequence_name, track_id, frame_idx - 1, frame_idx, frame_idx + 1, seq_map)
            row["frame_name"] = frame_paths[batch_idx][frame_idx].stem
            rows.append(row)
        for frame_idx in sorted(seq_map):
            if any(index not in seq_map for index in (frame_idx - 3, frame_idx - 2, frame_idx - 1, frame_idx)):
                continue
            row = temporal_jerk_row(base, hsi, gt, batch_idx, dataset_index, sequence_name, track_id, frame_idx - 3, frame_idx - 2, frame_idx - 1, frame_idx, seq_map)
            row["frame_name"] = frame_paths[batch_idx][frame_idx].stem
            rows.append(row)
    return rows


def temporal_pair_row(
    base: dict[str, torch.Tensor],
    hsi: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    batch_idx: int,
    dataset_index: int,
    sequence_name: str,
    track_id: int,
    prev_frame: int,
    curr_frame: int,
    prev: dict[str, int],
    curr: dict[str, int],
) -> dict[str, Any]:
    q0, q1 = prev["query"], curr["query"]
    g0, g1 = prev["gt_index"], curr["gt_index"]
    base_tv = base["transl"][batch_idx, curr_frame, q1] - base["transl"][batch_idx, prev_frame, q0]
    hsi_tv = hsi["transl"][batch_idx, curr_frame, q1] - hsi["transl"][batch_idx, prev_frame, q0]
    gt_tv = gt["transl"][batch_idx, curr_frame, g1] - gt["transl"][batch_idx, prev_frame, g0]
    base_jv = base["joints"][batch_idx, curr_frame, q1, :24] - base["joints"][batch_idx, prev_frame, q0, :24]
    hsi_jv = hsi["joints"][batch_idx, curr_frame, q1, :24] - hsi["joints"][batch_idx, prev_frame, q0, :24]
    gt_jv = gt["joints"][batch_idx, curr_frame, g1, :24] - gt["joints"][batch_idx, prev_frame, g0, :24]
    return {
        "dataset_index": int(dataset_index),
        "sequence_name": sequence_name,
        "track_id": int(track_id),
        "metric_type": "velocity",
        "frame_idx": int(curr_frame),
        "query_switch": int(q0 != q1),
        "base_transl_velocity_l1_m": safe_float(torch.abs(base_tv - gt_tv).mean()),
        "hsi_transl_velocity_l1_m": safe_float(torch.abs(hsi_tv - gt_tv).mean()),
        "base_joints_velocity_l1_m": safe_float(torch.abs(base_jv - gt_jv).mean()),
        "hsi_joints_velocity_l1_m": safe_float(torch.abs(hsi_jv - gt_jv).mean()),
        "base_joints_accel_l1_m": None,
        "hsi_joints_accel_l1_m": None,
        "base_joints_jerk_l1_m": None,
        "hsi_joints_jerk_l1_m": None,
    }


def temporal_accel_row(
    base: dict[str, torch.Tensor],
    hsi: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    batch_idx: int,
    dataset_index: int,
    sequence_name: str,
    track_id: int,
    frame0: int,
    frame1: int,
    frame2: int,
    seq_map: dict[int, dict[str, int]],
) -> dict[str, Any]:
    qs = [seq_map[frame]["query"] for frame in (frame0, frame1, frame2)]
    gs = [seq_map[frame]["gt_index"] for frame in (frame0, frame1, frame2)]
    base_acc = base["joints"][batch_idx, frame2, qs[2], :24] - 2.0 * base["joints"][batch_idx, frame1, qs[1], :24] + base["joints"][batch_idx, frame0, qs[0], :24]
    hsi_acc = hsi["joints"][batch_idx, frame2, qs[2], :24] - 2.0 * hsi["joints"][batch_idx, frame1, qs[1], :24] + hsi["joints"][batch_idx, frame0, qs[0], :24]
    gt_acc = gt["joints"][batch_idx, frame2, gs[2], :24] - 2.0 * gt["joints"][batch_idx, frame1, gs[1], :24] + gt["joints"][batch_idx, frame0, gs[0], :24]
    return {
        "dataset_index": int(dataset_index),
        "sequence_name": sequence_name,
        "track_id": int(track_id),
        "metric_type": "acceleration",
        "frame_idx": int(frame1),
        "query_switch": None,
        "base_transl_velocity_l1_m": None,
        "hsi_transl_velocity_l1_m": None,
        "base_joints_velocity_l1_m": None,
        "hsi_joints_velocity_l1_m": None,
        "base_joints_accel_l1_m": safe_float(torch.abs(base_acc - gt_acc).mean()),
        "hsi_joints_accel_l1_m": safe_float(torch.abs(hsi_acc - gt_acc).mean()),
        "base_joints_jerk_l1_m": None,
        "hsi_joints_jerk_l1_m": None,
    }


def temporal_jerk_row(
    base: dict[str, torch.Tensor],
    hsi: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    batch_idx: int,
    dataset_index: int,
    sequence_name: str,
    track_id: int,
    frame0: int,
    frame1: int,
    frame2: int,
    frame3: int,
    seq_map: dict[int, dict[str, int]],
) -> dict[str, Any]:
    qs = [seq_map[frame]["query"] for frame in (frame0, frame1, frame2, frame3)]
    gs = [seq_map[frame]["gt_index"] for frame in (frame0, frame1, frame2, frame3)]
    base_jerk = (
        base["joints"][batch_idx, frame3, qs[3], :24]
        - 3.0 * base["joints"][batch_idx, frame2, qs[2], :24]
        + 3.0 * base["joints"][batch_idx, frame1, qs[1], :24]
        - base["joints"][batch_idx, frame0, qs[0], :24]
    )
    hsi_jerk = (
        hsi["joints"][batch_idx, frame3, qs[3], :24]
        - 3.0 * hsi["joints"][batch_idx, frame2, qs[2], :24]
        + 3.0 * hsi["joints"][batch_idx, frame1, qs[1], :24]
        - hsi["joints"][batch_idx, frame0, qs[0], :24]
    )
    gt_jerk = (
        gt["joints"][batch_idx, frame3, gs[3], :24]
        - 3.0 * gt["joints"][batch_idx, frame2, gs[2], :24]
        + 3.0 * gt["joints"][batch_idx, frame1, gs[1], :24]
        - gt["joints"][batch_idx, frame0, gs[0], :24]
    )
    return {
        "dataset_index": int(dataset_index),
        "sequence_name": sequence_name,
        "track_id": int(track_id),
        "metric_type": "jerk",
        "frame_idx": int(frame3),
        "query_switch": None,
        "base_transl_velocity_l1_m": None,
        "hsi_transl_velocity_l1_m": None,
        "base_joints_velocity_l1_m": None,
        "hsi_joints_velocity_l1_m": None,
        "base_joints_accel_l1_m": None,
        "hsi_joints_accel_l1_m": None,
        "base_joints_jerk_l1_m": safe_float(torch.abs(base_jerk - gt_jerk).mean()),
        "hsi_joints_jerk_l1_m": safe_float(torch.abs(hsi_jerk - gt_jerk).mean()),
    }


def summarize_person_rows(
    frame_rows: list[dict[str, Any]],
    temporal_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        groups[(int(row["dataset_index"]), int(row["track_id"]))].append(row)
    temporal_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in temporal_rows:
        temporal_groups[(int(row["dataset_index"]), int(row["track_id"]))].append(row)

    summaries = []
    for key, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda item: int(item["frame_idx"]))
        first = rows[0]
        out: dict[str, Any] = {
            "dataset_index": int(key[0]),
            "sequence_name": first["sequence_name"],
            "track_id": int(key[1]),
            "track_source": first["track_source"],
            "track_quality_mean": mean_value(rows, "track_quality"),
            "num_frames": len(rows),
            "first_frame": int(rows[0]["frame_idx"]),
            "last_frame": int(rows[-1]["frame_idx"]),
            "query_switch_count": query_switch_count(rows),
        }
        for metric in PERSON_AGG_METRICS:
            out[f"{metric}_mean"] = mean_value(rows, metric)
            out[f"{metric}_median"] = median_value(rows, metric)
            out[f"{metric}_max"] = max_value(rows, metric)
        for metric in TEMPORAL_AGG_METRICS:
            trows = temporal_groups.get(key, [])
            out[f"{metric}_mean"] = mean_value(trows, metric)
        out["worst_hsi_mpjpe_frame"] = worst_frame(rows, "hsi_mpjpe_m")
        out["worst_hsi_transl_frame"] = worst_frame(rows, "hsi_transl_l2_m")
        out["hsi_better_mpjpe_ratio"] = lower_better_ratio(rows, "hsi_mpjpe_m", "base_mpjpe_m")
        out["hsi_better_transl_ratio"] = lower_better_ratio(rows, "hsi_transl_l2_m", "base_transl_l2_m")
        out["likely_source"] = classify_summary(out, args)
        summaries.append(out)
    return summaries


def build_global_summary(
    frame_rows: list[dict[str, Any]],
    depth_rows: list[dict[str, Any]],
    temporal_rows: list[dict[str, Any]],
    scales: list[float],
    biases: list[float],
    args: argparse.Namespace,
    selected_indices: list[int],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "selected_dataset_indices": [int(index) for index in selected_indices],
        "num_frame_person_rows": len(frame_rows),
        "num_frame_depth_rows": len(depth_rows),
        "num_temporal_rows": len(temporal_rows),
        "hsi_scene_scale": describe_array(np.asarray(scales, dtype=np.float64)),
        "hsi_scene_depth_bias": describe_array(np.asarray(biases, dtype=np.float64)),
    }
    for metric in GLOBAL_FRAME_METRICS:
        summary[f"{metric}_mean"] = mean_value(frame_rows, metric)
        summary[f"{metric}_median"] = median_value(frame_rows, metric)
    for metric in GLOBAL_DEPTH_METRICS:
        summary[f"{metric}_mean"] = mean_value(depth_rows, metric)
        summary[f"{metric}_median"] = median_value(depth_rows, metric)
    for metric in TEMPORAL_AGG_METRICS:
        summary[f"{metric}_mean"] = mean_value(temporal_rows, metric)
    summary["likely_source_counts"] = source_counts(frame_rows)
    summary["thresholds"] = {
        "smpl_bad_mpjpe_m": float(args.smpl_bad_mpjpe_m),
        "smpl_bad_transl_m": float(args.smpl_bad_transl_m),
        "depth_bad_human_roi_m": float(args.depth_bad_human_roi_m),
        "contact_bad_m": float(args.contact_bad_m),
    }
    return summary


def classify_row(row: dict[str, Any], args: argparse.Namespace) -> str:
    flags = []
    if above(row.get("hsi_mpjpe_m"), args.smpl_bad_mpjpe_m) or above(row.get("hsi_transl_l2_m"), args.smpl_bad_transl_m):
        flags.append("smpl_error")
    if above(row.get("hsi_human_roi_depth_l1_median_m"), args.depth_bad_human_roi_m):
        flags.append("depth_error")
    contact_value = max_optional(
        row.get("hsi_sole_float_m"),
        row.get("hsi_sole_penetration_m"),
        row.get("hsi_sole_plane_float_m"),
        row.get("hsi_sole_plane_penetration_m"),
    )
    if above(contact_value, args.contact_bad_m):
        flags.append("contact_error")
    if not flags:
        return "ok"
    return flags[0] if len(flags) == 1 else "mixed:" + "+".join(flags)


def classify_summary(row: dict[str, Any], args: argparse.Namespace) -> str:
    flags = []
    if above(row.get("hsi_mpjpe_m_mean"), args.smpl_bad_mpjpe_m) or above(row.get("hsi_transl_l2_m_mean"), args.smpl_bad_transl_m):
        flags.append("smpl_error")
    if above(row.get("hsi_human_roi_depth_l1_median_m_mean"), args.depth_bad_human_roi_m):
        flags.append("depth_error")
    contact_value = max_optional(
        row.get("hsi_sole_float_m_mean"),
        row.get("hsi_sole_penetration_m_mean"),
        row.get("hsi_sole_plane_float_m_mean"),
        row.get("hsi_sole_plane_penetration_m_mean"),
    )
    if above(contact_value, args.contact_bad_m):
        flags.append("contact_error")
    if not flags:
        return "ok"
    return flags[0] if len(flags) == 1 else "mixed:" + "+".join(flags)


def source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("likely_source", "unknown"))] += 1
    return dict(sorted(counts.items()))


def query_switch_count(rows: list[dict[str, Any]]) -> int:
    count = 0
    prev_query = None
    for row in sorted(rows, key=lambda item: int(item["frame_idx"])):
        query = int(row["query_idx"])
        if prev_query is not None and query != prev_query:
            count += 1
        prev_query = query
    return count


def worst_frame(rows: list[dict[str, Any]], key: str) -> int | None:
    best_frame = None
    best_value = None
    for row in rows:
        value = as_float(row.get(key))
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_frame = int(row["frame_idx"])
    return best_frame


def lower_better_ratio(rows: list[dict[str, Any]], hsi_key: str, base_key: str) -> float | None:
    values = []
    for row in rows:
        hsi = as_float(row.get(hsi_key))
        base = as_float(row.get(base_key))
        if hsi is not None and base is not None:
            values.append(float(hsi < base))
    return float(np.mean(values)) if values else None


def mean_value(rows: list[dict[str, Any]], key: str) -> float | None:
    values = numeric_values(rows, key)
    return float(np.mean(values)) if values else None


def median_value(rows: list[dict[str, Any]], key: str) -> float | None:
    values = numeric_values(rows, key)
    return float(np.median(values)) if values else None


def max_value(rows: list[dict[str, Any]], key: str) -> float | None:
    values = numeric_values(rows, key)
    return float(np.max(values)) if values else None


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out = []
    for row in rows:
        value = as_float(row.get(key))
        if value is not None:
            out.append(value)
    return out


def above(value: Any, threshold: float) -> bool:
    value_f = as_float(value)
    return value_f is not None and value_f > float(threshold)


def max_optional(*values: Any) -> float | None:
    numeric = [value for value in (as_float(item) for item in values) if value is not None]
    return max(numeric) if numeric else None


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if np.isfinite(value_f) else None


def safe_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            value = value.detach().float().mean()
        value = value.detach().float().cpu().item()
    return as_float(value)


def prediction_scalar(predictions: dict[str, torch.Tensor], key: str, batch_idx: int, frame_idx: int) -> float | None:
    value = predictions.get(key)
    if value is None:
        return None
    try:
        return safe_float(value[batch_idx, frame_idx].reshape(-1)[0])
    except (IndexError, RuntimeError):
        return None


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def print_human_summary(payload: dict[str, Any]) -> None:
    summary = payload["global_summary"]
    print("========== HSI sequence per-person diagnostics ==========")
    print(f"Sequences/frames       sequences={payload['num_sequences']} requested_frames={payload['num_frames_requested']}")
    print(f"Rows                   frame_person={summary['num_frame_person_rows']} frame_depth={summary['num_frame_depth_rows']} temporal={summary['num_temporal_rows']}")
    print(
        "SMPL 3D mean           "
        f"base_mpjpe={fmt(summary.get('base_mpjpe_m_mean'))}m "
        f"hsi_mpjpe={fmt(summary.get('hsi_mpjpe_m_mean'))}m "
        f"base_transl={fmt(summary.get('base_transl_l2_m_mean'))}m "
        f"hsi_transl={fmt(summary.get('hsi_transl_l2_m_mean'))}m"
    )
    print(
        "Human ROI depth median "
        f"raw={fmt(summary.get('raw_human_roi_depth_l1_median_m_mean'))}m "
        f"hsi={fmt(summary.get('hsi_human_roi_depth_l1_median_m_mean'))}m"
    )
    print(
        "Contact mean           "
        f"hsi_sole_float={fmt(summary.get('hsi_sole_float_m_mean'))}m "
        f"hsi_sole_pen={fmt(summary.get('hsi_sole_penetration_m_mean'))}m "
        f"hsi_plane_float={fmt(summary.get('hsi_sole_plane_float_m_mean'))}m "
        f"hsi_plane_pen={fmt(summary.get('hsi_sole_plane_penetration_m_mean'))}m"
    )
    print(
        "Temporal mean          "
        f"hsi_transl_vel={fmt(summary.get('hsi_transl_velocity_l1_m_mean'))}m "
        f"hsi_joint_vel={fmt(summary.get('hsi_joints_velocity_l1_m_mean'))}m "
        f"hsi_accel={fmt(summary.get('hsi_joints_accel_l1_m_mean'))}m "
        f"hsi_jerk={fmt(summary.get('hsi_joints_jerk_l1_m_mean'))}m"
    )
    print(f"Likely source counts   {summary.get('likely_source_counts', {})}")
    print(f"hsi scale              {summary['hsi_scene_scale']}")
    print(f"hsi bias               {summary['hsi_scene_depth_bias']}")
    print("Per-person brief:")
    for row in payload["per_person_summary"][:12]:
        print(
            f"  idx={row['dataset_index']} track={row['track_id']} frames={row['num_frames']} "
            f"source={row['likely_source']} "
            f"hsi_mpjpe={fmt(row.get('hsi_mpjpe_m_mean'))}m "
            f"hsi_transl={fmt(row.get('hsi_transl_l2_m_mean'))}m "
            f"hsi_depth_roi={fmt(row.get('hsi_human_roi_depth_l1_median_m_mean'))}m "
            f"sole_float={fmt(row.get('hsi_sole_float_m_mean'))}m "
            f"sole_pen={fmt(row.get('hsi_sole_penetration_m_mean'))}m"
        )


FRAME_FIELDS = [
    "dataset_index",
    "sequence_name",
    "frame_idx",
    "frame_name",
    "image_path",
    "track_id",
    "track_source",
    "track_source_code",
    "track_quality",
    "gt_slot",
    "query_idx",
    "pred_conf",
    "hsi_scene_scale",
    "hsi_scene_depth_bias_m",
    "base_mpjpe_m",
    "hsi_mpjpe_m",
    "hsi_mpjpe_delta_m",
    "base_pve_m",
    "hsi_pve_m",
    "hsi_pve_delta_m",
    "base_transl_l2_m",
    "hsi_transl_l2_m",
    "hsi_transl_delta_m",
    "base_pelvis_l2_m",
    "hsi_pelvis_l2_m",
    "hsi_pelvis_delta_m",
    "human_roi_depth_valid_pixels",
    "raw_human_roi_depth_l1_mean_m",
    "hsi_human_roi_depth_l1_mean_m",
    "raw_human_roi_depth_l1_median_m",
    "hsi_human_roi_depth_l1_median_m",
    "foot_contact_count",
    "base_foot_valid_count",
    "base_foot_abs_delta_m",
    "base_foot_float_m",
    "base_foot_penetration_m",
    "hsi_foot_valid_count",
    "hsi_foot_abs_delta_m",
    "hsi_foot_float_m",
    "hsi_foot_penetration_m",
    "sole_contact_count",
    "base_sole_valid_count",
    "base_sole_abs_delta_m",
    "base_sole_float_m",
    "base_sole_penetration_m",
    "hsi_sole_valid_count",
    "hsi_sole_abs_delta_m",
    "hsi_sole_float_m",
    "hsi_sole_penetration_m",
    "base_sole_plane_valid_count",
    "base_sole_plane_abs_signed_m",
    "base_sole_plane_float_m",
    "base_sole_plane_penetration_m",
    "hsi_sole_plane_valid_count",
    "hsi_sole_plane_abs_signed_m",
    "hsi_sole_plane_float_m",
    "hsi_sole_plane_penetration_m",
    "likely_source",
]

DEPTH_FIELDS = [
    "dataset_index",
    "sequence_name",
    "frame_idx",
    "frame_name",
    "image_path",
    "hsi_scene_scale",
    "hsi_scene_depth_bias_m",
    "full_depth_valid_pixels",
    "raw_full_depth_l1_mean_m",
    "hsi_full_depth_l1_mean_m",
    "raw_full_depth_l1_median_m",
    "hsi_full_depth_l1_median_m",
    "near_depth_valid_pixels",
    "raw_near_depth_l1_mean_m",
    "hsi_near_depth_l1_mean_m",
    "raw_near_depth_l1_median_m",
    "hsi_near_depth_l1_median_m",
]

TEMPORAL_FIELDS = [
    "dataset_index",
    "sequence_name",
    "track_id",
    "metric_type",
    "frame_idx",
    "frame_name",
    "query_switch",
    "base_transl_velocity_l1_m",
    "hsi_transl_velocity_l1_m",
    "base_joints_velocity_l1_m",
    "hsi_joints_velocity_l1_m",
    "base_joints_accel_l1_m",
    "hsi_joints_accel_l1_m",
    "base_joints_jerk_l1_m",
    "hsi_joints_jerk_l1_m",
]

PERSON_AGG_METRICS = [
    "base_mpjpe_m",
    "hsi_mpjpe_m",
    "hsi_mpjpe_delta_m",
    "base_pve_m",
    "hsi_pve_m",
    "hsi_pve_delta_m",
    "base_transl_l2_m",
    "hsi_transl_l2_m",
    "hsi_transl_delta_m",
    "base_pelvis_l2_m",
    "hsi_pelvis_l2_m",
    "raw_human_roi_depth_l1_median_m",
    "hsi_human_roi_depth_l1_median_m",
    "base_sole_float_m",
    "hsi_sole_float_m",
    "base_sole_penetration_m",
    "hsi_sole_penetration_m",
    "base_sole_plane_float_m",
    "hsi_sole_plane_float_m",
    "base_sole_plane_penetration_m",
    "hsi_sole_plane_penetration_m",
]

TEMPORAL_AGG_METRICS = [
    "base_transl_velocity_l1_m",
    "hsi_transl_velocity_l1_m",
    "base_joints_velocity_l1_m",
    "hsi_joints_velocity_l1_m",
    "base_joints_accel_l1_m",
    "hsi_joints_accel_l1_m",
    "base_joints_jerk_l1_m",
    "hsi_joints_jerk_l1_m",
]

GLOBAL_FRAME_METRICS = [
    "base_mpjpe_m",
    "hsi_mpjpe_m",
    "base_pve_m",
    "hsi_pve_m",
    "base_transl_l2_m",
    "hsi_transl_l2_m",
    "raw_human_roi_depth_l1_median_m",
    "hsi_human_roi_depth_l1_median_m",
    "hsi_sole_float_m",
    "hsi_sole_penetration_m",
    "hsi_sole_plane_float_m",
    "hsi_sole_plane_penetration_m",
]

GLOBAL_DEPTH_METRICS = [
    "raw_full_depth_l1_median_m",
    "hsi_full_depth_l1_median_m",
    "raw_near_depth_l1_median_m",
    "hsi_near_depth_l1_median_m",
]

PERSON_FIELDS = [
    "dataset_index",
    "sequence_name",
    "track_id",
    "track_source",
    "track_quality_mean",
    "num_frames",
    "first_frame",
    "last_frame",
    "query_switch_count",
    *[f"{metric}_{stat}" for metric in PERSON_AGG_METRICS for stat in ("mean", "median", "max")],
    *[f"{metric}_mean" for metric in TEMPORAL_AGG_METRICS],
    "worst_hsi_mpjpe_frame",
    "worst_hsi_transl_frame",
    "hsi_better_mpjpe_ratio",
    "hsi_better_transl_ratio",
    "likely_source",
]


if __name__ == "__main__":
    main()
