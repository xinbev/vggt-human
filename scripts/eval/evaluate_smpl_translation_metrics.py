#!/usr/bin/env python
"""Evaluate base vs ray-refined SMPL camera translation without HSI/depth."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
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

from scripts.train.train_smpl import apply_overrides, build_model, load_initial_checkpoint, load_yaml_config
from vggt_omega.data import BedlamDataset, bedlam_collate_fn
from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.training.config import deep_update, require_path
from vggt_omega.training.hungarian_losses import flatten_smpl_targets
from vggt_omega.training.smpl_matcher import HungarianSMPLMatcher
from vggt_omega.utils.rotation import rot6d_to_axis_angle


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args)
    model = build_model(config).to(device)
    load_initial_checkpoint(model, config, device)
    load_training_checkpoint(model, Path(args.checkpoint), device)
    model.eval()

    loader, selected_indices, frame_paths_by_index = build_eval_loader(config, args)
    matcher = HungarianSMPLMatcher(cost_conf=0.5, cost_bbox=8.0, cost_giou=4.0, cost_kpts=0.0, require_boxes=True, require_j2ds=False)
    smpl = SMPLLayer(require_path(config, "assets.smpl_model_dir", allow_empty=False)).to(device).eval()
    totals = MetricTotals()
    person_rows: list[dict[str, Any]] = []

    processed = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = int(batch["images"].shape[0])
            batch_indices = selected_indices[processed : processed + batch_size]
            batch = move_to_device(batch, device)
            predictions = model(
                batch["images"],
                smpl_query_boxes=batch["gt_boxes"] if args.use_gt_box_prior else None,
                smpl_query_boxes_mask=batch["boxes_mask"] if args.use_gt_box_prior else None,
                smpl_track_ids=batch.get("gt_track_ids", batch.get("person_ids")),
                smpl_track_mask=batch.get("gt_track_mask", batch.get("person_id_mask")),
            )
            evaluate_batch(
                predictions,
                batch,
                matcher,
                smpl,
                totals,
                person_rows,
                batch_indices,
                frame_paths_by_index,
                args.match_mode,
            )
            processed += batch_size
            if int(args.max_samples) > 0 and processed >= int(args.max_samples):
                break
            if args.log_interval > 0 and processed % args.log_interval == 0:
                print(f"[eval] processed={processed}")

    summary = totals.summary()
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "num_samples": processed,
            "use_gt_box_prior": bool(args.use_gt_box_prior),
            "match_mode": str(args.match_mode),
            "person_csv": str(output_dir / "smpl_translation_person_metrics.csv"),
            "worst_refined_transl_l2": top_rows(person_rows, "refined_transl_l2_m", args.top_worst),
            "worst_regression_transl_l2": top_rows(person_rows, "transl_l2_delta_m", args.top_worst),
        }
    )
    out_json = output_dir / "smpl_translation_metrics.json"
    out_csv = output_dir / "smpl_translation_person_metrics.csv"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(out_csv, person_rows, TRANSLATION_PERSON_FIELDS)
    print(json.dumps(summary, indent=2))
    print(f"Metrics: {out_json}")
    print(f"Person CSV: {out_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SMPL translation ray refinement")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_translation_ray_refine.yaml")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--output-dir", default="outputs/eval/smpl_translation_ray_refine")
    parser.add_argument("--device", default="")
    parser.add_argument("--split", default="Training")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--use-gt-box-prior", action="store_true")
    parser.add_argument(
        "--match-mode",
        choices=("hungarian", "slot"),
        default="hungarian",
        help="Use normal Hungarian box/conf matching or force q_i to match gt_i for GT-box-prior diagnostics.",
    )
    parser.add_argument("--subset-indices-csv", default="", help="Optional CSV with dataset_index rows to evaluate, e.g. old bad_frame_person_rows.csv")
    parser.add_argument("--subset-index-column", default="dataset_index")
    parser.add_argument("--subset-frame-csv", default="", help="Optional CSV with sequence_name/frame_name rows; selects windows containing those frames")
    parser.add_argument("--subset-sequence-column", default="sequence_name")
    parser.add_argument("--subset-frame-column", default="frame_name")
    parser.add_argument("--subset-unique", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--top-worst", type=int, default=20)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.train_config))
    config = apply_overrides(config, args.override)
    if args.baseline_checkpoint:
        config.setdefault("checkpoints", {})["vggt_baseline"] = args.baseline_checkpoint
    model_cfg = config.setdefault("model", {})
    model_cfg["enable_camera"] = True
    model_cfg["enable_depth"] = False
    model_cfg["enable_hsi_refine"] = False
    model_cfg.setdefault("smpl_enable_translation_refine", True)
    data_cfg = config.setdefault("data", {})
    data_cfg["require_boxes"] = True
    data_cfg["require_smpl"] = True
    data_cfg["require_depth"] = False
    return config


def build_eval_loader(config: dict[str, Any], args: argparse.Namespace) -> tuple[DataLoader, list[int], dict[int, list[Path]]]:
    data_cfg = config["data"]
    dataset = BedlamDataset(
        root=require_path(config, data_cfg.get("root_key", "datasets.bedlam_root")),
        split=args.split,
        sequence_length=int(data_cfg["sequence_length"]),
        stride=int(data_cfg["stride"]),
        image_size=int(data_cfg["image_size"]),
        max_humans=int(data_cfg["max_humans"]),
        require_smpl=True,
        require_depth=False,
        boxes_root=require_path(config, data_cfg["boxes_root_key"], allow_empty=False),
        require_boxes=True,
    )
    selected_indices = select_eval_indices(dataset, args)
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


def select_eval_indices(dataset: BedlamDataset, args: argparse.Namespace) -> list[int]:
    subset_frame_csv = str(args.subset_frame_csv or "").strip()
    if subset_frame_csv:
        path = Path(subset_frame_csv).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"--subset-frame-csv not found: {path}")
        targets = read_subset_frames_csv(path, str(args.subset_sequence_column), str(args.subset_frame_column))
        selected = []
        for dataset_index in range(len(dataset)):
            frame_paths = frame_paths_for_dataset_index(dataset, dataset_index)
            if any((path.parent.parent.name, path.stem) in targets for path in frame_paths):
                selected.append(dataset_index)
        if bool(args.subset_unique):
            seen = set()
            selected = [idx for idx in selected if not (idx in seen or seen.add(idx))]
        if not selected:
            raise ValueError(f"No windows matched frames from {path}")
        max_samples = int(args.max_samples)
        return selected[:max_samples] if max_samples > 0 else selected

    subset_csv = str(args.subset_indices_csv or "").strip()
    if subset_csv:
        path = Path(subset_csv).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"--subset-indices-csv not found: {path}")
        indices = read_subset_indices_csv(path, str(args.subset_index_column))
        if bool(args.subset_unique):
            seen = set()
            indices = [idx for idx in indices if not (idx in seen or seen.add(idx))]
        selected = [idx for idx in indices if 0 <= idx < len(dataset)]
        if not selected:
            raise ValueError(f"No valid dataset indices found in {path}")
        max_samples = int(args.max_samples)
        return selected[:max_samples] if max_samples > 0 else selected
    max_samples = int(args.max_samples)
    end = len(dataset) if max_samples <= 0 else min(len(dataset), int(args.start_index) + max_samples)
    return list(range(int(args.start_index), end))


def read_subset_indices_csv(path: Path, column: str) -> list[int]:
    indices: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"CSV {path} does not contain required column {column!r}")
        for row in reader:
            raw = str(row.get(column, "")).strip()
            if not raw:
                continue
            indices.append(int(float(raw)))
    return indices


def read_subset_frames_csv(path: Path, sequence_column: str, frame_column: str) -> set[tuple[str, str]]:
    frames: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV {path} has no header")
        missing = [column for column in (sequence_column, frame_column) if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV {path} missing required columns: {missing}")
        for row in reader:
            sequence_name = str(row.get(sequence_column, "")).strip()
            frame_name = str(row.get(frame_column, "")).strip()
            if sequence_name and frame_name:
                frames.add((sequence_name, frame_name))
    if not frames:
        raise ValueError(f"No frame keys found in {path}")
    return frames


def load_training_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if state_dict is None:
        raise ValueError(f"Training checkpoint missing 'model' state_dict: {checkpoint_path}")
    missing, unexpected = model.load_state_dict({key.removeprefix("module."): value for key, value in state_dict.items()}, strict=False)
    print(f"[ckpt] loaded training checkpoint: {checkpoint_path}")
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")


def evaluate_batch(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    matcher: HungarianSMPLMatcher,
    smpl: SMPLLayer,
    totals: "MetricTotals",
    person_rows: list[dict[str, Any]],
    batch_indices: list[int],
    frame_paths_by_index: dict[int, list[Path]],
    match_mode: str = "hungarian",
) -> None:
    pred_confs = flatten_prediction(predictions["pred_confs"], 3)
    pred_boxes = flatten_prediction(predictions["pred_boxes"], 3)
    pred_transl = flatten_prediction(predictions["pred_transl_cam"], 3)
    base_transl = flatten_prediction(predictions.get("base_pred_transl_cam", predictions["pred_transl_cam"]), 3)
    seed_transl = flatten_prediction(predictions.get("seed_pred_transl_cam"), 3)
    pred_pose6d = flatten_prediction(predictions["pred_pose_6d"], 3)
    pred_poses = flatten_prediction(predictions["pred_poses"], 3)
    pred_betas = flatten_prediction(predictions["pred_betas"], 3)
    box_prior_weight = predictions.get("pred_transl_box_prior_weight")
    flat_box_prior_weight = flatten_prediction(box_prior_weight, 3) if box_prior_weight is not None else None
    ray_dir = flatten_prediction(predictions.get("pred_transl_ray_dir"), 3)
    tangent_x = flatten_prediction(predictions.get("pred_transl_tangent_x"), 3)
    tangent_y = flatten_prediction(predictions.get("pred_transl_tangent_y"), 3)
    pred_ray_depth = flatten_prediction(predictions.get("pred_transl_ray_depth"), 3)
    pred_tangent = flatten_prediction(predictions.get("pred_transl_tangent"), 3)
    base_ray_depth = flatten_prediction(predictions.get("base_pred_transl_ray_depth"), 3)
    base_tangent = flatten_prediction(predictions.get("base_pred_transl_tangent"), 3)

    targets = flatten_smpl_targets(batch, device=pred_confs.device)
    indices = match_smpl_predictions(
        matcher=matcher,
        outputs={"pred_confs": pred_confs, "pred_boxes": pred_boxes},
        targets=targets,
        device=pred_confs.device,
        num_queries=int(pred_confs.shape[1]),
        match_mode=match_mode,
    )
    matched = collect_matches(indices, targets, pred_confs.device)
    if matched["frame_idx"].numel() == 0:
        return

    frame_idx = matched["frame_idx"]
    src_idx = matched["src_idx"]
    gt_transl = matched["transl_cam"].to(device=pred_transl.device, dtype=pred_transl.dtype)
    refined = pred_transl[frame_idx, src_idx]
    base = base_transl[frame_idx, src_idx]
    seed = seed_transl[frame_idx, src_idx] if seed_transl is not None else None
    totals.add_translation(base, refined, gt_transl)
    if seed is not None:
        totals.add_seed_translation(seed, refined, gt_transl)
    if ray_dir is not None and tangent_x is not None and tangent_y is not None:
        matched_ray = ray_dir[frame_idx, src_idx].to(dtype=refined.dtype)
        matched_tx = tangent_x[frame_idx, src_idx].to(dtype=refined.dtype)
        matched_ty = tangent_y[frame_idx, src_idx].to(dtype=refined.dtype)
        target_ray = (gt_transl * matched_ray).sum(dim=-1, keepdim=True)
        target_tangent = torch.cat(
            [
                (gt_transl * matched_tx).sum(dim=-1, keepdim=True),
                (gt_transl * matched_ty).sum(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        refined_ray = pred_ray_depth[frame_idx, src_idx].to(dtype=refined.dtype) if pred_ray_depth is not None else (refined * matched_ray).sum(dim=-1, keepdim=True)
        refined_tan = pred_tangent[frame_idx, src_idx].to(dtype=refined.dtype) if pred_tangent is not None else torch.cat(
            [
                (refined * matched_tx).sum(dim=-1, keepdim=True),
                (refined * matched_ty).sum(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        base_ray = base_ray_depth[frame_idx, src_idx].to(dtype=refined.dtype) if base_ray_depth is not None else (base * matched_ray).sum(dim=-1, keepdim=True)
        base_tan = base_tangent[frame_idx, src_idx].to(dtype=refined.dtype) if base_tangent is not None else torch.cat(
            [
                (base * matched_tx).sum(dim=-1, keepdim=True),
                (base * matched_ty).sum(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        totals.add_ray_components(base_ray, refined_ray, target_ray, base_tan, refined_tan, target_tangent)

    if flat_box_prior_weight is not None:
        totals.add("box_prior_weight_abs", flat_box_prior_weight[frame_idx, src_idx].abs().mean(), refined.shape[0])

    gt_pose = rot6d_to_axis_angle(matched["pose_6d"].to(device=pred_transl.device, dtype=pred_transl.dtype)).reshape(-1, 72)
    gt_betas = matched["betas"].to(device=pred_transl.device, dtype=pred_transl.dtype)
    pred_vertices, pred_joints = smpl(pred_poses[frame_idx, src_idx].reshape(-1, 72).float(), pred_betas[frame_idx, src_idx].float())
    gt_vertices, gt_joints = smpl(gt_pose.float(), gt_betas.float())
    pred_vertices = pred_vertices.to(dtype=pred_transl.dtype)
    gt_vertices = gt_vertices.to(dtype=pred_transl.dtype)
    pred_joints = pred_joints[:, :24].to(dtype=pred_transl.dtype)
    gt_joints = gt_joints[:, :24].to(dtype=pred_transl.dtype)
    gt_joints_cam = gt_joints + gt_transl[:, None, :]
    refined_joints_cam = pred_joints + refined[:, None, :]
    base_joints_cam = pred_joints + base[:, None, :]
    gt_vertices_cam = gt_vertices + gt_transl[:, None, :]
    refined_vertices_cam = pred_vertices + refined[:, None, :]
    base_vertices_cam = pred_vertices + base[:, None, :]
    base_mpjpe = torch.linalg.norm(base_joints_cam - gt_joints_cam, dim=-1).mean(dim=-1)
    refined_mpjpe = torch.linalg.norm(refined_joints_cam - gt_joints_cam, dim=-1).mean(dim=-1)
    base_pve = torch.linalg.norm(base_vertices_cam - gt_vertices_cam, dim=-1).mean(dim=-1)
    refined_pve = torch.linalg.norm(refined_vertices_cam - gt_vertices_cam, dim=-1).mean(dim=-1)
    totals.add("base_joints_mpjpe_m", base_mpjpe.mean(), refined.shape[0])
    totals.add("refined_joints_mpjpe_m", refined_mpjpe.mean(), refined.shape[0])
    totals.add("base_vertices_pve_m", base_pve.mean(), refined.shape[0])
    totals.add("refined_vertices_pve_m", refined_pve.mean(), refined.shape[0])
    totals.add("pose6d_l1", F.l1_loss(pred_pose6d[frame_idx, src_idx], matched["pose_6d"].to(device=pred_transl.device, dtype=pred_transl.dtype)), refined.shape[0])
    append_person_rows(
        rows=person_rows,
        batch=batch,
        batch_indices=batch_indices,
        frame_paths_by_index=frame_paths_by_index,
        matched=matched,
        src_idx=src_idx,
        base=base,
        seed=seed,
        refined=refined,
        target=gt_transl,
        base_mpjpe=base_mpjpe,
        refined_mpjpe=refined_mpjpe,
        base_pve=base_pve,
        refined_pve=refined_pve,
        base_ray=base_ray if ray_dir is not None and tangent_x is not None and tangent_y is not None else None,
        refined_ray=refined_ray if ray_dir is not None and tangent_x is not None and tangent_y is not None else None,
        target_ray=target_ray if ray_dir is not None and tangent_x is not None and tangent_y is not None else None,
        base_tan=base_tan if ray_dir is not None and tangent_x is not None and tangent_y is not None else None,
        refined_tan=refined_tan if ray_dir is not None and tangent_x is not None and tangent_y is not None else None,
        target_tan=target_tangent if ray_dir is not None and tangent_x is not None and tangent_y is not None else None,
        box_prior_weight=flat_box_prior_weight[frame_idx, src_idx] if flat_box_prior_weight is not None else None,
    )


def match_smpl_predictions(
    matcher: HungarianSMPLMatcher,
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    device: torch.device,
    num_queries: int,
    match_mode: str,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if match_mode == "hungarian":
        return matcher(outputs, targets)
    if match_mode != "slot":
        raise ValueError(f"Unsupported match mode: {match_mode}")
    indices: list[tuple[torch.Tensor, torch.Tensor]] = []
    for target in targets:
        num_targets = int(target["transl_cam"].shape[0])
        count = min(int(num_queries), num_targets)
        src_idx = torch.arange(count, dtype=torch.long, device=device)
        tgt_idx = torch.arange(count, dtype=torch.long, device=device)
        indices.append((src_idx, tgt_idx))
    return indices


def collect_matches(indices, targets: list[dict[str, torch.Tensor]], device: torch.device) -> dict[str, torch.Tensor]:
    frame_indices = []
    src_indices = []
    target_indices = []
    target_parts: dict[str, list[torch.Tensor]] = {
        "pose_6d": [],
        "betas": [],
        "transl_cam": [],
        "person_ids": [],
        "person_id_mask": [],
        "gt_track_source": [],
        "gt_track_quality": [],
    }
    for frame_idx, (src_idx, tgt_idx) in enumerate(indices):
        if src_idx.numel() == 0:
            continue
        frame_indices.append(torch.full_like(src_idx, frame_idx))
        src_indices.append(src_idx)
        target_indices.append(tgt_idx)
        target = targets[frame_idx]
        for key in target_parts:
            target_parts[key].append(target[key][tgt_idx])
    if not frame_indices:
        return {
            "frame_idx": torch.empty(0, dtype=torch.long, device=device),
            "src_idx": torch.empty(0, dtype=torch.long, device=device),
            "target_idx": torch.empty(0, dtype=torch.long, device=device),
        }
    out = {
        "frame_idx": torch.cat(frame_indices),
        "src_idx": torch.cat(src_indices),
        "target_idx": torch.cat(target_indices),
    }
    out.update({key: torch.cat(values) for key, values in target_parts.items()})
    return out


def append_person_rows(
    rows: list[dict[str, Any]],
    batch: dict[str, torch.Tensor],
    batch_indices: list[int],
    frame_paths_by_index: dict[int, list[Path]],
    matched: dict[str, torch.Tensor],
    src_idx: torch.Tensor,
    base: torch.Tensor,
    seed: torch.Tensor | None,
    refined: torch.Tensor,
    target: torch.Tensor,
    base_mpjpe: torch.Tensor,
    refined_mpjpe: torch.Tensor,
    base_pve: torch.Tensor,
    refined_pve: torch.Tensor,
    base_ray: torch.Tensor | None,
    refined_ray: torch.Tensor | None,
    target_ray: torch.Tensor | None,
    base_tan: torch.Tensor | None,
    refined_tan: torch.Tensor | None,
    target_tan: torch.Tensor | None,
    box_prior_weight: torch.Tensor | None,
) -> None:
    num_frames = int(batch["images"].shape[1])
    frame_idx = matched["frame_idx"]
    target_idx = matched["target_idx"]
    for row_idx in range(int(frame_idx.numel())):
        flat_frame = int(frame_idx[row_idx].detach().cpu())
        batch_idx = flat_frame // num_frames
        frame_offset = flat_frame % num_frames
        dataset_index = int(batch_indices[batch_idx]) if batch_idx < len(batch_indices) else -1
        frame_paths = frame_paths_by_index.get(dataset_index, [])
        frame_path = frame_paths[frame_offset] if frame_offset < len(frame_paths) else None
        base_err = base[row_idx] - target[row_idx]
        seed_err = seed[row_idx] - target[row_idx] if seed is not None else None
        refined_err = refined[row_idx] - target[row_idx]
        row = {
            "dataset_index": dataset_index,
            "sequence_name": frame_path.parent.parent.name if frame_path is not None else "",
            "frame_idx": frame_offset,
            "frame_name": frame_path.stem if frame_path is not None else "",
            "image_path": str(frame_path) if frame_path is not None else "",
            "query_idx": int(src_idx[row_idx].detach().cpu()),
            "gt_idx": int(target_idx[row_idx].detach().cpu()),
            "track_id": tensor_int(matched["person_ids"][row_idx]),
            "track_valid": int(bool(tensor_bool(matched["person_id_mask"][row_idx]))),
            "track_source": tensor_int(matched["gt_track_source"][row_idx]),
            "track_quality": tensor_float(matched["gt_track_quality"][row_idx]),
            "base_transl_x_m": tensor_float(base[row_idx, 0]),
            "base_transl_y_m": tensor_float(base[row_idx, 1]),
            "base_transl_z_m": tensor_float(base[row_idx, 2]),
            "seed_transl_x_m": tensor_float(seed[row_idx, 0]) if seed is not None else None,
            "seed_transl_y_m": tensor_float(seed[row_idx, 1]) if seed is not None else None,
            "seed_transl_z_m": tensor_float(seed[row_idx, 2]) if seed is not None else None,
            "refined_transl_x_m": tensor_float(refined[row_idx, 0]),
            "refined_transl_y_m": tensor_float(refined[row_idx, 1]),
            "refined_transl_z_m": tensor_float(refined[row_idx, 2]),
            "gt_transl_x_m": tensor_float(target[row_idx, 0]),
            "gt_transl_y_m": tensor_float(target[row_idx, 1]),
            "gt_transl_z_m": tensor_float(target[row_idx, 2]),
            "base_transl_l2_m": tensor_norm(base_err),
            "seed_transl_l2_m": tensor_norm(seed_err) if seed_err is not None else None,
            "refined_transl_l2_m": tensor_norm(refined_err),
            "transl_l2_delta_m": tensor_norm(refined_err) - tensor_norm(base_err),
            "seed_to_refined_l2_delta_m": (tensor_norm(refined_err) - tensor_norm(seed_err)) if seed_err is not None else None,
            "base_transl_z_l1_m": tensor_float(base_err[2].abs()),
            "seed_transl_z_l1_m": tensor_float(seed_err[2].abs()) if seed_err is not None else None,
            "refined_transl_z_l1_m": tensor_float(refined_err[2].abs()),
            "base_transl_xy_l2_m": tensor_norm(base_err[:2]),
            "refined_transl_xy_l2_m": tensor_norm(refined_err[:2]),
            "delta_transl_l2_m": tensor_norm(refined[row_idx] - base[row_idx]),
            "base_mpjpe_m": tensor_float(base_mpjpe[row_idx]),
            "refined_mpjpe_m": tensor_float(refined_mpjpe[row_idx]),
            "mpjpe_delta_m": tensor_float(refined_mpjpe[row_idx] - base_mpjpe[row_idx]),
            "base_pve_m": tensor_float(base_pve[row_idx]),
            "refined_pve_m": tensor_float(refined_pve[row_idx]),
            "pve_delta_m": tensor_float(refined_pve[row_idx] - base_pve[row_idx]),
        }
        if base_ray is not None and refined_ray is not None and target_ray is not None:
            row.update(
                {
                    "base_ray_depth_l1_m": tensor_float((base_ray[row_idx] - target_ray[row_idx]).abs().mean()),
                    "refined_ray_depth_l1_m": tensor_float((refined_ray[row_idx] - target_ray[row_idx]).abs().mean()),
                    "ray_depth_delta_m": tensor_float((refined_ray[row_idx] - target_ray[row_idx]).abs().mean() - (base_ray[row_idx] - target_ray[row_idx]).abs().mean()),
                }
            )
        if base_tan is not None and refined_tan is not None and target_tan is not None:
            row.update(
                {
                    "base_tangent_l2_m": tensor_norm(base_tan[row_idx] - target_tan[row_idx]),
                    "refined_tangent_l2_m": tensor_norm(refined_tan[row_idx] - target_tan[row_idx]),
                    "tangent_l2_delta_m": tensor_norm(refined_tan[row_idx] - target_tan[row_idx]) - tensor_norm(base_tan[row_idx] - target_tan[row_idx]),
                }
            )
        if box_prior_weight is not None:
            row["box_prior_weight"] = tensor_float(box_prior_weight[row_idx].reshape(-1)[0])
        rows.append(row)


def frame_paths_for_dataset_index(dataset: BedlamDataset, dataset_index: int) -> list[Path]:
    seq_idx, start_idx = dataset._index[int(dataset_index)]  # noqa: SLF001
    seq_dir, frame_ids = dataset._sequences[seq_idx]  # noqa: SLF001
    return [seq_dir / "rgb" / f"{frame_ids[start_idx + step * dataset.stride]}.png" for step in range(dataset.sequence_length)]


def top_rows(rows: list[dict[str, Any]], key: str, limit: int) -> list[dict[str, Any]]:
    sortable = [row for row in rows if isinstance(row.get(key), (int, float)) and np.isfinite(float(row[key]))]
    selected = sorted(sortable, key=lambda row: float(row[key]), reverse=True)[: max(int(limit), 0)]
    keys = [
        "dataset_index",
        "sequence_name",
        "frame_idx",
        "frame_name",
        "image_path",
        "query_idx",
        "gt_idx",
        "track_id",
        "base_transl_l2_m",
        "seed_transl_l2_m",
        "refined_transl_l2_m",
        "transl_l2_delta_m",
        "seed_to_refined_l2_delta_m",
        "base_transl_z_l1_m",
        "seed_transl_z_l1_m",
        "refined_transl_z_l1_m",
        "base_mpjpe_m",
        "refined_mpjpe_m",
        "mpjpe_delta_m",
    ]
    return [{name: row.get(name) for name in keys if name in row} for row in selected]


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


def tensor_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def tensor_int(value: torch.Tensor) -> int:
    return int(value.detach().cpu())


def tensor_bool(value: torch.Tensor) -> bool:
    return bool(value.detach().cpu())


def tensor_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.norm(value).detach().cpu())


def flatten_prediction(tensor: torch.Tensor | None, unframed_ndim: int) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim == unframed_ndim:
        return tensor
    if tensor.ndim == unframed_ndim + 1:
        return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])
    raise ValueError(f"Expected prediction with {unframed_ndim} or {unframed_ndim + 1} dims, got {tensor.shape}")


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


class MetricTotals:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, name: str, value: torch.Tensor, count: int) -> None:
        scalar = float(value.detach().cpu())
        if np.isfinite(scalar) and count > 0:
            self.totals[name] = self.totals.get(name, 0.0) + scalar * int(count)
            self.counts[name] = self.counts.get(name, 0) + int(count)

    def add_translation(self, base: torch.Tensor, refined: torch.Tensor, target: torch.Tensor) -> None:
        count = int(refined.shape[0])
        base_err = base - target
        refined_err = refined - target
        self.add("base_transl_l1_m", base_err.abs().mean(), count)
        self.add("refined_transl_l1_m", refined_err.abs().mean(), count)
        self.add("base_transl_l2_m", torch.linalg.norm(base_err, dim=-1).mean(), count)
        self.add("refined_transl_l2_m", torch.linalg.norm(refined_err, dim=-1).mean(), count)
        self.add("base_transl_z_l1_m", base_err[:, 2].abs().mean(), count)
        self.add("refined_transl_z_l1_m", refined_err[:, 2].abs().mean(), count)
        self.add("base_transl_x_l1_m", base_err[:, 0].abs().mean(), count)
        self.add("refined_transl_x_l1_m", refined_err[:, 0].abs().mean(), count)
        self.add("base_transl_y_l1_m", base_err[:, 1].abs().mean(), count)
        self.add("refined_transl_y_l1_m", refined_err[:, 1].abs().mean(), count)
        self.add("base_transl_z_bias_m", base_err[:, 2].mean(), count)
        self.add("refined_transl_z_bias_m", refined_err[:, 2].mean(), count)
        self.add("base_transl_xy_l2_m", torch.linalg.norm(base_err[:, :2], dim=-1).mean(), count)
        self.add("refined_transl_xy_l2_m", torch.linalg.norm(refined_err[:, :2], dim=-1).mean(), count)
        self.add("delta_transl_l2_m", torch.linalg.norm(refined - base, dim=-1).mean(), count)

    def add_seed_translation(self, seed: torch.Tensor, refined: torch.Tensor, target: torch.Tensor) -> None:
        count = int(refined.shape[0])
        seed_err = seed - target
        refined_err = refined - target
        self.add("seed_transl_l1_m", seed_err.abs().mean(), count)
        self.add("seed_transl_l2_m", torch.linalg.norm(seed_err, dim=-1).mean(), count)
        self.add("seed_transl_z_l1_m", seed_err[:, 2].abs().mean(), count)
        self.add("seed_transl_xy_l2_m", torch.linalg.norm(seed_err[:, :2], dim=-1).mean(), count)
        self.add("seed_to_refined_transl_l2_delta_m", torch.linalg.norm(refined_err, dim=-1).mean() - torch.linalg.norm(seed_err, dim=-1).mean(), count)

    def add_ray_components(
        self,
        base_ray: torch.Tensor,
        refined_ray: torch.Tensor,
        target_ray: torch.Tensor,
        base_tangent: torch.Tensor,
        refined_tangent: torch.Tensor,
        target_tangent: torch.Tensor,
    ) -> None:
        count = int(refined_ray.shape[0])
        self.add("base_ray_depth_l1_m", (base_ray - target_ray).abs().mean(), count)
        self.add("refined_ray_depth_l1_m", (refined_ray - target_ray).abs().mean(), count)
        self.add("base_tangent_l1_m", (base_tangent - target_tangent).abs().mean(), count)
        self.add("refined_tangent_l1_m", (refined_tangent - target_tangent).abs().mean(), count)
        self.add("base_tangent_l2_m", torch.linalg.norm(base_tangent - target_tangent, dim=-1).mean(), count)
        self.add("refined_tangent_l2_m", torch.linalg.norm(refined_tangent - target_tangent, dim=-1).mean(), count)

    def summary(self) -> dict[str, float]:
        out = {name: total / max(self.counts.get(name, 0), 1) for name, total in self.totals.items()}
        add_improvement(out, "transl_l2", "base_transl_l2_m", "refined_transl_l2_m")
        add_improvement(out, "transl_z_l1", "base_transl_z_l1_m", "refined_transl_z_l1_m")
        add_improvement(out, "transl_xy_l2", "base_transl_xy_l2_m", "refined_transl_xy_l2_m")
        add_improvement(out, "ray_depth_l1", "base_ray_depth_l1_m", "refined_ray_depth_l1_m")
        add_improvement(out, "tangent_l2", "base_tangent_l2_m", "refined_tangent_l2_m")
        add_improvement(out, "seed_to_refined_transl_l2", "seed_transl_l2_m", "refined_transl_l2_m")
        add_improvement(out, "joints_mpjpe", "base_joints_mpjpe_m", "refined_joints_mpjpe_m")
        add_improvement(out, "vertices_pve", "base_vertices_pve_m", "refined_vertices_pve_m")
        out["num_matched"] = float(max(self.counts.values(), default=0))
        return out


def add_improvement(out: dict[str, float], name: str, base_key: str, refined_key: str) -> None:
    if base_key not in out or refined_key not in out:
        return
    delta = out[base_key] - out[refined_key]
    out[f"{name}_improvement_m"] = delta
    out[f"{name}_improvement_percent"] = 100.0 * delta / max(out[base_key], 1e-8)


TRANSLATION_PERSON_FIELDS = [
    "dataset_index",
    "sequence_name",
    "frame_idx",
    "frame_name",
    "image_path",
    "query_idx",
    "gt_idx",
    "track_id",
    "track_valid",
    "track_source",
    "track_quality",
    "base_transl_x_m",
    "base_transl_y_m",
    "base_transl_z_m",
    "seed_transl_x_m",
    "seed_transl_y_m",
    "seed_transl_z_m",
    "refined_transl_x_m",
    "refined_transl_y_m",
    "refined_transl_z_m",
    "gt_transl_x_m",
    "gt_transl_y_m",
    "gt_transl_z_m",
    "base_transl_l2_m",
    "seed_transl_l2_m",
    "refined_transl_l2_m",
    "transl_l2_delta_m",
    "seed_to_refined_l2_delta_m",
    "base_transl_z_l1_m",
    "seed_transl_z_l1_m",
    "refined_transl_z_l1_m",
    "base_transl_xy_l2_m",
    "refined_transl_xy_l2_m",
    "delta_transl_l2_m",
    "base_ray_depth_l1_m",
    "refined_ray_depth_l1_m",
    "ray_depth_delta_m",
    "base_tangent_l2_m",
    "refined_tangent_l2_m",
    "tangent_l2_delta_m",
    "box_prior_weight",
    "base_mpjpe_m",
    "refined_mpjpe_m",
    "mpjpe_delta_m",
    "base_pve_m",
    "refined_pve_m",
    "pve_delta_m",
]


if __name__ == "__main__":
    main()
