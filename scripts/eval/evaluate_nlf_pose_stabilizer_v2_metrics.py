#!/usr/bin/env python3
"""Evaluate RGB -> VGGT -> NLF -> PoseTemporalStabilizer V2 on HMR4D labels.

Each dataset item is a nine-frame RGB window.  Only its centre frame is
scored, so every temporal-eligible frame is evaluated exactly once rather
than being over-counted by overlapping windows.  NLF base and V2-stabilised
results use the same matched centre detection and the same GT frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.evaluate_hmr4d_smpl_metrics import (  # noqa: E402
    UnsupportedLabelError,
    box_iou_cxcywh,
    extract_gt_smpl,
    move_to_device,
)
from scripts.train.train_smpl import apply_overrides, build_model, load_initial_checkpoint  # noqa: E402
from vggt_omega.data import HMR4DSupportEvalDataset, hmr4d_eval_collate_fn  # noqa: E402
from vggt_omega.data.geometry import resolve_image_size_config  # noqa: E402
from vggt_omega.models import PoseStabilizerConfig, PoseTemporalStabilizer  # noqa: E402
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.training.config import deep_update, load_yaml_config, require_path  # noqa: E402
from vggt_omega.utils.rotation import rot6d_to_axis_angle  # noqa: E402


WINDOW_SIZE = 9
CENTER_INDEX = WINDOW_SIZE // 2


def main() -> None:
    args = parse_args()
    requested_device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested ({requested_device}) but PyTorch cannot see a CUDA device. "
            "Check CUDA_VISIBLE_DEVICES_VALUE and the active torch/CUDA environment; do not run full VGGT/NLF evaluation on CPU."
        )
    device = torch.device(requested_device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.train_config))
    config = apply_overrides(config, args.override)
    config = apply_nlf_defaults(config)
    # A released VGGT baseline is a plain state_dict and is valid for this
    # RGB->VGGT(camera)->NLF evaluation.  ``load_initial_checkpoint`` handles
    # both that format and project checkpoint wrappers, so do not load the
    # same file again through the full-training-checkpoint loader.
    config.setdefault("checkpoints", {})["vggt_baseline"] = str(args.checkpoint)
    config.setdefault("checkpoint", {})["load_vggt_baseline"] = True
    model = build_model(config).to(device)
    load_initial_checkpoint(model, config, device)
    model.eval()
    stabilizer = load_pose_stabilizer(Path(args.temporal_checkpoint), device)
    smpl = SMPLLayer(require_path(config, "assets.smpl_model_dir", allow_empty=False)).to(device).eval()
    dataset = build_dataset(config, args)
    filtered_windows = apply_sequence_filter(dataset, args.sequence_filter)
    print_component_manifest(config, args, stabilizer, dataset, device)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=hmr4d_eval_collate_fn,
        drop_last=False,
    )

    totals = {"nlf_base": MetricTotals(), "nlf_pose_temporal": MetricTotals()}
    coverage = CoverageTotals()
    rows: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    processed_windows = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            predictions = model(
                batch["images"],
                smpl_query_boxes=None,
                smpl_query_boxes_mask=None,
                smpl_query_patch_masks=None,
                external_track_ids=None,
                external_track_mask=None,
                external_track_confidence=None,
            )
            try:
                gt = extract_gt_smpl(batch["eval_label"], device)
            except UnsupportedLabelError as exc:
                unsupported.append({"meta": batch.get("meta"), "reason": str(exc)})
                processed_windows += int(batch["images"].shape[0])
                continue
            evaluate_center_frames(predictions, batch, gt, stabilizer, smpl, totals, coverage, rows)
            processed_windows += int(batch["images"].shape[0])
            if int(args.max_windows) > 0 and processed_windows >= int(args.max_windows):
                break
            if int(args.log_interval) > 0 and processed_windows % int(args.log_interval) == 0:
                print(f"[eval] dataset={args.dataset} processed_windows={processed_windows} metric_rows={len(rows)}", flush=True)

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "vggt_checkpoint": str(args.checkpoint),
        "temporal_checkpoint": str(args.temporal_checkpoint),
        "num_windows_processed": int(processed_windows),
        "num_metric_rows": len(rows),
        "num_unsupported_windows": len(unsupported),
        "unsupported_examples": unsupported[:5],
        "coverage": coverage.summary(),
        "sequence_filter": str(args.sequence_filter or ""),
        "dataset_windows_after_filter": int(filtered_windows),
        "metric_protocol": "project_native_smpl24_pelvis_aligned",
        "metric_definition": {
            "PA-MPJPE": "24-joint Procrustes-aligned error in mm",
            "MPJPE": "24-joint pelvis-aligned error in mm",
            "PVE": "6890-vertex pelvis-aligned error in mm",
        },
        "input_protocol": "RGB->VGGT(camera)->NLF_internal_detector->PoseTemporalStabilizerV2; HSI/TRSTR disabled; no external sidecar",
        "temporal_protocol": "nine-frame window; centre frame scored once; only same assigned NLF track is stabilised; missing track context is temporal no-op",
        "rows_csv": str(output_dir / f"{args.dataset}_nlf_pose_stabilizer_v2_rows.csv"),
    }
    for name, metrics in totals.items():
        summary[name] = metrics.summary()
    (output_dir / f"{args.dataset}_nlf_pose_stabilizer_v2_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_rows(output_dir / f"{args.dataset}_nlf_pose_stabilizer_v2_rows.csv", rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RGB->VGGT->NLF->PoseTemporalStabilizerV2 on unique centre frames")
    parser.add_argument("--dataset", required=True, choices=["emdb1", "3dpw"])
    parser.add_argument("--checkpoint", required=True, help="VGGT/NLF checkpoint or the released VGGT baseline checkpoint")
    parser.add_argument("--temporal-checkpoint", required=True, help="V2 pose mixture/hard-finetune checkpoint")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/eval_nlf_pose_stabilizer_v2.yaml")
    parser.add_argument("--support-root", default="")
    parser.add_argument("--frames-root", default="")
    parser.add_argument("--sequence-filter", default="", help="Optional substring matched against support vid/vname; useful for a native 3DPW sequence smoke")
    parser.add_argument("--output-dir", default="outputs/eval/nlf_pose_stabilizer_v2")
    parser.add_argument("--device", default="")
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=0, help="Smoke/debug only; 0 evaluates every temporal-eligible centre frame")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def apply_nlf_defaults(config: dict[str, Any]) -> dict[str, Any]:
    model = config.setdefault("model", {})
    model.update(
        {
            "smpl_provider": "nlf",
            "nlf_use_detector": True,
            "nlf_require_boxes": False,
            "smpl_use_aggregator_queries": False,
            "smpl_use_external_track_prior": False,
            "smpl_track_assignment_mode": "base_smpl",
            "enable_hsi_refine": False,
            "enable_hsi_human_scene_align": False,
            "enable_hsi_translation_refine_v4": False,
            "enable_hsi_contact_refine": False,
            "enable_hsi_grounding": False,
            "enable_hsi_foot_contact_intent": False,
            "enable_hsi_trstr": False,
        }
    )
    return config


def load_pose_stabilizer(path: Path, device: torch.device) -> PoseTemporalStabilizer:
    if not path.is_file():
        raise FileNotFoundError(f"Pose stabilizer checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    valid_formats = {"smpl_temporal_stabilizer_v2_pose_mixture", "smpl_temporal_stabilizer_v2_pose_e0"}
    if checkpoint.get("format") not in valid_formats:
        raise ValueError(f"Expected V2 pose stabilizer checkpoint, got {checkpoint.get('format')!r}")
    stabilizer = PoseTemporalStabilizer(PoseStabilizerConfig(**checkpoint["model_config"])).to(device)
    if stabilizer.config.window_size != WINDOW_SIZE:
        raise ValueError(f"Current evaluator requires V2 window_size={WINDOW_SIZE}, checkpoint uses {stabilizer.config.window_size}")
    stabilizer.load_state_dict(checkpoint["model_state"], strict=True)
    stabilizer.eval()
    return stabilizer


def print_component_manifest(
    config: dict[str, Any],
    args: argparse.Namespace,
    stabilizer: PoseTemporalStabilizer,
    dataset: HMR4DSupportEvalDataset,
    device: torch.device,
) -> None:
    """Make every model asset and disabled branch explicit in evaluation logs."""
    model_cfg = config.get("model", {})
    checkpoints = config.get("checkpoints", {})
    assets = config.get("assets", {})
    print("========== Evaluation component manifest ==========", flush=True)
    print(f"device: {device}", flush=True)
    print(f"VGGT runtime camera checkpoint: {args.checkpoint}", flush=True)
    print(f"NLF detector checkpoint: {checkpoints.get('nlf_smpl', '<missing>')}", flush=True)
    print(
        "NLF runtime: "
        f"detector={model_cfg.get('nlf_use_detector')} "
        f"model_name={model_cfg.get('nlf_model_name')} "
        f"internal_batch_size={model_cfg.get('nlf_internal_batch_size')} "
        f"num_aug={model_cfg.get('nlf_num_aug')}",
        flush=True,
    )
    print(f"SMPL body model: {assets.get('smpl_model_dir', '<missing>')}", flush=True)
    print(
        "PoseTemporalStabilizer V2 checkpoint: "
        f"{args.temporal_checkpoint} "
        f"window={stabilizer.config.window_size} "
        f"proposal_hidden={stabilizer.config.proposal_hidden_dim} "
        f"gate_hidden={stabilizer.config.gate_hidden_dim} "
        f"max_blend={stabilizer.config.max_blend}",
        flush=True,
    )
    print("HSI scale checkpoint: not loaded (disabled)", flush=True)
    print("TRSTR checkpoint: not loaded (disabled)", flush=True)
    print("Contact/grounding checkpoints: not loaded (disabled)", flush=True)
    print(f"GT support root: {dataset.support_root}", flush=True)
    print(f"RGB frames root: {dataset.frames_root}", flush=True)
    print(f"Dataset records/windows: {len(dataset.records)} / {len(dataset)}", flush=True)
    print("Metric protocol: SMPL-24 pelvis-aligned PA-MPJPE / MPJPE / PVE; unique 9-frame centres", flush=True)


def build_dataset(config: dict[str, Any], args: argparse.Namespace) -> HMR4DSupportEvalDataset:
    data_cfg = config.get("data", {})
    support_root = args.support_root or require_path(
        config,
        "datasets.emdb_hmr4d_support_root" if args.dataset == "emdb1" else "datasets.threedpw_hmr4d_support_root",
    )
    frames_root = args.frames_root or require_path(config, "datasets.hmr4d_eval_frames_root")
    image_size, image_resolution = resolve_image_size_config(data_cfg, args.image_size)
    return HMR4DSupportEvalDataset(
        dataset=args.dataset,
        support_root=support_root,
        frames_root=frames_root,
        sidecar_root=None,
        sequence_length=WINDOW_SIZE,
        stride=1,
        image_size=image_size,
        image_resolution=image_resolution,
        resize_mode=str(data_cfg.get("resize_mode", "balanced")),
        max_humans=int(data_cfg.get("max_humans", config.get("model", {}).get("num_smpl_queries", 20))),
        patch_size=int(config.get("model", {}).get("patch_size", 16)),
        full_sequence=False,
    )


def apply_sequence_filter(dataset: HMR4DSupportEvalDataset, raw_filter: str) -> int:
    query = str(raw_filter or "").strip().lower()
    if not query:
        return len(dataset)
    record_indices = {
        index
        for index, record in enumerate(dataset.records)
        if query in str(record.vid).lower() or query in str(record.label.get("vname", "")).lower()
    }
    dataset._index = [item for item in dataset._index if item[0] in record_indices]
    if not dataset._index:
        raise ValueError(f"sequence_filter={raw_filter!r} matched no evaluation windows")
    return len(dataset)


def evaluate_center_frames(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, Any],
    gt: dict[str, torch.Tensor],
    stabilizer: PoseTemporalStabilizer,
    smpl: SMPLLayer,
    totals: dict[str, "MetricTotals"],
    coverage: "CoverageTotals",
    rows: list[dict[str, Any]],
) -> None:
    pose = require_prediction(predictions, "pred_pose_6d")
    betas = require_prediction(predictions, "pred_betas")
    transl = require_prediction(predictions, "pred_transl_cam")
    conf = require_prediction(predictions, "pred_confs")
    boxes = predictions.get("pred_boxes")
    track_ids = predictions.get("assigned_track_ids")
    track_mask = predictions.get("assigned_track_mask")
    batch_size, steps, queries = pose.shape[:3]
    if steps != WINDOW_SIZE:
        raise ValueError(f"Expected {WINDOW_SIZE} frames, got {steps}")
    eval_mask = batch.get("eval_mask", torch.ones(batch_size, steps, device=pose.device, dtype=torch.bool)).bool()
    for batch_index in range(batch_size):
        coverage.total_windows += 1
        if not bool(eval_mask[batch_index, CENTER_INDEX]):
            coverage.invalid_gt_center += 1
            continue
        query_index, selection_kind = select_centre_query(boxes, conf, batch, batch_index, queries)
        if query_index < 0:
            coverage.no_nlf_detection += 1
            continue
        coverage.metric_centres += 1
        if selection_kind == "iou":
            coverage.iou_matched_centres += 1
        else:
            coverage.confidence_fallback_centres += 1

        temporal_pose, temporal_applied = stabilise_center_track(
            pose[batch_index], conf[batch_index], track_ids[batch_index] if isinstance(track_ids, torch.Tensor) else None,
            track_mask[batch_index] if isinstance(track_mask, torch.Tensor) else None, query_index, stabilizer,
        )
        if temporal_applied:
            coverage.temporal_applied_centres += 1
        else:
            coverage.temporal_noop_centres += 1
        gt_pose = gt["poses"][batch_index, CENTER_INDEX].reshape(1, 72)
        gt_beta = gt["betas"][batch_index, CENTER_INDEX].reshape(1, 10)
        gt_transl = gt["transl"][batch_index, CENTER_INDEX].reshape(1, 3)
        base_values = metric_values(
            pose[batch_index, CENTER_INDEX, query_index].reshape(1, 144),
            betas[batch_index, CENTER_INDEX, query_index].reshape(1, 10),
            transl[batch_index, CENTER_INDEX, query_index].reshape(1, 3),
            gt_pose, gt_beta, gt_transl, smpl,
        )
        temporal_values = metric_values(
            temporal_pose.reshape(1, 144), betas[batch_index, CENTER_INDEX, query_index].reshape(1, 10),
            transl[batch_index, CENTER_INDEX, query_index].reshape(1, 3), gt_pose, gt_beta, gt_transl, smpl,
        )
        for name, values in (("nlf_base", base_values), ("nlf_pose_temporal", temporal_values)):
            for key, value in values.items():
                totals[name].add(key, value.mean(), int(value.numel()))
        append_row(rows, batch, batch_index, query_index, selection_kind, temporal_applied, base_values, temporal_values)


def select_centre_query(
    pred_boxes: torch.Tensor | None,
    conf: torch.Tensor,
    batch: dict[str, Any],
    batch_index: int,
    num_queries: int,
) -> tuple[int, str]:
    scores = conf[batch_index, CENTER_INDEX]
    scores = scores[..., 0] if scores.shape[-1] == 1 else scores
    valid = scores > 0.0
    gt_boxes = batch.get("gt_boxes")
    boxes_mask = batch.get("boxes_mask")
    if isinstance(pred_boxes, torch.Tensor) and isinstance(gt_boxes, torch.Tensor) and isinstance(boxes_mask, torch.Tensor):
        gt_valid = torch.nonzero(boxes_mask[batch_index, CENTER_INDEX].bool(), as_tuple=False).reshape(-1)
        if gt_valid.numel() > 0:
            target = gt_boxes[batch_index, CENTER_INDEX, gt_valid[0]].reshape(1, 4)
            iou = box_iou_cxcywh(pred_boxes[batch_index, CENTER_INDEX], target).reshape(-1)
            iou = iou.masked_fill(~valid, -1.0)
            if float(iou.max()) >= 0.0:
                return int(iou.argmax()), "iou"
    if bool(valid.any()):
        return int(scores.masked_fill(~valid, -1.0).argmax()), "confidence"
    return -1, "none"


def stabilise_center_track(
    pose: torch.Tensor,
    conf: torch.Tensor,
    track_ids: torch.Tensor | None,
    track_mask: torch.Tensor | None,
    center_query: int,
    stabilizer: PoseTemporalStabilizer,
) -> tuple[torch.Tensor, bool]:
    """Apply V2 only when the selected centre detection has valid local track context."""
    base = pose[CENTER_INDEX, center_query]
    if track_ids is None or track_mask is None or not bool(track_mask[CENTER_INDEX, center_query]):
        return base, False
    track_id = int(track_ids[CENTER_INDEX, center_query])
    if track_id < 0:
        return base, False
    same_id = (track_ids == track_id) & track_mask.bool()
    counts = same_id.sum(dim=1)
    valid = counts == 1
    slots = same_id.to(dtype=torch.long).argmax(dim=1)
    track_pose = torch.zeros(1, WINDOW_SIZE, 144, device=pose.device, dtype=pose.dtype)
    frames = torch.nonzero(valid, as_tuple=False).reshape(-1)
    if frames.numel() > 0:
        track_pose[0, frames] = pose[frames, slots[frames]]
    outputs = stabilizer(track_pose, valid.reshape(1, WINDOW_SIZE))
    if not bool(outputs["context_valid"][0, CENTER_INDEX]):
        return base, False
    return outputs["refined_pose_6d"][0, CENTER_INDEX], True


def metric_values(
    pred_pose6d: torch.Tensor,
    pred_betas: torch.Tensor,
    pred_transl: torch.Tensor,
    gt_pose: torch.Tensor,
    gt_betas: torch.Tensor,
    gt_transl: torch.Tensor,
    smpl: SMPLLayer,
) -> dict[str, torch.Tensor]:
    pred_vertices, pred_joints = smpl(rot6d_to_axis_angle(pred_pose6d.reshape(-1, 24, 6)).reshape(-1, 72).float(), pred_betas.float())
    gt_vertices, gt_joints = smpl(gt_pose.float(), gt_betas.float())
    pred_joints, gt_joints = pred_joints[:, :24], gt_joints[:, :24]
    pred_joints_cam, gt_joints_cam = pred_joints + pred_transl[:, None, :], gt_joints + gt_transl[:, None, :]
    pred_vertices_cam, gt_vertices_cam = pred_vertices + pred_transl[:, None, :], gt_vertices + gt_transl[:, None, :]
    pred_ja, gt_ja, pred_va, gt_va = align_by_pelvis(pred_joints_cam, gt_joints_cam, pred_vertices_cam, gt_vertices_cam)
    return {
        "pa_mpjpe_m": procrustes_mpjpe(pred_ja, gt_ja),
        "mpjpe_m": torch.linalg.norm(pred_ja - gt_ja, dim=-1).mean(dim=-1),
        "pve_m": torch.linalg.norm(pred_va - gt_va, dim=-1).mean(dim=-1),
        "cam_mpjpe_no_align_m": torch.linalg.norm(pred_joints_cam - gt_joints_cam, dim=-1).mean(dim=-1),
        "cam_pve_no_align_m": torch.linalg.norm(pred_vertices_cam - gt_vertices_cam, dim=-1).mean(dim=-1),
    }


def align_by_pelvis(pred_joints: torch.Tensor, gt_joints: torch.Tensor, pred_vertices: torch.Tensor, gt_vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_pelvis = 0.5 * (pred_joints[:, 1:2] + pred_joints[:, 2:3])
    gt_pelvis = 0.5 * (gt_joints[:, 1:2] + gt_joints[:, 2:3])
    return pred_joints - pred_pelvis, gt_joints - gt_pelvis, pred_vertices - pred_pelvis, gt_vertices - gt_pelvis


def procrustes_mpjpe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_center, target_center = pred.mean(dim=1, keepdim=True), target.mean(dim=1, keepdim=True)
    pred0, target0 = pred - pred_center, target - target_center
    pred_norm = torch.linalg.norm(pred0.reshape(pred0.shape[0], -1), dim=1).clamp_min(1e-8)
    target_norm = torch.linalg.norm(target0.reshape(target0.shape[0], -1), dim=1).clamp_min(1e-8)
    x, y = pred0 / pred_norm[:, None, None], target0 / target_norm[:, None, None]
    u, singular, vh = torch.linalg.svd(y.transpose(1, 2) @ x)
    v = vh.transpose(1, 2)
    rotation = v @ u.transpose(1, 2)
    det = torch.det(rotation)
    if bool((det < 0).any()):
        v, singular = v.clone(), singular.clone()
        v[det < 0, :, -1] *= -1.0
        singular[det < 0, -1] *= -1.0
        rotation = v @ u.transpose(1, 2)
    scale = singular.sum(dim=1) * target_norm / pred_norm
    aligned = scale[:, None, None] * (pred0 @ rotation) + target_center
    return torch.linalg.norm(aligned - target, dim=-1).mean(dim=-1)


def require_prediction(predictions: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    value = predictions.get(key)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"NLF prediction missing required tensor: {key}")
    return value


def append_row(
    rows: list[dict[str, Any]],
    batch: dict[str, Any],
    batch_index: int,
    query_index: int,
    selection_kind: str,
    temporal_applied: bool,
    base: dict[str, torch.Tensor],
    temporal: dict[str, torch.Tensor],
) -> None:
    meta = batch.get("meta", {})
    frame_indices = meta.get("frame_indices", [[]])[batch_index] if isinstance(meta, dict) else []
    rows.append(
        {
            "dataset": meta.get("dataset_key", [""])[batch_index] if isinstance(meta, dict) else "",
            "vid": meta.get("vid", [""])[batch_index] if isinstance(meta, dict) else "",
            "frame_index": int(frame_indices[CENTER_INDEX]) if len(frame_indices) > CENTER_INDEX else -1,
            "query_idx": int(query_index),
            "query_selection": selection_kind,
            "temporal_applied": int(temporal_applied),
            **{f"base_{key[:-2]}_mm": float(value[0].detach().cpu() * 1000.0) for key, value in base.items()},
            **{f"temporal_{key[:-2]}_mm": float(value[0].detach().cpu() * 1000.0) for key, value in temporal.items()},
        }
    )


class MetricTotals:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, key: str, value: torch.Tensor, count: int) -> None:
        self.sums[key] = self.sums.get(key, 0.0) + float(value.detach().cpu()) * int(count)
        self.counts[key] = self.counts.get(key, 0) + int(count)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, total in sorted(self.sums.items()):
            value = total / max(self.counts.get(key, 0), 1)
            out[key] = value
            if key.endswith("_m"):
                out[f"{key[:-2]}_mm"] = value * 1000.0
        out["count"] = dict(self.counts)
        return out


class CoverageTotals:
    def __init__(self) -> None:
        self.total_windows = 0
        self.invalid_gt_center = 0
        self.no_nlf_detection = 0
        self.metric_centres = 0
        self.iou_matched_centres = 0
        self.confidence_fallback_centres = 0
        self.temporal_applied_centres = 0
        self.temporal_noop_centres = 0

    def summary(self) -> dict[str, int | float]:
        out = dict(self.__dict__)
        denom = max(self.metric_centres, 1)
        out["temporal_applied_rate"] = self.temporal_applied_centres / denom
        out["iou_match_rate"] = self.iou_matched_centres / denom
        return out


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset", "vid", "frame_index", "query_idx", "query_selection", "temporal_applied",
        "base_pa_mpjpe_mm", "base_mpjpe_mm", "base_pve_mm", "base_cam_mpjpe_no_align_mm", "base_cam_pve_no_align_mm",
        "temporal_pa_mpjpe_mm", "temporal_mpjpe_mm", "temporal_pve_mm", "temporal_cam_mpjpe_no_align_mm", "temporal_cam_pve_no_align_mm",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
