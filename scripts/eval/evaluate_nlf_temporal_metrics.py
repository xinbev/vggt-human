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
    load_training_checkpoint,
    move_to_device,
)
from scripts.train.train_smpl import apply_overrides, build_model, load_initial_checkpoint  # noqa: E402
from vggt_omega.data import HMR4DSupportEvalDataset, hmr4d_eval_collate_fn  # noqa: E402
from vggt_omega.data.geometry import resolve_image_size_config  # noqa: E402
from vggt_omega.integrations.smpl_temporal_refiner import SMPLTemporalRefinementAdapter  # noqa: E402
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.training.config import deep_update, load_yaml_config, require_path  # noqa: E402
from vggt_omega.utils.rotation import rot6d_to_axis_angle  # noqa: E402


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.train_config))
    config = apply_overrides(config, args.override)
    config = apply_nlf_temporal_defaults(config)
    model = build_model(config).to(device)
    load_initial_checkpoint(model, config, device)
    load_training_checkpoint(model, Path(args.checkpoint), device)
    model.eval()
    temporal = SMPLTemporalRefinementAdapter.from_checkpoint(args.temporal_checkpoint, device=device)
    temporal_window = int(temporal.refiner.config.window_size)
    smpl = SMPLLayer(require_path(config, "assets.smpl_model_dir", allow_empty=False)).to(device).eval()

    dataset = build_dataset(config, args, sequence_length=temporal_window)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=hmr4d_eval_collate_fn,
        drop_last=False,
    )
    totals = {"base": MetricTotals(), "temporal": MetricTotals()}
    rows: list[dict[str, Any]] = []
    seen_frames: set[tuple[str, int]] = set()
    processed = 0
    unsupported: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            # Deliberately omit query boxes, masks and all external sidecars.
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
                processed += int(batch["images"].shape[0])
                continue
            refined = refine_predictions(predictions, temporal)
            evaluate_batch(predictions, refined, batch, gt, smpl, totals, rows, seen_frames)
            processed += int(batch["images"].shape[0])
            if int(args.max_windows) > 0 and processed >= int(args.max_windows):
                break
            if args.log_interval > 0 and processed % int(args.log_interval) == 0:
                print(f"[eval] processed_windows={processed}", flush=True)

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "checkpoint": str(args.checkpoint),
        "temporal_checkpoint": str(args.temporal_checkpoint),
        "num_windows_processed": int(processed),
        "num_metric_rows": len(rows),
        "num_unsupported_windows": len(unsupported),
        "unsupported_examples": unsupported[:5],
        "metric_protocol": "project_native_smpl24_pelvis_aligned",
        "input_protocol": "RGB->VGGT->NLF_detector->standalone_TemporalSMPLRefiner; no_HSI_scale; no_TRSTR; no_sidecar",
        "temporal_coordinate_assumption": "TemporalRefiner target_transl is in the same camera coordinate system as NLF pred_transl_cam; verify against training pkl convention before reporting temporal numbers.",
        "temporal_window_size": temporal_window,
        "rows_csv": str(output_dir / f"{args.dataset}_nlf_temporal_metric_rows.csv"),
    }
    for name, metric_totals in totals.items():
        summary[name] = metric_totals.summary()
    (output_dir / f"{args.dataset}_nlf_temporal_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / f"{args.dataset}_nlf_temporal_metric_rows.csv", rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RGB->VGGT->NLF->TemporalSMPLRefiner without HSI/TRSTR/sidecars")
    parser.add_argument("--dataset", required=True, choices=["emdb1", "emdb2", "rich", "3dpw"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--temporal-checkpoint", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/eval_nlf_temporal.yaml")
    parser.add_argument("--support-root", default="")
    parser.add_argument("--frames-root", default="")
    parser.add_argument("--output-dir", default="outputs/eval/nlf_temporal")
    parser.add_argument("--device", default="")
    parser.add_argument("--sequence-length", type=int, default=0, help="Must not exceed the temporal checkpoint window")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def apply_nlf_temporal_defaults(config: dict[str, Any]) -> dict[str, Any]:
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


def build_dataset(config: dict[str, Any], args: argparse.Namespace, sequence_length: int) -> HMR4DSupportEvalDataset:
    data_cfg = config.get("data", {})
    support_root = args.support_root or support_key(config, args.dataset)
    frames_root = args.frames_root or require_path(config, "datasets.hmr4d_eval_frames_root")
    image_size, image_resolution = resolve_image_size_config(data_cfg, args.image_size)
    requested = int(args.sequence_length or sequence_length)
    if requested > sequence_length:
        raise ValueError(f"sequence_length={requested} exceeds temporal checkpoint window={sequence_length}")
    return HMR4DSupportEvalDataset(
        dataset=args.dataset,
        support_root=support_root,
        frames_root=frames_root,
        sidecar_root=None,
        sequence_length=requested,
        stride=int(args.stride),
        image_size=image_size,
        image_resolution=image_resolution,
        resize_mode=str(data_cfg.get("resize_mode", "balanced")),
        max_humans=int(data_cfg.get("max_humans", config.get("model", {}).get("num_smpl_queries", 20))),
        patch_size=int(config.get("model", {}).get("patch_size", 16)),
        full_sequence=False,
    )


def support_key(config: dict[str, Any], dataset: str) -> str:
    return require_path(
        config,
        {
            "emdb1": "datasets.emdb_hmr4d_support_root",
            "emdb2": "datasets.emdb_hmr4d_support_root",
            "rich": "datasets.rich_hmr4d_support_root",
            "3dpw": "datasets.threedpw_hmr4d_support_root",
        }[dataset],
    )


def refine_predictions(predictions: dict[str, torch.Tensor], temporal: SMPLTemporalRefinementAdapter) -> dict[str, torch.Tensor]:
    pose = predictions.get("pred_pose_6d")
    transl = predictions.get("pred_transl_cam")
    betas = predictions.get("pred_betas")
    if not all(isinstance(value, torch.Tensor) for value in (pose, transl, betas)):
        raise KeyError("NLF predictions must contain pred_pose_6d, pred_transl_cam and pred_betas")
    track_ids = predictions.get("assigned_track_ids")
    track_mask = predictions.get("assigned_track_mask")
    conf = predictions.get("pred_confs")
    if not isinstance(track_ids, torch.Tensor):
        track_ids = torch.arange(pose.shape[2], device=pose.device).reshape(1, 1, -1).expand(pose.shape[:3])
    if not isinstance(track_mask, torch.Tensor):
        track_mask = torch.ones_like(track_ids, dtype=torch.bool)
    confidence = conf.squeeze(-1) if isinstance(conf, torch.Tensor) and conf.shape[-1] == 1 else conf
    return temporal.refine_tracked_batch(pose, transl, betas, track_ids, track_mask, confidence)


def evaluate_batch(
    predictions: dict[str, torch.Tensor],
    refined: dict[str, torch.Tensor],
    batch: dict[str, Any],
    gt: dict[str, torch.Tensor],
    smpl: SMPLLayer,
    totals: dict[str, "MetricTotals"],
    rows: list[dict[str, Any]],
    seen_frames: set[tuple[str, int]],
) -> None:
    pred_pose = predictions["pred_pose_6d"]
    pred_betas = predictions["pred_betas"]
    pred_transl = predictions["pred_transl_cam"]
    ref_pose = refined["smpl_temporal_refined_pose_6d"]
    ref_transl = refined["smpl_temporal_refined_pred_transl_cam"]
    conf = predictions.get("pred_confs")
    pred_boxes = predictions.get("pred_boxes")
    gt_boxes = batch.get("gt_boxes")
    boxes_mask = batch.get("boxes_mask")
    batch_size, num_frames, num_queries = pred_pose.shape[:3]
    for b in range(batch_size):
        valid = batch.get("eval_mask", torch.ones(batch_size, num_frames, dtype=torch.bool, device=pred_pose.device))[b].bool()
        if not bool(valid.any()):
            continue
        q_by_frame = choose_queries(pred_boxes, gt_boxes, boxes_mask, conf, b, num_frames, num_queries)
        frame_offsets = torch.where(valid)[0]
        q = q_by_frame[frame_offsets]
        pose_base = pred_pose[b, frame_offsets, q].reshape(-1, 144)
        pose_ref = ref_pose[b, frame_offsets, q].reshape(-1, 144)
        beta = pred_betas[b, frame_offsets, q]
        transl_base = pred_transl[b, frame_offsets, q]
        transl_ref = ref_transl[b, frame_offsets, q]
        gt_pose = gt["poses"][b, frame_offsets].reshape(-1, 72)
        gt_beta = gt["betas"][b, frame_offsets]
        gt_transl = gt["transl"][b, frame_offsets]
        base_metrics = metric_values(pose_base, beta, transl_base, gt_pose, gt_beta, gt_transl, smpl)
        ref_metrics = metric_values(pose_ref, beta, transl_ref, gt_pose, gt_beta, gt_transl, smpl)
        meta = batch.get("meta", {})
        vid = meta.get("vid", [""] * batch_size)[b] if isinstance(meta, dict) else ""
        frame_indices = meta.get("frame_indices", [[]] * batch_size)[b] if isinstance(meta, dict) else []
        keep = []
        for local_idx, frame_offset in enumerate(frame_offsets.tolist()):
            frame_number = int(frame_indices[frame_offset]) if frame_offset < len(frame_indices) else int(frame_offset)
            key = (str(vid), frame_number)
            if key not in seen_frames:
                seen_frames.add(key)
                keep.append(local_idx)
        if not keep:
            continue
        keep_tensor = torch.as_tensor(keep, device=pose_base.device, dtype=torch.long)
        for name, values in (("base", base_metrics), ("temporal", ref_metrics)):
            for key, value in values.items():
                selected = value[keep_tensor]
                totals[name].add(key, selected.mean(), int(selected.numel()))
        for local_idx in keep:
            frame_offset = int(frame_offsets[local_idx])
            rows.append(
                {
                    "dataset": batch.get("meta", {}).get("dataset_key", [""] * batch_size)[b],
                    "vid": vid,
                    "frame_index": int(frame_indices[frame_offset]) if frame_offset < len(frame_indices) else int(frame_offset),
                    "query_idx": int(q[local_idx]),
                    "base_pa_mpjpe_mm": float(base_metrics["pa_mpjpe_m"][local_idx] * 1000.0),
                    "base_mpjpe_mm": float(base_metrics["mpjpe_m"][local_idx] * 1000.0),
                    "base_pve_mm": float(base_metrics["pve_m"][local_idx] * 1000.0),
                    "temporal_pa_mpjpe_mm": float(ref_metrics["pa_mpjpe_m"][local_idx] * 1000.0),
                    "temporal_mpjpe_mm": float(ref_metrics["mpjpe_m"][local_idx] * 1000.0),
                    "temporal_pve_mm": float(ref_metrics["pve_m"][local_idx] * 1000.0),
                }
            )


def choose_queries(pred_boxes, gt_boxes, boxes_mask, conf, batch_idx: int, num_frames: int, num_queries: int) -> torch.Tensor:
    if isinstance(pred_boxes, torch.Tensor) and isinstance(gt_boxes, torch.Tensor) and isinstance(boxes_mask, torch.Tensor):
        iou = box_iou_cxcywh(pred_boxes[batch_idx, :, :, None, :], gt_boxes[batch_idx, :, None, :, :])
        iou = iou.masked_fill(~boxes_mask[batch_idx, :, None, :].bool(), -1.0)
        best = iou.max(dim=-1).values.argmax(dim=-1)
        return best
    if isinstance(conf, torch.Tensor):
        score = conf[batch_idx].squeeze(-1) if conf.shape[-1] == 1 else conf[batch_idx]
        return score.argmax(dim=-1)
    return torch.zeros(num_frames, dtype=torch.long, device=gt_boxes.device if isinstance(gt_boxes, torch.Tensor) else "cpu")


def metric_values(pred_pose6d, pred_betas, pred_transl, gt_pose, gt_betas, gt_transl, smpl):
    pred_v, pred_j = smpl(rot6d_to_axis_angle(pred_pose6d.reshape(-1, 24, 6)).reshape(-1, 72).float(), pred_betas.float())
    gt_v, gt_j = smpl(gt_pose.float(), gt_betas.float())
    pred_j, gt_j = pred_j[:, :24], gt_j[:, :24]
    pred_jc, gt_jc = pred_j + pred_transl[:, None, :], gt_j + gt_transl[:, None, :]
    pred_vc, gt_vc = pred_v + pred_transl[:, None, :], gt_v + gt_transl[:, None, :]
    pred_ja, gt_ja, pred_va, gt_va = align_by_pelvis(pred_jc, gt_jc, pred_vc, gt_vc)
    mpjpe = torch.linalg.norm(pred_ja - gt_ja, dim=-1).mean(dim=-1)
    pve = torch.linalg.norm(pred_va - gt_va, dim=-1).mean(dim=-1)
    pa = procrustes_mpjpe(pred_ja, gt_ja)
    return {"mpjpe_m": mpjpe, "pve_m": pve, "pa_mpjpe_m": pa}


def align_by_pelvis(pred_j, gt_j, pred_v, gt_v):
    pred_root = 0.5 * (pred_j[:, 1:2] + pred_j[:, 2:3])
    gt_root = 0.5 * (gt_j[:, 1:2] + gt_j[:, 2:3])
    return pred_j - pred_root, gt_j - gt_root, pred_v - pred_root, gt_v - gt_root


def procrustes_mpjpe(pred, target):
    pred_center = pred.mean(dim=1, keepdim=True)
    target_center = target.mean(dim=1, keepdim=True)
    pred0, target0 = pred - pred_center, target - target_center
    pred_norm = torch.linalg.norm(pred0.reshape(pred0.shape[0], -1), dim=1).clamp_min(1e-8)
    target_norm = torch.linalg.norm(target0.reshape(target0.shape[0], -1), dim=1).clamp_min(1e-8)
    x, y = pred0 / pred_norm[:, None, None], target0 / target_norm[:, None, None]
    u, singular, vh = torch.linalg.svd(y.transpose(1, 2) @ x)
    v = vh.transpose(1, 2)
    r = v @ u.transpose(1, 2)
    det = torch.det(r)
    if bool((det < 0).any()):
        v = v.clone()
        v[det < 0, :, -1] *= -1.0
        singular = singular.clone()
        singular[det < 0, -1] *= -1.0
        r = v @ u.transpose(1, 2)
    scale = singular.sum(dim=1) * target_norm / pred_norm
    aligned = scale[:, None, None] * (pred0 @ r) + target_center
    return torch.linalg.norm(aligned - target, dim=-1).mean(dim=-1)


class MetricTotals:
    def __init__(self):
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, key: str, value: torch.Tensor, count: int):
        self.sums[key] = self.sums.get(key, 0.0) + float(value.detach().cpu()) * int(count)
        self.counts[key] = self.counts.get(key, 0) + int(count)

    def summary(self):
        out = {}
        for key, total in sorted(self.sums.items()):
            value = total / max(self.counts.get(key, 0), 1)
            out[key] = value
            if key.endswith("_m"):
                out[f"{key[:-2]}_mm"] = value * 1000.0
        out["count"] = dict(self.counts)
        return out


def write_csv(path: Path, rows: list[dict[str, Any]]):
    fields = ["dataset", "vid", "frame_index", "query_idx", "base_pa_mpjpe_mm", "base_mpjpe_mm", "base_pve_mm", "temporal_pa_mpjpe_mm", "temporal_mpjpe_mm", "temporal_pve_mm"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
