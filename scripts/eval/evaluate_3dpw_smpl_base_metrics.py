from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_smpl import apply_overrides, build_model, forward_model, load_initial_checkpoint
from vggt_omega.data import ThreeDPWDataset, threedpw_collate_fn
from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.training.config import deep_update, load_yaml_config, require_path
from vggt_omega.training.hungarian_losses import flatten_smpl_targets
from vggt_omega.training.smpl_matcher import HungarianSMPLMatcher
from vggt_omega.utils.rotation import rot6d_to_axis_angle


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.train_config))
    config = apply_overrides(config, args.override)
    model = build_model(config).to(device)
    load_initial_checkpoint(model, config, device)
    load_training_checkpoint(model, Path(args.checkpoint), device)
    model.eval()
    smpl = SMPLLayer(require_path(config, "assets.smpl_model_dir"), gender="neutral").to(device).eval()
    dataset = build_dataset(config, args)
    indices = list(range(len(dataset)))
    if int(args.max_samples) > 0:
        indices = indices[int(args.start_index) : int(args.start_index) + int(args.max_samples)]
    else:
        indices = indices[int(args.start_index) :]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=threedpw_collate_fn,
        drop_last=False,
    )
    matcher = HungarianSMPLMatcher(cost_conf=0.5, cost_bbox=5.0, cost_giou=2.0, cost_kpts=0.0, require_boxes=True, require_j2ds=False)
    totals = MetricTotals()
    rows: list[dict[str, Any]] = []
    processed = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            predictions = forward_model(model, batch, config)
            evaluate_batch(predictions, batch, matcher, smpl, totals, rows, selected_indices=indices[processed : processed + int(batch["images"].shape[0])])
            processed += int(batch["images"].shape[0])
            if args.log_interval > 0 and processed % int(args.log_interval) == 0:
                print(f"[eval] processed={processed}", flush=True)
    summary = totals.summary()
    summary.update(
        {
            "dataset": "3dpw",
            "split": args.split,
            "checkpoint": str(args.checkpoint),
            "num_windows": int(processed),
            "num_matches": len(rows),
            "metric_protocol": "project_native_smpl24_camera_pelvis_aligned",
            "rows_csv": str(output_dir / "3dpw_smpl_base_metric_rows.csv"),
        }
    )
    (output_dir / "3dpw_smpl_base_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "3dpw_smpl_base_metric_rows.csv", rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 3DPW SMPL base checkpoint with PA-MPJPE/MPJPE/PVE")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_base_3dpw.yaml")
    parser.add_argument("--output-dir", default="outputs/eval/3dpw_smpl_base")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def build_dataset(config: dict[str, Any], args: argparse.Namespace) -> ThreeDPWDataset:
    data_cfg = config["data"]
    return ThreeDPWDataset(
        root=require_path(config, data_cfg.get("root_key", "datasets.threedpw_root")),
        annotation_root=require_path(config, data_cfg.get("annotation_root_key", "datasets.threedpw_smpl_base_root")),
        split=args.split,
        sequence_length=int(data_cfg.get("sequence_length", 1)),
        stride=int(data_cfg.get("stride", 1)),
        image_size=int(data_cfg.get("image_size", 518)),
        max_humans=int(data_cfg.get("max_humans", 2)),
        require_boxes=True,
        require_smpl=True,
    )


def load_training_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if state_dict is None:
        state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint does not contain a state_dict: {checkpoint_path}")
    missing, unexpected = model.load_state_dict({key.removeprefix("module."): value for key, value in state_dict.items()}, strict=False)
    print(f"[ckpt] loaded {checkpoint_path} missing={len(missing)} unexpected={len(unexpected)}", flush=True)


def evaluate_batch(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    matcher: HungarianSMPLMatcher,
    smpl: SMPLLayer,
    totals: "MetricTotals",
    rows: list[dict[str, Any]],
    selected_indices: list[int],
) -> None:
    pred_confs = flatten_prediction(predictions["pred_confs"], 3)
    pred_boxes = flatten_prediction(predictions["pred_boxes"], 3)
    pred_poses = flatten_prediction(predictions["pred_poses"], 3)
    pred_betas = flatten_prediction(predictions["pred_betas"], 3)
    targets = flatten_smpl_targets(batch, device=pred_confs.device)
    indices = matcher({"pred_confs": pred_confs, "pred_boxes": pred_boxes}, targets)
    matched = collect_matches(indices, targets, pred_confs.device)
    if matched["frame_idx"].numel() == 0:
        return
    frame_idx = matched["frame_idx"]
    src_idx = matched["src_idx"]
    gt_pose = rot6d_to_axis_angle(matched["pose_6d"]).reshape(-1, 72)
    gt_betas = matched["betas"]
    pred_vertices, pred_joints = smpl(pred_poses[frame_idx, src_idx].reshape(-1, 72).float(), pred_betas[frame_idx, src_idx].float())
    gt_vertices, gt_joints = smpl(gt_pose.float(), gt_betas.float())
    pred_joints = pred_joints[:, :24].to(dtype=pred_vertices.dtype)
    gt_joints = gt_joints[:, :24].to(dtype=gt_vertices.dtype)
    pred_joints_a, gt_joints_a, pred_vertices_a, gt_vertices_a = align_by_pelvis(pred_joints, gt_joints, pred_vertices, gt_vertices)
    mpjpe = torch.linalg.norm(pred_joints_a - gt_joints_a, dim=-1).mean(dim=-1)
    pve = torch.linalg.norm(pred_vertices_a - gt_vertices_a, dim=-1).mean(dim=-1)
    pa = procrustes_mpjpe(pred_joints_a, gt_joints_a)
    totals.add("mpjpe_mm", mpjpe.mean() * 1000.0, int(mpjpe.numel()))
    totals.add("pa_mpjpe_mm", pa.mean() * 1000.0, int(pa.numel()))
    totals.add("pve_mm", pve.mean() * 1000.0, int(pve.numel()))
    append_rows(rows, selected_indices, batch, matched, src_idx, mpjpe, pa, pve)


def collect_matches(indices, targets: list[dict[str, torch.Tensor]], device: torch.device) -> dict[str, torch.Tensor]:
    frame_indices = []
    src_indices = []
    target_indices = []
    parts: dict[str, list[torch.Tensor]] = {"pose_6d": [], "betas": [], "transl_cam": [], "person_ids": []}
    for frame_idx, (src_idx, tgt_idx) in enumerate(indices):
        if src_idx.numel() == 0:
            continue
        frame_indices.append(torch.full_like(src_idx, frame_idx))
        src_indices.append(src_idx)
        target_indices.append(tgt_idx)
        target = targets[frame_idx]
        for key in parts:
            parts[key].append(target[key][tgt_idx])
    if not frame_indices:
        return {
            "frame_idx": torch.empty(0, dtype=torch.long, device=device),
            "src_idx": torch.empty(0, dtype=torch.long, device=device),
            "target_idx": torch.empty(0, dtype=torch.long, device=device),
        }
    out = {"frame_idx": torch.cat(frame_indices), "src_idx": torch.cat(src_indices), "target_idx": torch.cat(target_indices)}
    out.update({key: torch.cat(value) for key, value in parts.items()})
    return out


def align_by_pelvis(
    pred_joints: torch.Tensor,
    gt_joints: torch.Tensor,
    pred_vertices: torch.Tensor,
    gt_vertices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_pelvis = 0.5 * (pred_joints[:, 1:2] + pred_joints[:, 2:3])
    gt_pelvis = 0.5 * (gt_joints[:, 1:2] + gt_joints[:, 2:3])
    return pred_joints - pred_pelvis, gt_joints - gt_pelvis, pred_vertices - pred_pelvis, gt_vertices - gt_pelvis


def procrustes_mpjpe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_center = pred.mean(dim=1, keepdim=True)
    target_center = target.mean(dim=1, keepdim=True)
    x = pred - pred_center
    y = target - target_center
    x_norm = torch.linalg.norm(x.reshape(x.shape[0], -1), dim=1).clamp_min(1e-8)
    y_norm = torch.linalg.norm(y.reshape(y.shape[0], -1), dim=1).clamp_min(1e-8)
    x = x / x_norm[:, None, None]
    y = y / y_norm[:, None, None]
    h = x.transpose(1, 2) @ y
    u, _, vh = torch.linalg.svd(h)
    r = vh.transpose(1, 2) @ u.transpose(1, 2)
    det = torch.det(r)
    if bool((det < 0).any()):
        vh = vh.clone()
        vh[det < 0, -1, :] *= -1.0
        r = vh.transpose(1, 2) @ u.transpose(1, 2)
    x_aligned = (x @ r) * y_norm[:, None, None] + target_center
    return torch.linalg.norm(x_aligned - target, dim=-1).mean(dim=-1)


def append_rows(
    rows: list[dict[str, Any]],
    selected_indices: list[int],
    batch: dict[str, torch.Tensor],
    matched: dict[str, torch.Tensor],
    src_idx: torch.Tensor,
    mpjpe: torch.Tensor,
    pa: torch.Tensor,
    pve: torch.Tensor,
) -> None:
    num_frames = int(batch["images"].shape[1])
    frame_idx = matched["frame_idx"]
    target_idx = matched["target_idx"]
    for row_idx in range(int(frame_idx.numel())):
        flat_frame = int(frame_idx[row_idx].detach().cpu())
        batch_idx = flat_frame // num_frames
        frame_offset = flat_frame % num_frames
        rows.append(
            {
                "dataset_index": int(selected_indices[batch_idx]) if batch_idx < len(selected_indices) else -1,
                "frame_offset": frame_offset,
                "query_idx": int(src_idx[row_idx].detach().cpu()),
                "gt_idx": int(target_idx[row_idx].detach().cpu()),
                "person_id": int(matched["person_ids"][row_idx].detach().cpu()),
                "mpjpe_mm": float(mpjpe[row_idx].detach().cpu() * 1000.0),
                "pa_mpjpe_mm": float(pa[row_idx].detach().cpu() * 1000.0),
                "pve_mm": float(pve[row_idx].detach().cpu() * 1000.0),
            }
        )


def flatten_prediction(value: torch.Tensor, unframed_ndim: int) -> torch.Tensor:
    if value.ndim == unframed_ndim:
        return value
    if value.ndim == unframed_ndim + 1:
        return value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
    raise ValueError(f"Unsupported prediction shape {tuple(value.shape)}")


class MetricTotals:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, key: str, value: torch.Tensor, count: int) -> None:
        self.sums[key] = self.sums.get(key, 0.0) + float(value.detach().cpu()) * int(count)
        self.counts[key] = self.counts.get(key, 0) + int(count)

    def summary(self) -> dict[str, Any]:
        out = {}
        for key, total in sorted(self.sums.items()):
            out[key] = total / max(self.counts.get(key, 0), 1)
        out["count"] = dict(self.counts)
        return out


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["dataset_index", "frame_offset", "query_idx", "gt_idx", "person_id", "mpjpe_mm", "pa_mpjpe_mm", "pve_mm"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
