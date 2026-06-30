import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_smpl import apply_overrides, build_model, load_initial_checkpoint
from vggt_omega.data import HMR4DSupportEvalDataset, hmr4d_eval_collate_fn
from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.training.config import deep_update, load_yaml_config, require_path


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
    smpl = SMPLLayer(require_path(config, "assets.smpl_model_dir", allow_empty=False)).to(device).eval()

    dataset = build_dataset(config, args)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=hmr4d_eval_collate_fn,
        drop_last=False,
    )
    totals = MetricTotals()
    rows: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    processed = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            predictions = model(
                batch["images"],
                smpl_query_boxes=batch.get("smpl_query_boxes"),
                smpl_query_boxes_mask=batch.get("smpl_query_boxes_mask"),
                smpl_query_patch_masks=batch.get("smpl_query_patch_masks"),
                external_track_ids=batch.get("external_track_ids"),
                external_track_mask=batch.get("external_track_mask"),
                external_track_confidence=batch.get("external_track_confidence"),
            )
            try:
                gt = extract_gt_smpl(batch["eval_label"], device)
            except UnsupportedLabelError as exc:
                unsupported.append({"meta": detach_meta(batch["meta"]), "reason": str(exc)})
                processed += int(batch["images"].shape[0])
                continue
            evaluate_batch(predictions, batch, gt, smpl, totals, rows, prefer_hsi=bool(args.prefer_hsi))
            processed += int(batch["images"].shape[0])
            if int(args.max_windows) > 0 and processed >= int(args.max_windows):
                break
            if args.log_interval > 0 and processed % args.log_interval == 0:
                print(f"[eval] processed_windows={processed}", flush=True)

    summary = totals.summary()
    summary.update(
        {
            "dataset": args.dataset,
            "checkpoint": str(args.checkpoint),
            "num_windows_processed": int(processed),
            "num_metric_rows": len(rows),
            "num_unsupported_windows": len(unsupported),
            "unsupported_examples": unsupported[:5],
            "metric_protocol": "project_native_smpl24",
            "note": "RICH exact metrics require SMPL-X to SMPL conversion assets; this script does not fabricate them.",
            "rows_csv": str(output_dir / f"{args.dataset}_smpl_metrics_rows.csv"),
        }
    )
    (output_dir / f"{args.dataset}_smpl_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / f"{args.dataset}_smpl_metrics_rows.csv", rows)
    if unsupported and not rows and not args.allow_missing_metrics:
        raise SystemExit("No supported SMPL labels were evaluated. Re-run with --allow-missing-metrics to export the summary only.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VGGT-Omega on EMDB/RICH/3DPW hmr4d_support adapters")
    parser.add_argument("--dataset", required=True, choices=["emdb1", "emdb2", "rich", "3dpw"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_hsi_full_system_restructure.yaml")
    parser.add_argument("--support-root", default="")
    parser.add_argument("--frames-root", default="")
    parser.add_argument("--sidecar-root", default="")
    parser.add_argument("--output-dir", default="outputs/eval/hmr4d_smpl_metrics")
    parser.add_argument("--device", default="")
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--max-humans", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--prefer-hsi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-missing-metrics", action="store_true")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def build_dataset(config: dict[str, Any], args: argparse.Namespace) -> HMR4DSupportEvalDataset:
    data_cfg = config.get("data", {})
    support_root = args.support_root or support_key(config, args.dataset)
    frames_root = args.frames_root or require_path(config, "datasets.hmr4d_eval_frames_root")
    sidecar_root = args.sidecar_root or str(config.get("datasets", {}).get("hmr4d_eval_tracks_root", "") or "")
    max_humans = int(args.max_humans or config.get("model", {}).get("num_smpl_queries", data_cfg.get("max_humans", 1)))
    return HMR4DSupportEvalDataset(
        dataset=args.dataset,
        support_root=support_root,
        frames_root=frames_root,
        sidecar_root=sidecar_root or None,
        sequence_length=int(args.sequence_length or data_cfg.get("sequence_length", 16)),
        stride=int(args.stride),
        image_size=int(args.image_size or data_cfg.get("image_size", 518)),
        max_humans=max_humans,
        patch_size=int(config.get("model", {}).get("patch_size", 16)),
        full_sequence=False,
    )


def support_key(config: dict[str, Any], dataset: str) -> str:
    key = {
        "emdb1": "datasets.emdb_hmr4d_support_root",
        "emdb2": "datasets.emdb_hmr4d_support_root",
        "rich": "datasets.rich_hmr4d_support_root",
        "3dpw": "datasets.threedpw_hmr4d_support_root",
    }[dataset]
    return require_path(config, key)


def load_training_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
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
    batch: dict[str, Any],
    gt: dict[str, torch.Tensor],
    smpl: SMPLLayer,
    totals: "MetricTotals",
    rows: list[dict[str, Any]],
    prefer_hsi: bool,
) -> None:
    pred_pose = select_prediction(predictions, "pred_poses", "hsi_refined_pred_poses", prefer_hsi)
    pred_betas = select_prediction(predictions, "pred_betas", "hsi_refined_pred_betas", prefer_hsi)
    pred_transl = select_prediction(predictions, "pred_transl_cam", "hsi_refined_pred_transl_cam", prefer_hsi)
    pred_confs = predictions.get("pred_confs")
    if pred_pose is None or pred_betas is None or pred_transl is None or pred_confs is None:
        raise KeyError("Model predictions missing SMPL pose/betas/transl/conf fields")

    batch_size, num_frames, num_queries = pred_confs.shape[:3]
    query_idx = choose_query_indices(predictions, batch)
    eval_mask = batch.get("eval_mask", torch.ones(batch_size, num_frames, dtype=torch.bool, device=pred_confs.device)).bool()
    for b in range(batch_size):
        q = query_idx[b]
        valid = eval_mask[b]
        if not bool(valid.any()):
            continue
        pred_pose_b = pred_pose[b, valid, q].reshape(-1, 72)
        pred_betas_b = pred_betas[b, valid, q]
        pred_transl_b = pred_transl[b, valid, q]
        gt_pose_b = gt["poses"][b, valid].reshape(-1, 72)
        gt_betas_b = gt["betas"][b, valid]
        gt_transl_b = gt["transl"][b, valid]

        pred_vertices, pred_joints = smpl(pred_pose_b.float(), pred_betas_b.float())
        gt_vertices, gt_joints = smpl(gt_pose_b.float(), gt_betas_b.float())
        pred_joints = pred_joints[:, :24].to(dtype=pred_transl_b.dtype)
        gt_joints = gt_joints[:, :24].to(dtype=pred_transl_b.dtype)
        pred_vertices = pred_vertices.to(dtype=pred_transl_b.dtype)
        gt_vertices = gt_vertices.to(dtype=pred_transl_b.dtype)
        pred_joints_cam = pred_joints + pred_transl_b[:, None, :]
        gt_joints_cam = gt_joints + gt_transl_b[:, None, :]
        pred_vertices_cam = pred_vertices + pred_transl_b[:, None, :]
        gt_vertices_cam = gt_vertices + gt_transl_b[:, None, :]

        mpjpe = torch.linalg.norm(root_align(pred_joints_cam) - root_align(gt_joints_cam), dim=-1).mean(dim=-1)
        cam_mpjpe = torch.linalg.norm(pred_joints_cam - gt_joints_cam, dim=-1).mean(dim=-1)
        pve = torch.linalg.norm(pred_vertices_cam - gt_vertices_cam, dim=-1).mean(dim=-1)
        pa_mpjpe = procrustes_mpjpe(pred_joints_cam, gt_joints_cam)
        totals.add("mpjpe_m", mpjpe.mean(), mpjpe.numel())
        totals.add("cam_mpjpe_m", cam_mpjpe.mean(), cam_mpjpe.numel())
        totals.add("pve_m", pve.mean(), pve.numel())
        totals.add("pa_mpjpe_m", pa_mpjpe.mean(), pa_mpjpe.numel())
        if pred_joints_cam.shape[0] >= 3:
            accel = acceleration_error(pred_joints_cam, gt_joints_cam)
            totals.add("accel_error_m", accel.mean(), accel.numel())
        append_rows(rows, batch["meta"], b, q, valid, mpjpe, cam_mpjpe, pve, pa_mpjpe)


def choose_query_indices(predictions: dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
    pred_boxes = predictions.get("pred_boxes")
    gt_boxes = batch.get("gt_boxes")
    boxes_mask = batch.get("boxes_mask")
    if isinstance(pred_boxes, torch.Tensor) and isinstance(gt_boxes, torch.Tensor) and isinstance(boxes_mask, torch.Tensor):
        iou = box_iou_cxcywh(pred_boxes[:, :, :, None, :], gt_boxes[:, :, None, :, :])
        iou = iou.masked_fill(~boxes_mask[:, :, None, :].bool(), -1.0)
        score = iou.max(dim=-1).values.mean(dim=1)
        return score.argmax(dim=-1)
    confs = predictions["pred_confs"]
    return confs.mean(dim=1).argmax(dim=-1)


def extract_gt_smpl(label: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    if "gt_smplx_params" in label and "smpl_params" not in label:
        raise UnsupportedLabelError("Found RICH SMPL-X labels but no SMPL conversion path. Provide SMPL-X/SMPL conversion assets before exact RICH metrics.")
    source = label.get("smpl_params", label)
    if not isinstance(source, dict):
        raise UnsupportedLabelError(f"Unsupported eval_label type: {type(source)!r}")
    pose = find_first(source, ("poses", "pose", "theta"))
    global_orient = find_first(source, ("global_orient", "global_orientation", "root_orient", "root_pose", "Rh"))
    body_pose = find_first(source, ("body_pose", "body_poses", "poses_body"))
    betas = find_first(source, ("betas", "beta", "shape", "smpl_betas"))
    transl = find_first(source, ("transl", "translation", "trans", "Th", "cam_trans", "cam_t"))
    if pose is None:
        if global_orient is None or body_pose is None:
            raise UnsupportedLabelError(f"Cannot find SMPL pose keys. Available keys: {sorted(source.keys())}")
        pose = assemble_pose(global_orient, body_pose)
    else:
        pose = normalize_pose_tensor(pose)
    if betas is None or transl is None:
        raise UnsupportedLabelError(f"Cannot find SMPL betas/transl keys. Available keys: {sorted(source.keys())}")
    pose = torch.as_tensor(pose, dtype=torch.float32, device=device)
    betas = normalize_betas(torch.as_tensor(betas, dtype=torch.float32, device=device), pose.shape[:2])
    transl = normalize_transl(torch.as_tensor(transl, dtype=torch.float32, device=device), pose.shape[:2])
    return {"poses": pose, "betas": betas, "transl": transl}


def assemble_pose(global_orient: Any, body_pose: Any) -> torch.Tensor:
    root = torch.as_tensor(global_orient, dtype=torch.float32)
    body = torch.as_tensor(body_pose, dtype=torch.float32)
    if root.shape[-2:] == (1, 3):
        root = root.reshape(*root.shape[:2], 1, 3)
    elif root.shape[-1] == 3:
        root = root.reshape(*root.shape[:2], 1, 3)
    else:
        raise UnsupportedLabelError(f"Unsupported SMPL root pose shape: {tuple(root.shape)}")
    body = body.reshape(*body.shape[:2], -1, 3)
    pose = torch.cat([root, body], dim=-2)
    if pose.shape[-2] < 24:
        pad = torch.zeros(*pose.shape[:-2], 24 - pose.shape[-2], 3, dtype=pose.dtype, device=pose.device)
        pose = torch.cat([pose, pad], dim=-2)
    return pose[..., :24, :].reshape(*pose.shape[:2], 72)


def normalize_pose_tensor(value: Any) -> torch.Tensor:
    pose = torch.as_tensor(value, dtype=torch.float32)
    if pose.shape[-1] == 72:
        return pose.reshape(*pose.shape[:2], 72)
    if pose.shape[-2:] == (24, 3):
        return pose.reshape(*pose.shape[:2], 72)
    if pose.shape[-1] in {63, 69}:
        joints = pose.reshape(*pose.shape[:2], -1, 3)
        pad = torch.zeros(*joints.shape[:-2], 24 - joints.shape[-2], 3, dtype=joints.dtype, device=joints.device)
        return torch.cat([joints, pad], dim=-2).reshape(*pose.shape[:2], 72)
    raise UnsupportedLabelError(f"Unsupported SMPL pose shape: {tuple(pose.shape)}")


def normalize_betas(betas: torch.Tensor, batch_frames: torch.Size) -> torch.Tensor:
    b, s = int(batch_frames[0]), int(batch_frames[1])
    if betas.ndim == 2:
        betas = betas[:, None, :].expand(b, s, -1)
    if betas.ndim == 3 and betas.shape[1] == 1:
        betas = betas.expand(b, s, -1)
    if betas.ndim != 3:
        raise UnsupportedLabelError(f"Unsupported SMPL beta shape: {tuple(betas.shape)}")
    return betas[..., :10]


def normalize_transl(transl: torch.Tensor, batch_frames: torch.Size) -> torch.Tensor:
    b, s = int(batch_frames[0]), int(batch_frames[1])
    if transl.ndim == 2:
        transl = transl[:, None, :].expand(b, s, -1)
    if transl.ndim != 3 or transl.shape[-1] != 3:
        raise UnsupportedLabelError(f"Unsupported SMPL translation shape: {tuple(transl.shape)}")
    return transl


def find_first(data: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in data:
            return data[key]
    return None


def select_prediction(predictions: dict[str, torch.Tensor], base_key: str, hsi_key: str, prefer_hsi: bool) -> torch.Tensor | None:
    if prefer_hsi and isinstance(predictions.get(hsi_key), torch.Tensor):
        return predictions[hsi_key]
    return predictions.get(base_key)


def root_align(joints: torch.Tensor) -> torch.Tensor:
    return joints - joints[:, :1, :]


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


def acceleration_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_accel = pred[2:] - 2.0 * pred[1:-1] + pred[:-2]
    target_accel = target[2:] - 2.0 * target[1:-1] + target[:-2]
    return torch.linalg.norm(pred_accel - target_accel, dim=-1).mean(dim=-1)


def box_iou_cxcywh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_xy = cxcywh_to_xyxy(a)
    b_xy = cxcywh_to_xyxy(b)
    lt = torch.maximum(a_xy[..., :2], b_xy[..., :2])
    rb = torch.minimum(a_xy[..., 2:], b_xy[..., 2:])
    wh = (rb - lt).clamp_min(0.0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a_xy[..., 2] - a_xy[..., 0]).clamp_min(0.0) * (a_xy[..., 3] - a_xy[..., 1]).clamp_min(0.0))
    area_b = ((b_xy[..., 2] - b_xy[..., 0]).clamp_min(0.0) * (b_xy[..., 3] - b_xy[..., 1]).clamp_min(0.0))
    return inter / (area_a + area_b - inter).clamp_min(1e-8)


def cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    half = box[..., 2:] * 0.5
    return torch.cat([box[..., :2] - half, box[..., :2] + half], dim=-1)


def append_rows(
    rows: list[dict[str, Any]],
    meta: dict[str, Any],
    batch_idx: int,
    query_idx: torch.Tensor,
    valid: torch.Tensor,
    mpjpe: torch.Tensor,
    cam_mpjpe: torch.Tensor,
    pve: torch.Tensor,
    pa_mpjpe: torch.Tensor,
) -> None:
    frame_indices = meta["frame_indices"][batch_idx]
    vid = meta["vid"][batch_idx]
    dataset = meta["dataset_key"][batch_idx]
    selected = torch.where(valid.detach().cpu())[0].tolist()
    for local_idx, frame_offset in enumerate(selected):
        rows.append(
            {
                "dataset": dataset,
                "vid": vid,
                "frame_index": int(frame_indices[frame_offset]),
                "query_idx": int(query_idx.detach().cpu()),
                "mpjpe_m": float(mpjpe[local_idx].detach().cpu()),
                "cam_mpjpe_m": float(cam_mpjpe[local_idx].detach().cpu()),
                "pve_m": float(pve[local_idx].detach().cpu()),
                "pa_mpjpe_m": float(pa_mpjpe[local_idx].detach().cpu()),
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
        out = {}
        for key, value in sorted(self.sums.items()):
            count = max(self.counts.get(key, 0), 1)
            out[key] = value / count
            out[f"{key}_mm"] = 1000.0 * out[key]
        out["count"] = dict(self.counts)
        return out


class UnsupportedLabelError(RuntimeError):
    pass


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def detach_meta(meta: Any) -> Any:
    if isinstance(meta, torch.Tensor):
        return meta.detach().cpu().tolist()
    if isinstance(meta, dict):
        return {key: detach_meta(value) for key, value in meta.items()}
    if isinstance(meta, list):
        return [detach_meta(value) for value in meta]
    return meta


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "vid", "frame_index", "query_idx", "mpjpe_m", "cam_mpjpe_m", "pve_m", "pa_mpjpe_m"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
