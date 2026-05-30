import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_smpl import apply_overrides, build_loader, build_model, load_initial_checkpoint, load_yaml_config
from vggt_omega.training.config import deep_update
from vggt_omega.training.smpl_matcher import cxcywh_to_xyxy


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args)
    loader = build_eval_loader(config, args)
    model = build_model(config).to(device)
    load_initial_checkpoint(model, config, device)
    load_training_checkpoint(model, Path(args.checkpoint).expanduser(), device)
    model.eval()

    conf_thresholds = sorted({float(args.conf_threshold), *[float(value) for value in args.conf_thresholds]})
    metrics = evaluate(
        model,
        loader,
        device,
        conf_thresholds,
        args.conf_threshold,
        args.iou_threshold,
        args.eval_nms,
        args.nms_iou_threshold,
        args.use_gt_box_prior,
    )
    output_path = output_dir / "smpl_box_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(output_path), **metrics}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SMPL query box/confidence metrics on a fixed BEDLAM subset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl.yaml")
    parser.add_argument("--output-dir", default="outputs/eval/smpl_train_subset")
    parser.add_argument("--device", default="")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--conf-threshold", type=float, default=0.30)
    parser.add_argument("--conf-thresholds", type=float, nargs="+", default=[0.25, 0.30, 0.40, 0.50])
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--eval-nms", action="store_true")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.50)
    parser.add_argument("--use-gt-box-prior", action="store_true", help="Pass dataset GT boxes as oracle SMPL query priors")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.train_config))
    config = apply_overrides(config, args.override)
    if args.baseline_checkpoint:
        config.setdefault("checkpoints", {})["vggt_baseline"] = args.baseline_checkpoint
    config.setdefault("data", {})["val_split"] = ""
    config.setdefault("optim", {})["batch_size"] = 1
    config.setdefault("model", {})["enable_smpl"] = True
    return config


def build_eval_loader(config: dict[str, Any], args: argparse.Namespace) -> DataLoader:
    dataset_loader = build_loader(config, split=config["data"]["train_split"], shuffle=False)
    dataset = dataset_loader.dataset
    if args.max_samples > 0 and len(dataset) > args.max_samples:
        indices = torch.linspace(0, len(dataset) - 1, args.max_samples).round().long().unique().tolist()
        dataset = Subset(dataset, indices)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(config.get("data", {}).get("num_workers", 0)),
        pin_memory=bool(config.get("data", {}).get("pin_memory", True)),
        collate_fn=dataset_loader.collate_fn,
    )


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


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    conf_thresholds: list[float],
    primary_conf_threshold: float,
    iou_threshold: float,
    eval_nms: bool,
    nms_iou_threshold: float,
    use_gt_box_prior: bool,
) -> dict[str, float]:
    conf_thresholds = sorted({float(value) for value in conf_thresholds})
    totals = {
        "num_frames": 0.0,
        "num_gt": 0.0,
        "best_gt_iou_sum": 0.0,
        "recall_topk": {3: 0.0, 5: 0.0, 10: 0.0, 20: 0.0},
        "thresholds": {threshold: _empty_threshold_counts() for threshold in conf_thresholds},
        "nms_thresholds": {threshold: _empty_threshold_counts() for threshold in conf_thresholds},
    }
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        predictions = _forward_model(model, batch, use_gt_box_prior)
        pred_boxes = predictions["pred_boxes"].detach().float().reshape(-1, predictions["pred_boxes"].shape[-2], 4)
        pred_confs = predictions["pred_confs"].detach().float().reshape(-1, predictions["pred_confs"].shape[-2])
        gt_boxes = batch["gt_boxes"].float().reshape(-1, batch["gt_boxes"].shape[-2], 4)
        gt_mask = (batch["smpl_mask"] & batch["boxes_mask"]).reshape(-1, batch["smpl_mask"].shape[-1])
        for frame_idx in range(pred_boxes.shape[0]):
            gt = gt_boxes[frame_idx, gt_mask[frame_idx]].clamp(0.0, 1.0)
            totals["num_frames"] += 1.0
            totals["num_gt"] += float(gt.shape[0])
            for threshold in conf_thresholds:
                selected = pred_confs[frame_idx] >= threshold
                if gt.numel() == 0:
                    totals["thresholds"][threshold]["num_pred"] += float(selected.sum().item())
                    if eval_nms:
                        nms_selected = _nms_keep_mask(pred_boxes[frame_idx], pred_confs[frame_idx], threshold, nms_iou_threshold)
                        totals["nms_thresholds"][threshold]["num_pred"] += float(nms_selected.sum().item())
                    continue
            if gt.numel() == 0:
                continue
            iou = pairwise_iou(cxcywh_to_xyxy(pred_boxes[frame_idx].clamp(0.0, 1.0)), cxcywh_to_xyxy(gt))
            best_gt_iou = iou.max(dim=0).values
            totals["best_gt_iou_sum"] += float(best_gt_iou.sum().item())
            for top_k in totals["recall_topk"]:
                topk_best = _best_gt_iou_from_topk(pred_boxes[frame_idx], pred_confs[frame_idx], gt, top_k)
                totals["recall_topk"][top_k] += float((topk_best >= iou_threshold).sum().item())
            for threshold in conf_thresholds:
                selected = pred_confs[frame_idx] >= threshold
                _add_counts(totals["thresholds"][threshold], _count_threshold_metrics(iou, selected, iou_threshold))
                if eval_nms:
                    nms_selected = _nms_keep_mask(pred_boxes[frame_idx], pred_confs[frame_idx], threshold, nms_iou_threshold)
                    _add_counts(totals["nms_thresholds"][threshold], _count_threshold_metrics(iou, nms_selected, iou_threshold))

    num_gt = max(totals["num_gt"], 1.0)
    metrics: dict[str, float] = {
        "num_frames": totals["num_frames"],
        "num_gt": totals["num_gt"],
        "mean_best_gt_iou": totals["best_gt_iou_sum"] / num_gt,
        "conf_threshold": float(primary_conf_threshold),
        "iou_threshold": float(iou_threshold),
        "use_gt_box_prior": bool(use_gt_box_prior),
    }
    for top_k, value in totals["recall_topk"].items():
        metrics[f"box_recall_iou50_top{top_k}"] = value / num_gt
    for threshold, counts in totals["thresholds"].items():
        _write_threshold_metrics(metrics, counts, threshold, num_gt, prefix="")
    primary_counts = totals["thresholds"].get(primary_conf_threshold, _closest_threshold_entry(totals["thresholds"], primary_conf_threshold))
    metrics["num_pred_conf"] = primary_counts["num_pred"]
    metrics["duplicate_predictions_per_gt"] = primary_counts["duplicate_predictions"] / num_gt
    legacy_counts = totals["thresholds"].get(0.30, _closest_threshold_entry(totals["thresholds"], 0.30))
    metrics["box_precision_iou50_conf030"] = legacy_counts["true_positive"] / max(legacy_counts["num_pred"], 1.0)
    if eval_nms:
        for threshold, counts in totals["nms_thresholds"].items():
            _write_threshold_metrics(metrics, counts, threshold, num_gt, prefix="nms_")
    return metrics


def _forward_model(model: torch.nn.Module, batch: dict[str, torch.Tensor], use_gt_box_prior: bool) -> dict[str, torch.Tensor]:
    if not use_gt_box_prior:
        return model(batch["images"])
    return model(
        batch["images"],
        smpl_query_boxes=batch["gt_boxes"],
        smpl_query_boxes_mask=batch["boxes_mask"],
    )


def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = area1[:, None] + area2[None] - inter
    return inter / union.clamp(min=1e-6)


def _empty_threshold_counts() -> dict[str, float]:
    return {"num_pred": 0.0, "true_positive": 0.0, "duplicate_predictions": 0.0}


def _add_counts(total: dict[str, float], increment: dict[str, float]) -> None:
    for key, value in increment.items():
        total[key] += value


def _count_threshold_metrics(iou: torch.Tensor, selected: torch.Tensor, iou_threshold: float) -> dict[str, float]:
    counts = _empty_threshold_counts()
    counts["num_pred"] = float(selected.sum().item())
    if not selected.any():
        return counts
    selected_iou = iou[selected]
    best_pred_iou, best_pred_gt = selected_iou.max(dim=1)
    true_positive = best_pred_iou >= iou_threshold
    counts["true_positive"] = float(true_positive.sum().item())
    if true_positive.any():
        gt_counts = torch.bincount(best_pred_gt[true_positive], minlength=iou.shape[1])
        counts["duplicate_predictions"] = float((gt_counts - 1).clamp(min=0).sum().item())
    return counts


def _best_gt_iou_from_topk(pred_boxes: torch.Tensor, pred_confs: torch.Tensor, gt: torch.Tensor, top_k: int) -> torch.Tensor:
    order = pred_confs.argsort(descending=True)
    keep = order[: min(top_k, order.numel())]
    if keep.numel() == 0:
        return pred_boxes.new_zeros((gt.shape[0],))
    iou = pairwise_iou(cxcywh_to_xyxy(pred_boxes[keep].clamp(0.0, 1.0)), cxcywh_to_xyxy(gt))
    return iou.max(dim=0).values


def _nms_keep_mask(pred_boxes: torch.Tensor, pred_confs: torch.Tensor, threshold: float, nms_iou_threshold: float) -> torch.Tensor:
    selected = pred_confs >= threshold
    if not selected.any():
        return selected
    selected_indices = selected.nonzero(as_tuple=False).squeeze(1)
    boxes = pred_boxes[selected_indices].clamp(0.0, 1.0)
    scores = pred_confs[selected_indices]
    order = scores.argsort(descending=True)
    kept_local = []
    while order.numel() > 0:
        current = order[0]
        kept_local.append(current)
        if order.numel() == 1:
            break
        rest = order[1:]
        current_box = boxes[current : current + 1]
        iou = pairwise_iou(cxcywh_to_xyxy(current_box), cxcywh_to_xyxy(boxes[rest]))[0]
        order = rest[iou <= nms_iou_threshold]
    keep = torch.zeros_like(selected)
    keep[selected_indices[torch.stack(kept_local)]] = True
    return keep


def _write_threshold_metrics(metrics: dict[str, float], counts: dict[str, float], threshold: float, num_gt: float, prefix: str) -> None:
    key = _format_threshold_key(threshold)
    metrics[f"{prefix}num_pred_conf{key}"] = counts["num_pred"]
    metrics[f"{prefix}box_precision_iou50_conf{key}"] = counts["true_positive"] / max(counts["num_pred"], 1.0)
    metrics[f"{prefix}duplicate_predictions_per_gt_conf{key}"] = counts["duplicate_predictions"] / num_gt


def _format_threshold_key(threshold: float) -> str:
    return f"{int(round(float(threshold) * 100)):03d}"


def _closest_threshold_entry(entries: dict[float, dict[str, float]], threshold: float) -> dict[str, float]:
    if not entries:
        return _empty_threshold_counts()
    key = min(entries, key=lambda value: abs(value - threshold))
    return entries[key]


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


if __name__ == "__main__":
    main()
