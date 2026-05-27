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

    metrics = evaluate(model, loader, device, args.conf_threshold, args.iou_threshold)
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
    parser.add_argument("--iou-threshold", type=float, default=0.50)
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
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, conf_threshold: float, iou_threshold: float) -> dict[str, float]:
    totals = {
        "num_frames": 0.0,
        "num_gt": 0.0,
        "num_pred_conf": 0.0,
        "num_true_positive": 0.0,
        "num_recalled_gt": 0.0,
        "best_gt_iou_sum": 0.0,
        "duplicate_predictions": 0.0,
    }
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        predictions = model(batch["images"])
        pred_boxes = predictions["pred_boxes"].detach().float().reshape(-1, predictions["pred_boxes"].shape[-2], 4)
        pred_confs = predictions["pred_confs"].detach().float().reshape(-1, predictions["pred_confs"].shape[-2])
        gt_boxes = batch["gt_boxes"].float().reshape(-1, batch["gt_boxes"].shape[-2], 4)
        gt_mask = (batch["smpl_mask"] & batch["boxes_mask"]).reshape(-1, batch["smpl_mask"].shape[-1])
        for frame_idx in range(pred_boxes.shape[0]):
            gt = gt_boxes[frame_idx, gt_mask[frame_idx]].clamp(0.0, 1.0)
            totals["num_frames"] += 1.0
            totals["num_gt"] += float(gt.shape[0])
            if gt.numel() == 0:
                totals["num_pred_conf"] += float((pred_confs[frame_idx] >= conf_threshold).sum().item())
                continue
            iou = pairwise_iou(cxcywh_to_xyxy(pred_boxes[frame_idx].clamp(0.0, 1.0)), cxcywh_to_xyxy(gt))
            best_gt_iou = iou.max(dim=0).values
            totals["best_gt_iou_sum"] += float(best_gt_iou.sum().item())
            totals["num_recalled_gt"] += float((best_gt_iou >= iou_threshold).sum().item())
            selected = pred_confs[frame_idx] >= conf_threshold
            totals["num_pred_conf"] += float(selected.sum().item())
            if selected.any():
                selected_iou = iou[selected]
                best_pred_iou, best_pred_gt = selected_iou.max(dim=1)
                true_positive = best_pred_iou >= iou_threshold
                totals["num_true_positive"] += float(true_positive.sum().item())
                if true_positive.any():
                    counts = torch.bincount(best_pred_gt[true_positive], minlength=gt.shape[0])
                    totals["duplicate_predictions"] += float((counts - 1).clamp(min=0).sum().item())

    num_gt = max(totals["num_gt"], 1.0)
    num_pred = max(totals["num_pred_conf"], 1.0)
    return {
        "num_frames": totals["num_frames"],
        "num_gt": totals["num_gt"],
        "num_pred_conf": totals["num_pred_conf"],
        "box_recall_iou50_top20": totals["num_recalled_gt"] / num_gt,
        "box_precision_iou50_conf030": totals["num_true_positive"] / num_pred,
        "mean_best_gt_iou": totals["best_gt_iou_sum"] / num_gt,
        "duplicate_predictions_per_gt": totals["duplicate_predictions"] / num_gt,
        "conf_threshold": float(conf_threshold),
        "iou_threshold": float(iou_threshold),
    }


def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = area1[:, None] + area2[None] - inter
    return inter / union.clamp(min=1e-6)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


if __name__ == "__main__":
    main()
