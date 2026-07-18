from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_smpl import (  # noqa: E402
    apply_overrides,
    build_loader,
    build_model,
    forward_model,
    load_initial_checkpoint,
    load_yaml_config,
    move_to_device,
)
from vggt_omega.training.config import deep_update  # noqa: E402


def main() -> None:
    args = parse_args()
    path_config = load_yaml_config(args.path_config)
    train_config = load_yaml_config(args.train_config)
    config = apply_overrides(deep_update(path_config, train_config), args.override)
    config.setdefault("model", {})["smpl_track_assignment_mode"] = "base_smpl"
    config["model"]["smpl_track_assign_id_weight"] = float(args.track_id_weight)
    config["model"]["smpl_track_assign_max_id_distance"] = float(args.max_id_distance)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).to(device)
    load_initial_checkpoint(model, config, device)
    load_partial_checkpoint(model, Path(args.checkpoint).expanduser(), device)
    model.eval()

    loader = build_loader(config, split=args.split, shuffle=False, role="val")
    metrics = evaluate(model, loader, config, device, args.max_batches)
    output_dir = Path(args.output_dir or Path(config["experiment"]["output_dir"]).parent / "nlf_id_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"[eval] wrote {output_path}", flush=True)


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader, config: dict[str, Any], device: torch.device, max_batches: int) -> dict[str, float | int]:
    total_valid = 0
    total_pairs = 0
    total_switches = 0
    pair_counts: Counter[tuple[int, int]] = Counter()
    positive_cosines: list[float] = []
    negative_cosines: list[float] = []

    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = move_to_device(batch, device)
        predictions = forward_model(model, batch, config)
        assigned_ids = predictions.get("assigned_track_ids")
        assigned_mask = predictions.get("assigned_track_mask")
        embeddings = predictions.get("pred_id_embed")
        gt_ids = batch.get("gt_track_ids", batch.get("person_ids"))
        gt_mask = batch.get("gt_track_mask", batch.get("person_id_mask"))
        boxes_mask = batch.get("boxes_mask")
        if not all(isinstance(value, torch.Tensor) for value in (assigned_ids, assigned_mask, gt_ids, gt_mask)):
            raise RuntimeError("ID evaluation requires assigned tracks and BEDLAM gt track ids")
        valid = assigned_mask.bool() & gt_mask.to(device=device).bool()
        if isinstance(boxes_mask, torch.Tensor):
            valid = valid & boxes_mask.to(device=device).bool()
        total_valid += int(valid.sum().item())

        for b in range(valid.shape[0]):
            previous_by_gt: dict[int, tuple[int, int]] = {}
            for s in range(valid.shape[1]):
                for q in range(valid.shape[2]):
                    if not bool(valid[b, s, q]):
                        continue
                    gt_id = int(gt_ids[b, s, q].item())
                    pred_id = int(assigned_ids[b, s, q].item())
                    pair_counts[(pred_id, gt_id)] += 1
                    previous = previous_by_gt.get(gt_id)
                    if previous is not None:
                        total_pairs += 1
                        total_switches += int(previous[1] != pred_id)
                    previous_by_gt[gt_id] = (s, pred_id)

        if isinstance(embeddings, torch.Tensor):
            emb = torch.nn.functional.normalize(embeddings.float(), dim=-1)
            for b in range(valid.shape[0]):
                items = [(s, q, int(gt_ids[b, s, q].item())) for s, q in valid[b].nonzero(as_tuple=False).tolist()]
                for left_idx, (s0, q0, gt0) in enumerate(items):
                    for s1, q1, gt1 in items[left_idx + 1 :]:
                        if s0 == s1:
                            continue
                        cosine = float(torch.dot(emb[b, s0, q0], emb[b, s1, q1]).item())
                        (positive_cosines if gt0 == gt1 else negative_cosines).append(cosine)

    total_association = sum(pair_counts.values())
    correct_association = 0
    by_pred: defaultdict[int, Counter[int]] = defaultdict(Counter)
    for (pred_id, gt_id), count in pair_counts.items():
        by_pred[pred_id][gt_id] += count
    for counts in by_pred.values():
        correct_association += max(counts.values())

    return {
        "num_valid_assignments": total_valid,
        "num_temporal_pairs": total_pairs,
        "num_id_switches": total_switches,
        "id_switch_rate": total_switches / max(total_pairs, 1),
        "majority_association_accuracy": correct_association / max(total_association, 1),
        "positive_embedding_cosine": _mean(positive_cosines),
        "negative_embedding_cosine": _mean(negative_cosines),
        "embedding_margin": _mean(positive_cosines) - _mean(negative_cosines),
    }


def load_partial_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint does not contain a state_dict: {path}")
    missing, unexpected = model.load_state_dict(
        {key.removeprefix("module."): value for key, value in state_dict.items()}, strict=False
    )
    print(f"[ckpt] loaded ID checkpoint {path} missing={len(missing)} unexpected={len(unexpected)}", flush=True)


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NLF SMPL identity tracking on BEDLAM clips")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_nlf_id_tracking.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="Training")
    parser.add_argument("--device", default="")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--track-id-weight", type=float, default=0.35)
    parser.add_argument("--max-id-distance", type=float, default=0.70)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    main()
