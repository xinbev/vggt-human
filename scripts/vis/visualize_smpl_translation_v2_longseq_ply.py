#!/usr/bin/env python
"""Export SMPL Translation V2 long-sequence PLY diagnostics for one target frame."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Compatibility patch for old chumpy on Python 3.11+.
import inspect
from collections import Counter, namedtuple

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

from scripts.train.train_smpl import build_model  # noqa: E402
from scripts.vis.visualize_hsi_clip_scene_affine import load_clip_images  # noqa: E402
from scripts.vis.visualize_smpl_inference import (  # noqa: E402
    add_projected_gt_smpl_joints,
    add_projected_smpl_joints,
    collect_predictions,
    draw_predictions,
    load_config,
    load_gt_smpl_for_image,
    load_training_checkpoint,
    load_vggt_baseline_for_camera,
    require_smpl_model_dir,
    write_ply_meshes,
)
from vggt_omega.data.bedlam import _build_box_targets, _load_box_persons, _load_persons  # noqa: E402
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.training.config import require_path  # noqa: E402
from vggt_omega.utils.rotation import rot6d_to_axis_angle  # noqa: E402


STAGE_COLORS = {
    "base": (255, 160, 32),
    "seed": (64, 192, 255),
    "refined": (255, 64, 64),
    "gt": (64, 255, 128),
}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args)
    config.setdefault("model", {})["enable_camera"] = True
    config.setdefault("model", {})["enable_depth"] = bool(args.export_scene_ply)
    config.setdefault("model", {})["enable_hsi_refine"] = False
    config.setdefault("data", {})["require_boxes"] = True
    config.setdefault("data", {})["require_smpl"] = True
    config.setdefault("data", {})["require_depth"] = False

    target_image = Path(args.image).expanduser()
    frame_paths, target_frame_idx = resolve_clip_frames(target_image, args)
    input_size = int(config["data"].get("image_size", args.image_size))
    image_tensor, orig_images = load_clip_images(frame_paths, input_size)
    priors = load_clip_priors_and_tracks(frame_paths, config, device)

    model = build_model(config).to(device)
    load_vggt_baseline_for_camera(model, config, device)
    load_training_checkpoint(model, Path(args.checkpoint).expanduser(), device)
    model.eval()

    with torch.no_grad():
        predictions = model(
            image_tensor.to(device),
            smpl_query_boxes=priors["gt_boxes"],
            smpl_query_boxes_mask=priors["boxes_mask"],
            smpl_track_ids=priors["gt_track_ids"],
            smpl_track_mask=priors["gt_track_mask"],
        )

    frame_pred = slice_predictions(predictions, target_frame_idx)
    selected = select_pairs_for_target(target_image, target_frame_idx, priors, frame_pred, args)
    if not selected:
        raise RuntimeError(f"No query/GT pairs selected for {target_image}")

    overlay_files = export_overlay(target_image, frame_pred, config, args, orig_images[target_frame_idx], input_size, output_dir, device)
    ply_files, records = export_stage_plys(selected, target_image, frame_pred, config, args, output_dir, device)

    summary = {
        "target_image": str(target_image),
        "target_frame_idx": int(target_frame_idx),
        "checkpoint": str(args.checkpoint),
        "num_frames": len(frame_paths),
        "frame_paths": [str(path) for path in frame_paths],
        "selected_pairs": selected,
        "records": records,
        "overlay_files": overlay_files,
        "ply_files": ply_files,
        "color_legend": {
            "base": "orange: legacy/base_pred_transl_cam",
            "seed": "cyan: ray/depth seed_pred_transl_cam",
            "refined": "red: final temporal pred_transl_cam",
            "gt": "green: BEDLAM GT SMPL/transl",
        },
    }
    out_json = output_dir / f"{target_image.stem}_translation_v2_ply_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(out_json), "ply_files": ply_files, "records": records}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Translation V2 long-sequence SMPL PLY diagnostics")
    parser.add_argument("--image", required=True, help="Target BEDLAM RGB frame to inspect")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_translation_v2_longseq.yaml")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--output-dir", default="outputs/vis/smpl_translation_v2_longseq_ply")
    parser.add_argument("--device", default="")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--num-frames", type=int, default=27)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--target-frame-offset", type=int, default=4, help="Place --image at this 0-based clip offset")
    parser.add_argument("--start-image", default="", help="Optional explicit clip start image; overrides --target-frame-offset")
    parser.add_argument("--person-csv", default="", help="Optional eval person CSV used to choose bad query/GT pairs")
    parser.add_argument("--max-people", type=int, default=3)
    parser.add_argument("--query-indices", default="", help="Comma-separated query indices; overrides --person-csv")
    parser.add_argument("--gt-indices", default="", help="Comma-separated GT indices paired with --query-indices")
    parser.add_argument("--conf-threshold", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--draw-smpl-joints", action="store_true")
    parser.add_argument("--draw-gt-smpl-joints", action="store_true")
    parser.add_argument("--smpl-model-dir", default="")
    parser.add_argument("--export-scene-ply", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def resolve_clip_frames(target_image: Path, args: argparse.Namespace) -> tuple[list[Path], int]:
    if not target_image.is_file():
        raise FileNotFoundError(f"Target image not found: {target_image}")
    if target_image.parent.name != "rgb":
        raise ValueError(f"--image must point to a BEDLAM rgb frame, got: {target_image}")
    frames = sorted(path.resolve() for path in target_image.parent.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    target_resolved = target_image.resolve()
    try:
        target_index = frames.index(target_resolved)
    except ValueError as exc:
        raise ValueError(f"Target image not found in its rgb directory listing: {target_image}") from exc

    stride = max(int(args.stride), 1)
    if str(args.start_image).strip():
        start_image = Path(args.start_image).expanduser().resolve()
        try:
            start_index = frames.index(start_image)
        except ValueError as exc:
            raise ValueError(f"--start-image not found in target sequence: {start_image}") from exc
        selected = take_frames(frames, start_index, int(args.num_frames), stride)
        target_frame_idx = next((idx for idx, path in enumerate(selected) if path == target_resolved), -1)
        if target_frame_idx < 0:
            raise ValueError(f"Target image {target_image} is not inside clip starting at {start_image}")
        return selected, target_frame_idx

    target_frame_idx = int(args.target_frame_offset)
    start_index = target_index - target_frame_idx * stride
    if start_index < 0:
        raise ValueError(f"target-frame-offset={target_frame_idx} would start before sequence beginning")
    selected = take_frames(frames, start_index, int(args.num_frames), stride)
    if target_frame_idx >= len(selected) or selected[target_frame_idx] != target_resolved:
        raise RuntimeError("Resolved clip does not contain target image at requested offset")
    return selected, target_frame_idx


def take_frames(frames: list[Path], start_index: int, num_frames: int, stride: int) -> list[Path]:
    selected = []
    for offset in range(num_frames):
        idx = start_index + offset * stride
        if idx >= len(frames):
            break
        selected.append(frames[idx])
    if not selected:
        raise RuntimeError("No frames selected")
    return selected


def load_clip_priors_and_tracks(frame_paths: list[Path], config: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    data_cfg = config.get("data", {})
    max_humans = int(data_cfg.get("max_humans", config.get("model", {}).get("num_smpl_queries", 20)))
    dataset_root = Path(require_path(config, data_cfg.get("root_key", "datasets.bedlam_root"), allow_empty=False)).expanduser()
    boxes_root = Path(require_path(config, data_cfg.get("boxes_root_key", "datasets.bedlam_boxes_root"), allow_empty=False)).expanduser()
    split = str(data_cfg.get("train_split", "Training"))

    persons_per_frame = []
    boxes_per_frame = []
    for image_path in frame_paths:
        seq_dir = image_path.parent.parent
        frame_id = image_path.stem
        sequence_rel = seq_dir.relative_to(dataset_root / split)
        persons = _load_persons(seq_dir / "smpl" / f"{frame_id}.pkl", require_smpl=True)
        box_persons = _load_box_persons(boxes_root / split / sequence_rel / "smpl_boxes" / f"{frame_id}.pkl", require_boxes=True)
        persons_per_frame.append(persons)
        boxes_per_frame.append(box_persons)
    boxes = _build_box_targets(boxes_per_frame, persons_per_frame, max_humans, require_boxes=True)
    return {
        "gt_boxes": boxes["boxes"].unsqueeze(0).to(device),
        "boxes_mask": boxes["boxes_mask"].unsqueeze(0).to(device),
        "gt_track_ids": boxes["gt_track_ids"].unsqueeze(0).to(device),
        "gt_track_mask": boxes["gt_track_mask"].unsqueeze(0).to(device),
    }


def slice_predictions(predictions: dict[str, torch.Tensor], frame_idx: int) -> dict[str, torch.Tensor]:
    sliced = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[0] == 1 and value.shape[1] > frame_idx:
            sliced[key] = value[:, frame_idx : frame_idx + 1].contiguous()
        else:
            sliced[key] = value
    return sliced


def select_pairs_for_target(
    target_image: Path,
    target_frame_idx: int,
    priors: dict[str, torch.Tensor],
    frame_pred: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    query_override = parse_int_list(args.query_indices)
    gt_override = parse_int_list(args.gt_indices)
    if query_override:
        if gt_override and len(gt_override) != len(query_override):
            raise ValueError("--gt-indices must have the same length as --query-indices")
        if not gt_override:
            gt_override = query_override
        return [
            {"query_idx": int(q), "gt_idx": int(g), "source": "manual", "rank": idx}
            for idx, (q, g) in enumerate(zip(query_override, gt_override, strict=True))
        ][: max(int(args.max_people), 1)]

    csv_path = Path(args.person_csv).expanduser() if str(args.person_csv).strip() else None
    if csv_path is not None and csv_path.is_file():
        rows = read_person_csv_rows(csv_path, target_image, target_frame_idx)
        if rows:
            rows.sort(key=lambda row: safe_float(row.get("refined_transl_l2_m")), reverse=True)
            selected = []
            seen_gt = set()
            for row in rows:
                gt_idx = safe_int(row.get("gt_idx"))
                query_idx = safe_int(row.get("query_idx"))
                if gt_idx is None or query_idx is None or gt_idx in seen_gt:
                    continue
                seen_gt.add(gt_idx)
                selected.append(
                    {
                        "query_idx": query_idx,
                        "gt_idx": gt_idx,
                        "source": "person_csv",
                        "rank": len(selected),
                        "csv_refined_l2_m": safe_float(row.get("refined_transl_l2_m")),
                        "csv_base_l2_m": safe_float(row.get("base_transl_l2_m")),
                        "csv_seed_l2_m": safe_float(row.get("seed_transl_l2_m")),
                        "csv_track_id": row.get("track_id", ""),
                    }
                )
                if len(selected) >= int(args.max_people):
                    break
            if selected:
                return selected

    valid = priors["boxes_mask"][0, target_frame_idx].detach().bool().cpu()
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten().tolist()
    conf = frame_pred.get("pred_confs")
    if isinstance(conf, torch.Tensor):
        conf_values = conf[0, 0, valid_indices, 0].detach().float().cpu().tolist() if valid_indices else []
        valid_indices = [idx for _, idx in sorted(zip(conf_values, valid_indices, strict=True), reverse=True)]
    return [{"query_idx": int(idx), "gt_idx": int(idx), "source": "slot_fallback", "rank": rank} for rank, idx in enumerate(valid_indices[: int(args.max_people)])]


def read_person_csv_rows(csv_path: Path, target_image: Path, target_frame_idx: int) -> list[dict[str, str]]:
    sequence_name = target_image.parent.parent.name
    frame_name = target_image.stem
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if row.get("sequence_name") != sequence_name or row.get("frame_name") != frame_name:
                continue
            if safe_int(row.get("frame_idx")) != int(target_frame_idx):
                continue
            rows.append(row)
    return rows


def export_overlay(
    target_image: Path,
    frame_pred: dict[str, torch.Tensor],
    config: dict[str, Any],
    args: argparse.Namespace,
    orig_image,
    input_size: int,
    output_dir: Path,
    device: torch.device,
) -> list[str]:
    results = collect_predictions(frame_pred, orig_image.size, args.conf_threshold, args.top_k)
    if args.draw_smpl_joints:
        add_projected_smpl_joints(results, frame_pred, config, args, orig_image.size, input_size, device)
    if args.draw_gt_smpl_joints:
        add_projected_gt_smpl_joints(results, target_image, frame_pred, config, args, orig_image.size, input_size, device)
    out_image = output_dir / f"{target_image.stem}_translation_v2_overlay.jpg"
    draw_predictions(orig_image, results, out_image)
    return [str(out_image)]


def export_stage_plys(
    selected: list[dict[str, Any]],
    target_image: Path,
    frame_pred: dict[str, torch.Tensor],
    config: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> tuple[list[str], list[dict[str, Any]]]:
    smpl_model_dir = require_smpl_model_dir(config, args)
    smpl = SMPLLayer(smpl_model_dir).to(device).eval()
    faces = np.asarray(smpl.faces, dtype=np.int64)
    gt = load_gt_smpl_for_image(target_image, config, args, device)
    query_indices = torch.as_tensor([int(item["query_idx"]) for item in selected], dtype=torch.long, device=device)
    gt_indices = torch.as_tensor([int(item["gt_idx"]) for item in selected], dtype=torch.long, device=device)

    for key in ("pred_poses", "pred_betas", "pred_transl_cam"):
        if key not in frame_pred:
            raise ValueError(f"Missing model output for PLY export: {key}")
    pred_poses = frame_pred["pred_poses"][0, 0, query_indices].detach()
    pred_betas = frame_pred["pred_betas"][0, 0, query_indices].detach()
    with torch.no_grad():
        pred_local_vertices, _ = smpl(pred_poses.reshape(-1, 72), pred_betas)
        gt_poses = rot6d_to_axis_angle(gt["poses_6d"][gt_indices]).reshape(-1, 72)
        gt_vertices, _ = smpl(gt_poses, gt["betas"][gt_indices])

    translations = {
        "base": tensor_stage(frame_pred.get("base_pred_transl_cam", frame_pred["pred_transl_cam"]), query_indices),
        "seed": tensor_stage(frame_pred.get("seed_pred_transl_cam", frame_pred["pred_transl_cam"]), query_indices),
        "refined": tensor_stage(frame_pred["pred_transl_cam"], query_indices),
        "gt": gt["transl_cam"][gt_indices].detach(),
    }
    gt_meshes = gt_vertices + translations["gt"][:, None, :]

    written: list[str] = []
    records: list[dict[str, Any]] = []
    all_stage_meshes: list[np.ndarray] = []
    all_stage_colors: list[tuple[int, int, int]] = []
    all_translation_only_meshes: list[np.ndarray] = []
    all_translation_only_colors: list[tuple[int, int, int]] = []

    for local_idx, item in enumerate(selected):
        stage_meshes = [
            (pred_local_vertices[local_idx] + translations["base"][local_idx][None, :]).detach().cpu().numpy(),
            (pred_local_vertices[local_idx] + translations["seed"][local_idx][None, :]).detach().cpu().numpy(),
            (pred_local_vertices[local_idx] + translations["refined"][local_idx][None, :]).detach().cpu().numpy(),
            gt_meshes[local_idx].detach().cpu().numpy(),
        ]
        stage_colors = [STAGE_COLORS[name] for name in ("base", "seed", "refined", "gt")]
        translation_only_meshes = [
            (pred_local_vertices[local_idx] + translations["base"][local_idx][None, :]).detach().cpu().numpy(),
            (pred_local_vertices[local_idx] + translations["seed"][local_idx][None, :]).detach().cpu().numpy(),
            (pred_local_vertices[local_idx] + translations["refined"][local_idx][None, :]).detach().cpu().numpy(),
            (pred_local_vertices[local_idx] + translations["gt"][local_idx][None, :]).detach().cpu().numpy(),
        ]

        query_idx = int(item["query_idx"])
        gt_idx = int(item["gt_idx"])
        prefix = f"{target_image.stem}_rank{int(item.get('rank', local_idx)):02d}_q{query_idx:02d}_gt{gt_idx:02d}"
        stage_path = output_dir / f"{prefix}_base_seed_refined_gt_mesh.ply"
        trans_path = output_dir / f"{prefix}_translation_only_compare.ply"
        write_ply_meshes(stage_path, stage_meshes, faces, stage_colors)
        write_ply_meshes(trans_path, translation_only_meshes, faces, stage_colors)
        written.extend([str(stage_path), str(trans_path)])

        all_stage_meshes.extend(stage_meshes)
        all_stage_colors.extend(stage_colors)
        all_translation_only_meshes.extend(translation_only_meshes)
        all_translation_only_colors.extend(stage_colors)
        record = build_record(item, translations, local_idx)
        records.append(record)

    combined_stage = output_dir / f"{target_image.stem}_all_selected_base_seed_refined_gt_mesh.ply"
    combined_trans = output_dir / f"{target_image.stem}_all_selected_translation_only_compare.ply"
    write_ply_meshes(combined_stage, all_stage_meshes, faces, all_stage_colors)
    write_ply_meshes(combined_trans, all_translation_only_meshes, faces, all_translation_only_colors)
    written.extend([str(combined_stage), str(combined_trans)])
    return written, records


def tensor_stage(value: torch.Tensor, query_indices: torch.Tensor) -> torch.Tensor:
    return value[0, 0, query_indices].detach()


def build_record(item: dict[str, Any], translations: dict[str, torch.Tensor], local_idx: int) -> dict[str, Any]:
    gt_vec = translations["gt"][local_idx].detach().float().cpu()
    out = dict(item)
    out["translations"] = {}
    out["errors_l2_m"] = {}
    for name, tensor in translations.items():
        vec = tensor[local_idx].detach().float().cpu()
        out["translations"][name] = [float(v) for v in vec.tolist()]
        out["errors_l2_m"][name] = float(torch.linalg.norm(vec - gt_vec).item())
    out["refined_minus_seed_l2_m"] = float(torch.linalg.norm(translations["refined"][local_idx] - translations["seed"][local_idx]).detach().cpu())
    out["refined_minus_base_l2_m"] = float(torch.linalg.norm(translations["refined"][local_idx] - translations["base"][local_idx]).detach().cpu())
    return out


def parse_int_list(value: str) -> list[int]:
    out = []
    for item in str(value or "").split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def safe_int(value: Any) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return -math.inf


if __name__ == "__main__":
    main()
