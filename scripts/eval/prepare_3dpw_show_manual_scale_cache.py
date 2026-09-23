#!/usr/bin/env python3
"""Run the 3DPW SHOW cascade once and persist replayable 200-frame windows.

The cache deliberately stores the selected prediction mesh and metric depth,
not model activations.  Manual calibration therefore never runs inference a
second time: a window scale is applied to the cached scene depth only, which
matches the project's existing RICH manual-scale semantics.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.evaluate_hmr4d_smpl_metrics import extract_gt_smpl, move_to_device  # noqa: E402
from scripts.eval.evaluate_show_human_scene_hmr4d import (  # noqa: E402
    apply_transform,
    build_cascade_config,
    build_dataset,
    choose_frame_query_indices,
    decode_gendered_gt_vertices,
    load_config,
)
from scripts.eval.evaluate_show_human_scene_3dpw import canonical_batch_depth, resolve_output_dir, resolve_branch, select_branch_vertices, validate_evaluation_config  # noqa: E402
from scripts.train.train_smpl import build_model  # noqa: E402
from vggt_omega.data import hmr4d_eval_collate_fn  # noqa: E402
from vggt_omega.evaluation.rich_physical_grounding import (  # noqa: E402
    configure_reference_cascade_model,
    load_evaluation_weights,
    run_metric_cascade,
)
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.training.config import require_path  # noqa: E402


CACHE_FORMAT = "vggt_omega_show_3dpw_manual_scale_cache_v1"
MANIFEST_NAME = "manifest.json"
FACES_NAME = "smpl_faces.npy"
INCOMPLETE_NAME = ".incomplete"


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if int(args.batch_size) != 1:
        raise ValueError("SHOW manual-scale cache requires --batch-size=1")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    config = load_config(args)
    branches, mask_source, metric_config = validate_evaluation_config(config["human_scene_evaluation"])
    if mask_source != "gt_smpl_projection":
        raise ValueError("The cache currently stores GT-SMPL projection inputs only; use mask_source=gt_smpl_projection")
    if "refined" not in branches:
        raise ValueError("The manual-scale cache is intended for the refined SHOW branch")
    cascade_config = build_cascade_config(config)
    config, checkpoint_report = configure_reference_cascade_model(
        config, args.checkpoint, max_humans=int(args.max_humans)
    )
    output_dir = resolve_output_dir(args.output_dir)
    cache_dir = output_dir / "cache"
    windows_dir = cache_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / INCOMPLETE_NAME).write_text("Cache export is in progress.\n", encoding="utf-8")

    model = build_model(config).to(device)
    checkpoint_audit = load_evaluation_weights(
        model=model,
        baseline_checkpoint=require_path(config, "checkpoints.vggt_baseline"),
        stage2_checkpoint=Path(args.checkpoint).expanduser(),
        scale_checkpoint=Path(args.scale_checkpoint).expanduser(),
        device=device,
    )
    model.eval()
    smpl_dir = require_path(config, "assets.smpl_model_dir")
    smpl = SMPLLayer(smpl_dir, gender="neutral").to(device).eval()
    gt_smpl_by_gender = {gender: SMPLLayer(smpl_dir, gender=gender).to(device).eval() for gender in ("neutral", "male", "female")}
    faces = np.asarray(smpl.faces, dtype=np.int32).reshape(-1, 3)
    np.save(cache_dir / FACES_NAME, faces, allow_pickle=False)

    dataset = build_dataset(config, args)
    indices = list(range(len(dataset)))
    if int(args.max_windows) > 0:
        indices = indices[: int(args.max_windows)]
    loader = DataLoader(
        Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=int(args.num_workers),
        pin_memory=True, collate_fn=hmr4d_eval_collate_fn, drop_last=False,
    )
    windows: list[dict[str, Any]] = []
    for position, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        predictions, _ = run_metric_cascade(model, smpl, batch["images"], cascade_config)
        frames = build_cached_frames(predictions, batch, smpl, gt_smpl_by_gender, metric_config, dataset)
        if not frames:
            continue
        meta = batch["meta"]
        vid = str(meta["vid"][0])
        start = int(meta["start"][0])
        frame_ids = [int(value) for value in meta["frame_indices"][0]]
        window_id = f"3dpw_{safe_name(vid)}_window_{position:04d}"
        frame_file = windows_dir / f"{window_id}.pkl"
        with frame_file.open("wb") as file:
            pickle.dump(frames, file, protocol=pickle.HIGHEST_PROTOCOL)
        windows.append({
            "window_id": window_id, "vid": vid, "window_index": len(windows),
            "dataset_index": int(indices[position]), "start_frame": start,
            "frame_count": len(frames), "source_frame_ids": frame_ids[: len(frames)],
            "cache_file": str(frame_file.relative_to(cache_dir)),
        })
        print(f"[show-cache] {len(windows)} windows={len(windows)} vid={vid} start={start} frames={len(frames)}", flush=True)

    manifest = {
        "format": CACHE_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "3dpw", "split": args.split, "window_size": int(args.sequence_length or config.get("data", {}).get("sequence_length", 200)),
        "branch": "refined", "mask_source": mask_source,
        "checkpoint": str(args.checkpoint), "scale_checkpoint": str(args.scale_checkpoint),
        "checkpoint_model_config": checkpoint_report, "checkpoint_load": checkpoint_audit,
        "faces_file": FACES_NAME, "metric_config": asdict(metric_config),
        "cascade": asdict(cascade_config), "windows": windows,
        "scale_scope": "one shared manual multiplier per non-overlapping window",
    }
    manifest_path = cache_dir / MANIFEST_NAME
    temporary = cache_dir / f"{MANIFEST_NAME}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(manifest_path)
    (cache_dir / INCOMPLETE_NAME).unlink(missing_ok=True)
    print(json.dumps({"manifest": str(manifest_path), "windows": len(windows)}, indent=2), flush=True)


def build_cached_frames(predictions: dict[str, Any], batch: dict[str, Any], smpl: SMPLLayer, gt_smpl_by_gender: dict[str, SMPLLayer], metric_config: Any, dataset: Any) -> list[dict[str, Any]]:
    gt = extract_gt_smpl(batch["eval_label"], predictions["pred_confs"].device)
    gt_world = decode_gendered_gt_vertices(gt, batch["eval_label"], gt_smpl_by_gender) + gt["transl"][..., None, :]
    transforms = torch.as_tensor(batch["eval_label"]["T_w2c"], device=gt_world.device, dtype=gt_world.dtype)
    gt_cam = apply_transform(gt_world, transforms)[0]
    branch = resolve_branch(predictions, "refined", smpl)
    depth = branch["depth"].reshape(-1, *branch["depth"].shape[-2:])
    confidence = canonical_batch_depth(predictions["depth_conf"]).reshape(-1, *depth.shape[-2:])
    image_hw = tuple(int(value) for value in depth.shape[-2:])
    from vggt_omega.utils.pose_enc import encoding_to_camera
    _, intrinsics = encoding_to_camera(predictions["pose_enc"].detach().float(), image_size_hw=image_hw, build_intrinsics=True)
    pred_k = intrinsics.reshape(-1, 3, 3)
    gt_k = batch["K_scal3r"].reshape(-1, 3, 3)
    query_indices = choose_frame_query_indices(predictions, batch)[0]
    eval_mask = batch.get("eval_mask", torch.ones(depth.shape[0], device=depth.device)).bool().reshape(-1)
    meta = batch["meta"]
    vid = str(meta["vid"][0])
    frame_ids = [int(value) for value in meta["frame_indices"][0]]
    record = next(item for item in dataset.records if item.vid == vid)
    frames: list[dict[str, Any]] = []
    for frame_offset, source_id in enumerate(frame_ids):
        query_idx = int(query_indices[frame_offset].detach().cpu())
        mesh = select_branch_vertices(branch, frame_offset, query_idx, "refined").detach().cpu().numpy().astype(np.float16)
        frame = {
            "source_frame_id": int(source_id),
            "image_path": str(dataset._frame_path(record, source_id)),
            "scene_depth": depth[frame_offset].detach().cpu().numpy().astype(np.float16),
            "scene_valid_mask": (confidence[frame_offset] > float(metric_config.depth_conf_threshold)).detach().cpu().numpy().astype(np.uint8),
            "pred_intrinsics": pred_k[frame_offset].detach().cpu().numpy().astype(np.float32),
            "gt_intrinsics": gt_k[frame_offset].detach().cpu().numpy().astype(np.float32),
            "pred_vertices_cam": mesh,
            "gt_vertices_cam": gt_cam[frame_offset].detach().cpu().numpy().astype(np.float16),
            "query_idx": query_idx, "person_id": 0, "vid": vid,
            "eval_valid": bool(eval_mask[frame_offset]),
        }
        frames.append(frame)
    return frames


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scale-checkpoint", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--model-config", default="configs/train_smpl_hsi_nlf_stage2_human_scene_align.yaml")
    parser.add_argument("--eval-config", default="configs/eval_show_human_scene_hmr4d.yaml")
    parser.add_argument("--support-root", default="")
    parser.add_argument("--frames-root", default="")
    parser.add_argument("--output-dir", default="outputs/eval/show_3dpw_manual_scale")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--max-humans", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--split", default="test")
    # The shared HMR4D builder uses this field to select the support labels.
    parser.set_defaults(dataset="3dpw")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    main()
