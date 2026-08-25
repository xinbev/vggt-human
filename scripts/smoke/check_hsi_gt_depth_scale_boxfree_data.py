#!/usr/bin/env python
"""Audit real BEDLAM windows for the box-free GT scale training contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_smpl import (  # noqa: E402
    apply_gt_smpl_online_visibility,
    build_coarse_residual_training_depth,
    build_loader,
    build_smpl_override_outputs,
    extract_state_dict,
    resolve_hsi_geometry_inputs,
)
from vggt_omega.training.config import deep_update, load_yaml_config  # noqa: E402


def main() -> None:
    args = parse_args()
    config = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.train_config))
    config["optim"]["batch_size"] = 1
    config["data"]["num_workers"] = 0
    config["data"]["persistent_workers"] = False
    assert_box_free_config(config)
    runtime_summary = assert_runtime_requirements(config)

    loader = build_loader(config, split=config["data"]["train_split"], shuffle=False, role="train")
    scanned_samples = 0
    original_people = 0
    online_visible_people = 0
    original_nonempty_samples = 0
    online_nonempty_samples = 0
    empty_visible_frames = 0
    batch = None
    original = None
    visible = None
    for batch_idx, candidate in enumerate(loader):
        if batch_idx >= int(args.num_batches):
            break
        candidate_original = candidate["smpl_mask"].clone()
        apply_gt_smpl_online_visibility(candidate, config)
        candidate_visible = candidate["smpl_mask"].bool()
        if bool((candidate_visible & ~candidate_original.bool()).any()):
            raise AssertionError("Online visibility introduced a person outside the dataset SMPL mask")
        scanned_samples += int(candidate_visible.shape[0])
        original_people += int(candidate_original.sum())
        online_visible_people += int(candidate_visible.sum())
        original_nonempty_samples += int(candidate_original.any(dim=-1).any(dim=-1).sum())
        online_nonempty_samples += int(candidate_visible.any(dim=-1).any(dim=-1).sum())
        empty_visible_frames += int((~candidate_visible.any(dim=-1)).sum())
        if batch is None and int(candidate_visible.sum()) > 0:
            batch = candidate
            original = candidate_original
            visible = candidate_visible
    if batch is None or original is None or visible is None:
        raise AssertionError(f"No online-visible GT SMPL people in the first {args.num_batches} BEDLAM windows")

    override = build_smpl_override_outputs(batch, config, epoch=0, is_training=True)
    if int(torch.count_nonzero(override["pred_boxes"])) != 0:
        raise AssertionError("Box-free GT override unexpectedly contains person boxes")
    geometry = resolve_hsi_geometry_inputs(
        batch,
        config,
        using_gt_override=True,
        epoch=0,
        is_training=True,
    )
    scale = geometry["diagnostics"]["hsi_gt_depth_perturb_scale"]
    if not torch.isfinite(scale).all() or not bool((scale > 0).all()):
        raise AssertionError("GT depth perturbation produced an invalid multiplicative scale")

    coarse_config = deep_update(config, {"training_prior": {"hsi_scale_training_mode": "coarse_residual_stratified"}})
    coarse_depth, coarse_diagnostics = build_coarse_residual_training_depth(batch, coarse_config)
    coarse_valid = coarse_diagnostics["hsi_coarse_valid_mask"].bool()
    if not bool(coarse_valid.any()):
        raise AssertionError("Real BEDLAM smoke batch produced no valid traditional coarse frame")
    if not torch.isfinite(coarse_depth).all() or not bool((coarse_depth >= 0).all()):
        raise AssertionError("Coarse-residual training produced invalid depth")
    absolute_target = coarse_diagnostics["hsi_absolute_scale_target"][coarse_valid]
    coarse_estimate = coarse_diagnostics["hsi_coarse_scale_estimate"][coarse_valid]
    coarse_log_l1 = float(torch.abs(torch.log(coarse_estimate) - torch.log(absolute_target)).mean())

    output_dir = ROOT / "outputs" / "debug" / "hsi_gt_depth_scale_boxfree_data_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "ok",
        "dataset_samples": len(loader.dataset),
        "scanned_samples": scanned_samples,
        "original_people": original_people,
        "online_visible_people": online_visible_people,
        "visibility_keep_ratio": online_visible_people / max(original_people, 1),
        "original_nonempty_samples": original_nonempty_samples,
        "online_nonempty_samples": online_nonempty_samples,
        "dropped_no_visible_samples": scanned_samples - online_nonempty_samples,
        "dropped_no_visible_sample_rate": (scanned_samples - online_nonempty_samples) / max(scanned_samples, 1),
        "empty_visible_frames": empty_visible_frames,
        "empty_visible_frame_rate": empty_visible_frames / max(scanned_samples * int(config["data"]["sequence_length"]), 1),
        "depth_perturb_scale": scale.reshape(-1).tolist(),
        "boxes_root_key": config["data"].get("boxes_root_key", ""),
        "matching_mode": config["matching"].get("mode"),
        "coarse_residual": {
            "valid_frames": int(coarse_valid.sum()),
            "absolute_scale_target": absolute_target.reshape(-1).tolist(),
            "coarse_scale_estimate": coarse_estimate.reshape(-1).tolist(),
            "coarse_scale_used": coarse_diagnostics["hsi_coarse_scale_used"][coarse_valid].reshape(-1).tolist(),
            "residual_scale_target": coarse_diagnostics["hsi_residual_scale_target"][coarse_valid].reshape(-1).tolist(),
            "coarse_log_l1": coarse_log_l1,
        },
        "runtime": runtime_summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[ok] box-free GT scale data smoke passed")
    print(json.dumps({"summary": str(summary_path), **summary}, indent=2))


def assert_box_free_config(config: dict) -> None:
    checks = {
        "data.boxes_root_key": not bool(config["data"].get("boxes_root_key")),
        "data.require_boxes": not bool(config["data"].get("require_boxes")),
        "data.box_free_gt_slots": bool(config["data"].get("box_free_gt_slots")),
        "model.gt_smpl_box_free": bool(config["model"].get("gt_smpl_box_free")),
        "model.gt_smpl_online_visibility": bool(config["model"].get("gt_smpl_online_visibility")),
        "model.smpl_query_box_prior": not bool(config["model"].get("smpl_query_box_prior")),
        "model.smpl_use_aggregator_queries": not bool(
            config["model"].get("smpl_use_aggregator_queries", True)
        ),
        "matching.mode": str(config["matching"].get("mode")) == "gt_slots",
        "optim.drop_no_visible_gt_samples": bool(config["optim"].get("drop_no_visible_gt_samples")),
        "optim.required_supervision_metric": str(config["optim"].get("required_supervision_metric"))
        == "metric_hsi_smpl_scale_teacher_valid_points",
    }
    failed = [key for key, valid in checks.items() if not valid]
    if failed:
        raise AssertionError(f"Training config is not box-free: {failed}")


def assert_runtime_requirements(config: dict) -> dict:
    baseline = resolve_project_path(config.get("checkpoints", {}).get("vggt_baseline", ""))
    resume = resolve_project_path(config.get("checkpoint", {}).get("resume", ""))
    smpl_dir = resolve_project_path(config.get("assets", {}).get("smpl_model_dir", ""))
    if not baseline.is_file():
        raise FileNotFoundError(f"VGGT baseline checkpoint not found: {baseline}")
    if not resume.is_file():
        raise FileNotFoundError(f"Scale continuation checkpoint not found: {resume}")
    if not smpl_dir.is_dir():
        raise FileNotFoundError(f"SMPL model directory not found: {smpl_dir}")
    if bool(config.get("logging", {}).get("wandb", {}).get("enabled", False)):
        if importlib.util.find_spec("wandb") is None:
            raise ImportError("W&B is enabled in the training config but the wandb package is not installed")

    state_dict = extract_state_dict(torch.load(resume, map_location="cpu"))
    required_prefixes = (
        "hsi_refinement_head.scale_delta.",
        "hsi_refinement_head.bias_delta.",
    )
    missing = [prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in state_dict)]
    if missing:
        raise RuntimeError(f"Continuation checkpoint is missing required scene-affine prefixes: {missing}")
    return {
        "vggt_baseline": str(baseline),
        "resume_checkpoint": str(resume),
        "smpl_model_dir": str(smpl_dir),
        "resume_model_tensors": len(state_dict),
        "wandb_available": importlib.util.find_spec("wandb") is not None,
    }


def resolve_project_path(value: object) -> Path:
    path = Path(str(value or "")).expanduser()
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_hsi_gt_depth_scale_scene_affine.yaml")
    parser.add_argument("--num-batches", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    main()
