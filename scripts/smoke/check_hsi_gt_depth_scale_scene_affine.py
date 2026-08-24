#!/usr/bin/env python
"""Smoke checks for GT-depth scale perturbation scene-affine training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics.check_hsi_stage2_transl_perturb_interface import make_batch  # noqa: E402
from scripts.train.train_smpl import (  # noqa: E402
    build_smpl_override_outputs,
    maybe_perturb_gt_metric_depth,
    missing_required_supervision,
    prepare_box_free_gt_scale_batch,
    resolve_hsi_geometry_inputs,
)
from vggt_omega.training.hungarian_losses import (  # noqa: E402
    _gt_slot_matching_indices,
    flatten_smpl_targets,
)


def main() -> None:
    torch.manual_seed(11)
    batch = make_batch()
    batch["gt_depth"] = make_gt_depth()
    config = {
        "model": {
            "hsi_geometry_mode": "gt_metric",
            "gt_smpl_box_free": True,
        },
        "training_prior": {
            "smpl_perturb_mode": "translation",
            "smpl_transl_ray_noise_schedule": "0.0",
            "smpl_transl_tangent_noise_schedule_m": "0.0",
            "smpl_transl_ray_noise_clean_prob": 1.0,
            "hsi_gt_depth_log_scale_std_schedule": "0.30",
            "hsi_gt_depth_scale_noise_mode": "lognormal",
            "hsi_gt_depth_scale_noise_unit": "sequence",
        },
    }
    smpl_override = build_smpl_override_outputs(batch, config, epoch=0, is_training=True)
    if not torch.allclose(smpl_override["pred_transl_cam"], batch["gt_transl_cam"]):
        raise AssertionError("GT-depth scale training should keep GT SMPL translation clean")
    if int(torch.count_nonzero(smpl_override["pred_boxes"])) != 0:
        raise AssertionError("Box-free GT override must not encode person boxes")

    original_boxes_mask = batch["boxes_mask"].clone()
    batch["boxes_mask"] = torch.zeros_like(batch["boxes_mask"])
    box_free_override = build_smpl_override_outputs(batch, config, epoch=0, is_training=True)
    if not torch.equal(box_free_override["gt_smpl_provider_mask"], batch["smpl_mask"].bool()):
        raise AssertionError("Box-free GT override must not filter SMPL slots with boxes_mask")
    targets = flatten_smpl_targets(batch, device=batch["gt_pose_6d"].device, use_boxes_mask=False)
    flat_confs = box_free_override["pred_confs"].reshape(-1, box_free_override["pred_confs"].shape[2], 1)
    indices = _gt_slot_matching_indices(batch, flat_confs, targets)
    expected_slots = torch.nonzero(batch["smpl_mask"].reshape(-1, batch["smpl_mask"].shape[-1])[0], as_tuple=False).reshape(-1)
    if not torch.equal(indices[0][0], expected_slots):
        raise AssertionError("GT-slot matching did not preserve dataset SMPL slot indices")
    batch["boxes_mask"] = original_boxes_mask
    check_empty_batch_filtering()

    geometry = resolve_hsi_geometry_inputs(
        batch,
        config,
        using_gt_override=True,
        epoch=0,
        is_training=False,
    )
    scale = geometry["diagnostics"]["hsi_gt_depth_perturb_scale"]
    target = geometry["diagnostics"]["hsi_gt_depth_perturb_target_scale"]
    if not torch.allclose(scale, torch.ones_like(scale), atol=1e-6):
        raise AssertionError(f"Evaluation must keep GT depth clean: {scale.flatten().tolist()}")
    if not torch.allclose(target, torch.ones_like(target), atol=1e-6):
        raise AssertionError(f"Unexpected clean evaluation target: {target.flatten().tolist()}")
    expected_depth = batch["gt_depth"] * scale.unsqueeze(2)
    if not torch.allclose(geometry["depth"], expected_depth, atol=1e-6):
        raise AssertionError("Perturbed GT depth does not equal gt_depth * scale")
    if geometry["intrinsics"] is not batch["K_scal3r"]:
        raise AssertionError("GT metric path should pass dataset intrinsics")
    if geometry["depth_is_metric"] is not True:
        raise AssertionError("GT metric depth must be marked metric")

    distribution_depth = torch.ones(4096, 1, 1, 1, 1)
    distribution_config = {
        "training_prior": {
            "hsi_gt_depth_log_scale_std_schedule": "0.30",
            "hsi_gt_depth_scale_noise_mode": "lognormal",
            "hsi_gt_depth_scale_noise_unit": "sequence",
            "hsi_gt_depth_scale_clean_prob": 0.0,
            "hsi_gt_depth_scale_min": 0.0,
            "hsi_gt_depth_scale_max": 0.0,
        }
    }
    _, distribution_diagnostics = maybe_perturb_gt_metric_depth(
        distribution_depth,
        distribution_config,
        epoch=0,
        is_training=True,
    )
    sampled_log_scale = torch.log(distribution_diagnostics["hsi_gt_depth_perturb_scale"].reshape(-1))
    sampled_log_mean = float(sampled_log_scale.mean())
    sampled_log_std = float(sampled_log_scale.std(unbiased=False))
    if abs(sampled_log_mean) > 0.03:
        raise AssertionError(f"Log-scale Gaussian mean drifted: {sampled_log_mean:.6f}")
    if not 0.27 <= sampled_log_std <= 0.33:
        raise AssertionError(f"Unexpected log-scale Gaussian std: {sampled_log_std:.6f}")

    out_dir = ROOT / "outputs" / "debug" / "hsi_gt_depth_scale_scene_affine_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "ok",
        "depth_shape": list(geometry["depth"].shape),
        "clean_eval_depth_scale": float(scale.mean().item()),
        "clean_eval_target_scale": float(target.mean().item()),
        "sampled_log_scale_mean": sampled_log_mean,
        "sampled_log_scale_std": sampled_log_std,
        "smpl_override_clean": True,
        "box_free_gt_slots": True,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[ok] HSI GT-depth scale scene-affine smoke passed")
    print(json.dumps({"summary": str(out_dir / "summary.json")}, indent=2))


def make_gt_depth() -> torch.Tensor:
    batch_size, num_frames, height, width = 1, 2, 16, 16
    y = torch.linspace(2.0, 8.0, height).reshape(1, 1, 1, height, 1)
    x = torch.linspace(0.0, 0.5, width).reshape(1, 1, 1, 1, width)
    return (y + x).expand(batch_size, num_frames, 1, height, width).contiguous()


def check_empty_batch_filtering() -> None:
    visible = torch.zeros(2, 2, 3, dtype=torch.bool)
    visible[1, 0, 0] = True
    batch = {
        "images": torch.zeros(2, 2, 3, 8, 8),
        "smpl_mask": visible.clone(),
        "gt_smpl_online_visibility_mask": visible.clone(),
        "dataset_index": torch.tensor([10, 11]),
    }
    config = {
        "model": {
            "smpl_provider": "gt_perturbed",
            "gt_smpl_online_visibility": True,
        },
        "optim": {
            "drop_no_visible_gt_samples": True,
            "required_supervision_metric": "metric_hsi_smpl_scale_teacher_valid_points",
            "required_supervision_min": 0.0,
        },
    }
    filtered, stats = prepare_box_free_gt_scale_batch(batch, config)
    if stats["dropped_no_visible_samples"] != 1 or stats["kept_samples"] != 1:
        raise AssertionError(f"Unexpected no-visible sample filtering stats: {stats}")
    if filtered["dataset_index"].tolist() != [11]:
        raise AssertionError("No-visible sample filtering retained the wrong sample")

    empty = {key: value.clone() for key, value in batch.items()}
    empty["smpl_mask"].zero_()
    empty["gt_smpl_online_visibility_mask"].zero_()
    filtered_empty, empty_stats = prepare_box_free_gt_scale_batch(empty, config)
    if empty_stats["kept_samples"] != 0 or int(filtered_empty["images"].shape[0]) != 0:
        raise AssertionError("All-empty batch was not reduced to zero samples")

    zero_losses = {"metric_hsi_smpl_scale_teacher_valid_points": torch.tensor(0.0)}
    valid_losses = {"metric_hsi_smpl_scale_teacher_valid_points": torch.tensor(128.0)}
    if missing_required_supervision(zero_losses, config) is None:
        raise AssertionError("Zero scale supervision should skip the batch")
    if missing_required_supervision(valid_losses, config) is not None:
        raise AssertionError("Positive scale supervision should keep the batch")


if __name__ == "__main__":
    main()
