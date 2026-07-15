#!/usr/bin/env python
"""Smoke checks for Stage2 GT-SMPL ray-depth translation perturbation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_smpl import build_smpl_override_outputs  # noqa: E402
from vggt_omega.utils.rotation import axis_angle_to_rot6d  # noqa: E402


def main() -> None:
    torch.manual_seed(7)
    batch = make_batch()
    config = {
        "training_prior": {
            "smpl_transl_ray_noise_schedule": "0.10",
            "smpl_transl_ray_noise_clean_prob": 0.0,
            "smpl_transl_ray_noise_mode": "uniform",
        }
    }
    outputs = build_smpl_override_outputs(batch, config, epoch=0)
    assert_shape(outputs["pred_pose_6d"], (1, 2, 4, 144), "pred_pose_6d")
    assert_shape(outputs["pred_poses"], (1, 2, 4, 72), "pred_poses")
    assert_shape(outputs["pred_betas"], (1, 2, 4, 10), "pred_betas")
    assert_shape(outputs["pred_transl_cam"], (1, 2, 4, 3), "pred_transl_cam")
    assert_shape(outputs["pred_confs"], (1, 2, 4, 1), "pred_confs")

    valid = outputs["gt_smpl_provider_mask"]
    clean = outputs["base_clean_pred_transl_cam"][valid]
    perturbed = outputs["perturbed_pred_transl_cam"][valid]
    ratios = outputs["transl_noise_ratio"][valid].reshape(-1)
    if not torch.all((ratios >= 0.90) & (ratios <= 1.10)):
        raise AssertionError(f"noise ratios outside expected range: {ratios.tolist()}")
    if torch.allclose(clean, perturbed):
        raise AssertionError("perturbed translations unexpectedly equal clean translations")

    projected_clean = project_roots(clean, batch["K_scal3r"].reshape(-1, 3, 3)[0].expand(clean.shape[0], -1, -1))
    projected_perturbed = project_roots(perturbed, batch["K_scal3r"].reshape(-1, 3, 3)[0].expand(perturbed.shape[0], -1, -1))
    projection_delta = (projected_clean - projected_perturbed).abs().max().item()
    if projection_delta > 1e-4:
        raise AssertionError(f"ray-depth perturbation changed projection center by {projection_delta}")
    if not torch.allclose(outputs["pred_transl_cam"], outputs["base_pred_transl_cam"]):
        raise AssertionError("base_pred_transl_cam must reflect the perturbed HSI input")

    out_dir = ROOT / "outputs" / "debug" / "hsi_stage2_transl_perturb_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "ok",
        "valid_slots": int(valid.sum().item()),
        "projection_delta_px_max": projection_delta,
        "noise_ratio_min": float(ratios.min().item()),
        "noise_ratio_max": float(ratios.max().item()),
        "keys": sorted(outputs.keys()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[ok] HSI Stage2 translation perturbation interface smoke passed")
    print(json.dumps({"summary": str(out_dir / "summary.json")}, indent=2))


def make_batch() -> dict[str, torch.Tensor]:
    batch_size, num_frames, num_queries = 1, 2, 4
    aa = torch.zeros(batch_size, num_frames, num_queries, 24, 3)
    pose6d = axis_angle_to_rot6d(aa).reshape(batch_size, num_frames, num_queries, 144)
    betas = torch.zeros(batch_size, num_frames, num_queries, 10)
    transl = torch.tensor(
        [
            [
                [[0.10, 0.05, 4.0], [-0.20, 0.08, 5.5], [0.0, 0.0, 0.0], [0.25, -0.10, 7.0]],
                [[0.12, 0.04, 4.2], [-0.18, 0.10, 5.7], [0.0, 0.0, 0.0], [0.28, -0.08, 7.1]],
            ]
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[[True, True, False, True], [True, True, False, True]]])
    boxes = torch.tensor(
        [
            [
                [[0.45, 0.48, 0.18, 0.42], [0.55, 0.50, 0.16, 0.36], [0.0, 0.0, 0.0, 0.0], [0.62, 0.52, 0.12, 0.30]],
                [[0.46, 0.48, 0.18, 0.42], [0.56, 0.50, 0.16, 0.36], [0.0, 0.0, 0.0, 0.0], [0.63, 0.52, 0.12, 0.30]],
            ]
        ],
        dtype=torch.float32,
    )
    k = torch.tensor([[700.0, 0.0, 256.0], [0.0, 700.0, 256.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    return {
        "gt_pose_6d": pose6d,
        "gt_betas": betas,
        "gt_transl_cam": transl,
        "gt_cam_trans": transl,
        "smpl_mask": mask,
        "boxes_mask": mask,
        "gt_boxes": boxes,
        "smpl_query_boxes": boxes,
        "K_scal3r": k.reshape(1, 1, 3, 3).expand(batch_size, num_frames, -1, -1).clone(),
    }


def project_roots(points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    z = points[:, 2].clamp(min=1e-6)
    x = intrinsics[:, 0, 0] * points[:, 0] / z + intrinsics[:, 0, 2]
    y = intrinsics[:, 1, 1] * points[:, 1] / z + intrinsics[:, 1, 2]
    return torch.stack([x, y], dim=-1)


def assert_shape(tensor: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != shape:
        raise AssertionError(f"{name} shape {tuple(tensor.shape)} != {shape}")


if __name__ == "__main__":
    main()
