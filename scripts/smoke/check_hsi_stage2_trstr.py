#!/usr/bin/env python
"""Server smoke for the standalone translation-only TRSTR head."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.models.heads.hsi_regional_translation_refiner import HSIRegionalTranslationRefiner  # noqa: E402
from vggt_omega.utils.rotation import axis_angle_to_rot6d  # noqa: E402


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    head = HSIRegionalTranslationRefiner(
        smpl_model_dir=args.smpl_model_dir,
        hidden_dim=64,
        region_embedding_dim=16,
        num_regions=args.num_regions,
        representative_vertices=4,
        num_iters=2,
        patch_sizes=(3, 7),
        max_ray_delta_m=0.20,
        max_tangent_delta_m=0.10,
        max_person_delta_m=0.30,
        image_size=128,
    ).to(device)
    head.train()

    batch_size, frames, queries, height, width = 1, 1, 3, 128, 128
    pose = axis_angle_to_rot6d(torch.zeros(batch_size, frames, queries, 24, 3, device=device)).reshape(
        batch_size, frames, queries, 144
    )
    betas = torch.zeros(batch_size, frames, queries, 10, device=device)
    transl = torch.tensor(
        [[[[0.0, 0.0, 5.0], [0.35, 0.0, 5.0], [-0.35, 0.0, 5.0]]]],
        device=device,
    )
    valid = torch.tensor([[[True, True, False]]], device=device)
    depth = torch.full((batch_size, frames, 1, height, width), 5.0, device=device)
    intrinsics = torch.tensor(
        [[[[100.0, 0.0, 64.0], [0.0, 100.0, 64.0], [0.0, 0.0, 1.0]]]],
        device=device,
    )
    predictions = {
        "pred_pose_6d": pose,
        "pred_betas": betas,
        "pred_transl_cam": transl,
        "pred_confs": valid[..., None].to(dtype=torch.float32),
    }
    outputs = head(
        predictions=predictions,
        depth=depth,
        pose_enc=None,
        intrinsics_override=intrinsics,
        image_size_hw=(height, width),
        depth_is_metric=True,
        person_valid=valid,
    )

    assert head.region_bank.num_regions == args.num_regions
    assert torch.unique(head.region_bank.vertex_region_ids).numel() == args.num_regions
    assert tuple(outputs["hsi_trstr_region_vote"].shape) == (batch_size, frames, queries, args.num_regions, 3)
    assert tuple(outputs["hsi_trstr_refined_pred_transl_cam"].shape) == transl.shape
    assert tuple(outputs["hsi_trstr_iteration_transl"].shape) == (3, batch_size, frames, queries, 3)
    if not torch.equal(outputs["hsi_trstr_refined_pred_transl_cam"][0, 0, 2], transl[0, 0, 2]):
        raise AssertionError("Invalid person slot was changed by TRSTR")

    target = transl + torch.tensor([0.05, -0.03, 0.08], device=device).reshape(1, 1, 1, 3)
    vote_gate_loss = F.binary_cross_entropy(
        outputs["hsi_trstr_region_gate"].clamp(1e-5, 1.0 - 1e-5),
        outputs["hsi_trstr_region_valid"].to(dtype=torch.float32)[..., None],
    )
    loss = F.smooth_l1_loss(outputs["hsi_trstr_refined_pred_transl_cam"][valid], target[valid]) + vote_gate_loss
    loss.backward()
    for prefix in ("vote_head", "gate_head"):
        gradients = [param.grad for name, param in head.named_parameters() if name.startswith(prefix)]
        if not any(grad is not None and torch.isfinite(grad).all() and bool((grad.abs() > 0).any()) for grad in gradients):
            raise AssertionError(f"Missing finite non-zero gradient for {prefix}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "ok",
        "device": str(device),
        "num_regions": head.region_bank.num_regions,
        "vertex_count": int(head.region_bank.vertex_region_ids.numel()),
        "region_vote_shape": list(outputs["hsi_trstr_region_vote"].shape),
        "iteration_translation_shape": list(outputs["hsi_trstr_iteration_transl"].shape),
        "valid_region_ratio": float(outputs["hsi_trstr_region_valid"].float().mean().detach().cpu()),
        "invalid_person_noop": True,
        "gradient_contract": ["vote_head", "gate_head"],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[ok] TRSTR standalone forward/gradient smoke passed")
    print(json.dumps({"summary": str(summary_path), **summary}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smpl-model-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/debug/hsi_stage2_trstr_smoke")
    parser.add_argument("--num-regions", type=int, default=96, choices=(48, 72, 96))
    return parser.parse_args()


if __name__ == "__main__":
    main()
