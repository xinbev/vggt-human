#!/usr/bin/env python
"""Server smoke for spatial, overlap, temporal, and ablation TRSTR contracts."""

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
from vggt_omega.tracking import HSITRSTRTrackMemory  # noqa: E402
from vggt_omega.utils.rotation import axis_angle_to_rot6d  # noqa: E402


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    region_summaries = {}
    for budget in (48, 72, 96):
        candidate = build_head(args.smpl_model_dir, device, budget, enable_temporal=False)
        assert candidate.region_bank.num_regions == budget
        assert torch.unique(candidate.region_bank.vertex_region_ids).numel() == budget
        assert candidate.scene_probe.scale_names == ("fixed_3", "fixed_7", "adaptive", "annulus")
        assert candidate.scene_probe.num_tokens == 8
        region_summaries[str(budget)] = {
            "vertices": int(candidate.region_bank.vertex_region_ids.numel()),
            "probe_tokens": candidate.scene_probe.num_tokens,
        }
        del candidate

    head = build_head(args.smpl_model_dir, device, args.num_regions, enable_temporal=args.check_temporal)
    head.train()
    inputs = make_clip_inputs(device, frames=3 if args.check_temporal else 1)
    original_pose = inputs["predictions"]["pred_pose_6d"].clone()
    original_betas = inputs["predictions"]["pred_betas"].clone()
    outputs = head(**inputs)

    batch_size, frames, queries = inputs["person_valid"].shape
    regions = args.num_regions
    assert tuple(outputs["hsi_trstr_region_vote"].shape) == (batch_size, frames, queries, regions, 3)
    assert tuple(outputs["hsi_trstr_region_valid_ratios"].shape) == (
        batch_size, frames, queries, regions, 8
    )
    assert tuple(outputs["hsi_trstr_iteration_transl"].shape) == (3, batch_size, frames, queries, 3)
    assert torch.equal(inputs["predictions"]["pred_pose_6d"], original_pose)
    assert torch.equal(inputs["predictions"]["pred_betas"], original_betas)
    invalid = ~inputs["person_valid"]
    assert torch.equal(
        outputs["hsi_trstr_refined_pred_transl_cam"][invalid],
        inputs["predictions"]["pred_transl_cam"][invalid],
    )
    if args.check_temporal:
        if not bool(outputs["hsi_trstr_temporal_valid"][:, 1:, :2].all()):
            raise AssertionError("Stable same-ID tracks did not activate temporal fusion after frame zero")
        if bool(outputs["hsi_trstr_temporal_valid"][:, 0].any()):
            raise AssertionError("Frame zero unexpectedly consumed temporal history")
    elif bool(outputs["hsi_trstr_temporal_valid"].any()):
        raise AssertionError("Spatial-only smoke unexpectedly activated temporal fusion")
    if float(outputs["hsi_trstr_other_human_ratio"][:, :, 0].max().detach().cpu()) <= 0.0:
        raise AssertionError("Overlapping front person was not excluded from the rear person's evidence")

    target = inputs["predictions"]["pred_transl_cam"] + torch.tensor(
        [0.03, -0.02, 0.05], device=device
    ).reshape(1, 1, 1, 3)
    valid = inputs["person_valid"]
    gate_target = outputs["hsi_trstr_region_valid"].to(dtype=torch.float32)[..., None]
    loss = F.smooth_l1_loss(outputs["hsi_trstr_refined_pred_transl_cam"][valid], target[valid])
    loss = loss + F.binary_cross_entropy(
        outputs["hsi_trstr_region_gate"].clamp(1e-5, 1.0 - 1e-5), gate_target
    )
    loss.backward()
    gradient_prefixes = ("vote_head", "gate_head", "temporal_gate_head") if args.check_temporal else (
        "vote_head",
        "gate_head",
    )
    assert_gradients(head, gradient_prefixes)

    memory_summary = (
        check_persistent_memory(args.smpl_model_dir, device, args.num_regions)
        if args.check_temporal
        else {"checked": False, "reason": "spatial-only smoke"}
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "ok",
        "device": str(device),
        "region_ablation": region_summaries,
        "probe_scales": list(head.scene_probe.scale_names),
        "probe_channels": list(head.scene_probe.channel_names),
        "region_vote_shape": list(outputs["hsi_trstr_region_vote"].shape),
        "valid_region_ratio": float(outputs["hsi_trstr_region_valid"].float().mean().detach().cpu()),
        "other_human_ratio_max": float(outputs["hsi_trstr_other_human_ratio"].max().detach().cpu()),
        "temporal_valid_ratio": float(outputs["hsi_trstr_temporal_valid"].float().mean().detach().cpu()),
        "invalid_person_noop": True,
        "pose_betas_read_only": True,
        "gradient_contract": list(gradient_prefixes),
        "temporal_checked": bool(args.check_temporal),
        "persistent_memory": memory_summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[ok] TRSTR spatial/overlap/temporal smoke passed")
    print(json.dumps({"summary": str(summary_path), **summary}, indent=2))


def build_head(
    smpl_model_dir: str,
    device: torch.device,
    num_regions: int,
    enable_temporal: bool,
) -> HSIRegionalTranslationRefiner:
    return HSIRegionalTranslationRefiner(
        smpl_model_dir=smpl_model_dir,
        hidden_dim=64,
        region_embedding_dim=16,
        num_regions=num_regions,
        representative_vertices=4,
        num_iters=2,
        patch_sizes=(3, 7),
        probe_token_dim=16,
        adaptive_radius_max=8,
        annulus_width=2,
        human_depth_dilation_px=4,
        enable_temporal=enable_temporal,
        max_ray_delta_m=0.20,
        max_tangent_delta_m=0.08,
        max_person_delta_m=0.22,
        image_size=128,
    ).to(device)


def make_clip_inputs(device: torch.device, frames: int = 3) -> dict[str, object]:
    batch_size, queries, height, width = 1, 3, 128, 128
    pose = axis_angle_to_rot6d(
        torch.zeros(batch_size, frames, queries, 24, 3, device=device)
    ).reshape(batch_size, frames, queries, 144)
    betas = torch.zeros(batch_size, frames, queries, 10, device=device)
    transl = torch.zeros(batch_size, frames, queries, 3, device=device)
    for frame_idx in range(frames):
        transl[0, frame_idx, 0] = torch.tensor([0.02 * frame_idx, 0.0, 5.0], device=device)
        transl[0, frame_idx, 1] = torch.tensor([0.04 + 0.02 * frame_idx, 0.0, 4.70], device=device)
    valid = torch.tensor([[[True, True, False]] * frames], device=device)
    depth = torch.full((batch_size, frames, 1, height, width), 4.70, device=device)
    intrinsics = torch.tensor(
        [[[[100.0, 0.0, 64.0], [0.0, 100.0, 64.0], [0.0, 0.0, 1.0]]] * frames],
        device=device,
    )
    predictions = {
        "pred_pose_6d": pose,
        "pred_betas": betas,
        "pred_transl_cam": transl,
        "pred_confs": valid[..., None].to(dtype=torch.float32),
    }
    return {
        "predictions": predictions,
        "depth": depth,
        "pose_enc": None,
        "intrinsics_override": intrinsics,
        "image_size_hw": (height, width),
        "depth_is_metric": True,
        "person_valid": valid,
        "track_ids": torch.tensor([[[10, 20, -1]] * frames], device=device),
        "track_quality": torch.ones(batch_size, frames, queries, device=device),
    }


def check_persistent_memory(
    smpl_model_dir: str,
    device: torch.device,
    num_regions: int,
) -> dict[str, object]:
    head = build_head(smpl_model_dir, device, num_regions, enable_temporal=True).eval()
    memory = HSITRSTRTrackMemory(max_gap=8)
    clip = make_clip_inputs(device, frames=3)

    def select_frame(source: dict[str, object], frame_slice: slice, track_id: int) -> dict[str, object]:
        predictions = {
            key: value[:, frame_slice].clone()
            for key, value in source["predictions"].items()
        }
        return {
            "predictions": predictions,
            "depth": source["depth"][:, frame_slice],
            "pose_enc": None,
            "intrinsics_override": source["intrinsics_override"][:, frame_slice],
            "image_size_hw": source["image_size_hw"],
            "depth_is_metric": True,
            "person_valid": source["person_valid"][:, frame_slice],
            "track_ids": torch.tensor([[[track_id, 20, -1]]], device=device),
            "track_quality": source["track_quality"][:, frame_slice],
            "track_memory": memory,
        }

    with torch.no_grad():
        first = head(**select_frame(clip, slice(0, 1), track_id=10), frame_offset=0)
        if bool(first["hsi_trstr_temporal_valid"].any()):
            raise AssertionError("Fresh persistent memory unexpectedly produced history")
        same_id = head(**select_frame(clip, slice(1, 2), track_id=10), frame_offset=1)
        if not bool(same_id["hsi_trstr_temporal_valid"][0, 0, 0]):
            raise AssertionError("Persistent memory was not reused for the same track ID")
        new_id = head(**select_frame(clip, slice(1, 2), track_id=99), frame_offset=1)
        if bool(new_id["hsi_trstr_temporal_valid"][0, 0, 0]):
            raise AssertionError("Temporal memory leaked from one track ID into another")
    return {
        "same_id_reused": True,
        "different_id_isolated": True,
        "stored_tracks": len(memory),
    }


def assert_gradients(head: torch.nn.Module, prefixes: tuple[str, ...]) -> None:
    for prefix in prefixes:
        gradients = [
            parameter.grad
            for name, parameter in head.named_parameters()
            if name.startswith(prefix)
        ]
        if not any(
            gradient is not None
            and torch.isfinite(gradient).all()
            and bool((gradient.abs() > 0).any())
            for gradient in gradients
        ):
            raise AssertionError(f"Missing finite non-zero gradient for {prefix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smpl-model-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/debug/hsi_stage2_trstr_smoke")
    parser.add_argument("--num-regions", type=int, default=96, choices=(48, 72, 96))
    parser.add_argument("--check-temporal", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
