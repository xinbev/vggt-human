#!/usr/bin/env python3
"""Export unchunked two-pass EMDB-2 stride-7 world predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.emdb2_global.data import (  # noqa: E402
    EMDB2Sequence,
    decode_gt_world_joints,
    load_emdb2_sequences,
)
from scripts.train.train_smpl import build_model  # noqa: E402
from scripts.vis.serve_nlf_hsi_vggt_sequence_viewer import (  # noqa: E402
    load_checkpoint_prefix_overlay,
    load_sequence_images,
    run_model,
)
from scripts.vis.visualize_smpl_inference import (  # noqa: E402
    load_training_checkpoint,
    load_vggt_baseline_for_camera,
)
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.training.config import deep_update, load_yaml_config, require_path  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({args.device}) but unavailable")
    device = torch.device(args.device)
    cfg = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.inference_config))
    cfg.setdefault("model", {})["num_smpl_queries"] = int(args.max_humans)
    cfg.setdefault("data", {})["max_humans"] = int(args.max_humans)
    emdb_root = Path(args.emdb_root or require_path(cfg, "datasets.emdb_root")).expanduser()
    sequences = load_emdb2_sequences(emdb_root, args.sequence_filter)
    if args.max_sequences > 0:
        sequences = sequences[: args.max_sequences]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(device).eval()
    load_vggt_baseline_for_camera(model, cfg, device)
    load_training_checkpoint(model, Path(args.checkpoint), device)
    load_checkpoint_prefix_overlay(
        model,
        Path(args.scale_checkpoint),
        device,
        ("hsi_refinement_head.",),
    )
    smpl = SMPLLayer(require_path(cfg, "assets.smpl_model_dir", allow_empty=False)).to(device).eval()
    gt_smpl_layers = {
        gender: SMPLLayer(
            require_path(cfg, "assets.smpl_model_dir", allow_empty=False), gender=gender
        ).to(device).eval()
        for gender in ("male", "female")
    }
    manifest: list[dict[str, Any]] = []
    for sequence in sequences:
        archive = export_sequence(
            sequence,
            emdb_root,
            model,
            smpl,
            gt_smpl_layers[sequence.gender],
            cfg,
            args,
            device,
            output_root,
        )
        manifest.append(archive)
        torch.cuda.empty_cache() if device.type == "cuda" else None
    summary = {
        "protocol": f"EMDB-2-S{args.subsample_stride} unchunked-two-pass",
        "checkpoint": str(Path(args.checkpoint)),
        "scale_checkpoint": str(Path(args.scale_checkpoint)),
        "sequence_count": len(manifest),
        "output_dir": str(output_root),
        "sequences": manifest,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scale-checkpoint", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument(
        "--inference-config",
        default="benchmarks/emdb2_global/inference_config.yaml",
    )
    parser.add_argument("--emdb-root", default="")
    parser.add_argument("--output-dir", default="outputs/eval/emdb2_global_stride7/predictions")
    parser.add_argument("--subsample-stride", type=int, default=7)
    parser.add_argument("--max-input-frames", type=int, default=500)
    parser.add_argument("--sequence-filter", default="")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-humans", type=int, default=8)
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument("--match-iou-threshold", type=float, default=0.05)
    parser.add_argument("--trstr-frame-chunk", type=int, default=16)
    parser.add_argument("--coarse-scale-min", type=float, default=0.10)
    parser.add_argument("--coarse-scale-max", type=float, default=25.0)
    parser.add_argument("--coarse-anchor-stride", type=int, default=8)
    parser.add_argument("--coarse-min-anchor-pixels", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def export_sequence(
    sequence: EMDB2Sequence,
    emdb_root: Path,
    model: torch.nn.Module,
    smpl: SMPLLayer,
    gt_smpl: SMPLLayer,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    output_root: Path,
) -> dict[str, Any]:
    stride = max(int(args.subsample_stride), 1)
    frame_indices = sequence.good_frame_indices[::stride]
    if frame_indices.size > int(args.max_input_frames):
        raise RuntimeError(
            f"{sequence.name}: stride={stride} selects {frame_indices.size} frames, "
            f"exceeding single-forward limit {args.max_input_frames}"
        )
    frame_paths = sequence_frame_paths(sequence, frame_indices)
    data_cfg = cfg.get("data", {})
    images, geometries = load_sequence_images(
        frame_paths,
        image_resolution=int(data_cfg.get("image_resolution", 512)),
        patch_size=int(cfg.get("model", {}).get("patch_size", 16)),
        resize_mode=str(data_cfg.get("resize_mode", "balanced")),
    )
    image_sequence = images.unsqueeze(0).to(device)
    run_args = SimpleNamespace(
        scene_scale_prealign="smpl_median",
        coarse_min_anchor_pixels=int(args.coarse_min_anchor_pixels),
        coarse_scale_min=float(args.coarse_scale_min),
        coarse_scale_max=float(args.coarse_scale_max),
        coarse_anchor_stride=int(args.coarse_anchor_stride),
        coarse_fallback="sequence_median",
        conf_threshold=float(args.conf_threshold),
        cascade_effective_affine_mode="clip_median",
        trstr_frame_chunk=max(int(args.trstr_frame_chunk), 1),
    )
    predictions = run_model(model, image_sequence, None, smpl=smpl, args=run_args)
    gt_keypoints = projected_gt_smpl_keypoints(
        sequence=sequence,
        frame_indices=frame_indices,
        geometries=geometries,
        gt_smpl=gt_smpl,
        neutral_joint_regressor=smpl.layer.J_regressor.detach(),
        device=device,
    )
    joints_cam_by_stage, valid, selected_queries = decode_selected_joints_cam_stages(
        predictions=predictions,
        smpl=smpl,
        gt_keypoints=gt_keypoints,
        image_size_hw=tuple(int(value) for value in image_sequence.shape[-2:]),
        conf_threshold=float(args.conf_threshold),
        match_iou_threshold=float(args.match_iou_threshold),
    )
    coarse_scale = predictions.get("hsi_coarse_scene_scale")
    if not isinstance(coarse_scale, torch.Tensor):
        raise RuntimeError("Multi-stage export requires hsi_coarse_scene_scale")
    shared_coarse_scale = torch.exp(
        torch.log(coarse_scale.detach().float().clamp(min=1e-6)).median(dim=1, keepdim=True).values
    ).expand_as(coarse_scale)
    coarse_t_c2w = predicted_camera_to_world(
        predictions,
        image_sequence.shape[-2:],
        camera_translation_scale=shared_coarse_scale,
    )
    metric_t_c2w = predicted_camera_to_world(
        predictions,
        image_sequence.shape[-2:],
        camera_translation_scale=predictions["hsi_scene_scale"],
    )
    stage_camera = {
        "vggt_nlf": coarse_t_c2w,
        "vggt_nlf_hsi_scale": metric_t_c2w,
        "vggt_nlf_hsi_scale_trstr": metric_t_c2w,
    }
    joints_world_by_stage = {
        stage: camera_joints_to_world(joints_cam_by_stage[stage], stage_camera[stage])
        for stage in joints_cam_by_stage
    }
    path = output_root / f"{sequence.safe_name}.npz"
    np.savez_compressed(
        path,
        sequence_name=np.asarray(sequence.name),
        frame_indices=frame_indices.astype(np.int64, copy=False),
        stage_names=np.asarray(tuple(joints_world_by_stage.keys())),
        pred_joints_cam__vggt_nlf=joints_cam_by_stage["vggt_nlf"][0].detach().float().cpu().numpy(),
        pred_T_c2w__vggt_nlf=coarse_t_c2w[0].detach().float().cpu().numpy(),
        pred_joints_world__vggt_nlf=joints_world_by_stage["vggt_nlf"][0].detach().float().cpu().numpy(),
        pred_joints_cam__vggt_nlf_hsi_scale=joints_cam_by_stage["vggt_nlf_hsi_scale"][0].detach().float().cpu().numpy(),
        pred_T_c2w__vggt_nlf_hsi_scale=metric_t_c2w[0].detach().float().cpu().numpy(),
        pred_joints_world__vggt_nlf_hsi_scale=joints_world_by_stage["vggt_nlf_hsi_scale"][0].detach().float().cpu().numpy(),
        pred_joints_cam__vggt_nlf_hsi_scale_trstr=joints_cam_by_stage["vggt_nlf_hsi_scale_trstr"][0].detach().float().cpu().numpy(),
        pred_T_c2w__vggt_nlf_hsi_scale_trstr=metric_t_c2w[0].detach().float().cpu().numpy(),
        pred_joints_world__vggt_nlf_hsi_scale_trstr=joints_world_by_stage["vggt_nlf_hsi_scale_trstr"][0].detach().float().cpu().numpy(),
        valid=valid[0].detach().cpu().numpy().astype(bool, copy=False),
        selected_query=selected_queries[0].detach().cpu().numpy().astype(np.int64, copy=False),
        joint_format=np.asarray("smpl24"),
        units=np.asarray("m"),
        subsample_stride=np.asarray(stride, dtype=np.int64),
        matching_protocol=np.asarray("human3r_gt_smpl2d_iou_v1"),
    )
    trstr = predictions.get("_viewer_trstr_summary", {})
    result = {
        "sequence": sequence.name,
        "archive": str(path),
        "original_good_frames": int(sequence.good_frame_indices.size),
        "selected_frames": int(frame_indices.size),
        "valid_predictions": int(valid.sum().detach().cpu()),
        "coverage": float(valid.float().mean().detach().cpu()),
        "matching_protocol": "human3r_gt_smpl2d_iou_v1",
        "effective_scale": float(predictions["hsi_scene_scale"][0, 0, 0].detach().cpu()),
        "analytic_coarse_scale": float(shared_coarse_scale[0, 0, 0].detach().cpu()),
        "stages": list(joints_world_by_stage.keys()),
        "trstr": trstr,
    }
    print(
        f"[export] {sequence.name} frames={result['selected_frames']} "
        f"valid={result['valid_predictions']} scale={result['effective_scale']:.5g}",
        flush=True,
    )
    del predictions, image_sequence, images, joints_cam_by_stage, joints_world_by_stage
    del coarse_t_c2w, metric_t_c2w, shared_coarse_scale
    return result


def sequence_frame_paths(sequence: EMDB2Sequence, frame_indices: np.ndarray) -> list[Path]:
    images_dir = sequence.annotation_path.parent / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"EMDB image directory not found: {images_dir}")
    images = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(images) < sequence.frame_count:
        raise RuntimeError(
            f"{sequence.name}: images={len(images)} fewer than annotation frames={sequence.frame_count}"
        )
    return [images[int(index)] for index in frame_indices.tolist()]


def decode_selected_joints_cam_stages(
    predictions: dict[str, torch.Tensor],
    smpl: SMPLLayer,
    gt_keypoints: torch.Tensor,
    image_size_hw: tuple[int, int],
    conf_threshold: float,
    match_iou_threshold: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    pose = predictions.get("pred_poses")
    betas = predictions.get("pred_betas")
    base_transl = predictions.get("pred_transl_cam")
    refined_transl = predictions.get("hsi_refined_pred_transl_cam")
    confs = predictions.get("pred_confs")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (pose, betas, base_transl, refined_transl, confs)
    ):
        raise RuntimeError("Inference output is missing base/refined pose/betas/transl/conf")
    confidence = confs[..., 0]
    flat_pose = pose.reshape(-1, pose.shape[-1])
    flat_betas = betas.reshape(-1, betas.shape[-1])
    _, all_joints = smpl(flat_pose.float(), flat_betas.float())
    all_joints = all_joints[:, :24].reshape(*pose.shape[:3], 24, 3)
    all_joints = all_joints.to(dtype=base_transl.dtype)
    all_base_joints_cam = all_joints + base_transl[..., None, :]
    _, predicted_intrinsics = encoding_to_camera(
        predictions["pose_enc"].detach().float(),
        image_size_hw=image_size_hw,
        build_intrinsics=True,
    )
    query, valid = match_emdb_person_by_2d(
        pred_joints_cam=all_base_joints_cam,
        confidence=confidence,
        intrinsics=predicted_intrinsics.to(dtype=all_base_joints_cam.dtype),
        gt_keypoints=gt_keypoints.to(
            device=all_base_joints_cam.device, dtype=all_base_joints_cam.dtype
        ),
        conf_threshold=conf_threshold,
        iou_threshold=match_iou_threshold,
    )
    gather_transl = query[..., None].expand(*query.shape, 3)
    selected_base_transl = base_transl.gather(2, gather_transl.unsqueeze(2)).squeeze(2)
    selected_refined_transl = refined_transl.gather(2, gather_transl.unsqueeze(2)).squeeze(2)
    gather_joints = query[..., None, None, None].expand(*query.shape, 1, 24, 3)
    expected_gather_shape = (*query.shape, 1, 24, 3)
    if gather_joints.shape != expected_gather_shape:
        raise RuntimeError(
            f"Joint gather index shape mismatch: got={tuple(gather_joints.shape)} "
            f"expected={expected_gather_shape}"
        )
    joints = all_joints.gather(2, gather_joints).squeeze(2)
    base_joints_cam = joints + selected_base_transl[..., None, :]
    refined_joints_cam = joints + selected_refined_transl[..., None, :]
    return {
        "vggt_nlf": base_joints_cam,
        "vggt_nlf_hsi_scale": base_joints_cam,
        "vggt_nlf_hsi_scale_trstr": refined_joints_cam,
    }, valid, query


def projected_gt_smpl_keypoints(
    sequence: EMDB2Sequence,
    frame_indices: np.ndarray,
    geometries: list[Any],
    gt_smpl: SMPLLayer,
    neutral_joint_regressor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if len(geometries) != len(frame_indices):
        raise ValueError("Geometry/frame count mismatch for EMDB 2D matching")
    joints_world = decode_gt_world_joints(
        sequence,
        frame_indices,
        gt_smpl,
        device,
        joint_regressor=neutral_joint_regressor,
    )
    world_to_camera = sequence.world_to_camera[frame_indices]
    joints_cam = np.einsum(
        "fij,fnj->fni", world_to_camera[:, :3, :3], joints_world
    ) + world_to_camera[:, None, :3, 3]
    z = np.maximum(joints_cam[..., 2], 1e-6)
    points_2d = np.stack(
        [
            sequence.intrinsics[0, 0] * joints_cam[..., 0] / z + sequence.intrinsics[0, 2],
            sequence.intrinsics[1, 1] * joints_cam[..., 1] / z + sequence.intrinsics[1, 2],
        ],
        axis=-1,
    )
    rows: list[np.ndarray] = []
    for points, geometry in zip(points_2d, geometries, strict=True):
        points = points.copy()
        x1, y1, _, _ = geometry.crop_xyxy
        pad_left, pad_top, _, _ = geometry.pad_xyxy
        sx, sy = geometry.scale_xy
        points[:, 0] = (points[:, 0] - float(x1)) * float(sx) + float(pad_left)
        points[:, 1] = (points[:, 1] - float(y1)) * float(sy) + float(pad_top)
        rows.append(points)
    return torch.from_numpy(np.stack(rows, axis=0)).unsqueeze(0)


def match_emdb_person_by_2d(
    pred_joints_cam: torch.Tensor,
    confidence: torch.Tensor,
    intrinsics: torch.Tensor,
    gt_keypoints: torch.Tensor,
    conf_threshold: float,
    iou_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the single EMDB GT person using Human3R's projected-joint rule."""
    z = pred_joints_cam[..., 2].clamp(min=1e-6)
    x = intrinsics[:, :, None, None, 0, 0] * pred_joints_cam[..., 0] / z
    x = x + intrinsics[:, :, None, None, 0, 2]
    y = intrinsics[:, :, None, None, 1, 1] * pred_joints_cam[..., 1] / z
    y = y + intrinsics[:, :, None, None, 1, 2]
    pred_2d = torch.stack([x, y], dim=-1)
    distance = torch.linalg.norm(pred_2d - gt_keypoints[:, :, None], dim=(-1, -2))

    pred_min, pred_max = pred_2d.amin(dim=-2), pred_2d.amax(dim=-2)
    gt_min, gt_max = gt_keypoints.amin(dim=-2), gt_keypoints.amax(dim=-2)
    inter_min = torch.maximum(pred_min, gt_min[:, :, None])
    inter_max = torch.minimum(pred_max, gt_max[:, :, None])
    inter_wh = (inter_max - inter_min + 1.0).clamp(min=0.0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    pred_wh = (pred_max - pred_min + 1.0).clamp(min=0.0)
    gt_wh = (gt_max - gt_min + 1.0).clamp(min=0.0)
    union = pred_wh[..., 0] * pred_wh[..., 1]
    union = union + gt_wh[:, :, None, 0] * gt_wh[:, :, None, 1] - inter
    iou = inter / union.clamp(min=1e-8)

    candidate = (
        (confidence >= float(conf_threshold))
        & torch.isfinite(distance)
        & torch.isfinite(pred_joints_cam).all(dim=(-1, -2))
        & (pred_joints_cam[..., 2] > 1e-6).all(dim=-1)
        & (iou >= float(iou_threshold))
    )
    cost = torch.where(candidate, distance, torch.full_like(distance, float("inf")))
    return cost.argmin(dim=-1), candidate.any(dim=-1)


def predicted_camera_to_world(
    predictions: dict[str, torch.Tensor],
    image_size_hw: tuple[int, int],
    camera_translation_scale: torch.Tensor,
) -> torch.Tensor:
    pose_enc = predictions.get("pose_enc")
    if not isinstance(pose_enc, torch.Tensor):
        raise RuntimeError("Inference output is missing pose_enc")
    if not isinstance(camera_translation_scale, torch.Tensor):
        raise RuntimeError("Camera export requires an explicit metric scale")
    world_to_camera, _ = encoding_to_camera(
        pose_enc.detach().float(), image_size_hw=image_size_hw, build_intrinsics=False
    )
    batch, frames = world_to_camera.shape[:2]
    homogeneous = torch.eye(4, device=world_to_camera.device, dtype=world_to_camera.dtype)
    homogeneous = homogeneous.reshape(1, 1, 4, 4).repeat(batch, frames, 1, 1)
    homogeneous[:, :, :3] = world_to_camera
    scale = camera_translation_scale.to(dtype=homogeneous.dtype).reshape(batch, frames, 1)
    homogeneous[:, :, :3, 3] = homogeneous[:, :, :3, 3] * scale
    return torch.linalg.inv(homogeneous)


def camera_joints_to_world(joints_cam: torch.Tensor, t_c2w: torch.Tensor) -> torch.Tensor:
    if joints_cam.shape[:2] != t_c2w.shape[:2]:
        raise ValueError(
            f"Camera/joint frame mismatch: joints={tuple(joints_cam.shape)} T={tuple(t_c2w.shape)}"
        )
    return torch.einsum(
        "bsij,bsnj->bsni", t_c2w[:, :, :3, :3], joints_cam
    ) + t_c2w[:, :, None, :3, 3]


if __name__ == "__main__":
    main()
