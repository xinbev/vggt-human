#!/usr/bin/env python3
"""Export EMDB-2 NLF predictions using native GT intrinsics and extrinsics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.emdb2_global.data import EMDB2Sequence, load_emdb2_sequences  # noqa: E402
from benchmarks.emdb2_global.export_stride7 import (  # noqa: E402
    match_emdb_person_by_2d,
    projected_gt_smpl_keypoints,
    sequence_frame_paths,
)
from benchmarks.emdb2_global.metrics import transform_points  # noqa: E402
from scripts.vis.serve_nlf_hsi_vggt_sequence_viewer import load_sequence_images  # noqa: E402
from vggt_omega.data.geometry import transform_intrinsics  # noqa: E402
from vggt_omega.integrations.nlf_smpl_provider import NLFSMPLProvider  # noqa: E402
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.training.config import deep_update, load_yaml_config, require_path  # noqa: E402


ORACLE_PROTOCOL = "nlf_gt_intrinsics_gt_extrinsics_v1"
MATCHING_PROTOCOL = "human3r_gt_smpl2d_iou_v1"


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({args.device}) but unavailable")
    device = torch.device(args.device)
    cfg = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.inference_config))
    emdb_root = Path(args.emdb_root or require_path(cfg, "datasets.emdb_root")).expanduser()
    sequences = load_emdb2_sequences(emdb_root, args.sequence_filter)
    if args.max_sequences > 0:
        sequences = sequences[: args.max_sequences]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    model_cfg = cfg.get("model", {})
    nlf = NLFSMPLProvider(
        model_path=str(model_cfg.get("nlf_model_path", require_path(cfg, "checkpoints.nlf_smpl"))),
        third_party_root=str(model_cfg.get("nlf_third_party_root", cfg.get("third_party", {}).get("nlf_root", "third_party/nlf"))),
        model_name=str(model_cfg.get("nlf_model_name", "smpl")),
        use_detector=True,
        require_boxes=False,
        internal_batch_size=int(model_cfg.get("nlf_internal_batch_size", 128)),
        num_aug=int(model_cfg.get("nlf_num_aug", 1)),
        detector_threshold=float(model_cfg.get("nlf_detector_threshold", 0.3)),
        detector_nms_iou_threshold=float(model_cfg.get("nlf_detector_nms_iou_threshold", 0.7)),
        max_detections=int(model_cfg.get("nlf_max_detections", 150)),
    ).to(device).eval()
    smpl_root = require_path(cfg, "assets.smpl_model_dir", allow_empty=False)
    neutral_smpl = SMPLLayer(smpl_root).to(device).eval()
    gt_smpl_layers = {
        gender: SMPLLayer(smpl_root, gender=gender).to(device).eval()
        for gender in ("male", "female")
    }

    manifest: list[dict[str, Any]] = []
    for sequence in sequences:
        manifest.append(
            export_sequence(
                sequence=sequence,
                nlf=nlf,
                neutral_smpl=neutral_smpl,
                gt_smpl=gt_smpl_layers[sequence.gender],
                cfg=cfg,
                args=args,
                device=device,
                output_root=output_root,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "protocol": ORACLE_PROTOCOL,
        "description": "NLF is run with processed EMDB GT K; camera-space joints are transformed by EMDB GT T_c2w.",
        "sequence_count": len(manifest),
        "subsample_stride": int(args.subsample_stride),
        "output_dir": str(output_root),
        "sequences": manifest,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--inference-config", default="benchmarks/emdb2_global/inference_config.yaml")
    parser.add_argument("--emdb-root", default="")
    parser.add_argument("--output-dir", default="outputs/eval/emdb2_s7_nlf_gt_camera/predictions")
    parser.add_argument("--subsample-stride", type=int, default=7)
    parser.add_argument("--max-input-frames", type=int, default=500)
    parser.add_argument("--sequence-filter", default="")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-humans", type=int, default=8)
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument("--match-iou-threshold", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def export_sequence(
    sequence: EMDB2Sequence,
    nlf: NLFSMPLProvider,
    neutral_smpl: SMPLLayer,
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
            f"exceeding limit {args.max_input_frames}"
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
    gt_intrinsics = processed_gt_intrinsics(sequence, frame_indices, geometries)
    gt_intrinsics_t = torch.from_numpy(gt_intrinsics).unsqueeze(0).to(device=device)
    predictions = nlf.forward_with_intrinsics(
        images=image_sequence,
        intrinsics=gt_intrinsics_t,
        max_humans=int(args.max_humans),
    )
    gt_keypoints = projected_gt_smpl_keypoints(
        sequence=sequence,
        frame_indices=frame_indices,
        geometries=geometries,
        gt_smpl=gt_smpl,
        neutral_joint_regressor=neutral_smpl.layer.J_regressor.detach(),
        device=device,
    )
    selected_joints_cam, valid, selected_queries = select_nlf_joints_cam(
        predictions=predictions,
        smpl=neutral_smpl,
        intrinsics=gt_intrinsics_t,
        gt_keypoints=gt_keypoints,
        conf_threshold=float(args.conf_threshold),
        match_iou_threshold=float(args.match_iou_threshold),
    )
    gt_t_c2w = np.linalg.inv(sequence.world_to_camera[frame_indices]).astype(np.float32)
    joints_cam_np = selected_joints_cam[0].detach().float().cpu().numpy()
    joints_world = transform_points(gt_t_c2w, joints_cam_np).astype(np.float32, copy=False)

    path = output_root / f"{sequence.safe_name}.npz"
    np.savez_compressed(
        path,
        sequence_name=np.asarray(sequence.name),
        frame_indices=frame_indices.astype(np.int64, copy=False),
        pred_joints_cam=joints_cam_np,
        pred_T_c2w_gt=gt_t_c2w,
        pred_joints_world=joints_world,
        valid=valid[0].detach().cpu().numpy().astype(bool, copy=False),
        selected_query=selected_queries[0].detach().cpu().numpy().astype(np.int64, copy=False),
        processed_gt_intrinsics=gt_intrinsics,
        joint_format=np.asarray("smpl24"),
        units=np.asarray("m"),
        subsample_stride=np.asarray(stride, dtype=np.int64),
        matching_protocol=np.asarray(MATCHING_PROTOCOL),
        oracle_camera_protocol=np.asarray(ORACLE_PROTOCOL),
    )
    valid_count = int(valid.sum().detach().cpu())
    result = {
        "sequence": sequence.name,
        "archive": str(path),
        "original_good_frames": int(sequence.good_frame_indices.size),
        "selected_frames": int(frame_indices.size),
        "valid_predictions": valid_count,
        "coverage": valid_count / max(int(frame_indices.size), 1),
    }
    print(
        f"[export-gt-camera] {sequence.name} frames={result['selected_frames']} "
        f"valid={result['valid_predictions']}",
        flush=True,
    )
    return result


def processed_gt_intrinsics(
    sequence: EMDB2Sequence,
    frame_indices: np.ndarray,
    geometries: list[Any],
) -> np.ndarray:
    """Map native EMDB K into each processed/padded NLF image plane."""
    if len(geometries) != int(frame_indices.size):
        raise ValueError("Geometry/frame count mismatch while transforming GT intrinsics")
    output: list[np.ndarray] = []
    for geometry in geometries:
        output.append(transform_intrinsics(sequence.intrinsics, geometry).numpy())
    return np.stack(output, axis=0)


def select_nlf_joints_cam(
    predictions: dict[str, torch.Tensor],
    smpl: SMPLLayer,
    intrinsics: torch.Tensor,
    gt_keypoints: torch.Tensor,
    conf_threshold: float,
    match_iou_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pose = predictions["pred_poses"]
    betas = predictions["pred_betas"]
    transl = predictions["pred_transl_cam"]
    confidence = predictions["pred_confs"][..., 0]
    _, joints = smpl(pose.reshape(-1, pose.shape[-1]).float(), betas.reshape(-1, betas.shape[-1]).float())
    joints = joints[:, :24].reshape(*pose.shape[:3], 24, 3).to(dtype=transl.dtype)
    all_joints_cam = joints + transl[..., None, :]
    query, valid = match_emdb_person_by_2d(
        pred_joints_cam=all_joints_cam,
        confidence=confidence,
        intrinsics=intrinsics.to(dtype=all_joints_cam.dtype),
        gt_keypoints=gt_keypoints.to(device=all_joints_cam.device, dtype=all_joints_cam.dtype),
        conf_threshold=conf_threshold,
        iou_threshold=match_iou_threshold,
    )
    gather_index = query[..., None, None, None].expand(*query.shape, 1, 24, 3)
    return all_joints_cam.gather(2, gather_index).squeeze(2), valid, query


if __name__ == "__main__":
    main()
