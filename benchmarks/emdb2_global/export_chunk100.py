#!/usr/bin/env python3
"""Export EMDB-2 predictions with 100-frame inference windows.

Every window is sent to VGGT without temporal subsampling.  A small overlap
between neighbouring windows is used only to stitch their prediction-local
world frames with a prediction-only rigid transform.  No EMDB GT is used for
this stitching step.
"""

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
    load_emdb2_sequences,
)
from benchmarks.emdb2_global.export_stride7 import (  # noqa: E402
    camera_joints_to_world,
    decode_selected_joints_cam_stages,
    predicted_camera_to_world,
    projected_gt_smpl_keypoints,
    sequence_frame_paths,
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
from benchmarks.emdb2_global.metrics import align_points, apply_similarity  # noqa: E402


STAGE_ORDER = (
    "vggt_nlf",
    "vggt_nlf_hsi_scale",
    "vggt_nlf_hsi_scale_trstr",
)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({args.device}) but unavailable")
    if int(args.chunk_size) < 2:
        raise ValueError("chunk_size must be at least 2")
    if not 0 <= int(args.chunk_overlap) < int(args.chunk_size):
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
    if int(args.max_input_frames) < int(args.chunk_size):
        raise ValueError("max_input_frames must be >= chunk_size")

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
    smpl_root = require_path(cfg, "assets.smpl_model_dir", allow_empty=False)
    smpl = SMPLLayer(smpl_root).to(device).eval()
    gt_smpl_layers = {
        gender: SMPLLayer(smpl_root, gender=gender).to(device).eval()
        for gender in ("male", "female")
    }

    manifest: list[dict[str, Any]] = []
    for sequence in sequences:
        manifest.append(
            export_sequence(
                sequence=sequence,
                model=model,
                smpl=smpl,
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
        "protocol": "EMDB-2 chunk100 no-subsampling prediction-only-stitch",
        "checkpoint": str(Path(args.checkpoint)),
        "scale_checkpoint": str(Path(args.scale_checkpoint)),
        "chunk_size": int(args.chunk_size),
        "chunk_overlap": int(args.chunk_overlap),
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
    parser.add_argument("--inference-config", default="benchmarks/emdb2_global/inference_config.yaml")
    parser.add_argument("--emdb-root", default="")
    parser.add_argument("--output-dir", default="outputs/eval/emdb2_global_chunk100/predictions")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--chunk-overlap", type=int, default=8)
    parser.add_argument("--max-input-frames", type=int, default=100)
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
    parser.add_argument("--min-stitch-overlap-frames", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def export_sequence(
    sequence: EMDB2Sequence,
    model: torch.nn.Module,
    smpl: SMPLLayer,
    gt_smpl: SMPLLayer,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    output_root: Path,
) -> dict[str, Any]:
    # Human3R's EMDB protocol uses good frames; no further temporal sampling is
    # applied here.  The stored frame IDs remain the original EMDB frame IDs.
    frame_indices = sequence.good_frame_indices.copy()
    ranges = make_chunk_ranges(
        total=int(frame_indices.size),
        chunk_size=int(args.chunk_size),
        overlap=int(args.chunk_overlap),
    )
    chunks: list[dict[str, Any]] = []
    for chunk_id, (start, end) in enumerate(ranges):
        chunks.append(
            infer_chunk(
                sequence=sequence,
                frame_indices=frame_indices[start:end],
                chunk_id=chunk_id,
                start=start,
                end=end,
                model=model,
                smpl=smpl,
                gt_smpl=gt_smpl,
                cfg=cfg,
                args=args,
                device=device,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    stitched_by_stage = {
        "vggt_nlf": stitch_stage_chunks(
            chunks=chunks,
            stage="vggt_nlf",
            total_frames=int(frame_indices.size),
            min_overlap_frames=int(args.min_stitch_overlap_frames),
        ),
        "vggt_nlf_hsi_scale": stitch_stage_chunks(
            chunks=chunks,
            stage="vggt_nlf_hsi_scale",
            total_frames=int(frame_indices.size),
            min_overlap_frames=int(args.min_stitch_overlap_frames),
        ),
        # TRSTR changes only person translation. Reuse the HSI camera-world
        # stitch so its correction cannot influence or be hidden by stitching.
        "vggt_nlf_hsi_scale_trstr": stitch_stage_chunks(
            chunks=chunks,
            stage="vggt_nlf_hsi_scale_trstr",
            total_frames=int(frame_indices.size),
            min_overlap_frames=int(args.min_stitch_overlap_frames),
            transform_source_stage="vggt_nlf_hsi_scale",
        ),
    }

    # Overlap frames may be valid in either copy. The stitcher keeps the first
    # valid copy, so the archive mask must be an OR over inference windows.
    merged_valid = np.zeros(frame_indices.size, dtype=bool)
    for chunk in chunks:
        merged_valid[chunk["start"] : chunk["end"]] |= chunk["valid"]
    selected_query = np.full(frame_indices.size, -1, dtype=np.int64)
    for chunk in chunks:
        start, end = chunk["start"], chunk["end"]
        local_query = chunk["selected_query"]
        unset = selected_query[start:end] < 0
        selected_query[start:end][unset] = local_query[unset]

    path = output_root / f"{sequence.safe_name}.npz"
    save_kwargs: dict[str, Any] = {
        "sequence_name": np.asarray(sequence.name),
        "frame_indices": frame_indices.astype(np.int64, copy=False),
        "stage_names": np.asarray(STAGE_ORDER),
        "valid": merged_valid,
        "selected_query": selected_query,
        "joint_format": np.asarray("smpl24"),
        "units": np.asarray("m"),
        "subsample_stride": np.asarray(1, dtype=np.int64),
        "matching_protocol": np.asarray("human3r_gt_smpl2d_iou_v1"),
        "inference_chunk_size": np.asarray(int(args.chunk_size), dtype=np.int64),
        "inference_chunk_overlap": np.asarray(int(args.chunk_overlap), dtype=np.int64),
        "stitch_protocol": np.asarray("prediction_only_overlap_se3"),
        "chunk_id": chunk_ids_for_ranges(ranges, frame_indices.size),
    }
    for stage in STAGE_ORDER:
        save_kwargs[f"pred_joints_world__{stage}"] = stitched_by_stage[stage].astype(
            np.float32, copy=False
        )
    np.savez_compressed(path, **save_kwargs)

    result = {
        "sequence": sequence.name,
        "archive": str(path),
        "original_frames": int(sequence.frame_count),
        "good_frames": int(frame_indices.size),
        "inference_chunks": len(chunks),
        "chunk_size": int(args.chunk_size),
        "chunk_overlap": int(args.chunk_overlap),
        "valid_predictions": int(merged_valid.sum()),
        "coverage": float(merged_valid.mean()) if merged_valid.size else 0.0,
        "matching_protocol": "human3r_gt_smpl2d_iou_v1",
        "stitch_protocol": "prediction_only_overlap_se3",
        "stitch_transforms": {
            stage: [chunk["stitch_summary"][stage] for chunk in chunks]
            for stage in STAGE_ORDER
        },
    }
    print(
        f"[export] {sequence.name} good_frames={frame_indices.size} "
        f"chunks={len(chunks)} valid={int(merged_valid.sum())} "
        f"coverage={float(merged_valid.mean()):.4f}",
        flush=True,
    )
    return result


def infer_chunk(
    sequence: EMDB2Sequence,
    frame_indices: np.ndarray,
    chunk_id: int,
    start: int,
    end: int,
    model: torch.nn.Module,
    smpl: SMPLLayer,
    gt_smpl: SMPLLayer,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    if frame_indices.size > int(args.max_input_frames):
        raise RuntimeError(
            f"{sequence.name}: chunk {chunk_id} has {frame_indices.size} frames, "
            f"exceeding max_input_frames={args.max_input_frames}"
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
        raise RuntimeError("Chunk100 export requires hsi_coarse_scene_scale")
    shared_coarse_scale = torch.exp(
        torch.log(coarse_scale.detach().float().clamp(min=1e-6)).median(dim=1, keepdim=True).values
    ).expand_as(coarse_scale)
    coarse_t_c2w = predicted_camera_to_world(
        predictions, image_sequence.shape[-2:], camera_translation_scale=shared_coarse_scale
    )
    metric_t_c2w = predicted_camera_to_world(
        predictions, image_sequence.shape[-2:], camera_translation_scale=predictions["hsi_scene_scale"]
    )
    stage_camera = {
        "vggt_nlf": coarse_t_c2w,
        "vggt_nlf_hsi_scale": metric_t_c2w,
        "vggt_nlf_hsi_scale_trstr": metric_t_c2w,
    }
    world_by_stage = {
        stage: camera_joints_to_world(joints_cam_by_stage[stage], stage_camera[stage])[0]
        .detach()
        .float()
        .cpu()
        .numpy()
        for stage in STAGE_ORDER
    }
    valid_np = valid[0].detach().cpu().numpy().astype(bool, copy=False)
    query_np = selected_queries[0].detach().cpu().numpy().astype(np.int64, copy=False)
    result = {
        "chunk_id": int(chunk_id),
        "start": int(start),
        "end": int(end),
        "frame_indices": frame_indices.copy(),
        "valid": valid_np,
        "selected_query": query_np,
        "world_by_stage": world_by_stage,
        "camera_by_stage": {
            stage: stage_camera[stage][0].detach().float().cpu().numpy()
            for stage in STAGE_ORDER
        },
        "stitch_summary": {},
        "stitch_transform": {},
    }
    del predictions, image_sequence, images, joints_cam_by_stage, world_by_stage
    del coarse_t_c2w, metric_t_c2w, shared_coarse_scale
    return result


def stitch_stage_chunks(
    chunks: list[dict[str, Any]],
    stage: str,
    total_frames: int,
    min_overlap_frames: int,
    transform_source_stage: str | None = None,
) -> np.ndarray:
    merged = np.full((total_frames, 24, 3), np.nan, dtype=np.float64)
    merged_valid = np.zeros(total_frames, dtype=bool)
    previous: dict[str, Any] | None = None
    for chunk in chunks:
        local = np.asarray(chunk["world_by_stage"][stage], dtype=np.float64)
        transform = (1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
        overlap_start = 0
        overlap_end = 0
        if transform_source_stage is not None:
            if transform_source_stage not in chunk["stitch_transform"]:
                raise RuntimeError(
                    f"Missing source stitch transform {transform_source_stage!r} "
                    f"for chunk {chunk['chunk_id']}"
                )
            transform = chunk["stitch_transform"][transform_source_stage]
            if previous is not None:
                overlap_start = max(int(previous["start"]), int(chunk["start"]))
                overlap_end = min(int(previous["end"]), int(chunk["end"]))
        elif previous is not None:
            overlap_start = max(int(previous["start"]), int(chunk["start"]))
            overlap_end = min(int(previous["end"]), int(chunk["end"]))
            prev_local_start = overlap_start - int(previous["start"])
            curr_local_start = overlap_start - int(chunk["start"])
            overlap_len = max(overlap_end - overlap_start, 0)
            if overlap_len < int(min_overlap_frames):
                raise RuntimeError(
                    f"Cannot stitch stage={stage}: chunk {chunk['chunk_id']} has "
                    f"only {overlap_len} overlap frames; require {min_overlap_frames}"
                )
            target_camera = np.asarray(
                previous["global_camera_by_stage"][stage], dtype=np.float64
            )[prev_local_start : prev_local_start + overlap_len]
            source_camera = np.asarray(chunk["camera_by_stage"][stage], dtype=np.float64)[
                curr_local_start : curr_local_start + overlap_len
            ]
            target = camera_pose_landmarks(target_camera).reshape(-1, 3)
            source = camera_pose_landmarks(source_camera).reshape(-1, 3)
            _, rotation, translation = align_points(target, source, fixed_scale=True)
            transform = (1.0, rotation, translation)
        transformed = apply_similarity(local, *transform)
        local_camera = np.asarray(chunk["camera_by_stage"][stage], dtype=np.float64)
        transformed_camera = transform_camera_poses(local_camera, transform[1], transform[2])
        chunk.setdefault("global_camera_by_stage", {})[stage] = transformed_camera
        chunk["stitch_transform"][stage] = transform
        stitch_rmse = 0.0
        if previous is not None and overlap_end > overlap_start:
            prev_local_start = overlap_start - int(previous["start"])
            curr_local_start = overlap_start - int(chunk["start"])
            overlap_len = overlap_end - overlap_start
            target_camera = np.asarray(previous["global_camera_by_stage"][stage], dtype=np.float64)[
                prev_local_start : prev_local_start + overlap_len
            ]
            source_camera = transformed_camera[curr_local_start : curr_local_start + overlap_len]
            residual = camera_pose_landmarks(target_camera) - camera_pose_landmarks(source_camera)
            stitch_rmse = float(np.sqrt(np.mean(np.square(residual))))
        chunk["stitch_summary"][stage] = {
            "scale": float(transform[0]),
            "rotation": np.asarray(transform[1]).round(8).tolist(),
            "translation_m": np.asarray(transform[2]).round(8).tolist(),
            "overlap_frames": int(overlap_end - overlap_start),
            "camera_landmark_rmse_m": stitch_rmse,
            "used_overlap_se3": bool(previous is not None and overlap_end > overlap_start),
        }
        start, end = int(chunk["start"]), int(chunk["end"])
        valid = np.asarray(chunk["valid"], dtype=bool)
        write = valid & ~merged_valid[start:end]
        merged[start:end][write] = transformed[write]
        merged_valid[start:end][write] = True
        previous = chunk
    if not merged_valid.all():
        # Keep invalid entries finite for the archive contract; the evaluator
        # removes them with the shared valid mask before metric computation.
        merged[~merged_valid] = 0.0
    return merged.astype(np.float32, copy=False)


def make_chunk_ranges(total: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    if total <= 0:
        raise ValueError("Cannot infer an empty sequence")
    if total <= chunk_size:
        return [(0, total)]
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(start + chunk_size, total)
        ranges.append((start, end))
        if end >= total:
            break
        start = end - overlap
    return ranges


def camera_pose_landmarks(camera_to_world: np.ndarray, axis_length_m: float = 0.25) -> np.ndarray:
    """Represent camera centers and axes as non-degenerate 3D landmarks."""
    camera = np.asarray(camera_to_world, dtype=np.float64)
    if camera.ndim != 3 or camera.shape[-2:] != (4, 4):
        raise ValueError(f"Expected camera poses [F,4,4], got {camera.shape}")
    centers = camera[:, :3, 3]
    axes = camera[:, :3, :3].transpose(0, 2, 1)
    axis_points = centers[:, None, :] + float(axis_length_m) * axes
    return np.concatenate([centers[:, None, :], axis_points], axis=1)


def transform_camera_poses(
    camera_to_world: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    camera = np.asarray(camera_to_world, dtype=np.float64)
    output = camera.copy()
    output[:, :3, :3] = np.einsum("ij,fjk->fik", rotation, camera[:, :3, :3])
    output[:, :3, 3] = np.einsum("ij,fj->fi", rotation, camera[:, :3, 3]) + translation
    return output


def chunk_ids_for_ranges(ranges: list[tuple[int, int]], total: int) -> np.ndarray:
    result = np.full(total, -1, dtype=np.int64)
    for chunk_id, (start, end) in enumerate(ranges):
        unset = result[start:end] < 0
        result[start:end][unset] = int(chunk_id)
    return result


if __name__ == "__main__":
    main()
