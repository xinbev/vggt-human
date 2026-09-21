#!/usr/bin/env python3
"""RICH labeled test views: VGGT + NLF, 100-frame windows, global metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.emdb2_global.export_chunk100 import make_chunk_ranges, stitch_stage_chunks
from benchmarks.emdb2_global.export_stride7 import (
    camera_joints_to_world, match_emdb_person_by_2d, predicted_camera_to_world,
)
from benchmarks.emdb2_global.metrics import (
    first_two_frame_align, global_align, joint_position_error, root_translation_error,
)
from scripts.train.train_smpl import build_model
from scripts.vis.serve_nlf_hsi_vggt_sequence_viewer import load_sequence_images
from scripts.vis.visualize_smpl_inference import (
    estimate_scene_to_smpl_scale, load_vggt_baseline_for_camera,
)
from vggt_omega.data.rich_physical_grounding import (
    RICH_PROTOCOL_HMR4D_VIEWS, RichPhysicalGroundingDataset,
)
from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.evaluation.rich_physical_grounding import canonical_depth
from vggt_omega.training.config import deep_update, load_yaml_config, require_path
from vggt_omega.utils.pose_enc import encoding_to_camera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--inference-config", default="benchmarks/emdb2_global/inference_config.yaml")
    parser.add_argument("--official-root", default="")
    parser.add_argument("--support-root", default="")
    parser.add_argument("--smpl-model-dir", default="")
    parser.add_argument("--smplx-model-dir", required=True)
    parser.add_argument("--smplx-to-smpl", required=True)
    parser.add_argument("--output-dir", default="outputs/eval/rich_global_chunk100")
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-frames-per-sequence", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--max-humans", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--min-iou", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_conversion(path: Path, device: torch.device) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Missing SMPL-X to SMPL vertex conversion: {path}")
    if path.suffix == ".pt":
        value = torch.load(path, map_location="cpu", weights_only=False)
    elif path.suffix == ".pkl":
        import pickle
        with path.open("rb") as handle:
            value = pickle.load(handle, encoding="latin1")
    else:
        raise ValueError("Conversion must be a .pt sparse matrix or .pkl matrix")
    if isinstance(value, dict):
        value = value["matrix"]
    if hasattr(value, "toarray"):
        value = value.toarray()
    matrix = torch.as_tensor(value, dtype=torch.float32).to_dense().to(device)
    if matrix.shape != (6890, 10475):
        raise ValueError(f"Expected SMPL-X -> SMPL [6890,10475], got {tuple(matrix.shape)}")
    return matrix


@torch.no_grad()
def gt_joints(record, positions, smplx_model, joint_mapping, device):
    params = record.label.get("gt_smplx_params")
    if not isinstance(params, dict):
        raise ValueError(f"{record.vid}: missing GT SMPL-X parameters")
    chunks = []
    for start in range(0, len(positions), 32):
        ids = positions[start:start + 32]
        values = {}
        for name in ("global_orient", "body_pose", "betas", "transl"):
            if name not in params:
                raise KeyError(f"{record.vid}: missing GT {name}")
            source = torch.as_tensor(params[name], dtype=torch.float32)
            if name == "betas" and (source.ndim == 1 or source.shape[0] == 1):
                values[name] = source.reshape(1, -1).expand(len(ids), -1).to(device)
            else:
                if source.shape[0] != len(record.frame_ids):
                    raise ValueError(f"{record.vid}: GT {name} length {source.shape[0]} != labeled frames {len(record.frame_ids)}")
                values[name] = source[ids].to(device)
        values["global_orient"] = values["global_orient"].reshape(len(ids), 3)
        values["body_pose"] = values["body_pose"].reshape(len(ids), 63)
        values["betas"] = values["betas"].reshape(len(ids), -1)[:, :10]
        values["transl"] = values["transl"].reshape(len(ids), 3)
        vertices = smplx_model(**values).vertices
        joints = torch.einsum("jv,bvc->bjc", joint_mapping, vertices)
        chunks.append(joints.cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def target_keypoints(record, positions, geometries, gt_world, cameras):
    scene = record.recording.split("_", 1)[0]
    camera_key = f"{scene}_{record.camera_id}"
    if camera_key not in cameras:
        raise KeyError(f"{record.vid}: no RICH calibration entry {camera_key!r}")
    world_to_camera, intrinsics = cameras[camera_key]
    extrinsic = np.asarray(world_to_camera, dtype=np.float64).reshape(4, 4)
    k = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    cam = np.einsum("ij,fnj->fni", extrinsic[:3, :3], gt_world) + extrinsic[None, None, :3, 3]
    z = cam[..., 2]
    projected = np.empty((len(positions), 24, 2), dtype=np.float32)
    projected[..., 0] = k[0, 0] * cam[..., 0] / np.maximum(z, 1e-6) + k[0, 2]
    projected[..., 1] = k[1, 1] * cam[..., 1] / np.maximum(z, 1e-6) + k[1, 2]
    points = []
    valid = []
    mask = torch.as_tensor(record.label.get("mask", torch.ones(len(record.frame_ids))), dtype=torch.bool)
    for index, (pos, geometry) in enumerate(zip(positions, geometries, strict=True)):
        if pos >= len(mask):
            raise ValueError(f"{record.vid}: RICH visibility mask shorter than labels")
        xy = projected[index].copy()
        x1, y1, _, _ = geometry.crop_xyxy
        pad_left, pad_top, _, _ = geometry.pad_xyxy
        sx, sy = geometry.scale_xy
        xy[:, 0] = (xy[:, 0] - x1) * sx + pad_left
        xy[:, 1] = (xy[:, 1] - y1) * sy + pad_top
        points.append(xy)
        valid.append(bool(mask[pos]) and bool(np.isfinite(xy).all()) and bool((z[index] > 0).all()))
    return torch.from_numpy(np.stack(points))[None], np.asarray(valid, dtype=bool)


@torch.no_grad()
def infer_chunk(record, positions, gt_world, cameras, start, end, model, smpl, regressor, cfg, args, device):
    paths = [record.image_paths[record.frame_ids[pos]] for pos in positions]
    images, geometries = load_sequence_images(
        paths, int(cfg["data"]["image_resolution"]),
        int(cfg["model"]["patch_size"]), str(cfg["data"]["resize_mode"]),
    )
    images = images[None].to(device)
    pred = model(images)
    pose, betas, transl = pred["pred_poses"], pred["pred_betas"], pred["pred_transl_cam"]
    b, frames, people = pose.shape[:3]
    if b != 1:
        raise ValueError("Expected one video per VGGT inference")
    vertices, _ = smpl(pose.reshape(-1, pose.shape[-1]).float(), betas.reshape(-1, betas.shape[-1]).float())
    joints = torch.einsum("jv,bvc->bjc", regressor, vertices)
    vertices = vertices.reshape(b, frames, people, -1, 3) + transl[..., None, :]
    joints = joints[:, :24].reshape(b, frames, people, 24, 3) + transl[..., None, :]
    depth = canonical_depth(pred["depth"])
    scales = depth.new_ones((b, frames))
    accepted = []
    for frame in range(frames):
        people_ok = pred["pred_confs"][0, frame, :, 0] >= args.confidence
        result = (
            estimate_scene_to_smpl_scale(
                vertices[0, frame, people_ok], depth[0, frame],
                pred["pose_enc"][:, frame:frame + 1],
                input_size=max(depth.shape[-2:]), min_anchor_pixels=32,
                scale_min=0.10, scale_max=25.0, anchor_stride=8,
            ) if bool(people_ok.any()) else {"applied": False}
        )
        accepted.append(bool(result["applied"]))
        if accepted[-1]:
            scales[0, frame] = result["scale"]
    if any(accepted):
        median = torch.exp(torch.log(scales[0, accepted]).median())
        scales[0, ~torch.as_tensor(accepted, device=device)] = median
    else:
        raise RuntimeError(f"{record.vid} frames {start}:{end}: no analytic coarse-scale anchors")
    shared = torch.exp(torch.log(scales.clamp(min=1e-6)).median(dim=1, keepdim=True).values).expand_as(scales)
    camera = predicted_camera_to_world(pred, images.shape[-2:], shared)
    keypoints, target_valid = target_keypoints(record, positions, geometries, gt_world, cameras)
    _, intrinsics = encoding_to_camera(pred["pose_enc"].float(), image_size_hw=images.shape[-2:], build_intrinsics=True)
    query, matched = match_emdb_person_by_2d(
        joints, pred["pred_confs"][..., 0], intrinsics,
        keypoints.to(device), args.confidence, args.min_iou,
    )
    index = query[..., None, None, None].expand(b, frames, 1, 24, 3)
    selected = joints.gather(2, index).squeeze(2)
    world = camera_joints_to_world(selected, camera)[0]
    valid = matched[0].cpu().numpy() & target_valid
    result = {
        "start": start, "end": end, "chunk_id": start,
        "valid": valid,
        "selected_query": query[0].cpu().numpy(),
        "world_by_stage": {"vggt_nlf": world.float().cpu().numpy()},
        "camera_by_stage": {"vggt_nlf": camera[0].float().cpu().numpy()},
        "stitch_summary": {}, "stitch_transform": {},
    }
    return result


@torch.no_grad()
def main():
    args = parse_args()
    if args.chunk_size != 100 or args.overlap < 2 or args.overlap >= 100:
        raise ValueError("This protocol requires chunk_size=100 and 2 <= overlap < 100")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    cfg = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.inference_config))
    cfg["data"]["max_humans"] = args.max_humans
    cfg["model"].update(
        num_smpl_queries=args.max_humans, enable_hsi_refine=False,
        enable_hsi_trstr=False, enable_hsi_human_scene_align=False,
        enable_hsi_translation_refine_v4=False, enable_hsi_contact_refine=False,
        enable_hsi_grounding=False, enable_hsi_foot_contact_intent=False,
        smpl_track_assignment_mode="none", nlf_use_detector=True,
    )
    official = args.official_root or require_path(cfg, "datasets.rich_official_root")
    support = args.support_root or require_path(cfg, "datasets.rich_hmr4d_support_root")
    cameras_path = Path(support) / "cam2params.pt"
    if not cameras_path.is_file():
        raise FileNotFoundError(f"Missing RICH GT camera parameters for 2D matching: {cameras_path}")
    cameras = torch.load(cameras_path, map_location="cpu", weights_only=False)
    dataset = RichPhysicalGroundingDataset(
        official, support, protocol=RICH_PROTOCOL_HMR4D_VIEWS,
        sequence_filters=args.sequence, max_sequences=args.max_sequences,
        max_frames_per_sequence=args.max_frames_per_sequence,
        max_humans=args.max_humans,
    )
    conversion = load_conversion(Path(args.smplx_to_smpl), device)
    import smplx
    models = {
        gender: smplx.create(
            args.smplx_model_dir, model_type="smplx", gender=gender,
            num_betas=10, use_pca=False, flat_hand_mean=True,
        ).to(device).eval()
        for gender in ("male", "female")
    }
    smpl_root = args.smpl_model_dir or require_path(cfg, "assets.smpl_model_dir")
    smpl = SMPLLayer(smpl_root).to(device).eval()
    regressor = smpl.layer.J_regressor[:24].to(device=device, dtype=torch.float32)
    joint_mapping = regressor @ conversion
    del conversion
    model = build_model(cfg).to(device).eval()
    load_vggt_baseline_for_camera(model, cfg, device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, frame_errors = [], {key: [] for key in ("W-MPJPE_mm", "WA-MPJPE_mm", "RTE_percent")}
    for record in dataset:
        positions = dataset.selected_label_positions(record)
        gender = str(record.label["gender"]).lower()
        if gender not in models:
            raise ValueError(f"{record.vid}: unsupported RICH gender {gender!r}")
        gt = gt_joints(record, positions, models[gender], joint_mapping, device)
        ranges = make_chunk_ranges(len(positions), args.chunk_size, args.overlap)
        chunks = []
        for start, end in ranges:
            chunks.append(infer_chunk(
                record, positions[start:end], gt[start:end], cameras, start, end,
                model, smpl, regressor, cfg, args, device,
            ))
            if device.type == "cuda":
                torch.cuda.empty_cache()
        stitched = stitch_stage_chunks(chunks, "vggt_nlf", len(positions), 2)
        valid = np.zeros(len(positions), dtype=bool)
        for chunk in chunks:
            valid[chunk["start"]:chunk["end"]] |= chunk["valid"]
        if valid.sum() < 2:
            raise RuntimeError(f"{record.vid}: fewer than two matched RICH frames")
        # Each 100-frame evaluation window is aligned independently. RTE uses
        # the complete prediction-only-stitched trajectory and one rigid fit.
        w = np.full(len(positions), np.nan)
        wa = np.full(len(positions), np.nan)
        for start in range(0, len(positions), 100):
            indices = np.flatnonzero(valid[start:start + 100]) + start
            if len(indices) < 2:
                continue
            w[indices] = joint_position_error(gt[indices], first_two_frame_align(gt[indices], stitched[indices])) * 1000
            wa[indices] = joint_position_error(gt[indices], global_align(gt[indices], stitched[indices])) * 1000
        eval_valid = valid & np.isfinite(w) & np.isfinite(wa)
        if eval_valid.sum() < 2:
            raise RuntimeError(f"{record.vid}: no evaluable 100-frame windows")
        rte = root_translation_error(gt[eval_valid, 0], stitched[eval_valid, 0], fixed_scale=True) * 100
        metrics = {
            "W-MPJPE_mm": float(np.mean(w[eval_valid])),
            "WA-MPJPE_mm": float(np.mean(wa[eval_valid])),
            "RTE_percent": float(np.mean(rte)),
        }
        for key, values in zip(frame_errors, (w[eval_valid], wa[eval_valid], rte)):
            frame_errors[key].append(values)
        rows.append({"sequence": record.vid, "frames": len(positions), "matched": int(eval_valid.sum()), **metrics})
        print(f"{record.vid},{len(positions)},{int(eval_valid.sum())},{metrics['W-MPJPE_mm']:.3f},{metrics['WA-MPJPE_mm']:.3f},{metrics['RTE_percent']:.5f}", flush=True)
    summary = {key: float(np.concatenate(values).mean()) for key, values in frame_errors.items()}
    with (output / "sequence_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps({
        "dataset": "RICH HMR4D labeled test camera views", "stage": "RGB-VGGT-NLF (analytic coarse gauge)",
        "inference_chunk": 100, "overlap": args.overlap, "evaluation_window": 100,
        "stitch": "prediction-only camera SE(3)", "matched_sequences": len(rows),
        "frame_weighted": summary, "sequences": rows,
    }, indent=2), encoding="utf-8")
    print(f"TOTAL,,{sum(row['matched'] for row in rows)},{summary['W-MPJPE_mm']:.3f},{summary['WA-MPJPE_mm']:.3f},{summary['RTE_percent']:.5f}", flush=True)


if __name__ == "__main__":
    main()
