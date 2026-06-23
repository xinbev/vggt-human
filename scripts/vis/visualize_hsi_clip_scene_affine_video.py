#!/usr/bin/env python
"""Create a side-by-side GIF for clip-level HSI scene affine modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Compatibility patch for old chumpy on Python 3.11+.
import inspect
from collections import namedtuple

if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = getargspec

if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "complex"):
    np.complex = complex

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_smpl import build_model  # noqa: E402
from scripts.vis.visualize_hsi_clip_scene_affine import (  # noqa: E402
    build_affine_modes,
    clip_median_affine,
    compute_depth_metrics,
    ema_affine,
    load_clip_box_priors,
    load_clip_images,
    load_gt_depth,
    select_sequence_frames,
    temporal_affine_stats,
    to_float_list,
)
from scripts.vis.visualize_smpl_inference import (  # noqa: E402
    add_projected_gt_smpl_joints,
    add_projected_smpl_joints,
    collect_predictions,
    draw_predictions,
    export_prediction_gt_ply,
    export_scene_ply,
    load_config,
    load_training_checkpoint,
    load_vggt_baseline_for_camera,
    make_noisy_gt_box_prior,
)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir).expanduser()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args)
    config["model"]["enable_camera"] = True
    config["model"]["enable_depth"] = True
    config["model"]["enable_hsi_refine"] = True
    model = build_model(config).to(device)
    load_vggt_baseline_for_camera(model, config, device)
    load_training_checkpoint(model, Path(args.checkpoint).expanduser(), device)
    model.eval()

    input_size = int(config["data"].get("image_size", args.image_size))
    frame_paths = select_sequence_frames(Path(args.image).expanduser(), int(args.num_frames), int(args.stride))
    image_tensor, orig_images = load_clip_images(frame_paths, input_size)
    prior_boxes, prior_mask = load_clip_box_priors(frame_paths, config, args, device) if args.use_gt_box_prior else (None, None)
    if prior_boxes is not None and prior_mask is not None:
        prior_boxes, prior_mask = make_noisy_gt_box_prior(
            prior_boxes,
            prior_mask,
            args.gt_box_prior_center_noise,
            args.gt_box_prior_size_noise,
            args.gt_box_prior_drop_prob,
        )

    with torch.no_grad():
        predictions = model(
            image_tensor.to(device),
            smpl_query_boxes=prior_boxes,
            smpl_query_boxes_mask=prior_mask,
        )

    if "hsi_scene_scale" not in predictions or "hsi_scene_depth_bias" not in predictions:
        raise ValueError("Checkpoint/model did not produce HSI scene affine outputs")

    scale_source = predictions.get("hsi_frame_scene_scale", predictions["hsi_scene_scale"])
    bias_source = predictions.get("hsi_frame_scene_depth_bias", predictions["hsi_scene_depth_bias"])
    scale = scale_source[0, :, 0].detach().float().cpu()
    bias = bias_source[0, :, 0].detach().float().cpu()
    clip_scale, clip_bias = clip_median_affine(scale, bias)
    ema_scale, ema_bias = ema_affine(scale, bias, alpha=float(args.ema_alpha))
    affine_modes = build_affine_modes(scale, bias, clip_scale, clip_bias, ema_scale, ema_bias)
    depth_metrics = compute_depth_metrics(predictions, frame_paths, affine_modes, args)
    ply_frame_indices = selected_ply_frame_indices(args, frame_paths)

    raw_depth = canonical_depth_numpy(predictions["depth"].detach().float().cpu())[0]
    gt_depths = [load_gt_depth(path, raw_depth.shape[-2], raw_depth.shape[-1]) for path in frame_paths]
    depth_range = choose_depth_range(raw_depth, gt_depths, float(args.depth_max_m))
    error_range = choose_error_range(raw_depth, gt_depths, affine_modes, float(args.depth_max_m))

    gif_frames: list[Image.Image] = []
    frame_files: list[str] = []
    ply_files: list[str] = []
    per_frame_stats = []
    for frame_idx, frame_path in enumerate(frame_paths):
        frame_pred_raw = slice_predictions(predictions, frame_idx)
        frame_pred = frame_pred_raw
        if args.use_hsi_refined:
            frame_pred = hsi_as_primary_smpl(frame_pred)
        results = collect_predictions(frame_pred, orig_images[frame_idx].size, args.conf_threshold, args.top_k)
        if args.draw_smpl_joints:
            add_projected_smpl_joints(results, frame_pred, config, args, orig_images[frame_idx].size, input_size, device)
        if args.draw_gt_smpl_joints:
            add_projected_gt_smpl_joints(results, frame_path, frame_pred, config, args, orig_images[frame_idx].size, input_size, device)

        rgb_overlay_path = frames_dir / f"{frame_idx:04d}_{frame_path.stem}_rgb_overlay.png"
        draw_predictions(orig_images[frame_idx], results, rgb_overlay_path)
        rgb_overlay = Image.open(rgb_overlay_path).convert("RGB")

        board, stats = make_frame_board(
            rgb_overlay=rgb_overlay,
            raw_depth=raw_depth[frame_idx],
            gt_depth=gt_depths[frame_idx],
            affine_modes=affine_modes,
            frame_idx=frame_idx,
            frame_name=frame_path.stem,
            depth_range=depth_range,
            error_range=error_range,
            panel_size=int(args.panel_size),
            depth_max_m=float(args.depth_max_m),
            include_error_panels=bool(args.include_error_panels),
        )
        frame_out = frames_dir / f"{frame_idx:04d}_{frame_path.stem}_affine_compare.png"
        board.save(frame_out)
        frame_files.append(str(frame_out))
        gif_frames.append(board)
        per_frame_stats.append(stats)
        if frame_idx in ply_frame_indices:
            frame_ply_dir = output_dir / "ply" / f"{frame_idx:04d}_{frame_path.stem}"
            frame_ply_dir.mkdir(parents=True, exist_ok=True)
            frame_image_tensor = image_tensor[:, frame_idx : frame_idx + 1].contiguous()
            frame_ply_files = export_prediction_gt_ply(
                results,
                frame_path,
                frame_pred_raw,
                config,
                args,
                frame_ply_dir,
                device,
            )
            scene_export = export_scene_ply(
                results,
                frame_image_tensor,
                frame_path,
                frame_pred_raw,
                config,
                args,
                frame_ply_dir,
                input_size,
                device,
            )
            frame_ply_files.extend(scene_export.get("ply_files", []))
            ply_files.extend(frame_ply_files)
            stats["ply_files"] = frame_ply_files

    gif_path = output_dir / "hsi_clip_scene_affine_compare.gif"
    if gif_frames:
        duration_ms = int(round(1000.0 / max(float(args.fps), 1e-6)))
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )

    summary = {
        "checkpoint": str(args.checkpoint),
        "image_start": str(Path(args.image).expanduser()),
        "num_frames": len(frame_paths),
        "stride": int(args.stride),
        "fps": float(args.fps),
        "frame_paths": [str(path) for path in frame_paths],
        "gif": str(gif_path),
        "frame_files": frame_files,
        "ply_frame_indices": sorted(ply_frame_indices),
        "ply_files": ply_files,
        "affine": {
            "per_frame_scale": to_float_list(scale),
            "per_frame_bias": to_float_list(bias),
            "clip_median_scale": float(clip_scale.detach().cpu()),
            "clip_median_bias": float(clip_bias.detach().cpu()),
            "ema_alpha": float(args.ema_alpha),
            "ema_scale": to_float_list(ema_scale),
            "ema_bias": to_float_list(ema_bias),
        },
        "temporal_stats": {name: temporal_affine_stats(values["scale"], values["bias"]) for name, values in affine_modes.items()},
        "depth_metrics": depth_metrics,
        "per_frame_stats": per_frame_stats,
    }
    out_json = output_dir / "hsi_clip_scene_affine_video_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_human_summary(summary)
    print(json.dumps({"output_gif": str(gif_path), "output_json": str(out_json), "num_frames": len(frame_files)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create side-by-side GIF for HSI clip scene affine modes")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_hsi_refine.yaml")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--output-dir", default="outputs/vis/hsi_clip_scene_affine_video")
    parser.add_argument("--device", default="")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--num-frames", type=int, default=27)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--panel-size", type=int, default=256)
    parser.add_argument("--ema-alpha", type=float, default=0.25)
    parser.add_argument("--conf-threshold", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--draw-smpl-joints", action="store_true")
    parser.add_argument("--draw-gt-smpl-joints", action="store_true")
    parser.add_argument("--use-gt-box-prior", action="store_true")
    parser.add_argument("--gt-box-prior-center-noise", type=float, default=0.0)
    parser.add_argument("--gt-box-prior-size-noise", type=float, default=0.0)
    parser.add_argument("--gt-box-prior-drop-prob", type=float, default=0.0)
    parser.add_argument("--smpl-model-dir", default="")
    parser.add_argument("--use-hsi-refined", action="store_true")
    parser.add_argument("--depth-max-m", type=float, default=30.0)
    parser.add_argument("--include-error-panels", action="store_true")
    parser.add_argument("--export-selected-ply", action="store_true", help="Export PLY files for selected clip frames")
    parser.add_argument(
        "--ply-frame-indices",
        default="",
        help="Comma-separated 0-based clip frame indices for --export-selected-ply, e.g. 7,20",
    )
    parser.add_argument(
        "--ply-frame-stems",
        default="",
        help="Comma-separated image stems for --export-selected-ply, e.g. seq_000000_0035,seq_000000_0100",
    )
    parser.add_argument("--export-ply", action="store_true", help="Compatibility flag used by shared PLY helpers")
    parser.add_argument("--export-scene-ply", action="store_true", help="Compatibility flag used by shared PLY helpers")
    parser.add_argument("--align-scene-to-smpl", action="store_true")
    parser.add_argument("--align-min-anchor-pixels", type=int, default=64)
    parser.add_argument("--align-scale-min", type=float, default=0.25)
    parser.add_argument("--align-scale-max", type=float, default=20.0)
    parser.add_argument("--align-anchor-stride", type=int, default=8)
    parser.add_argument("--align-use-gt-smpl-anchors", action="store_true")
    parser.add_argument("--ply-top-k", type=int, default=3)
    parser.add_argument("--export-hsi-comparison", action="store_true")
    parser.add_argument("--export-pre-refine-comparison", action="store_true")
    parser.add_argument("--export-translation-debug-json", action="store_true")
    parser.add_argument("--hsi-align-scene", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def slice_predictions(predictions: dict[str, torch.Tensor], frame_idx: int) -> dict[str, torch.Tensor]:
    sliced = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[0] == 1 and value.shape[1] > frame_idx:
            sliced[key] = value[:, frame_idx : frame_idx + 1].contiguous()
        else:
            sliced[key] = value
    return sliced


def hsi_as_primary_smpl(predictions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = dict(predictions)
    mapping = {
        "hsi_refined_pred_poses": "pred_poses",
        "hsi_refined_pred_betas": "pred_betas",
        "hsi_refined_pred_transl_cam": "pred_transl_cam",
        "hsi_refined_pred_pose_6d": "pred_pose_6d",
    }
    for src, dst in mapping.items():
        if src in out:
            out[dst] = out[src]
    return out


def selected_ply_frame_indices(args: argparse.Namespace, frame_paths: list[Path]) -> set[int]:
    if not bool(args.export_selected_ply):
        return set()
    selected: set[int] = set()
    if str(args.ply_frame_indices).strip():
        for item in str(args.ply_frame_indices).split(","):
            item = item.strip()
            if not item:
                continue
            idx = int(item)
            if idx < 0 or idx >= len(frame_paths):
                raise ValueError(f"PLY frame index out of range: {idx}, valid=[0,{len(frame_paths) - 1}]")
            selected.add(idx)
    stems = {path.stem: idx for idx, path in enumerate(frame_paths)}
    if str(args.ply_frame_stems).strip():
        for item in str(args.ply_frame_stems).split(","):
            stem = item.strip()
            if not stem:
                continue
            if stem not in stems:
                raise ValueError(f"PLY frame stem not found in selected clip: {stem}")
            selected.add(stems[stem])
    if not selected:
        raise ValueError("--export-selected-ply requires --ply-frame-indices or --ply-frame-stems")
    return selected


def canonical_depth_numpy(depth: torch.Tensor) -> np.ndarray:
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 4:
        return depth.numpy()
    raise ValueError(f"Unsupported depth shape: {tuple(depth.shape)}")


def make_frame_board(
    rgb_overlay: Image.Image,
    raw_depth: np.ndarray,
    gt_depth: np.ndarray | None,
    affine_modes: dict[str, dict[str, torch.Tensor]],
    frame_idx: int,
    frame_name: str,
    depth_range: tuple[float, float],
    error_range: tuple[float, float],
    panel_size: int,
    depth_max_m: float,
    include_error_panels: bool,
) -> tuple[Image.Image, dict[str, Any]]:
    panels: list[tuple[str, Image.Image]] = [(f"RGB + HSI SMPL | {frame_idx:03d}", rgb_overlay)]
    stats: dict[str, Any] = {"frame_idx": int(frame_idx), "frame": frame_name, "modes": {}}
    valid = np.isfinite(raw_depth) & (raw_depth > 1e-6)
    if gt_depth is not None:
        valid &= np.isfinite(gt_depth) & (gt_depth > 1e-6)
        if depth_max_m > 0:
            valid &= gt_depth <= depth_max_m

    for mode_name in ("per_frame", "clip_median", "ema"):
        mode = affine_modes[mode_name]
        scale = float(mode["scale"][frame_idx].detach().cpu())
        bias = float(mode["bias"][frame_idx].detach().cpu())
        aligned = raw_depth * scale + bias
        label = f"{mode_name} depth\ns={scale:.3f} b={bias:.3f}"
        if gt_depth is not None and valid.any():
            abs_err = np.abs(aligned[valid] - gt_depth[valid])
            stats["modes"][mode_name] = {
                "scale": scale,
                "bias": bias,
                "depth_l1_median": float(np.median(abs_err)),
                "depth_l1_mean": float(abs_err.mean()),
            }
            label += f"\nmed err={np.median(abs_err):.3f}m"
        else:
            stats["modes"][mode_name] = {"scale": scale, "bias": bias}
        panels.append((label, colorize(aligned, depth_range, np.isfinite(aligned))))

    if include_error_panels and gt_depth is not None and valid.any():
        for mode_name in ("per_frame", "clip_median", "ema"):
            mode = affine_modes[mode_name]
            aligned = raw_depth * float(mode["scale"][frame_idx].detach().cpu()) + float(mode["bias"][frame_idx].detach().cpu())
            err = np.abs(aligned - gt_depth)
            median = stats["modes"][mode_name].get("depth_l1_median")
            panels.append((f"{mode_name} error\nmed={median:.3f}m", colorize(err, error_range, valid, heat=True)))
    return make_board(panels, panel_size=panel_size), stats


def colorize(values: np.ndarray, value_range: tuple[float, float], valid: np.ndarray, heat: bool = False) -> Image.Image:
    lo, hi = value_range
    norm = (values.astype(np.float32) - lo) / max(hi - lo, 1e-6)
    norm = np.clip(norm, 0.0, 1.0)
    if heat:
        rgb = np.stack([255 * norm, 255 * np.sqrt(norm), 40 * (1.0 - norm)], axis=-1)
    else:
        rgb = np.stack([255 * norm, 255 * (1.0 - np.abs(norm - 0.5) * 2.0), 255 * (1.0 - norm)], axis=-1)
    rgb[~valid] = np.array([0, 0, 0], dtype=np.float32)
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def make_board(panels: list[tuple[str, Image.Image]], panel_size: int) -> Image.Image:
    label_h = 46
    cols = min(4, len(panels))
    rows = int(np.ceil(len(panels) / cols))
    board = Image.new("RGB", (cols * panel_size, rows * (panel_size + label_h)), (18, 18, 18))
    draw = ImageDraw.Draw(board)
    font = ImageFont.load_default()
    for idx, (label, image) in enumerate(panels):
        x = (idx % cols) * panel_size
        y = (idx // cols) * (panel_size + label_h)
        image = image.resize((panel_size, panel_size), Image.BILINEAR)
        board.paste(image, (x, y + label_h))
        for line_idx, line in enumerate(str(label).splitlines()[:3]):
            draw.text((x + 6, y + 6 + 13 * line_idx), line, fill=(242, 242, 242), font=font)
    return board


def choose_depth_range(raw_depth: np.ndarray, gt_depths: list[np.ndarray | None], depth_max_m: float) -> tuple[float, float]:
    values = []
    values.append(raw_depth.reshape(-1))
    for gt in gt_depths:
        if gt is not None:
            valid = np.isfinite(gt) & (gt > 1e-6)
            if depth_max_m > 0:
                valid &= gt <= depth_max_m
            values.append(gt[valid])
    return robust_range(np.concatenate([v[np.isfinite(v)] for v in values if v.size > 0]))


def choose_error_range(
    raw_depth: np.ndarray,
    gt_depths: list[np.ndarray | None],
    affine_modes: dict[str, dict[str, torch.Tensor]],
    depth_max_m: float,
) -> tuple[float, float]:
    errors = []
    for frame_idx, gt in enumerate(gt_depths):
        if gt is None:
            continue
        valid = np.isfinite(gt) & (gt > 1e-6)
        if depth_max_m > 0:
            valid &= gt <= depth_max_m
        if not valid.any():
            continue
        for mode in affine_modes.values():
            aligned = raw_depth[frame_idx] * float(mode["scale"][frame_idx].detach().cpu()) + float(mode["bias"][frame_idx].detach().cpu())
            errors.append(np.abs(aligned[valid] - gt[valid]))
    if not errors:
        return (0.0, 1.0)
    arr = np.concatenate(errors)
    hi = float(np.percentile(arr[np.isfinite(arr)], 95)) if np.isfinite(arr).any() else 1.0
    return (0.0, max(hi, 1e-6))


def robust_range(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0)
    lo = float(np.percentile(values, 2))
    hi = float(np.percentile(values, 98))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def print_human_summary(summary: dict[str, Any]) -> None:
    print("========== HSI clip scene affine GIF ==========")
    print(f"frames: {summary['num_frames']} fps={summary['fps']}")
    print(f"gif: {summary['gif']}")
    for name, stats in summary["temporal_stats"].items():
        print(
            f"{name:12s} "
            f"log_delta={stats['log_scale_abs_delta']:.6f} "
            f"scale_range={stats['scale_range']:.6f} "
            f"bias_delta={stats['bias_abs_delta_m']:.6f} "
            f"bias_range={stats['bias_range_m']:.6f}"
        )
    depth_metrics = summary.get("depth_metrics", {})
    if depth_metrics.get("valid_gt_depth_frames", 0):
        for name in ("raw", "per_frame", "clip_median", "ema"):
            key = f"{name}_l1_median"
            if key in depth_metrics:
                print(f"{name:12s} depth median L1: {depth_metrics[key]}")


if __name__ == "__main__":
    main()
