#!/usr/bin/env python
"""Visualize clip-level HSI scene affine aggregation on a BEDLAM sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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
from scripts.vis.visualize_smpl_inference import (  # noqa: E402
    add_projected_gt_smpl_joints,
    add_projected_smpl_joints,
    collect_predictions,
    draw_predictions,
    export_scene_ply,
    load_config,
    load_gt_box_prior,
    load_image,
    load_training_checkpoint,
    load_vggt_baseline_for_camera,
    make_noisy_gt_box_prior,
)
from vggt_omega.data.geometry import resolve_image_size_config  # noqa: E402


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args)
    config["model"]["enable_camera"] = True
    config["model"]["enable_depth"] = True
    config["model"]["enable_hsi_refine"] = True
    model = build_model(config).to(device)
    load_vggt_baseline_for_camera(model, config, device)
    load_training_checkpoint(model, Path(args.checkpoint).expanduser(), device)
    model.eval()

    _, input_resolution = resolve_image_size_config(config.get("data", {}), args.image_size)
    frame_paths = select_sequence_frames(Path(args.image).expanduser(), int(args.num_frames), int(args.stride))
    image_tensor, orig_images = load_clip_images(frame_paths, input_resolution)
    input_hw = (int(image_tensor.shape[-2]), int(image_tensor.shape[-1]))
    input_size = max(input_hw)
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
    predictions["images"] = image_tensor.to(device)

    if "hsi_scene_scale" not in predictions or "hsi_scene_depth_bias" not in predictions:
        raise ValueError("Checkpoint/model did not produce HSI scene affine outputs")

    scale_source = predictions.get("hsi_frame_scene_scale", predictions["hsi_scene_scale"])
    bias_source = predictions.get("hsi_frame_scene_depth_bias", predictions["hsi_scene_depth_bias"])
    scale = scale_source[0, :, 0].detach().float()
    bias = bias_source[0, :, 0].detach().float()
    clip_scale, clip_bias = clip_median_affine(scale, bias)
    ema_scale, ema_bias = ema_affine(scale, bias, alpha=float(args.ema_alpha))
    affine_modes = build_affine_modes(scale, bias, clip_scale, clip_bias, ema_scale, ema_bias)

    depth_metrics = compute_depth_metrics(predictions, frame_paths, affine_modes, args)
    ply_files: list[str] = []
    overlay_files: list[str] = []
    export_indices = select_export_indices(len(frame_paths), int(args.export_stride), int(args.max_export_frames))
    for frame_idx in export_indices:
        frame_pred = slice_predictions(predictions, frame_idx)
        frame_image = image_tensor[:, frame_idx : frame_idx + 1].to(device)
        frame_path = frame_paths[frame_idx]
        results = collect_predictions(frame_pred, orig_images[frame_idx].size, args.conf_threshold, args.top_k)
        if args.draw_smpl_joints:
            add_projected_smpl_joints(results, frame_pred, config, args, orig_images[frame_idx].size, input_size, device)
        if args.draw_gt_smpl_joints:
            add_projected_gt_smpl_joints(results, frame_path, frame_pred, config, args, orig_images[frame_idx].size, input_size, device)
        if args.draw_overlays:
            overlay_path = output_dir / "overlays" / f"{frame_path.stem}_clip_affine_overlay.jpg"
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            draw_predictions(orig_images[frame_idx], results, overlay_path)
            overlay_files.append(str(overlay_path))
        if args.export_scene_ply:
            for mode_name in args.export_affine_modes:
                mode_scale, mode_bias = affine_for_frame(affine_modes, mode_name, frame_idx, device=device, dtype=scale.dtype)
                mode_pred = override_frame_affine(frame_pred, mode_scale, mode_bias)
                mode_dir = output_dir / "ply" / mode_name
                scene_export = export_scene_ply(results, frame_image, frame_path, mode_pred, config, args, mode_dir, input_size, device)
                ply_files.extend(scene_export.get("ply_files", []))

    summary = {
        "checkpoint": str(args.checkpoint),
        "image_start": str(Path(args.image).expanduser()),
        "num_frames": len(frame_paths),
        "stride": int(args.stride),
        "frame_paths": [str(path) for path in frame_paths],
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
        "exported_frame_indices": export_indices,
        "overlay_files": overlay_files,
        "ply_files": ply_files,
    }
    out_json = output_dir / "hsi_clip_scene_affine_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_human_summary(summary)
    print(json.dumps({"output_json": str(out_json), "num_frames": len(frame_paths), "num_ply_files": len(ply_files)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize clip-level HSI scene affine aggregation")
    parser.add_argument("--image", required=True, help="Start RGB image path in a BEDLAM sequence")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--train-config", default="configs/train_smpl_hsi_refine.yaml")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--output-dir", default="outputs/vis/hsi_clip_scene_affine")
    parser.add_argument("--device", default="")
    parser.add_argument("--image-size", type=int, default=0, help="Legacy explicit geometry override; default uses data.image_resolution or 512")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--ema-alpha", type=float, default=0.25)
    parser.add_argument("--conf-threshold", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--ply-top-k", type=int, default=3)
    parser.add_argument("--export-scene-ply", action="store_true")
    parser.add_argument("--export-affine-modes", nargs="+", default=["per_frame", "clip_median"], choices=["per_frame", "clip_median", "ema"])
    parser.add_argument("--export-stride", type=int, default=4)
    parser.add_argument("--max-export-frames", type=int, default=4)
    parser.add_argument("--draw-overlays", action="store_true")
    parser.add_argument("--draw-smpl-joints", action="store_true")
    parser.add_argument("--draw-gt-smpl-joints", action="store_true")
    parser.add_argument("--use-gt-box-prior", action="store_true")
    parser.add_argument("--gt-box-prior-center-noise", type=float, default=0.0)
    parser.add_argument("--gt-box-prior-size-noise", type=float, default=0.0)
    parser.add_argument("--gt-box-prior-drop-prob", type=float, default=0.0)
    parser.add_argument("--smpl-model-dir", default="")
    parser.add_argument("--use-hsi-refined", action="store_true")
    parser.add_argument("--hsi-align-scene", action="store_true", default=True)
    parser.add_argument("--align-scene-to-smpl", action="store_true")
    parser.add_argument("--align-min-anchor-pixels", type=int, default=64)
    parser.add_argument("--align-scale-min", type=float, default=0.25)
    parser.add_argument("--align-scale-max", type=float, default=20.0)
    parser.add_argument("--align-anchor-stride", type=int, default=8)
    parser.add_argument("--align-use-gt-smpl-anchors", action="store_true")
    parser.add_argument("--depth-max-m", type=float, default=30.0)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def select_sequence_frames(start_image: Path, num_frames: int, stride: int) -> list[Path]:
    if not start_image.is_file():
        raise FileNotFoundError(f"Start image not found: {start_image}")
    if start_image.parent.name != "rgb":
        raise ValueError(f"--image must point to a BEDLAM rgb frame, got: {start_image}")
    frames = sorted(path for path in start_image.parent.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    try:
        start_idx = frames.index(start_image.resolve())
    except ValueError:
        resolved = start_image.resolve()
        frames = [path.resolve() for path in frames]
        start_idx = frames.index(resolved)
    selected = []
    for offset in range(int(num_frames)):
        idx = start_idx + offset * int(stride)
        if idx >= len(frames):
            break
        selected.append(frames[idx])
    if not selected:
        raise RuntimeError(f"No frames selected from {start_image}")
    return selected


def load_clip_images(frame_paths: list[Path], input_size: int) -> tuple[torch.Tensor, list[Image.Image]]:
    tensors = []
    originals = []
    for path in frame_paths:
        tensor, image = load_image(path, input_size)
        tensors.append(tensor[:, 0])
        originals.append(image)
    return torch.stack(tensors, dim=1), originals


def load_clip_box_priors(
    frame_paths: list[Path],
    config: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    boxes = []
    masks = []
    for path in frame_paths:
        frame_boxes, frame_mask = load_gt_box_prior(path, config, args, device)
        boxes.append(frame_boxes[:, 0])
        masks.append(frame_mask[:, 0])
    return torch.stack(boxes, dim=1), torch.stack(masks, dim=1)


def clip_median_affine(scale: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    log_scale = torch.log(scale.clamp(min=1e-6))
    return torch.exp(log_scale.median()), bias.median()


def ema_affine(scale: torch.Tensor, bias: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    log_scale = torch.log(scale.clamp(min=1e-6))
    ema_log = []
    ema_bias = []
    prev_log = log_scale[0]
    prev_bias = bias[0]
    for idx in range(scale.numel()):
        if idx > 0:
            prev_log = alpha * log_scale[idx] + (1.0 - alpha) * prev_log
            prev_bias = alpha * bias[idx] + (1.0 - alpha) * prev_bias
        ema_log.append(prev_log)
        ema_bias.append(prev_bias)
    return torch.exp(torch.stack(ema_log, dim=0)), torch.stack(ema_bias, dim=0)


def build_affine_modes(
    scale: torch.Tensor,
    bias: torch.Tensor,
    clip_scale: torch.Tensor,
    clip_bias: torch.Tensor,
    ema_scale: torch.Tensor,
    ema_bias: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "per_frame": {"scale": scale, "bias": bias},
        "clip_median": {"scale": torch.full_like(scale, clip_scale), "bias": torch.full_like(bias, clip_bias)},
        "ema": {"scale": ema_scale, "bias": ema_bias},
    }


def slice_predictions(predictions: dict[str, torch.Tensor], frame_idx: int) -> dict[str, torch.Tensor]:
    sliced = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[0] == 1 and value.shape[1] > frame_idx:
            sliced[key] = value[:, frame_idx : frame_idx + 1].contiguous()
        else:
            sliced[key] = value
    return sliced


def affine_for_frame(
    affine_modes: dict[str, dict[str, torch.Tensor]],
    mode_name: str,
    frame_idx: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    mode = affine_modes[mode_name]
    return mode["scale"][frame_idx].to(device=device, dtype=dtype), mode["bias"][frame_idx].to(device=device, dtype=dtype)


def override_frame_affine(frame_pred: dict[str, torch.Tensor], scale: torch.Tensor, bias: torch.Tensor) -> dict[str, torch.Tensor]:
    out = dict(frame_pred)
    out["hsi_scene_scale"] = scale.reshape(1, 1, 1)
    out["hsi_scene_depth_bias"] = bias.reshape(1, 1, 1)
    return out


def select_export_indices(num_frames: int, export_stride: int, max_export_frames: int) -> list[int]:
    stride = max(int(export_stride), 1)
    indices = list(range(0, int(num_frames), stride))
    if (num_frames - 1) not in indices:
        indices.append(num_frames - 1)
    return indices[: max(int(max_export_frames), 0)]


def compute_depth_metrics(
    predictions: dict[str, torch.Tensor],
    frame_paths: list[Path],
    affine_modes: dict[str, dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    depth = canonical_depth(predictions["depth"].detach().float().cpu())
    metrics: dict[str, list[float]] = {f"{name}_l1_mean": [] for name in affine_modes}
    metrics.update({f"{name}_l1_median": [] for name in affine_modes})
    metrics["raw_l1_mean"] = []
    metrics["raw_l1_median"] = []
    valid_frames = 0
    for frame_idx, path in enumerate(frame_paths):
        gt = load_gt_depth(path, depth.shape[-2], depth.shape[-1])
        if gt is None:
            continue
        pred_depth = depth[0, frame_idx]
        valid = np.isfinite(gt) & (gt > 1e-6) & np.isfinite(pred_depth)
        if float(args.depth_max_m) > 0:
            valid &= gt <= float(args.depth_max_m)
        if not valid.any():
            continue
        valid_frames += 1
        raw_abs = np.abs(pred_depth[valid] - gt[valid])
        metrics["raw_l1_mean"].append(float(raw_abs.mean()))
        metrics["raw_l1_median"].append(float(np.median(raw_abs)))
        for mode_name, mode in affine_modes.items():
            scale = float(mode["scale"][frame_idx].detach().cpu())
            bias = float(mode["bias"][frame_idx].detach().cpu())
            aligned = pred_depth * scale + bias
            abs_err = np.abs(aligned[valid] - gt[valid])
            metrics[f"{mode_name}_l1_mean"].append(float(abs_err.mean()))
            metrics[f"{mode_name}_l1_median"].append(float(np.median(abs_err)))
    return {"valid_gt_depth_frames": valid_frames, **{key: describe(values) for key, values in metrics.items()}}


def canonical_depth(depth: torch.Tensor) -> np.ndarray:
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 4:
        return depth.numpy()
    raise ValueError(f"Unsupported depth shape for metrics: {tuple(depth.shape)}")


def load_gt_depth(image_path: Path, height: int, width: int) -> np.ndarray | None:
    depth_path = image_path.parent.parent / "depth" / f"{image_path.stem}.npy"
    if not depth_path.is_file():
        return None
    depth = np.load(depth_path).astype(np.float32).squeeze()
    if depth.ndim != 2:
        return None
    image = Image.fromarray(depth, mode="F").resize((int(width), int(height)), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def temporal_affine_stats(scale: torch.Tensor, bias: torch.Tensor) -> dict[str, float]:
    log_scale = torch.log(scale.detach().float().clamp(min=1e-6))
    out = {
        "log_scale_abs_delta": float(torch.abs(log_scale[1:] - log_scale[:-1]).mean().cpu()) if scale.numel() > 1 else 0.0,
        "log_scale_seq_abs": float(torch.abs(log_scale - log_scale.median()).mean().cpu()),
        "scale_range": float((scale.detach().float().max() - scale.detach().float().min()).cpu()),
        "bias_abs_delta_m": float(torch.abs(bias[1:] - bias[:-1]).mean().cpu()) if bias.numel() > 1 else 0.0,
        "bias_seq_abs_m": float(torch.abs(bias.detach().float() - bias.detach().float().median()).mean().cpu()),
        "bias_range_m": float((bias.detach().float().max() - bias.detach().float().min()).cpu()),
    }
    return out


def describe(values: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def to_float_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().reshape(-1)]


def print_human_summary(summary: dict[str, Any]) -> None:
    print("========== HSI clip scene affine ==========")
    print(f"frames: {summary['num_frames']}")
    affine = summary["affine"]
    print(f"clip median scale/bias: {affine['clip_median_scale']:.6f} / {affine['clip_median_bias']:.6f}")
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
