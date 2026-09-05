#!/usr/bin/env python3
"""Export Bonn depth predictions for the two UniSH comparison stages.

This script is intended for the Linux server. It writes one ``.npy`` depth
map per selected Bonn frame under ``<output-root>/<sequence>/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from scripts.train.train_smpl import (  # noqa: E402
    apply_overrides,
    build_model,
    load_yaml_config,
    make_state_dict_loadable,
)
from scripts.vis.visualize_smpl_inference import estimate_scene_to_smpl_scale  # noqa: E402
from vggt_omega.models.smpl_layer import SMPLLayer  # noqa: E402
from vggt_omega.training.config import deep_update, require_path  # noqa: E402

SEQUENCES = ("balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--stage", choices=["pure_vggt", "vggt_traditional_hsi_scale"], required=True)
    parser.add_argument("--sequence", choices=SEQUENCES, action="append", default=None)
    parser.add_argument("--start-frame", type=int, default=30)
    parser.add_argument("--num-frames", type=int, default=110)
    parser.add_argument("--chunk-size", type=int, default=25, help="Frames per forward pass; <=0 means one pass")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--resize-mode", choices=["balanced", "max_size"], default="balanced")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="", help="Pure VGGT checkpoint or baseline checkpoint")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--hsi-config", default="configs/infer_smpl_hsi_v3_trstr_spatial.yaml")
    parser.add_argument("--stage2-checkpoint", default="", help="Stage2 HSI/scene-align checkpoint")
    parser.add_argument("--scale-checkpoint", default="", help="HSI coarse-residual scale checkpoint")
    parser.add_argument("--smpl-model-dir", default="", help="SMPL model directory; defaults to config")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else (ROOT / value).resolve()


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state dict")
    state = {}
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            state[str(key).removeprefix("module.")] = value
    if not state:
        raise ValueError("No tensor weights found in checkpoint")
    return state


def load_pure_model(checkpoint: Path, device: torch.device) -> VGGTOmega:
    model = VGGTOmega().to(device).eval()
    state = extract_state_dict(torch.load(checkpoint, map_location="cpu"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[pure] checkpoint={checkpoint} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return model


def load_hsi_model(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], SMPLLayer]:
    config = deep_update(load_yaml_config(resolve(args.path_config)), load_yaml_config(resolve(args.hsi_config)))
    config = apply_overrides(config, [])
    model_cfg = config.setdefault("model", {})
    # This benchmark isolates the traditional scale + HSI scene-scale branch.
    model_cfg["enable_hsi_refine"] = True
    model_cfg["enable_hsi_human_scene_align"] = False
    model_cfg["enable_hsi_translation_refine_v4"] = False
    model_cfg["enable_hsi_contact_refine"] = False
    model_cfg["enable_hsi_grounding"] = False
    model_cfg["enable_hsi_foot_contact_intent"] = False
    model_cfg["enable_hsi_trstr"] = False
    model_cfg["enable_smpl"] = True
    model_cfg["enable_depth"] = True
    model_cfg["enable_camera"] = True
    config.setdefault("data", {})["image_resolution"] = int(args.image_resolution)
    model = build_model(config).to(device).eval()

    baseline_path = Path(args.checkpoint).expanduser() if args.checkpoint else resolve(require_path(config, "checkpoints.vggt_baseline"))
    baseline_state = extract_state_dict(torch.load(baseline_path, map_location="cpu"))
    baseline_state, report = make_state_dict_loadable(baseline_state, model.state_dict(), adapt_query_tensors=False)
    missing, unexpected = model.load_state_dict(baseline_state, strict=False)
    print(f"[hsi] baseline={baseline_path} missing={len(missing)} unexpected={len(unexpected)} skipped={len(report['skipped'])}", flush=True)

    for checkpoint_text in (args.stage2_checkpoint, args.scale_checkpoint):
        if not checkpoint_text:
            continue
        checkpoint = resolve(checkpoint_text)
        payload = torch.load(checkpoint, map_location="cpu")
        state = extract_state_dict(payload)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[hsi] overlay={checkpoint} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    smpl_dir = resolve(args.smpl_model_dir) if args.smpl_model_dir else resolve(require_path(config, "assets.smpl_model_dir"))
    return model, config, SMPLLayer(smpl_dir).to(device).eval()


def frame_paths(dataset_root: Path, sequence: str, start: int, count: int) -> list[Path]:
    paths = sorted((dataset_root / f"rgbd_bonn_{sequence}" / "rgb").glob("*.png"))[start : start + count]
    if len(paths) != count:
        raise ValueError(f"{sequence}: expected {count} RGB frames after [{start}:{start + count}], found {len(paths)}")
    return paths


def tensor_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 5 and depth.shape[-1] == 1:
        return depth[..., 0]
    if depth.ndim == 4:
        return depth
    raise ValueError(f"Unexpected depth shape: {tuple(depth.shape)}")


def save_depths(depth: torch.Tensor, output_dir: Path, frame_list: list[Path], diagnostics: list[dict[str, Any]] | None = None) -> None:
    depth = tensor_depth(depth).detach().float().cpu()
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, image_path in enumerate(frame_list):
        np.save(output_dir / f"{index:04d}_{image_path.stem}.npy", depth[0, index].numpy().astype(np.float32))
    if diagnostics is not None:
        (output_dir / "inference_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")


def run_pure(args: argparse.Namespace, model: VGGTOmega, sequence: str, frames: list[Path], device: torch.device) -> None:
    output_dir = resolve(args.output_root) / sequence
    if output_dir.exists() and not args.overwrite and list(output_dir.glob("*.npy")):
        print(f"[skip] {sequence}: predictions already exist; use --overwrite", flush=True)
        return
    all_depth: list[torch.Tensor] = []
    chunk_size = len(frames) if args.chunk_size <= 0 else args.chunk_size
    for start in range(0, len(frames), chunk_size):
        chunk = frames[start : start + chunk_size]
        images = load_and_preprocess_images([str(p) for p in chunk], mode=args.resize_mode, image_resolution=args.image_resolution).to(device)
        with torch.inference_mode():
            predictions = model(images[None])
            if not isinstance(predictions, dict) or not isinstance(predictions.get("depth"), torch.Tensor):
                raise RuntimeError("Pure VGGT inference did not return a tensor at predictions['depth']")
            all_depth.append(tensor_depth(predictions["depth"]))
        print(f"[pure] {sequence}: {min(start + len(chunk), len(frames))}/{len(frames)}", flush=True)
    save_depths(torch.cat(all_depth, dim=1), output_dir, frames)


def run_hsi(args: argparse.Namespace, model: torch.nn.Module, config: dict[str, Any], smpl: SMPLLayer, sequence: str, frames: list[Path], device: torch.device) -> None:
    output_dir = resolve(args.output_root) / sequence
    if output_dir.exists() and not args.overwrite and list(output_dir.glob("*.npy")):
        print(f"[skip] {sequence}: predictions already exist; use --overwrite", flush=True)
        return
    chunk_size = len(frames) if args.chunk_size <= 0 else args.chunk_size
    all_final: list[torch.Tensor] = []
    diagnostics: list[dict[str, Any]] = []
    for start in range(0, len(frames), chunk_size):
        chunk = frames[start : start + chunk_size]
        images = load_and_preprocess_images([str(p) for p in chunk], mode=args.resize_mode, image_resolution=args.image_resolution).to(device)
        with torch.inference_mode():
            base = model(images[None])
        raw_5d = base["depth"]
        raw = tensor_depth(raw_5d)
        coarse_scales: list[float | None] = []
        chunk_diagnostics: list[dict[str, Any]] = []
        for frame_idx in range(len(chunk)):
            scale_info = {"scale": 1.0, "applied": False, "reason": "missing_smpl"}
            if all(key in base for key in ("pred_poses", "pred_betas", "pred_transl_cam", "pred_confs")):
                conf = base["pred_confs"][0, frame_idx, :, 0]
                query = int(torch.argmax(conf).item())
                poses = base["pred_poses"][0, frame_idx]
                betas = base["pred_betas"][0, frame_idx]
                transl = base["pred_transl_cam"][0, frame_idx]
                vertices, _ = smpl(poses, betas)
                vertices = vertices + transl[:, None, :]
                scale_info = estimate_scene_to_smpl_scale(
                    smpl_vertices=vertices[query : query + 1],
                    depth=raw[0, frame_idx],
                    pose_enc=base["pose_enc"][:, frame_idx : frame_idx + 1],
                    input_size=int(images.shape[-1]),
                    min_anchor_pixels=32,
                    scale_min=0.05,
                    scale_max=25.0,
                    anchor_stride=8,
                )
                scale_info["query"] = query
            scale = float(scale_info.get("scale", 1.0)) if bool(scale_info.get("applied", False)) else None
            coarse_scales.append(scale)
            chunk_diagnostics.append({"sequence": sequence, "frame": start + frame_idx, "image": str(chunk[frame_idx]), "coarse_scale_raw": scale, "coarse_applied": bool(scale_info.get("applied", False)), "coarse_reason": scale_info.get("reason", "unknown")})
        valid_scales = [value for value in coarse_scales if value is not None and value > 0]
        if not valid_scales:
            raise RuntimeError(f"{sequence} frames {start}:{start + len(chunk)}: traditional coarse scale failed on every frame")
        fallback_scale = float(np.exp(np.median(np.log(np.asarray(valid_scales, dtype=np.float64)))))
        filled_scales = [fallback_scale if value is None else value for value in coarse_scales]
        for item, scale in zip(chunk_diagnostics, filled_scales, strict=True):
            item["coarse_scale"] = float(scale)
            item["coarse_fallback"] = item["coarse_scale_raw"] is None
            item["coarse_fallback_scale"] = fallback_scale
        diagnostics.extend(chunk_diagnostics)
        coarse = raw_5d * raw_5d.new_tensor(filled_scales).reshape(1, len(chunk), 1, 1, 1)
        with torch.inference_mode():
            refined = model(images[None], hsi_depth_override=coarse, hsi_depth_is_metric=True)
        residual_scale = refined.get("hsi_scene_scale")
        residual_bias = refined.get("hsi_scene_depth_bias")
        if not isinstance(residual_scale, torch.Tensor) or not isinstance(residual_bias, torch.Tensor):
            raise RuntimeError("HSI model did not return hsi_scene_scale/hsi_scene_depth_bias")
        residual_scale = residual_scale.to(device=coarse.device, dtype=coarse.dtype).reshape(1, len(chunk), 1, 1, 1)
        residual_bias = residual_bias.to(device=coarse.device, dtype=coarse.dtype).reshape(1, len(chunk), 1, 1, 1)
        final = coarse * residual_scale + residual_bias
        all_final.append(tensor_depth(final))
        for idx, scale in enumerate(filled_scales):
            diagnostics[-len(chunk) + idx]["hsi_residual_scale"] = float(residual_scale[0, idx, 0, 0, 0].cpu())
            diagnostics[-len(chunk) + idx]["effective_scale"] = float(scale * residual_scale[0, idx, 0, 0, 0].cpu())
        print(f"[hsi] {sequence}: {min(start + len(chunk), len(frames))}/{len(frames)}", flush=True)
    save_depths(torch.cat(all_final, dim=1), output_dir, frames, diagnostics)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    sequences = args.sequence or list(SEQUENCES)
    checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else resolve(require_path(load_yaml_config(resolve(args.path_config)), "checkpoints.vggt_baseline"))
    if args.stage == "pure_vggt":
        model = load_pure_model(checkpoint, device)
        for sequence in sequences:
            run_pure(args, model, sequence, frame_paths(resolve(args.dataset_root), sequence, args.start_frame, args.num_frames), device)
    else:
        model, config, smpl = load_hsi_model(args, device)
        for sequence in sequences:
            run_hsi(args, model, config, smpl, sequence, frame_paths(resolve(args.dataset_root), sequence, args.start_frame, args.num_frames), device)


if __name__ == "__main__":
    main()
