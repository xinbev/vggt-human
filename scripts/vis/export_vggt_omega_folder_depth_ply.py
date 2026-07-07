#!/usr/bin/env python
"""Export native VGGT-Omega depth point clouds for a folder of frames.

This script intentionally instantiates the pure camera/depth VGGT-Omega model
(`VGGTOmega()` defaults to enable_smpl=False) and does not build or call any
SMPL/Human pipeline components.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.tracking.io import iter_image_files  # noqa: E402
from vggt_omega.training.config import load_yaml_config, require_path  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    frames_dir = resolve_project_path(args.frames_dir)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = resolve_checkpoint(args)
    frame_paths = select_frames(iter_image_files(frames_dir), args)
    if not frame_paths:
        raise RuntimeError(f"No image frames found under {frames_dir}")

    model = VGGTOmega().to(device).eval()
    load_checkpoint(model, checkpoint, device=device, strict=args.strict_load)

    manifest: dict[str, Any] = {
        "purpose": "Native VGGT-Omega camera/depth export without SMPL.",
        "frames_dir": str(frames_dir),
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "resize_mode": args.resize_mode,
        "image_resolution": int(args.image_resolution),
        "patch_size": int(args.patch_size),
        "sequence_length": int(args.sequence_length),
        "effective_sequence_policy": "all_selected_frames" if int(args.sequence_length) <= 0 else "fixed_chunks",
        "coordinate_frame": args.coordinate_frame,
        "depth_point_stride": int(args.depth_point_stride),
        "min_depth_conf": float(args.min_depth_conf),
        "max_depth": float(args.max_depth),
        "frames": [],
    }

    chunks = list(chunk_paths(frame_paths, int(args.sequence_length)))
    for chunk_index, chunk in enumerate(chunks):
        records = process_chunk(
            chunk=chunk,
            model=model,
            output_dir=output_dir,
            chunk_index=chunk_index,
            args=args,
            device=device,
        )
        manifest["frames"].extend(records)
        if args.log_interval > 0:
            sequence_length = len(frame_paths) if int(args.sequence_length) <= 0 else int(args.sequence_length)
            done = min((chunk_index + 1) * sequence_length, len(frame_paths))
            if done == len(frame_paths) or (chunk_index + 1) % int(args.log_interval) == 0:
                print(f"[export] {done}/{len(frame_paths)} frames", flush=True)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "manifest": str(manifest_path), "num_frames": len(manifest["frames"])}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", required=True, help="Folder containing RGB frames.")
    parser.add_argument("--checkpoint", default="", help="Native VGGT-Omega checkpoint. Defaults to configs/path.yaml checkpoints.vggt_baseline.")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--output-dir", default="outputs/vis/vggt_omega_folder_depth_ply")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resize-mode", default="balanced", choices=["balanced", "max_size"])
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=0,
        help="Number of frames inferred together. <=0 means one VGGT forward over all selected frames.",
    )
    parser.add_argument("--coordinate-frame", default="world", choices=["world", "camera"])
    parser.add_argument("--depth-point-stride", type=int, default=2, help="Subsample depth pixels before writing PLY.")
    parser.add_argument("--max-depth", type=float, default=30.0, help="Discard points deeper than this value; <=0 disables this filter.")
    parser.add_argument("--min-depth-conf", type=float, default=0.0, help="Discard points with depth_conf below this value; <=0 disables this filter.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means export all selected frames.")
    parser.add_argument("--strict-load", action="store_true", help="Require checkpoint keys to exactly match pure VGGTOmega().")
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return resolve_project_path(args.checkpoint)
    config = load_yaml_config(resolve_project_path(args.path_config))
    return resolve_project_path(require_path(config, "checkpoints.vggt_baseline", allow_empty=False))


def select_frames(paths: list[Path], args: argparse.Namespace) -> list[Path]:
    start = max(0, int(args.start_index))
    stride = max(1, int(args.frame_stride))
    selected = paths[start::stride]
    if int(args.max_frames) > 0:
        selected = selected[: int(args.max_frames)]
    return selected


def chunk_paths(paths: list[Path], chunk_size: int) -> list[list[Path]]:
    if chunk_size <= 0:
        return [paths]
    return [paths[idx : idx + chunk_size] for idx in range(0, len(paths), chunk_size)]


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device, strict: bool) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    print(
        f"[checkpoint] loaded {checkpoint_path} strict={strict} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if missing:
        print("[checkpoint] first missing keys:", list(missing)[:8], flush=True)
    if unexpected:
        print("[checkpoint] first unexpected keys:", list(unexpected)[:8], flush=True)


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "module"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state_dict or a dict containing one")

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean_key = str(key)
        for prefix in ("module.", "model."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        state_dict[clean_key] = value
    if not state_dict:
        raise ValueError("No tensor weights found in checkpoint")
    return state_dict


def process_chunk(
    chunk: list[Path],
    model: VGGTOmega,
    output_dir: Path,
    chunk_index: int,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    images = load_and_preprocess_images(
        [str(path) for path in chunk],
        mode=args.resize_mode,
        image_resolution=int(args.image_resolution),
        patch_size=int(args.patch_size),
    ).to(device)

    with torch.inference_mode():
        predictions = model(images)
        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"],
            image_size_hw=predictions["images"].shape[-2:],
            build_intrinsics=True,
        )

    depth = predictions["depth"].detach().float().cpu()
    depth_conf = predictions.get("depth_conf")
    depth_conf_cpu = depth_conf.detach().float().cpu() if isinstance(depth_conf, torch.Tensor) else None
    rgb = predictions["images"].detach().float().cpu()
    extrinsics_cpu = extrinsics.detach().float().cpu()
    intrinsics_cpu = intrinsics.detach().float().cpu()

    records: list[dict[str, Any]] = []
    for local_index, image_path in enumerate(chunk):
        record = export_frame_point_cloud(
            image_path=image_path,
            output_dir=output_dir,
            chunk_index=chunk_index,
            local_index=local_index,
            depth=select_sequence_tensor(depth, local_index),
            depth_conf=select_sequence_tensor(depth_conf_cpu, local_index) if depth_conf_cpu is not None else None,
            rgb=select_sequence_tensor(rgb, local_index),
            extrinsic=extrinsics_cpu[0, local_index].numpy(),
            intrinsic=intrinsics_cpu[0, local_index].numpy(),
            args=args,
        )
        records.append(record)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records


def select_sequence_tensor(tensor: torch.Tensor, index: int) -> torch.Tensor:
    if tensor.ndim >= 2 and tensor.shape[0] == 1:
        return tensor[0, index]
    return tensor[index]


def export_frame_point_cloud(
    image_path: Path,
    output_dir: Path,
    chunk_index: int,
    local_index: int,
    depth: torch.Tensor,
    depth_conf: torch.Tensor | None,
    rgb: torch.Tensor,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    depth_np = tensor_to_depth_hw(depth)
    rgb_np = tensor_to_rgb_hw3(rgb, depth_np.shape)
    conf_np = tensor_to_depth_hw(depth_conf) if depth_conf is not None else None

    points = unproject_depth(depth_np, intrinsic)
    if args.coordinate_frame == "world":
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]
        points = np.einsum("ij,hwj->hwi", rotation.T, points - translation[None, None, :])

    stride = max(1, int(args.depth_point_stride))
    points = points[::stride, ::stride]
    colors = rgb_np[::stride, ::stride]
    depth_sampled = depth_np[::stride, ::stride]
    conf_sampled = conf_np[::stride, ::stride] if conf_np is not None else None

    valid = np.isfinite(points).all(axis=-1) & np.isfinite(depth_sampled) & (depth_sampled > 0)
    if float(args.max_depth) > 0:
        valid &= depth_sampled <= float(args.max_depth)
    if conf_sampled is not None and float(args.min_depth_conf) > 0:
        valid &= np.isfinite(conf_sampled) & (conf_sampled >= float(args.min_depth_conf))

    vertices = points[valid].astype(np.float32, copy=False)
    vertex_colors = colors[valid].astype(np.uint8, copy=False)

    stem = f"{chunk_index:04d}_{local_index:03d}_{image_path.stem}"
    ply_path = output_dir / f"{stem}_depth_{args.coordinate_frame}.ply"
    write_ply_vertices(ply_path, vertices, vertex_colors)

    return {
        "image": str(image_path),
        "ply": str(ply_path),
        "model_hw": [int(depth_np.shape[0]), int(depth_np.shape[1])],
        "point_count": int(vertices.shape[0]),
        "coordinate_frame": args.coordinate_frame,
        "chunk_index": int(chunk_index),
        "chunk_local_index": int(local_index),
    }


def tensor_to_depth_hw(tensor: torch.Tensor | None) -> np.ndarray:
    if tensor is None:
        raise ValueError("Depth tensor is required")
    array = tensor.detach().float().cpu().numpy()
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected depth/conf tensor with shape [H,W] or [H,W,1], got {array.shape}")
    return array.astype(np.float32, copy=False)


def tensor_to_rgb_hw3(tensor: torch.Tensor, depth_hw: tuple[int, int]) -> np.ndarray:
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected RGB tensor [3,H,W], got {tuple(tensor.shape)}")
    rgb = tensor.detach().float().cpu()
    if tuple(rgb.shape[-2:]) != tuple(depth_hw):
        rgb = F.interpolate(rgb[None], size=depth_hw, mode="bilinear", align_corners=False)[0]
    array = rgb.permute(1, 2, 0).clamp(0.0, 1.0).numpy()
    return (array * 255.0 + 0.5).astype(np.uint8)


def unproject_depth(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    z = depth.astype(np.float32, copy=False)
    x = (xx - cx) / max(fx, 1e-8) * z
    y = (yy - cy) / max(fy, 1e-8) * z
    return np.stack([x, y, z], axis=-1)


def write_ply_vertices(path: Path, vertices: np.ndarray, colors: np.ndarray) -> None:
    if vertices.shape[0] != colors.shape[0]:
        raise ValueError("vertices and colors must have the same length")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {vertices.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")
        for vertex, color in zip(vertices, colors):
            file.write(
                f"{float(vertex[0]):.6f} {float(vertex[1]):.6f} {float(vertex[2]):.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


if __name__ == "__main__":
    main()
