#!/usr/bin/env python3
"""Run VGGT-Omega on prepared TUM-Dynamics prefixes and export trajectories.

This is the project-native counterpart of Human3R's relpose inference step.
It intentionally performs one forward pass per sequence prefix so that the
ATE curve measures the effect of the requested number of input views.  No GT
pose, GT scale, or Human3R code is used during inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402


def parse_lengths(raw: str) -> tuple[int, ...]:
    lengths = tuple(dict.fromkeys(int(token.strip()) for token in raw.split(",") if token.strip()))
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("--lengths must contain positive comma-separated integers")
    return lengths


def load_state_dict(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError(f"Could not find a model state_dict in {checkpoint_path}")
    return {str(key).removeprefix("module."): value for key, value in checkpoint.items()}


def rotation_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert one 3x3 rotation matrix to a scalar-first quaternion."""

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * np.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 1e-12))
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
        elif index == 1:
            scale = 2.0 * np.sqrt(max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 1e-12))
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
        else:
            scale = 2.0 * np.sqrt(max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 1e-12))
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
    quaternion = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    return quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)


def export_trajectory(pose_encoding: torch.Tensor, output_path: Path) -> None:
    """Convert VGGT camera-from-world extrinsics to TUM camera centers."""

    if pose_encoding.ndim == 3:
        pose_encoding = pose_encoding[0]
    extrinsics, _ = encoding_to_camera(pose_encoding[None].float(), image_size_hw=(1, 1), build_intrinsics=False)
    extrinsics = extrinsics[0].detach().cpu().numpy()
    rows: list[str] = []
    for index, extrinsic in enumerate(extrinsics):
        rotation_w2c = extrinsic[:, :3]
        translation_w2c = extrinsic[:, 3]
        camera_center = -rotation_w2c.T @ translation_w2c
        rotation_c2w = rotation_w2c.T
        quaternion_wxyz = rotation_to_wxyz(rotation_c2w)
        values = [float(index), *camera_center.tolist(), *quaternion_wxyz.tolist()]
        rows.append(" ".join(f"{value:.9f}" for value in values))
    if not rows:
        raise ValueError("VGGT returned an empty camera trajectory")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path, help="Prepared root from prepare_tum_dynamics.sh")
    parser.add_argument("--checkpoint", required=True, type=Path, help="VGGT-Omega checkpoint")
    parser.add_argument("--output-parent", required=True, type=Path, help="Parent for tum_<N>_<model>/")
    parser.add_argument("--lengths", default="50,100,150,200,300,400,500,600,700,800,900,1000")
    parser.add_argument("--model-name", default="vggt")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--image-mode", choices=("balanced", "max_size"), default="balanced")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence", action="append", default=None)
    parser.add_argument("--max-sequences", type=int, default=0)
    args = parser.parse_args()
    lengths = parse_lengths(args.lengths)
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Prepared dataset root does not exist: {args.dataset_root}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"VGGT checkpoint does not exist: {args.checkpoint}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({device}) but is not available")

    model = VGGTOmega().to(device).eval()
    missing, unexpected = model.load_state_dict(load_state_dict(args.checkpoint, device), strict=False)
    if missing:
        print(f"[warning] missing checkpoint keys: {len(missing)}", flush=True)
    if unexpected:
        print(f"[warning] unexpected checkpoint keys: {len(unexpected)}", flush=True)

    sequences = sorted(path for path in args.dataset_root.iterdir() if path.is_dir())
    if args.sequence:
        requested = set(args.sequence)
        sequences = [path for path in sequences if path.name in requested]
    if args.max_sequences > 0:
        sequences = sequences[: args.max_sequences]
    if not sequences:
        raise RuntimeError("No prepared TUM sequence directories selected")

    for length in lengths:
        for sequence in sequences:
            image_dir = sequence / f"rgb_{length}"
            image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}) if image_dir.is_dir() else []
            if not image_paths:
                raise FileNotFoundError(f"No RGB frames found: {image_dir}")
            images = load_and_preprocess_images(image_paths, mode=args.image_mode, image_resolution=args.image_resolution).to(device)
            with torch.inference_mode():
                predictions = model(images)
            pose_encoding = predictions.get("pose_enc")
            if pose_encoding is None:
                raise RuntimeError("VGGT output does not contain pose_enc; camera ATE cannot be computed")
            output_path = args.output_parent / f"tum_{length}_{args.model_name}" / sequence.name / "pred_traj.txt"
            export_trajectory(pose_encoding, output_path)
            print(f"[inferred] length={length} sequence={sequence.name} frames={len(image_paths)} -> {output_path}", flush=True)


if __name__ == "__main__":
    main()

