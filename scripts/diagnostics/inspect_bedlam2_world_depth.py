#!/usr/bin/env python3
"""Identify the geometry stored in a BEDLAM2 WorldDepth EXR without writing data.

BEDLAM2 Movie Render Queue stores ``WorldDepth`` as an HxWx4 payload.  The
first three components may be a point vector rather than a scalar depth map.
This diagnostic tests candidate coordinate conventions against the per-frame
label intrinsics: the correct interpretation reprojects to its source pixel
coordinates with a very small error and has positive camera Z.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preprocess.prepare_bedlam2_scene import select_depth_channel  # noqa: E402


def main() -> None:
    args = parse_args()
    labels_path = Path(args.labels).expanduser()
    depth_root = Path(args.depth_root).expanduser()
    if not labels_path.is_file():
        raise FileNotFoundError(f"Label NPZ not found: {labels_path}")
    if not depth_root.is_dir():
        raise FileNotFoundError(f"Depth root not found: {depth_root}")
    labels = np.load(labels_path, allow_pickle=False)
    for key in ("imgname", "cam_int", "cam_ext"):
        if key not in labels:
            raise KeyError(f"Label NPZ misses {key!r}; available={sorted(labels.files)}")
    row, imgname = find_label_row(labels["imgname"], args.imgname)
    exr_path = depth_root / "exr_depth" / PurePosixPath(imgname).with_suffix(".exr")
    points, channel = read_vector_payload(exr_path, args.exr_channel)
    K = np.asarray(labels["cam_int"][row], dtype=np.float64)
    ext = np.asarray(labels["cam_ext"][row], dtype=np.float64)
    if K.shape != (3, 3) or ext.shape != (4, 4):
        raise ValueError(f"Invalid label matrices at row={row}: K={K.shape}, cam_ext={ext.shape}")

    pixels = sample_pixels(points.shape[:2], args.sample_stride)
    sampled_points = points[pixels[:, 1], pixels[:, 0]].astype(np.float64, copy=False)
    candidates = evaluate_candidates(sampled_points, pixels, K, ext)
    candidates.sort(key=lambda item: (item["median_reprojection_error_px"], -item["positive_z_fraction"]))
    best = candidates[0]
    report = {
        "imgname": imgname,
        "label_row": row,
        "exr_path": str(exr_path),
        "exr_channel": channel,
        "payload_shape": list(points.shape),
        "sample_stride": args.sample_stride,
        "sample_count": int(pixels.shape[0]),
        "cam_int": K.tolist(),
        "cam_ext": ext.tolist(),
        "candidates": candidates,
        "recommended": best,
        "auto_accept": bool(
            best["positive_z_fraction"] >= 0.99 and best["median_reprojection_error_px"] <= args.max_accept_error_px
        ),
        "acceptance_threshold_px": args.max_accept_error_px,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["auto_accept"]:
        raise RuntimeError(
            "No candidate passed the coordinate-consistency threshold. Do not convert depth; inspect this report and depth metadata."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate BEDLAM2 WorldDepth EXR coordinate semantics")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--depth-root", required=True)
    parser.add_argument("--imgname", default="", help="Label imgname; default is the first frame")
    parser.add_argument("--exr-channel", default="FinalImageMovieRenderQueue_WorldDepth")
    parser.add_argument("--sample-stride", type=int, default=8)
    parser.add_argument("--max-accept-error-px", type=float, default=2.0)
    args = parser.parse_args()
    if args.sample_stride <= 0:
        parser.error("--sample-stride must be positive")
    return args


def find_label_row(imgnames: np.ndarray, requested: str) -> tuple[int, str]:
    names = [normalise_name(value) for value in imgnames]
    if requested:
        wanted = normalise_name(requested)
        try:
            return names.index(wanted), wanted
        except ValueError as exc:
            raise KeyError(f"Requested imgname {wanted!r} not found in labels") from exc
    return 0, names[0]


def normalise_name(value: Any) -> str:
    name = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    return name.replace("\\", "/").lstrip("./")


def read_vector_payload(path: Path, requested_channel: str) -> tuple[np.ndarray, str]:
    try:
        import OpenEXR
    except ImportError as exc:  # pragma: no cover - server-only dependency.
        raise ImportError("OpenEXR is required in the server environment") from exc
    if not path.is_file():
        raise FileNotFoundError(f"EXR corresponding to label imgname was not found: {path}")
    exr = OpenEXR.File(str(path))
    if not exr.parts:
        raise ValueError(f"EXR has no parts: {path}")
    channels = exr.parts[0].channels
    channel = select_depth_channel(channels, requested_channel)
    payload = np.asarray(channels[channel].pixels, dtype=np.float32)
    if payload.ndim != 3 or payload.shape[-1] < 3:
        raise ValueError(f"Expected HxWx3+ WorldDepth payload from {path}, got {payload.shape}")
    return payload[..., :3], channel


def sample_pixels(hw: tuple[int, int], stride: int) -> np.ndarray:
    height, width = hw
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    # These coordinates are used for NumPy indexing before reprojection.
    return np.stack((xs.reshape(-1), ys.reshape(-1)), axis=1).astype(np.int64)


def evaluate_candidates(points: np.ndarray, pixels: np.ndarray, K: np.ndarray, ext: np.ndarray) -> list[dict[str, Any]]:
    # UE world vector -> project OpenCV convention, matching the established
    # conversion used by the old BEDLAM adapter: [x,y,z]_cv=[y,-z,x]_ue.
    ue_to_cv = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    vectors = {
        "raw_xyz": points,
        "ue_to_cv_xyz": points @ ue_to_cv.T,
    }
    candidates: list[dict[str, Any]] = []
    for vector_name, vector in vectors.items():
        # Direct tests cover a camera-space vector payload. Unit does not affect
        # reprojection here, but the reported median Z clarifies its scale.
        candidates.append(score_candidate(f"camera_{vector_name}", vector, pixels, K, raw_scale=1.0))
        for raw_scale in (1.0, 0.01):
            world = vector * raw_scale
            candidates.append(
                score_candidate(
                    f"world_{vector_name}_cam_ext_w2c_scale_{raw_scale:g}",
                    transform_points(ext, world),
                    pixels,
                    K,
                    raw_scale=raw_scale,
                )
            )
            candidates.append(
                score_candidate(
                    f"world_{vector_name}_cam_ext_c2w_scale_{raw_scale:g}",
                    transform_points(np.linalg.inv(ext), world),
                    pixels,
                    K,
                    raw_scale=raw_scale,
                )
            )
    return candidates


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def score_candidate(name: str, camera_points: np.ndarray, pixels: np.ndarray, K: np.ndarray, raw_scale: float) -> dict[str, Any]:
    finite = np.isfinite(camera_points).all(axis=1)
    z = camera_points[:, 2]
    valid = finite & (z > 1e-6)
    if not bool(valid.any()):
        return {
            "name": name,
            "raw_vector_scale": raw_scale,
            "positive_z_fraction": 0.0,
            "median_z": None,
            "median_reprojection_error_px": float("inf"),
            "p95_reprojection_error_px": float("inf"),
        }
    points_valid = camera_points[valid]
    observed = pixels[valid]
    projected_x = K[0, 0] * points_valid[:, 0] / points_valid[:, 2] + K[0, 2]
    projected_y = K[1, 1] * points_valid[:, 1] / points_valid[:, 2] + K[1, 2]
    error = np.hypot(projected_x - observed[:, 0], projected_y - observed[:, 1])
    return {
        "name": name,
        "raw_vector_scale": raw_scale,
        "positive_z_fraction": float(valid.mean()),
        "median_z": float(np.median(points_valid[:, 2])),
        "median_reprojection_error_px": float(np.median(error)),
        "p95_reprojection_error_px": float(np.percentile(error, 95)),
    }


if __name__ == "__main__":
    main()
