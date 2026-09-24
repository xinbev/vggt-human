#!/usr/bin/env python3
"""Export a cached full-viewer depth map as a false-color PNG.

The input is one ``frame_XXXX.pkl`` produced by ``full_viewer_cache_io``.
No model, checkpoint, or GPU is required.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


COLORMAP_STOPS: dict[str, list[list[int]]] = {
    "turbo": [
        [48, 18, 59], [58, 82, 166], [32, 159, 181], [72, 193, 110],
        [245, 231, 65], [245, 135, 48], [180, 35, 38],
    ],
    "inferno": [
        [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
        [212, 72, 66], [245, 125, 21], [252, 255, 164],
    ],
    "magma": [
        [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129],
        [181, 54, 122], [229, 80, 100], [252, 253, 191],
    ],
    "viridis": [
        [68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98],
        [253, 231, 37],
    ],
    "teal": [[29, 74, 120], [21, 132, 160], [40, 177, 150], [139, 213, 168]],
}


def load_depth(path: Path, source: str) -> np.ndarray:
    with path.open("rb") as file:
        frame = pickle.load(file)  # noqa: S301 - trusted local cache artifact
    if not isinstance(frame, dict):
        raise TypeError(f"Expected a cached frame dict, got {type(frame).__name__}")
    key = {"raw": "raw_depth_map", "hsi": "hsi_depth_map"}[source]
    if key not in frame:
        alternatives = ", ".join(sorted(k for k in frame if "depth" in str(k).lower()))
        raise KeyError(f"Missing {key!r} in {path}; depth-like fields: {alternatives or '<none>'}")
    depth = frame[key]
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth map after squeeze, got shape {depth.shape}")
    return depth


def colorize(depth: np.ndarray, lo: float, hi: float, colormap: str) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(depth) & (depth > 1e-6)
    stops = np.asarray(COLORMAP_STOPS[colormap], dtype=np.float32)
    t = np.nan_to_num((depth - lo) / max(hi - lo, 1e-6), nan=0.0, posinf=1.0, neginf=0.0)
    t = np.clip(t, 0.0, 1.0)
    position = t * float(len(stops) - 1)
    index = np.floor(position).astype(np.int32).clip(0, len(stops) - 2)
    fraction = (position - index)[..., None]
    rgb = ((1.0 - fraction) * stops[index] + fraction * stops[index + 1]).clip(0, 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb, valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-pkl", required=True, help="Path to one cached frame_XXXX.pkl")
    parser.add_argument("--output", required=True, help="Output color PNG path")
    parser.add_argument("--source", choices=sorted(("raw", "hsi")), default="hsi")
    parser.add_argument("--colormap", choices=sorted(COLORMAP_STOPS), default="turbo")
    parser.add_argument("--percentiles", nargs=2, type=float, metavar=("LOW", "HIGH"), default=(2.0, 98.0))
    parser.add_argument("--min-depth", type=float, default=None, help="Override lower color scale bound in meters")
    parser.add_argument("--max-depth", type=float, default=None, help="Override upper color scale bound in meters")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_path = Path(args.frame_pkl).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    depth = load_depth(frame_path, args.source)
    valid_values = depth[np.isfinite(depth) & (depth > 1e-6)]
    if valid_values.size == 0:
        raise ValueError(f"No positive finite depth values found in {frame_path}")
    low_q, high_q = map(float, args.percentiles)
    if not 0.0 <= low_q < high_q <= 100.0:
        raise ValueError("--percentiles must satisfy 0 <= LOW < HIGH <= 100")
    lo = float(args.min_depth) if args.min_depth is not None else float(np.percentile(valid_values, low_q))
    hi = float(args.max_depth) if args.max_depth is not None else float(np.percentile(valid_values, high_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"Invalid color scale bounds: lo={lo}, hi={hi}")

    rgb, valid = colorize(depth, lo, hi, args.colormap)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_path)
    metadata: dict[str, Any] = {
        "frame_pkl": str(frame_path),
        "source": args.source,
        "colormap": args.colormap,
        "image_size_hw": [int(depth.shape[0]), int(depth.shape[1])],
        "valid_pixels": int(valid.sum()),
        "depth_min_valid": float(valid_values.min()),
        "depth_max_valid": float(valid_values.max()),
        "color_scale_min": lo,
        "color_scale_max": hi,
        "percentiles": [low_q, high_q],
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[depth-export] source={args.source} shape={depth.shape} valid={int(valid.sum())}")
    print(f"[depth-export] color range=[{lo:.6g}, {hi:.6g}] m")
    print(f"[depth-export] png={output_path}")
    print(f"[depth-export] metadata={metadata_path}")


if __name__ == "__main__":
    main()
