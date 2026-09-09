#!/usr/bin/env python3
"""Overlay an exact VGGT ATE curve on the released comparison raster.

The reference raster has a linear plot area spanning x=[0, 1050] views and
y=[0, 0.2] metres.  The pixel anchors below are measured from the supplied
638x460 image.  Drawing is performed at 4x resolution and downsampled for
anti-aliasing; a JSON sidecar records the exact data-to-pixel transform.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


VIEWS = (50, 100, 150, 200, 300, 400, 500)
ATE_METRES = (
    0.0029932083135804973,
    0.0043409809895220515,
    0.004930351035055057,
    0.005909471782128872,
    0.006341055280197789,
    0.006451712633021363,
    0.006916918533247201,
)

# Inclusive plot-border anchors measured on the user's 638x460 reference.
PLOT_LEFT = 86.0
PLOT_RIGHT = 610.0
PLOT_TOP = 36.0
PLOT_BOTTOM = 403.0
X_MIN = 0.0
X_MAX = 1050.0
Y_MIN = 0.0
Y_MAX = 0.2


def data_to_pixel(x: float, y: float) -> tuple[float, float]:
    px = PLOT_LEFT + (x - X_MIN) / (X_MAX - X_MIN) * (PLOT_RIGHT - PLOT_LEFT)
    py = PLOT_BOTTOM - (y - Y_MIN) / (Y_MAX - Y_MIN) * (PLOT_BOTTOM - PLOT_TOP)
    return px, py


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/timesbd.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    image = Image.open(args.reference).convert("RGBA")
    if image.size != (638, 460):
        raise ValueError(f"Expected the supplied 638x460 reference, got {image.size}")

    scale = 4
    canvas = image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(canvas, "RGBA")
    points = [data_to_pixel(x, y) for x, y in zip(VIEWS, ATE_METRES)]
    scaled_points = [(round(x * scale), round(y * scale)) for x, y in points]
    blue = (20, 91, 180, 255)
    draw.line(scaled_points, fill=blue, width=3 * scale, joint="curve")
    radius = 4.2 * scale
    for x, y in scaled_points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=blue, outline=(255, 255, 255, 255), width=1 * scale)

    # A separate bottom-right legend avoids altering the six-entry reference
    # legend, the original top-right OOM annotation, and the new curve ending
    # at x=500 (pixel x=335.5).
    legend = (421 * scale, 364 * scale, 601 * scale, 396 * scale)
    draw.rounded_rectangle(legend, radius=4 * scale, fill=(255, 255, 255, 235), outline=(190, 190, 190, 255), width=1 * scale)
    legend_y = 380.5 * scale
    draw.line(((433 * scale, legend_y), (467 * scale, legend_y)), fill=blue, width=3 * scale)
    draw.ellipse((450 * scale - radius, legend_y - radius, 450 * scale + radius, legend_y + radius), fill=blue, outline=(255, 255, 255, 255), width=1 * scale)
    draw.text((475 * scale, 369 * scale), "VGGT-Omega", font=font(15 * scale), fill=(25, 25, 25, 255))

    final = canvas.resize(image.size, Image.Resampling.LANCZOS).convert("RGB")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.save(args.output, quality=100)
    sidecar = {
        "reference": str(args.reference),
        "output": str(args.output),
        "plot_pixel_bounds": {"left": PLOT_LEFT, "right": PLOT_RIGHT, "top": PLOT_TOP, "bottom": PLOT_BOTTOM},
        "axis_bounds": {"x": [X_MIN, X_MAX], "y_m": [Y_MIN, Y_MAX]},
        "series": "VGGT-Omega",
        "points": [
            {"views": views, "ate_m": ate, "pixel_x": pixel[0], "pixel_y": pixel[1]}
            for views, ate, pixel in zip(VIEWS, ATE_METRES, points)
        ],
    }
    args.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
