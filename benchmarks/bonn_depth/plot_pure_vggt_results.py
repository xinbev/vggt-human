#!/usr/bin/env python3
"""Plot pure VGGT raw/GT-scaled results alongside the Human3R-style curves."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from plot_human3r_figure9b_comparison import REFERENCE, FRAMES, load_font, marker


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    frames = np.asarray([int(row["requested_frames"]) for row in rows])
    abs_rel = np.asarray([float(row["Abs Rel Human3R pixel-weighted"]) for row in rows])
    delta = np.asarray([float(row["delta<1.25 Human3R pixel-weighted"]) for row in rows])
    if not np.array_equal(frames, FRAMES):
        raise ValueError(f"Expected frame positions {FRAMES.tolist()}, got {frames.tolist()} in {path}")
    return frames, abs_rel, delta


def plot_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    ylabel: str,
    domain: tuple[float, float],
    ticks: list[float],
    key: str,
    raw: np.ndarray,
    scaled: np.ndarray,
    enhanced: np.ndarray,
) -> None:
    left, top, right, bottom = box
    x_min, x_max = 30.0, 520.0
    y_min, y_max = domain
    tick_font = load_font(16)
    title_font = load_font(25)
    axis_font = load_font(20)
    value_font = load_font(13, bold=True)

    def px(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * (right - left)

    def py(y: float) -> float:
        return bottom - (y - y_min) / (y_max - y_min) * (bottom - top)

    grid = "#d7d7d7"
    for xt in (50, 100, 200, 300, 400, 500):
        x = px(xt)
        draw.line((x, top, x, bottom), fill=grid, width=1)
        label = str(xt)
        bb = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - (bb[2] - bb[0]) / 2, bottom + 8), label, font=tick_font, fill="#333333")
    for yt in ticks:
        y = py(yt)
        draw.line((left, y, right, y), fill=grid, width=1)
        label = f"{yt:.2f}"
        bb = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((left - (bb[2] - bb[0]) - 10, y - (bb[3] - bb[1]) / 2), label, font=tick_font, fill="#333333")
    draw.rectangle(box, outline="#444444", width=2)
    bb = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((left + right - (bb[2] - bb[0])) / 2, top - 42), title, font=title_font, fill="#222222")
    xlabel = "Requested Number of Input Views"
    bb = draw.textbbox((0, 0), xlabel, font=axis_font)
    draw.text(((left + right - (bb[2] - bb[0])) / 2, bottom + 37), xlabel, font=axis_font, fill="#222222")
    bb = draw.textbbox((0, 0), ylabel, font=axis_font)
    layer = Image.new("RGBA", (bb[2] - bb[0] + 12, bb[3] - bb[1] + 12), (255, 255, 255, 0))
    ImageDraw.Draw(layer).text((6, 6), ylabel, font=axis_font, fill="#222222")
    rotated = layer.rotate(90, expand=True)
    draw._image.paste(rotated, (left - 78, int((top + bottom - rotated.height) / 2)), rotated)

    # Reference curves remain approximate digitizations from the supplied image.
    for series in REFERENCE.values():
        points = [(px(float(x)), py(float(y))) for x, y in zip(FRAMES, series[key], strict=True)]
        draw.line(points, fill=series["color"], width=3, joint="curve")
        for point in points:
            marker(draw, point, str(series["marker"]), str(series["color"]), radius=6)

    lines = [
        (raw, "#4b4b4b", "s", "raw"),
        (scaled, "#16835b", "o", "gt-scale"),
        (enhanced, "#b3242a", "X", "hsi"),
    ]
    for values, color, mark, label in lines:
        points = [(px(float(x)), py(float(y))) for x, y in zip(FRAMES, values, strict=True)]
        width = 5 if label != "raw" else 4
        draw.line(points, fill=color, width=width, joint="curve")
        for idx, (point, value) in enumerate(zip(points, values, strict=True)):
            marker(draw, point, mark, color, radius=8, width=4)
            # Labels on raw metric are intentionally omitted in the crowded lower panel.
            if label != "raw":
                text = f"{float(value):.3f}"
                bb = draw.textbbox((0, 0), text, font=value_font)
                ty = point[1] - 25 if idx % 2 == 0 else point[1] + 10
                tx = point[0] - (bb[2] - bb[0]) / 2
                draw.rounded_rectangle((tx - 2, ty - 1, tx + bb[2] - bb[0] + 2, ty + bb[3] - bb[1] + 2), radius=2, fill="#ffffff")
                draw.text((tx, ty), text, font=value_font, fill=color)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, default=Path("outputs/eval/bonn_depth_pure_vggt/metric/pure_vggt_curve_points.csv"))
    parser.add_argument("--scaled-csv", type=Path, default=Path("outputs/eval/bonn_depth_pure_vggt/scale/pure_vggt_scale_curve_points.csv"))
    parser.add_argument("--enhanced-csv", type=Path, default=Path("outputs/eval/bonn_depth_curve/vggt_traditional_hsi_scale_curve_points.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/vis/bonn_depth_curve/pure_vggt_raw_vs_gt_scale.png"))
    args = parser.parse_args()
    frames, raw_abs, raw_delta = read_curve(args.raw_csv)
    _, scaled_abs, scaled_delta = read_curve(args.scaled_csv)
    _, enhanced_abs, enhanced_delta = read_curve(args.enhanced_csv)

    canvas = Image.new("RGB", (1900, 900), "white")
    draw = ImageDraw.Draw(canvas)
    # Dynamic axes are required because raw VGGT is non-metric and has Abs Rel ~0.57, delta ~0.005.
    plot_panel(draw, (145, 175, 895, 730), "Absolute Relative Error ↓", "Abs Rel", (0.0, 0.65), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "abs_rel", raw_abs, scaled_abs, enhanced_abs)
    plot_panel(draw, (1035, 175, 1785, 730), "Threshold Accuracy (δ < 1.25) ↑", "δ < 1.25", (0.0, 1.0), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "delta", raw_delta, scaled_delta, enhanced_delta)

    legend_font = load_font(17)
    legend = [
        ("Pure VGGT raw metric", "#4b4b4b", "s"),
        ("Pure VGGT + GT scale (oracle diagnostic)", "#16835b", "o"),
        ("VGGT + traditional + HSI scale", "#b3242a", "X"),
        ("Human3R reference curves (digitized)", "#777777", "-"),
    ]
    x = 90
    for label, color, mark in legend:
        draw.line((x, 95, x + 44, 95), fill=color, width=5)
        if mark != "-":
            marker(draw, (x + 22, 95), mark, color, radius=8, width=4)
        bb = draw.textbbox((0, 0), label, font=legend_font)
        draw.text((x + 55, 82), label, font=legend_font, fill="#222222")
        x += 70 + bb[2] - bb[0]
    caption = "Exact project CSV values; no GT scale/shift for raw metric or HSI metric. GT-scale curve uses one fitted scale per sequence; reference curves are approximate digitizations."
    cap_font = load_font(14)
    bb = draw.textbbox((0, 0), caption, font=cap_font)
    draw.text(((1900 - (bb[2] - bb[0])) / 2, 842), caption, font=cap_font, fill="#444444")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
