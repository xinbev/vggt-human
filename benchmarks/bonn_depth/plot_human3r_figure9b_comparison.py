#!/usr/bin/env python3
"""Replot Human3R Figure 9b with exact project Bonn curve measurements.

Reference series are digitized from the user-provided raster and are therefore
approximate. The project series is read directly from the benchmark CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FRAMES = np.asarray([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])

# Approximate values digitized from the supplied Human3R Figure 9b raster.
REFERENCE = {
    "Human3R": {
        "abs_rel": [0.08448, 0.08614, 0.09214, 0.09897, 0.10124, 0.10579, 0.10331, 0.10207, 0.10021, 0.09907],
        "delta": [0.95421, 0.95408, 0.93842, 0.90763, 0.90763, 0.90961, 0.91368, 0.91974, 0.92289, 0.92605],
        "color": "#d99b9b",
        "marker": "o",
    },
    "TTT3R": {
        "abs_rel": [0.08759, 0.08934, 0.09462, 0.10103, 0.10269, 0.10745, 0.10497, 0.10331, 0.10145, 0.09959],
        "delta": [0.94553, 0.94289, 0.93053, 0.90158, 0.90105, 0.90368, 0.90921, 0.91474, 0.91842, 0.92263],
        "color": "#dec487",
        "marker": "*",
    },
    "CUT3R": {
        "abs_rel": [0.10579, 0.10486, 0.10497, 0.10848, 0.10393, 0.10786, 0.10848, 0.10600, 0.10145, 0.09897],
        "delta": [0.89526, 0.87763, 0.87276, 0.85921, 0.87789, 0.88553, 0.88579, 0.89316, 0.89974, 0.90421],
        "color": "#aa94ba",
        "marker": "^",
    },
    "Point3R": {
        "abs_rel": [0.14428, 0.13641, 0.15028, 0.15276, 0.15276, 0.15938, 0.15545, 0.15152, 0.14697, 0.14552],
        "delta": [0.93868, 0.94974, 0.91763, 0.91342, 0.91711, 0.91211, 0.91474, 0.91816, 0.91921, 0.92079],
        "color": "#78aab8",
        "marker": "D",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curve-csv",
        type=Path,
        default=Path("outputs/eval/bonn_depth_curve/vggt_traditional_hsi_scale_curve_points.csv"),
    )
    parser.add_argument(
        "--pure-vggt-csv",
        type=Path,
        default=None,
        help="Optional pure_vggt metric curve CSV to add as a diagnostic line.",
    )
    parser.add_argument("--ours-label", default="VGGT + traditional + HSI scale (ours)")
    parser.add_argument("--left-y-min", type=float, default=0.08)
    parser.add_argument("--left-y-max", type=float, default=0.22)
    parser.add_argument("--right-y-min", type=float, default=0.62)
    parser.add_argument("--right-y-max", type=float, default=0.97)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vis/bonn_depth_curve/human3r_figure9b_with_ours.png"),
    )
    return parser.parse_args()


def read_ours(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    frames = np.asarray([int(row["requested_frames"]) for row in rows])
    abs_rel = np.asarray([float(row["Abs Rel Human3R pixel-weighted"]) for row in rows])
    delta = np.asarray([float(row["delta<1.25 Human3R pixel-weighted"]) for row in rows])
    if not np.array_equal(frames, FRAMES):
        raise ValueError(f"Expected frames {FRAMES.tolist()}, got {frames.tolist()}")
    return frames, abs_rel, delta


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf",
        "timesbd.ttf" if bold else "times.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def marker(draw: ImageDraw.ImageDraw, xy: tuple[float, float], kind: str, color: str, radius: int, width: int = 2) -> None:
    x, y = xy
    if kind == "o":
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    elif kind == "^":
        draw.polygon([(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)], fill=color)
    elif kind == "D":
        draw.polygon([(x, y - radius), (x - radius, y), (x, y + radius), (x + radius, y)], fill=color)
    elif kind == "s":
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color)
    elif kind == "*":
        points = []
        for index in range(10):
            angle = -math.pi / 2 + index * math.pi / 5
            r = radius if index % 2 == 0 else radius * 0.42
            points.append((x + math.cos(angle) * r, y + math.sin(angle) * r))
        draw.polygon(points, fill=color)
    else:
        draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=width)
        draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=width)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    ylabel: str,
    y_domain: tuple[float, float],
    y_ticks: list[float],
    reference_key: str,
    ours: np.ndarray,
    pure_vggt: np.ndarray | None = None,
) -> None:
    left, top, right, bottom = box
    x_min, x_max = 30.0, 520.0
    y_min, y_max = y_domain
    axis_font = load_font(22)
    tick_font = load_font(17)
    title_font = load_font(27)
    value_font = load_font(14, bold=True)

    def px(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * (right - left)

    def py(y: float) -> float:
        return bottom - (y - y_min) / (y_max - y_min) * (bottom - top)

    grid = "#d7d7d7"
    for x_tick in (50, 100, 200, 300, 400, 500):
        x = px(x_tick)
        draw.line((x, top, x, bottom), fill=grid, width=1)
        label = str(x_tick)
        bounds = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - (bounds[2] - bounds[0]) / 2, bottom + 10), label, font=tick_font, fill="#333333")
    for y_tick in y_ticks:
        y = py(y_tick)
        draw.line((left, y, right, y), fill=grid, width=1)
        label = f"{y_tick:.2f}"
        bounds = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((left - (bounds[2] - bounds[0]) - 12, y - (bounds[3] - bounds[1]) / 2), label, font=tick_font, fill="#333333")
    draw.rectangle(box, outline="#444444", width=2)

    title_bounds = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((left + right - (title_bounds[2] - title_bounds[0])) / 2, top - 48), title, font=title_font, fill="#202020")
    x_label = "Requested Number of Input Views"
    x_bounds = draw.textbbox((0, 0), x_label, font=axis_font)
    draw.text(((left + right - (x_bounds[2] - x_bounds[0])) / 2, bottom + 42), x_label, font=axis_font, fill="#202020")
    # Pillow has no universal rotated text API across versions; draw on a transparent layer.
    y_bounds = draw.textbbox((0, 0), ylabel, font=axis_font)
    y_layer = Image.new("RGBA", (y_bounds[2] - y_bounds[0] + 12, y_bounds[3] - y_bounds[1] + 12), (255, 255, 255, 0))
    ImageDraw.Draw(y_layer).text((6, 6), ylabel, font=axis_font, fill="#202020")
    rotated = y_layer.rotate(90, expand=True)
    draw._image.paste(rotated, (left - 88, int((top + bottom - rotated.height) / 2)), rotated)

    for name, series in REFERENCE.items():
        points = [(px(float(x)), py(float(y))) for x, y in zip(FRAMES, series[reference_key], strict=True)]
        draw.line(points, fill=series["color"], width=4, joint="curve")
        for point in points:
            marker(draw, point, str(series["marker"]), str(series["color"]), radius=7)

    if pure_vggt is not None:
        pure_color = "#555555"
        pure_points = [(px(float(x)), py(float(y))) for x, y in zip(FRAMES, pure_vggt, strict=True)]
        for start, end in zip(pure_points[:-1], pure_points[1:], strict=True):
            # Dashed line keeps raw-scale VGGT visually distinct from metric methods.
            segments = 10
            for index in range(0, segments, 2):
                a = index / segments
                b = min((index + 1) / segments, 1.0)
                draw.line(
                    (
                        start[0] + (end[0] - start[0]) * a,
                        start[1] + (end[1] - start[1]) * a,
                        start[0] + (end[0] - start[0]) * b,
                        start[1] + (end[1] - start[1]) * b,
                    ),
                    fill=pure_color,
                    width=5,
                )
        for point in pure_points:
            marker(draw, point, "s", pure_color, radius=7)

    ours_color = "#b3242a"
    ours_points = [(px(float(x)), py(float(y))) for x, y in zip(FRAMES, ours, strict=True)]
    draw.line(ours_points, fill=ours_color, width=7, joint="curve")
    for index, (point, value) in enumerate(zip(ours_points, ours, strict=True)):
        marker(draw, point, "X", ours_color, radius=9, width=5)
        text = f"{float(value):.3f}"
        bounds = draw.textbbox((0, 0), text, font=value_font)
        text_y = point[1] - 28 if index % 2 == 0 else point[1] + 12
        text_x = point[0] - (bounds[2] - bounds[0]) / 2
        draw.rounded_rectangle((text_x - 3, text_y - 2, text_x + bounds[2] - bounds[0] + 3, text_y + bounds[3] - bounds[1] + 3), radius=3, fill="#ffffff")
        draw.text((text_x, text_y), text, font=value_font, fill=ours_color)


def main() -> None:
    args = parse_args()
    frames, ours_abs, ours_delta = read_ours(args.curve_csv)
    pure_abs = None
    pure_delta = None
    if args.pure_vggt_csv is not None:
        pure_frames, pure_abs, pure_delta = read_ours(args.pure_vggt_csv)
        if not np.array_equal(pure_frames, frames):
            raise ValueError("Pure VGGT and enhanced curves use different frame positions")
    if not np.array_equal(frames, FRAMES):
        raise ValueError("Unexpected frame positions")
    canvas = Image.new("RGB", (1800, 780), "white")
    draw = ImageDraw.Draw(canvas)
    abs_candidates = [float(np.max(ours_abs)) * 1.08, args.left_y_max]
    if pure_abs is not None:
        abs_candidates.append(float(np.max(pure_abs)) * 1.08)
    abs_max = max(abs_candidates)
    abs_step = 0.05 if abs_max > 0.30 else 0.02
    abs_max = math.ceil(abs_max / abs_step) * abs_step
    abs_candidates_min = [float(np.min(ours_abs)) * 0.90, args.left_y_min]
    if pure_abs is not None:
        abs_candidates_min.append(float(np.min(pure_abs)) * 0.90)
    abs_min = min(abs_candidates_min)
    abs_ticks = np.arange(abs_min, abs_max + abs_step * 0.5, abs_step).tolist()
    delta_candidates_min = [float(np.min(ours_delta)) - 0.03, args.right_y_min]
    if pure_delta is not None:
        delta_candidates_min.append(float(np.min(pure_delta)) - 0.03)
    delta_min_value = min(delta_candidates_min)
    delta_step = 0.10 if delta_min_value < 0.50 else 0.05
    delta_min = max(0.0, math.floor(delta_min_value / delta_step) * delta_step)
    delta_ticks = np.arange(delta_min, 0.951, delta_step).tolist()
    draw_panel(draw, (135, 145, 845, 640), "Absolute Relative Error ↓", "Abs Rel", (abs_min, abs_max), abs_ticks, "abs_rel", ours_abs, pure_abs)
    draw_panel(draw, (990, 145, 1700, 640), "Threshold Accuracy (δ < 1.25) ↑", "δ < 1.25", (delta_min, args.right_y_max), delta_ticks, "delta", ours_delta, pure_delta)

    legend_font = load_font(18)
    legend_items = [(name, str(series["color"]), str(series["marker"])) for name, series in REFERENCE.items()]
    if pure_abs is not None:
        legend_items.append(("Pure VGGT (raw scale, diagnostic)", "#555555", "s"))
    legend_items.append((args.ours_label, "#b3242a", "X"))
    x = 70
    legend_y = 64
    for name, color, kind in legend_items:
        bounds = draw.textbbox((0, 0), name, font=legend_font)
        item_width = 72 + bounds[2] - bounds[0]
        if x + item_width > 1740:
            x = 70
            legend_y += 34
        draw.line((x, legend_y + 13, x + 42, legend_y + 13), fill=color, width=6 if kind == "X" else 4)
        marker(draw, (x + 21, legend_y + 13), kind, color, radius=8, width=4)
        draw.text((x + 52, legend_y), name, font=legend_font, fill="#222222")
        x += item_width

    caption_font = load_font(14)
    if "GT scale" in args.ours_label or "oracle" in args.ours_label.lower():
        ours_note = "Selected curve uses one GT-fitted scale per sequence (oracle diagnostic)."
    else:
        ours_note = "Selected curve uses no GT scale/shift alignment."
    caption = (
        "Reference curves: digitized approximately from the supplied Human3R raster.  "
        f"{ours_note} Requested N is capped for short sequences."
    )
    bounds = draw.textbbox((0, 0), caption, font=caption_font)
    draw.text(((1800 - (bounds[2] - bounds[0])) / 2, 744), caption, font=caption_font, fill="#444444")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
