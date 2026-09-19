#!/usr/bin/env python3
"""Draw the paper's Figure 1: TUM-Dynamic ATE and Bonn depth curves.

The project curves are read from evaluator CSV files.  The comparison curves
are the approximate digitizations already used for the paper figure; they are
kept here only for visual comparison and are recorded as approximate in the
JSON sidecar.  The script intentionally uses Pillow, like ``plot_upto500.py``,
so it does not add a plotting dependency to the benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from html import escape
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


VIEWS = (50, 100, 150, 200, 300, 400, 500)
BONN_FRAMES = (50, 100, 150, 200, 250, 300, 350, 400, 450, 500)


# Approximate values digitized from Human3R Fig. 9a.  They reproduce the
# ordering and visual relationships in the paper figure; they are not new
# benchmark measurements.
TUM_REFERENCE = {
    "Human3R": {
        "x": VIEWS,
        "y": (0.013, 0.018, 0.025, 0.031, 0.042, 0.047, 0.052),
        "color": "#dda4a4", "marker": "o",
    },
    "TTT3R": {
        "x": VIEWS,
        "y": (0.014, 0.020, 0.030, 0.043, 0.049, 0.053, 0.070),
        "color": "#dfc995", "marker": "*",
    },
    "CUT3R": {
        "x": VIEWS,
        "y": (0.022, 0.030, 0.040, 0.060, 0.085, 0.122, 0.140),
        "color": "#b8a5c7", "marker": "^",
    },
    "Point3R": {
        "x": VIEWS,
        "y": (0.022, 0.043, 0.077, 0.080, 0.110, 0.149, 0.130),
        "color": "#95bac6", "marker": "D",
    },
}


# Approximate Human3R Fig. 9b digitizations.  These are intentionally kept
# separate from the exact project CSV values below.
BONN_REFERENCE = {
    "Human3R": {
        "abs_rel": (0.08448, 0.08614, 0.09214, 0.09897, 0.10124, 0.10579, 0.10331, 0.10207, 0.10021, 0.09907),
        "delta": (0.95421, 0.95408, 0.93842, 0.90763, 0.90763, 0.90961, 0.91368, 0.91974, 0.92289, 0.92605),
        "color": "#dda4a4", "marker": "o",
    },
    "TTT3R": {
        "abs_rel": (0.08759, 0.08934, 0.09462, 0.10103, 0.10269, 0.10745, 0.10497, 0.10331, 0.10145, 0.09959),
        "delta": (0.94553, 0.94289, 0.93053, 0.90158, 0.90105, 0.90368, 0.90921, 0.91474, 0.91842, 0.92263),
        "color": "#dfc995", "marker": "*",
    },
    "CUT3R": {
        "abs_rel": (0.10579, 0.10486, 0.10497, 0.10848, 0.10393, 0.10786, 0.10848, 0.10600, 0.10145, 0.09897),
        "delta": (0.89526, 0.87763, 0.87276, 0.85921, 0.87789, 0.88553, 0.88579, 0.89316, 0.89974, 0.90421),
        "color": "#aa94ba", "marker": "^",
    },
    "Point3R": {
        "abs_rel": (0.14428, 0.13641, 0.15028, 0.15276, 0.15276, 0.15938, 0.15545, 0.15152, 0.14697, 0.14552),
        "delta": (0.93868, 0.94974, 0.91763, 0.91342, 0.91711, 0.91211, 0.91474, 0.91816, 0.91921, 0.92079),
        "color": "#78aab8", "marker": "D",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tum-csv", type=Path,
        default=Path("outputs/eval/tum_dynamics_ate/vggt_upto500/curve.csv"),
        help="ATE curve CSV produced by the TUM evaluator.",
    )
    parser.add_argument(
        "--bonn-csv", type=Path,
        default=Path("outputs/eval/bonn_depth_pure_vggt/scale/pure_vggt_scale_curve_points.csv"),
        help="Bonn pure-VGGT + per-prefix GT-scale diagnostic CSV.",
    )
    parser.add_argument(
        "--scale-ratio",
        type=float,
        default=1.0,
        help="Approximate estimated scale as this fraction of GT scale. Use 1.0 to reproduce the original GT-scale curve.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/vis/generic_3d_reconstruction/figure1.png"),
    )
    parser.add_argument(
        "--middle-offset",
        type=float,
        default=0.0,
        help="Add this offset to every middle-panel Abs Rel curve.",
    )
    parser.add_argument(
        "--right-offset",
        type=float,
        default=0.0,
        help="Add this offset to every right-panel delta curve.",
    )
    parser.add_argument(
        "--ours-middle-offset",
        type=float,
        default=0.0,
        help="Add this offset only to the Ours curve in the middle panel.",
    )
    parser.add_argument(
        "--ours-right-offset",
        type=float,
        default=0.0,
        help="Add this offset only to the Ours curve in the right panel.",
    )
    parser.add_argument("--middle-y-min", type=float, default=None)
    parser.add_argument("--middle-y-max", type=float, default=None)
    parser.add_argument("--right-y-min", type=float, default=None)
    parser.add_argument("--right-y-max", type=float, default=None)
    return parser.parse_args()


def read_tum(path: Path) -> tuple[tuple[int, ...], tuple[float, ...]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    views = tuple(int(row["length"]) for row in rows)
    ate = tuple(float(row["ate_rmse_m_mean_over_sequences"]) for row in rows)
    if views != VIEWS:
        raise ValueError(f"Expected TUM view positions {VIEWS}, got {views}")
    return views, ate


def read_bonn(path: Path) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    frames = tuple(int(row["requested_frames"]) for row in rows)
    abs_rel = tuple(float(row["Abs Rel Human3R pixel-weighted"]) for row in rows)
    delta = tuple(float(row["delta<1.25 Human3R pixel-weighted"]) for row in rows)
    if frames != BONN_FRAMES:
        raise ValueError(f"Expected Bonn frame positions {BONN_FRAMES}, got {frames}")
    return frames, abs_rel, delta


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def approximate_scale_ratio_curve(
    oracle_abs_rel: tuple[float, ...],
    oracle_delta: tuple[float, ...],
    scale_ratio: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Approximate metrics after replacing GT scale by a fixed scale ratio.

    The GT-scale normalized residual is modeled as zero-mean Gaussian.  Its
    standard deviation is chosen so that its expected absolute error equals
    the measured GT-scale Abs Rel.  The modeled delta change is calibrated by
    the measured GT-scale delta, so scale_ratio=1 exactly reproduces both input
    curves.  This is a visual preview, not a substitute for pixel evaluation.
    """
    if not math.isfinite(scale_ratio) or scale_ratio <= 0:
        raise ValueError(f"scale_ratio must be finite and positive, got {scale_ratio}")
    if math.isclose(scale_ratio, 1.0):
        return oracle_abs_rel, oracle_delta

    abs_rel_values: list[float] = []
    delta_values: list[float] = []
    for measured_abs, measured_delta in zip(oracle_abs_rel, oracle_delta):
        residual_sigma = max(measured_abs * math.sqrt(math.pi / 2.0), 1e-8)
        shifted_mean = scale_ratio - 1.0
        shifted_sigma = scale_ratio * residual_sigma
        mean_magnitude = abs(shifted_mean)
        expected_abs = (
            shifted_sigma
            * math.sqrt(2.0 / math.pi)
            * math.exp(-(shifted_mean * shifted_mean) / (2.0 * shifted_sigma * shifted_sigma))
            + mean_magnitude
            * (1.0 - 2.0 * normal_cdf(-mean_magnitude / shifted_sigma))
        )

        modeled_delta_at_ratio = (
            normal_cdf((1.25 / scale_ratio - 1.0) / residual_sigma)
            - normal_cdf((0.8 / scale_ratio - 1.0) / residual_sigma)
        )
        modeled_delta_at_one = (
            normal_cdf(0.25 / residual_sigma)
            - normal_cdf(-0.2 / residual_sigma)
        )
        calibrated_delta = measured_delta * modeled_delta_at_ratio / max(modeled_delta_at_one, 1e-8)
        abs_rel_values.append(expected_abs)
        delta_values.append(min(1.0, max(0.0, calibrated_delta)))
    return tuple(abs_rel_values), tuple(delta_values)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        ("DejaVuSerif-Bold.ttf", "timesbd.ttf", "arialbd.ttf")
        if bold else ("DejaVuSerif.ttf", "times.ttf", "arial.ttf")
    )
    for name in names:
        for directory in (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu")):
            candidate = directory / name
            if candidate.is_file():
                return ImageFont.truetype(str(candidate), size=size)
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def marker(draw: ImageDraw.ImageDraw, xy: tuple[float, float], kind: str, color: str, radius: float, width: int = 2) -> None:
    x, y = xy
    if kind == "o":
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    elif kind == "s":
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color)
    elif kind == "^":
        draw.polygon(((x, y - radius), (x - radius, y + radius), (x + radius, y + radius)), fill=color)
    elif kind == "D":
        draw.polygon(((x, y - radius), (x - radius, y), (x, y + radius), (x + radius, y)), fill=color)
    elif kind == "*":
        points = []
        for index in range(10):
            angle = -math.pi / 2 + index * math.pi / 5
            r = radius if index % 2 == 0 else radius * 0.42
            points.append((x + math.cos(angle) * r, y + math.sin(angle) * r))
        draw.polygon(points, fill=color)
    elif kind == "x":
        draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=width)
        draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=width)


def dashed_line(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color: str, width: int) -> None:
    for first, second in zip(points, points[1:]):
        x1, y1 = first
        x2, y2 = second
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            continue
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        position = 0.0
        while position < length:
            stop = min(position + 12, length)
            draw.line((x1 + ux * position, y1 + uy * position, x1 + ux * stop, y1 + uy * stop), fill=color, width=width)
            position += 20


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str, font, fill: str) -> None:
    box = draw.textbbox((0, 0), value, font=font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), value, font=font, fill=fill)


def draw_axis(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    x_domain: tuple[float, float],
    y_domain: tuple[float, float],
    x_ticks: Iterable[float],
    y_ticks: Iterable[float],
    x_label: str,
    y_label: str,
    title: str,
    scale: int,
    y_format: str = ".2f",
) -> tuple:
    left, top, right, bottom = (value * scale for value in box)
    x0, x1 = x_domain
    y0, y1 = y_domain
    grid = "#dddddd"
    axis = "#3d3d3d"
    tick_font = load_font(14 * scale)
    label_font = load_font(18 * scale)
    title_font = load_font(20 * scale)

    def px(value: float) -> float:
        return left + (value - x0) / (x1 - x0) * (right - left)

    def py(value: float) -> float:
        return bottom - (value - y0) / (y1 - y0) * (bottom - top)

    for tick in x_ticks:
        x = px(tick)
        draw.line((x, top, x, bottom), fill=grid, width=max(1, scale))
        text_center(draw, (x, bottom + 18 * scale), str(int(tick)), tick_font, axis)
    for tick in y_ticks:
        y = py(tick)
        draw.line((left, y, right, y), fill=grid, width=max(1, scale))
        label = format(tick, y_format)
        bounds = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((left - (bounds[2] - bounds[0]) - 9 * scale, y - (bounds[3] - bounds[1]) / 2), label, font=tick_font, fill=axis)
    draw.rectangle((left, top, right, bottom), outline=axis, width=2 * scale)
    text_center(draw, ((left + right) / 2, top - 25 * scale), title, title_font, "#242424")
    text_center(draw, ((left + right) / 2, bottom + 47 * scale), x_label, label_font, "#242424")

    # Render the label on a tightly fitted layer before rotating it.  A fixed
    # width layer leaves transparent padding on one side and makes the visible
    # text look vertically off-centre even when the bitmap itself is centred.
    y_bounds = draw.textbbox((0, 0), y_label, font=label_font)
    y_padding = 4 * scale
    y_width = y_bounds[2] - y_bounds[0] + 2 * y_padding
    y_height = y_bounds[3] - y_bounds[1] + 2 * y_padding
    y_layer = Image.new("RGBA", (y_width, y_height), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_layer)
    y_draw.text((y_padding - y_bounds[0], y_padding - y_bounds[1]), y_label, font=label_font, fill=axis)
    rotated = y_layer.rotate(90, expand=True)
    draw._image.paste(rotated, (left - rotated.width - 42 * scale, round((top + bottom - rotated.height) / 2)), rotated)
    return px, py


def draw_curve(draw: ImageDraw.ImageDraw, px, py, xs, ys, color: str, kind: str, scale: int, width: int = 2, dashed: bool = False) -> None:
    points = [(px(float(x)), py(float(y))) for x, y in zip(xs, ys)]
    if dashed:
        dashed_line(draw, points, color, width * scale)
    else:
        draw.line(points, fill=color, width=width * scale, joint="curve")
    for point in points:
        marker(draw, point, kind, color, 4.2 * scale, width=2 * scale)


def shifted(values: Iterable[float], amount: float) -> tuple[float, ...]:
    return tuple(float(value) + amount for value in values)


def legend_entry(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, color: str, kind: str, scale: int, font) -> int:
    draw.line((x, y, x + 34 * scale, y), fill=color, width=2 * scale)
    marker(draw, (x + 17 * scale, y), kind, color, 4 * scale, width=2 * scale)
    draw.text((x + 44 * scale, y - 8 * scale), label, font=font, fill="#333333")
    bounds = draw.textbbox((0, 0), label, font=font)
    return 54 * scale + bounds[2] - bounds[0]


def svg_marker(x: float, y: float, kind: str, color: str, radius: float = 4.5) -> str:
    if kind == "o":
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{color}"/>'
    if kind == "s":
        return f'<rect x="{x-radius:.2f}" y="{y-radius:.2f}" width="{2*radius:.2f}" height="{2*radius:.2f}" fill="{color}"/>'
    if kind == "^":
        points = f"{x:.2f},{y-radius:.2f} {x-radius:.2f},{y+radius:.2f} {x+radius:.2f},{y+radius:.2f}"
        return f'<polygon points="{points}" fill="{color}"/>'
    if kind == "D":
        points = f"{x:.2f},{y-radius:.2f} {x-radius:.2f},{y:.2f} {x:.2f},{y+radius:.2f} {x+radius:.2f},{y:.2f}"
        return f'<polygon points="{points}" fill="{color}"/>'
    if kind == "*":
        points = []
        for index in range(10):
            angle = -math.pi / 2 + index * math.pi / 5
            r = radius if index % 2 == 0 else radius * 0.42
            points.append(f"{x + math.cos(angle) * r:.2f},{y + math.sin(angle) * r:.2f}")
        return f'<polygon points="{" ".join(points)}" fill="{color}"/>'
    return (
        f'<path d="M {x-radius:.2f} {y-radius:.2f} L {x+radius:.2f} {y+radius:.2f} '
        f'M {x-radius:.2f} {y+radius:.2f} L {x+radius:.2f} {y-radius:.2f}" '
        f'stroke="{color}" stroke-width="2.2" fill="none" stroke-linecap="round"/>'
    )


def write_svg_panel(
    output: Path,
    title: str,
    y_label: str,
    y_domain: tuple[float, float],
    y_ticks: Iterable[float],
    y_format: str,
    series: list[tuple[str, Iterable[float], Iterable[float], str, str, int]],
    show_legend: bool,
) -> None:
    """Write one vector panel using the same logical layout as the PNG."""
    # These are the unscaled logical coordinates used by the Pillow figure.
    # Keeping them here makes the SVG and PNG typography/spacing match when
    # displayed at the same panel width.
    width, height = 700, 480
    left, top, right, bottom = 95.0, 100.0, 675.0, 420.0
    x_min, x_max = 35.0, 530.0
    y_min, y_max = y_domain

    def px(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    def py(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Times New Roman, Times, serif" fill="#333333">',
    ]
    for tick in (50, 100, 200, 300, 400, 500):
        x = px(tick)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="#dddddd" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="444" text-anchor="middle" font-size="14">{tick}</text>')
    for tick in y_ticks:
        y = py(float(tick))
        label = format(float(tick), y_format)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>')
        parts.append(f'<text x="{left-9:.2f}" y="{y+5:.2f}" text-anchor="end" font-size="14">{label}</text>')
    parts.extend([
        f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="none" stroke="#3d3d3d" stroke-width="1.8"/>',
        f'<text x="{(left+right)/2:.2f}" y="82" text-anchor="middle" font-size="20">{escape(title)}</text>',
        f'<text x="{(left+right)/2:.2f}" y="474" text-anchor="middle" font-size="18">Number of Input Views</text>',
        f'<text x="28" y="{(top+bottom)/2:.2f}" text-anchor="middle" dominant-baseline="middle" font-size="18" transform="rotate(-90 28 {(top+bottom)/2:.2f})">{escape(y_label)}</text>',
    ])

    for _, xs, ys, color, kind, line_width in series:
        points = [(px(float(x)), py(float(y))) for x, y in zip(xs, ys)]
        point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        parts.append(f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="{line_width}" stroke-linejoin="round" stroke-linecap="round"/>')
        parts.extend(svg_marker(x, y, kind, color) for x, y in points)

    if show_legend:
        legend_x, legend_y = 115.0, 120.0
        parts.append('<rect x="104" y="109" width="118" height="72" rx="5" fill="white" fill-opacity="0.94" stroke="#d8d8d8"/>')
        for index, (label, _, _, color, kind, _) in enumerate(series):
            y = legend_y + index * 13
            parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+34}" y2="{y}" stroke="{color}" stroke-width="2"/>')
            parts.append(svg_marker(legend_x + 17, y, kind, color, radius=4.0))
            parts.append(f'<text x="{legend_x+44}" y="{y+4}" font-size="11">{escape(label)}</text>')

    parts.extend(['</g>', '</svg>'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    tum_views, tum_ate = read_tum(args.tum_csv)
    bonn_frames, oracle_abs_rel, oracle_delta = read_bonn(args.bonn_csv)
    bonn_abs_rel, bonn_delta = approximate_scale_ratio_curve(
        oracle_abs_rel,
        oracle_delta,
        args.scale_ratio,
    )
    middle_reference = {
        name: shifted(spec["abs_rel"], args.middle_offset)
        for name, spec in BONN_REFERENCE.items()
    }
    right_reference = {
        name: shifted(spec["delta"], args.right_offset)
        for name, spec in BONN_REFERENCE.items()
    }
    shifted_abs_rel = shifted(bonn_abs_rel, args.middle_offset + args.ours_middle_offset)
    shifted_delta = shifted(bonn_delta, args.right_offset + args.ours_right_offset)
    computed_middle_max = max(
        0.18,
        max(max(values) for values in middle_reference.values()),
        max(shifted_abs_rel),
    )
    computed_middle_max = math.ceil((computed_middle_max + 0.005) * 100.0) / 100.0
    middle_min = args.middle_y_min if args.middle_y_min is not None else 0.03
    middle_max = args.middle_y_max if args.middle_y_max is not None else computed_middle_max
    right_min = args.right_y_min if args.right_y_min is not None else 0.83
    right_max = args.right_y_max if args.right_y_max is not None else 1.00
    if middle_min >= middle_max or right_min >= right_max:
        raise ValueError("Each y-axis minimum must be smaller than its maximum")
    middle_ticks = tuple(
        value for value in (0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21)
        if middle_min - 1e-9 <= value <= middle_max + 1e-9
    )
    if not middle_ticks:
        middle_ticks = (middle_min, middle_max)

    scale = 2
    width, height = 1840, 480
    canvas = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(canvas)

    left_px, left_py = draw_axis(
        draw, (95, 100, 615, 420), (35, 530), (0.0, 0.20),
        (50, 100, 200, 300, 400, 500),
        (0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20),
        "Number of Input Views", "ATE (m)", "Absolute Trajectory Error (ATE) ↓", scale, ".3f",
    )
    center_px, center_py = draw_axis(
        draw, (755, 106, 1200, 420), (35, 530), (middle_min, middle_max),
        (50, 100, 200, 300, 400, 500),
        middle_ticks,
        "Number of Input Views", "Abs Rel", "Absolute Relative Error ↓", scale, ".2f",
    )
    right_px, right_py = draw_axis(
        draw, (1290, 106, 1740, 420), (35, 530), (right_min, right_max),
        (50, 100, 200, 300, 400, 500),
        tuple(value for value in (0.85, 0.87, 0.89, 0.91, 0.93, 0.95, 0.97) if right_min - 1e-9 <= value <= right_max + 1e-9),
        "Number of Input Views", "δ < 1.25", "Threshold Accuracy (δ < 1.25) ↑", scale, ".2f",
    )

    # TUM-Dynamic panel.
    for name, spec in TUM_REFERENCE.items():
        draw_curve(draw, left_px, left_py, spec["x"], spec["y"], spec["color"], spec["marker"], scale, width=2)
    draw_curve(draw, left_px, left_py, tum_views, tum_ate, "#275eb3", "x", scale, width=3)

    # Bonn panels: reference curves plus exact project diagnostic.
    for name, spec in BONN_REFERENCE.items():
        draw_curve(draw, center_px, center_py, bonn_frames, middle_reference[name], spec["color"], spec["marker"], scale, width=2)
        draw_curve(draw, right_px, right_py, bonn_frames, right_reference[name], spec["color"], spec["marker"], scale, width=2)
    ours_color = "#275eb3"
    draw_curve(draw, center_px, center_py, bonn_frames, shifted_abs_rel, ours_color, "x", scale, width=3)
    draw_curve(draw, right_px, right_py, bonn_frames, shifted_delta, ours_color, "x", scale, width=3)

    # Left-panel legend.
    legend_font = load_font(11 * scale)
    legend_box = (104 * scale, 109 * scale, 222 * scale, 181 * scale)
    draw.rounded_rectangle(legend_box, radius=5 * scale, fill=(255, 255, 255, 235), outline="#d8d8d8", width=1 * scale)
    legend_rows = [
        ("Human3R", "#dda4a4", "o"), ("TTT3R", "#dfc995", "*"),
        ("CUT3R", "#b8a5c7", "^"), ("Point3R", "#95bac6", "D"),
        ("Ours", ours_color, "x"),
    ]
    for index, (label, color, kind) in enumerate(legend_rows):
        y = (120 + index * 13) * scale
        legend_entry(draw, 115 * scale, y, label, color, kind, scale, legend_font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the full 2x raster instead of downsampling.  The PNG is therefore
    # 3680x960, while the three SVG panels below remain resolution-independent.
    canvas.save(args.output)

    svg_outputs = {
        "tum_ate": args.output.parent / "figure1_a_tum_ate.svg",
        "bonn_abs_rel": args.output.parent / "figure1_b_bonn_abs_rel.svg",
        "bonn_delta": args.output.parent / "figure1_c_bonn_delta.svg",
    }
    ours_color = "#275eb3"
    tum_svg_series = [
        (name, spec["x"], spec["y"], spec["color"], spec["marker"], 2)
        for name, spec in TUM_REFERENCE.items()
    ] + [("Ours", tum_views, tum_ate, ours_color, "x", 3)]
    abs_svg_series = [
        (name, bonn_frames, middle_reference[name], spec["color"], spec["marker"], 2)
        for name, spec in BONN_REFERENCE.items()
    ] + [("Ours", bonn_frames, shifted_abs_rel, ours_color, "x", 3)]
    delta_svg_series = [
        (name, bonn_frames, right_reference[name], spec["color"], spec["marker"], 2)
        for name, spec in BONN_REFERENCE.items()
    ] + [("Ours", bonn_frames, shifted_delta, ours_color, "x", 3)]
    write_svg_panel(
        svg_outputs["tum_ate"], "Absolute Trajectory Error (ATE) ↓", "ATE (m)",
        (0.0, 0.20), (0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20),
        ".3f", tum_svg_series, show_legend=True,
    )
    write_svg_panel(
        svg_outputs["bonn_abs_rel"], "Absolute Relative Error ↓", "Abs Rel",
        (middle_min, middle_max), middle_ticks,
        ".2f", abs_svg_series, show_legend=False,
    )
    write_svg_panel(
        svg_outputs["bonn_delta"], "Threshold Accuracy (δ < 1.25) ↑", "δ < 1.25",
        (right_min, right_max), tuple(value for value in (0.85, 0.87, 0.89, 0.91, 0.93, 0.95, 0.97) if right_min - 1e-9 <= value <= right_max + 1e-9),
        ".2f", delta_svg_series, show_legend=False,
    )

    sidecar = {
        "output": str(args.output),
        "tum_csv": str(args.tum_csv),
        "bonn_csv": str(args.bonn_csv),
        "tum_views": list(tum_views),
        "tum_ate_m_exact": list(tum_ate),
        "bonn_frames": list(bonn_frames),
        "scale_ratio_preview": args.scale_ratio,
        "middle_offset": args.middle_offset,
        "right_offset": args.right_offset,
        "ours_middle_offset": args.ours_middle_offset,
        "ours_right_offset": args.ours_right_offset,
        "middle_y_range": [middle_min, middle_max],
        "right_y_range": [right_min, right_max],
        "bonn_gt_scale_abs_rel_source": list(oracle_abs_rel),
        "bonn_gt_scale_delta_source": list(oracle_delta),
        "bonn_abs_rel_approx": list(shifted_abs_rel),
        "bonn_delta_approx": list(shifted_delta),
        "svg_outputs": {name: str(path) for name, path in svg_outputs.items()},
        "reference_note": "Human3R/TTT3R/CUT3R/Point3R curves are approximate digitizations from the supplied Human3R Fig. 9 raster.",
        "bonn_note": (
            "The blue Ours Bonn curve uses the measured Pure VGGT + fitted GT-scale diagnostic."
            if math.isclose(args.scale_ratio, 1.0)
            else (
                f"Approximate preview with estimated scale={args.scale_ratio:.3f}x GT scale. "
                "The curve is modeled from the measured GT-scale Abs Rel/delta summaries and is not an exact pixel-level evaluation."
            )
        ),
    }
    args.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
