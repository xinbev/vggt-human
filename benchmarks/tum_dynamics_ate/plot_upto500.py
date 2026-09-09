#!/usr/bin/env python3
"""Redraw the supplied ATE comparison with x limited to 500 views."""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT = Path("outputs/vis/tum_dynamics_ate_curve/vggt_omega_ate_upto500_redrawn.png")

# Baseline values are digitized from the supplied comparison raster.  The
# VGGT-Omega values are the exact evaluator output supplied by the user.
SERIES = {
    "Human3R": {
        "x": [50, 100, 150, 200, 300, 400, 500],
        "y": [0.013, 0.018, 0.025, 0.031, 0.042, 0.047, 0.052],
        "color": (223, 167, 167), "marker": "circle",
    },
    "TTT3R": {
        "x": [50, 100, 150, 200, 300, 400, 500],
        "y": [0.014, 0.020, 0.030, 0.043, 0.049, 0.053, 0.070],
        "color": (230, 210, 163), "marker": "star",
    },
    "CUT3R": {
        "x": [50, 100, 150, 200, 300, 400, 500],
        "y": [0.022, 0.030, 0.040, 0.060, 0.085, 0.122, 0.140],
        "color": (184, 165, 199), "marker": "triangle",
    },
    "Point3R": {
        "x": [50, 100, 150, 200, 300, 400, 500],
        "y": [0.022, 0.043, 0.077, 0.080, 0.110, 0.149, 0.130],
        "color": (149, 186, 198), "marker": "pentagon",
    },
    "StreamVGGT": {
        "x": [50, 100, 150, 200],
        "y": [0.012, 0.020, 0.034, 0.045],
        "color": (202, 233, 187), "marker": "diamond",
    },
    "VGGT (offline)": {
        "x": [50, 100, 150],
        "y": [0.006, 0.009, 0.010],
        "color": (150, 150, 150), "marker": "square", "dash": True,
    },
    "VGGT-Omega (ours)": {
        "x": [50, 100, 150, 200, 300, 400, 500],
        "y": [
            0.0029932083135804973,
            0.0043409809895220515,
            0.004930351035055057,
            0.005909471782128872,
            0.006341055280197789,
            0.006451712633021363,
            0.006916918533247201,
        ],
        "color": (20, 91, 180), "marker": "circle", "width": 4,
    },
}


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["timesbd.ttf", "times.ttf"] if bold else ["times.ttf", "timesbd.ttf"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    linux = Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
    return ImageFont.truetype(str(linux), size) if linux.is_file() else ImageFont.load_default()


def polygon(cx: float, cy: float, radius: float, sides: int, angle: float = -math.pi / 2):
    return [
        (cx + radius * math.cos(angle + 2 * math.pi * index / sides), cy + radius * math.sin(angle + 2 * math.pi * index / sides))
        for index in range(sides)
    ]


def draw_marker(draw: ImageDraw.ImageDraw, point, marker: str, color, radius: float) -> None:
    x, y = point
    outline = (255, 255, 255, 240)
    if marker == "circle":
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline=outline, width=2)
    elif marker == "square":
        draw.rectangle((x-radius, y-radius, x+radius, y+radius), fill=color, outline=outline, width=2)
    elif marker == "triangle":
        draw.polygon(polygon(x, y, radius*1.2, 3), fill=color, outline=outline)
    elif marker == "diamond":
        draw.polygon([(x, y-radius*1.25), (x+radius*1.1, y), (x, y+radius*1.25), (x-radius*1.1, y)], fill=color, outline=outline)
    elif marker == "pentagon":
        draw.polygon(polygon(x, y, radius*1.15, 5), fill=color, outline=outline)
    elif marker == "star":
        pts = []
        for index in range(10):
            r = radius*1.35 if index % 2 == 0 else radius*0.58
            a = -math.pi/2 + index*math.pi/5
            pts.append((x+r*math.cos(a), y+r*math.sin(a)))
        draw.polygon(pts, fill=color, outline=outline)


def dashed_line(draw, points, fill, width, dash=16, gap=10):
    for start, end in zip(points, points[1:]):
        x1, y1 = start; x2, y2 = end
        length = math.hypot(x2-x1, y2-y1)
        ux, uy = (x2-x1)/length, (y2-y1)/length
        position = 0.0
        while position < length:
            stop = min(position+dash, length)
            draw.line((x1+ux*position, y1+uy*position, x1+ux*stop, y1+uy*stop), fill=fill, width=width)
            position += dash+gap


def main() -> None:
    scale = 2
    width, height = 1400*scale, 950*scale
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    left, right, top, bottom = 150*scale, 1340*scale, 95*scale, 820*scale

    def xy(x, y):
        return left + x/500*(right-left), bottom - y/0.2*(bottom-top)

    grid = (222, 222, 222, 255)
    axis = (35, 35, 35, 255)
    tick_font = get_font(25*scale)
    label_font = get_font(34*scale)
    title_font = get_font(38*scale)
    legend_font = get_font(23*scale)

    for y_value in [index*0.025 for index in range(9)]:
        _, py = xy(0, y_value)
        draw.line((left, py, right, py), fill=grid, width=2)
        label = f"{y_value:.3f}"
        box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((left-17*scale-(box[2]-box[0]), py-(box[3]-box[1])/2), label, font=tick_font, fill=axis)
    for x_value in range(0, 501, 100):
        px, _ = xy(x_value, 0)
        draw.line((px, top, px, bottom), fill=grid, width=2)
        label = str(x_value)
        box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((px-(box[2]-box[0])/2, bottom+13*scale), label, font=tick_font, fill=axis)
    draw.rectangle((left, top, right, bottom), outline=axis, width=3)

    for name, spec in SERIES.items():
        points = [xy(x, y) for x, y in zip(spec["x"], spec["y"])]
        line_width = int(spec.get("width", 3)*scale)
        if spec.get("dash"):
            dashed_line(draw, points, spec["color"]+(255,), line_width, 13*scale, 8*scale)
        else:
            draw.line(points, fill=spec["color"]+(255,), width=line_width, joint="curve")
        for point in points:
            draw_marker(draw, point, spec["marker"], spec["color"]+(255,), 7*scale)

    # OOM markers copied from the supplied raster protocol.
    for x_value, y_value, color in [(200, 0.052, (164, 211, 139)), (150, 0.014, (140, 140, 140))]:
        px, py = xy(x_value, y_value)
        r = 11*scale
        draw.line((px-r, py-r, px+r, py+r), fill=color+(255,), width=5*scale)
        draw.line((px-r, py+r, px+r, py-r), fill=color+(255,), width=5*scale)
        draw.text((px+13*scale, py-18*scale), "OOM", font=get_font(24*scale, bold=True), fill=color+(255,))

    title = "Absolute Trajectory Error (ATE) ↓"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width-(title_box[2]-title_box[0]))/2, 25*scale), title, font=title_font, fill=axis)
    x_label = "Number of Input Views"
    x_box = draw.textbbox((0, 0), x_label, font=label_font)
    draw.text(((width-(x_box[2]-x_box[0]))/2, 875*scale), x_label, font=label_font, fill=axis)

    y_label = Image.new("RGBA", (650*scale, 70*scale), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label)
    y_draw.text((0, 0), "ATE (m)", font=label_font, fill=axis)
    y_label = y_label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(y_label, (25*scale, int((height-y_label.height)/2)), y_label)
    draw = ImageDraw.Draw(image, "RGBA")

    # Compact two-column legend.
    legend_left, legend_top = 175*scale, 115*scale
    legend_width, legend_height = 690*scale, 205*scale
    draw.rounded_rectangle((legend_left, legend_top, legend_left+legend_width, legend_top+legend_height), radius=7*scale, fill=(255,255,255,238), outline=(190,190,190,255), width=2)
    entries = list(SERIES.items())
    for index, (name, spec) in enumerate(entries):
        column, row = index // 4, index % 4
        x0 = legend_left + (20+column*340)*scale
        y0 = legend_top + (27+row*45)*scale
        draw.line((x0, y0, x0+48*scale, y0), fill=spec["color"]+(255,), width=int(spec.get("width",3)*scale))
        draw_marker(draw, (x0+24*scale, y0), spec["marker"], spec["color"]+(255,), 6*scale)
        draw.text((x0+62*scale, y0-14*scale), name, font=legend_font, fill=axis)

    final = image.resize((1400, 950), Image.Resampling.LANCZOS)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    final.save(OUTPUT)
    metadata = {
        "x_limit": [0, 500], "y_limit_m": [0, 0.2],
        "series": {name: {"views": spec["x"], "ate_m": spec["y"]} for name, spec in SERIES.items()},
        "note": "VGGT-Omega values are exact evaluator outputs; comparison baselines are digitized from the supplied raster.",
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()

