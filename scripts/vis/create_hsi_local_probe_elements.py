#!/usr/bin/env python3
"""Create paper-figure elements for HSI local scene probing and residuals.

The assets are deterministic diagram components for the HSI architecture
figure. They intentionally contain no text so labels and formulas can stay
editable in the final SVG/PDF composition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


BG_TRANSPARENT = (255, 255, 255, 0)
HUMAN = (233, 138, 58)
HUMAN_LIGHT = (249, 205, 170)
SCENE = (60, 125, 217)
SCENE_LIGHT = (190, 220, 250)
SCENE_DARK = (37, 99, 180)
HSI = (42, 168, 107)
HSI_LIGHT = (185, 236, 208)
CALIB = (26, 166, 166)
RESIDUAL = (220, 64, 64)
GRID = (203, 213, 225)
MUTED = (102, 112, 133)
PANEL = (248, 250, 252)
WHITE = (255, 255, 255)


def rgba(color: tuple[int, int, int], alpha: int = 255) -> tuple[int, int, int, int]:
    return (*color, int(alpha))


def new_canvas(width: int, height: int, scale: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA", (width * scale, height * scale), BG_TRANSPARENT)
    return image, ImageDraw.Draw(image, "RGBA")


def save_downsampled(image: Image.Image, path: Path, width: int, height: int, scale: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if scale != 1:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    image.save(path)


def s_point(point: tuple[float, float], scale: int) -> tuple[float, float]:
    return (point[0] * scale, point[1] * scale)


def s_rect(rect: tuple[float, float, float, float], scale: int) -> tuple[float, float, float, float]:
    return tuple(float(v) * scale for v in rect)  # type: ignore[return-value]


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    scale: int,
    width: float = 3.0,
    head: float = 14.0,
    alpha: int = 255,
    dashed: bool = False,
) -> None:
    x1, y1 = s_point(start, scale)
    x2, y2 = s_point(end, scale)
    stroke = rgba(color, alpha)
    line_width = max(1, round(width * scale))
    if dashed:
        draw_dashed_line(draw, (x1, y1), (x2, y2), stroke, line_width, dash=9 * scale)
    else:
        draw.line([(x1, y1), (x2, y2)], fill=stroke, width=line_width)
    dx = x2 - x1
    dy = y2 - y1
    length = max(float(np.hypot(dx, dy)), 1e-6)
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    h = head * scale
    base = (x2 - ux * h, y2 - uy * h)
    p1 = (base[0] + px * h * 0.45, base[1] + py * h * 0.45)
    p2 = (base[0] - px * h * 0.45, base[1] - py * h * 0.45)
    draw.polygon([(x2, y2), p1, p2], fill=stroke)


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int, int],
    width: int,
    dash: int,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = max(float(np.hypot(x2 - x1, y2 - y1)), 1e-6)
    steps = int(length // dash) + 1
    for i in range(steps):
        if i % 2:
            continue
        t0 = i / steps
        t1 = min((i + 0.65) / steps, 1.0)
        p0 = (x1 + (x2 - x1) * t0, y1 + (y2 - y1) * t0)
        p1 = (x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1)
        draw.line([p0, p1], fill=color, width=width)


def draw_depth_grid(
    draw: ImageDraw.ImageDraw,
    origin: tuple[float, float],
    cell: float,
    cols: int,
    rows: int,
    scale: int,
    selected_center: tuple[int, int] = (5, 3),
    selected_radius: int = 1,
    alpha: int = 235,
) -> None:
    ox, oy = origin
    cx, cy = selected_center
    for row in range(rows):
        for col in range(cols):
            t = (row / max(rows - 1, 1)) * 0.55 + (col / max(cols - 1, 1)) * 0.45
            fill = blend(SCENE_LIGHT, SCENE_DARK, t)
            x1 = ox + col * cell
            y1 = oy + row * cell
            rect = (x1, y1, x1 + cell - 2, y1 + cell - 2)
            draw.rounded_rectangle(
                s_rect(rect, scale),
                radius=4 * scale,
                fill=rgba(fill, alpha),
                outline=rgba(GRID, 210),
                width=max(1, scale),
            )
    for row in range(cy - selected_radius, cy + selected_radius + 1):
        for col in range(cx - selected_radius, cx + selected_radius + 1):
            if row < 0 or row >= rows or col < 0 or col >= cols:
                continue
            x1 = ox + col * cell
            y1 = oy + row * cell
            rect = (x1, y1, x1 + cell - 2, y1 + cell - 2)
            draw.rounded_rectangle(
                s_rect(rect, scale),
                radius=4 * scale,
                fill=rgba(HSI_LIGHT, 185),
                outline=rgba(HSI, 255),
                width=max(2, 2 * scale),
            )
    px = ox + (cx + 0.5) * cell
    py = oy + (cy + 0.5) * cell
    draw.ellipse(s_rect((px - 7, py - 7, px + 7, py + 7), scale), fill=rgba(SCENE, 255))


def blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    arr = (1.0 - t) * np.asarray(a, dtype=np.float32) + t * np.asarray(b, dtype=np.float32)
    return tuple(int(x) for x in arr.clip(0, 255))


def draw_anchor_symbol(draw: ImageDraw.ImageDraw, center: tuple[float, float], scale: int) -> None:
    x, y = center
    draw.line([s_point((x - 80, y + 90), scale), s_point((x, y), scale)], fill=rgba(HUMAN, 230), width=7 * scale)
    draw.line([s_point((x + 52, y + 84), scale), s_point((x, y), scale)], fill=rgba(HUMAN, 230), width=7 * scale)
    draw.ellipse(s_rect((x - 14, y - 14, x + 14, y + 14), scale), fill=rgba(HSI, 255), outline=rgba(WHITE, 255), width=3 * scale)
    draw.ellipse(s_rect((x - 32, y - 32, x + 32, y + 32), scale), outline=rgba(HSI, 90), width=3 * scale)


def create_depth_window(path: Path, scale: int) -> dict:
    width, height = 560, 390
    image, draw = new_canvas(width, height, scale)
    draw.rounded_rectangle(s_rect((18, 20, 542, 368), scale), radius=18 * scale, fill=rgba(PANEL, 235), outline=rgba(GRID, 230), width=2 * scale)
    draw_depth_grid(draw, (62, 55), 42, cols=10, rows=7, scale=scale, selected_center=(5, 3))
    save_downsampled(image, path, width, height, scale)
    return {"width": width, "height": height}


def create_anchor_projection(path: Path, scale: int) -> dict:
    width, height = 820, 420
    image, draw = new_canvas(width, height, scale)
    draw_anchor_symbol(draw, (145, 215), scale)
    draw_depth_grid(draw, (430, 70), 36, cols=8, rows=6, scale=scale, selected_center=(3, 2))
    draw_arrow(draw, (174, 210), (556, 160), HSI, scale, width=3.2, head=15)
    draw_arrow(draw, (556, 160), (556, 252), SCENE, scale, width=2.3, head=12, alpha=210)
    save_downsampled(image, path, width, height, scale)
    return {"width": width, "height": height}


def create_local_probe_composite(path: Path, scale: int) -> dict:
    width, height = 980, 500
    image, draw = new_canvas(width, height, scale)
    draw.rounded_rectangle(s_rect((18, 24, 962, 476), scale), radius=22 * scale, fill=rgba(PANEL, 210), outline=rgba(GRID, 180), width=2 * scale)
    draw_anchor_symbol(draw, (170, 260), scale)
    draw_depth_grid(draw, (452, 84), 40, cols=9, rows=7, scale=scale, selected_center=(4, 3))
    draw_arrow(draw, (202, 255), (634, 225), HSI, scale, width=3.4, head=16)
    for idx, (dx, dy) in enumerate([(-52, 120), (0, 138), (52, 120)]):
        start = (634 + dx * 0.15, 225 + dy * 0.1)
        end = (720 + dx, 360 + dy * 0.04)
        draw_arrow(draw, start, end, SCENE, scale, width=2.1, head=10, alpha=180)
        draw.ellipse(s_rect((end[0] - 8, end[1] - 8, end[0] + 8, end[1] + 8), scale), fill=rgba(SCENE, 230))
    save_downsampled(image, path, width, height, scale)
    return {"width": width, "height": height}


def create_body_scene_residual(path: Path, scale: int) -> dict:
    width, height = 820, 500
    image, draw = new_canvas(width, height, scale)
    # Local scene surface.
    surface = [(390, 305), (700, 225), (760, 300), (445, 392)]
    draw.polygon([s_point(p, scale) for p in surface], fill=rgba(SCENE_LIGHT, 180), outline=rgba(SCENE, 240))
    for t in np.linspace(0.15, 0.85, 5):
        p0 = lerp2(surface[0], surface[1], t)
        p1 = lerp2(surface[3], surface[2], t)
        draw.line([s_point(p0, scale), s_point(p1, scale)], fill=rgba(SCENE, 105), width=max(1, scale))
        q0 = lerp2(surface[0], surface[3], t)
        q1 = lerp2(surface[1], surface[2], t)
        draw.line([s_point(q0, scale), s_point(q1, scale)], fill=rgba(SCENE, 105), width=max(1, scale))

    anchor = (220, 230)
    scene = (528, 300)
    z_base = (220, 300)
    draw_anchor_symbol(draw, anchor, scale)
    draw.ellipse(s_rect((scene[0] - 13, scene[1] - 13, scene[0] + 13, scene[1] + 13), scale), fill=rgba(SCENE, 255), outline=rgba(WHITE, 255), width=3 * scale)
    draw_arrow(draw, anchor, scene, RESIDUAL, scale, width=3.4, head=16)
    draw_arrow(draw, scene, (575, 214), CALIB, scale, width=2.8, head=14)
    draw_arrow(draw, anchor, z_base, MUTED, scale, width=2.2, head=10, alpha=180, dashed=True)
    draw_arrow(draw, z_base, scene, HUMAN, scale, width=2.2, head=10, alpha=190, dashed=True)
    draw.ellipse(s_rect((z_base[0] - 6, z_base[1] - 6, z_base[0] + 6, z_base[1] + 6), scale), fill=rgba(HUMAN, 180))
    save_downsampled(image, path, width, height, scale)
    return {"width": width, "height": height}


def lerp2(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
    return (a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t)


def write_svg(path: Path, kind: str) -> None:
    if kind == "depth_window":
        svg = depth_window_svg()
    elif kind == "residual":
        svg = residual_svg()
    else:
        svg = local_probe_svg()
    path.write_text(svg, encoding="utf-8")


def depth_window_svg() -> str:
    rects = []
    ox, oy, cell = 62, 55, 42
    for row in range(7):
        for col in range(10):
            selected = abs(row - 3) <= 1 and abs(col - 5) <= 1
            fill = "#B9ECD0" if selected else "#C8DFF7"
            stroke = "#2AA86B" if selected else "#CBD5E1"
            rects.append(f'<rect x="{ox + col * cell}" y="{oy + row * cell}" width="40" height="40" rx="4" fill="{fill}" fill-opacity="0.82" stroke="{stroke}" stroke-width="2"/>')
    return '<svg xmlns="http://www.w3.org/2000/svg" width="560" height="390" viewBox="0 0 560 390">\n' + "\n".join(rects) + "\n</svg>\n"


def local_probe_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" width="820" height="420" viewBox="0 0 820 420">
<circle cx="145" cy="215" r="14" fill="#2AA86B" stroke="#FFFFFF" stroke-width="3"/>
<path d="M174 210 L556 160" stroke="#2AA86B" stroke-width="4" fill="none" marker-end="url(#arrow)"/>
<rect x="430" y="70" width="286" height="214" rx="10" fill="#F8FAFC" fill-opacity="0.4"/>
<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 Z" fill="#2AA86B"/></marker></defs>
</svg>
"""


def residual_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" width="820" height="500" viewBox="0 0 820 500">
<polygon points="390,305 700,225 760,300 445,392" fill="#BEDCFA" fill-opacity="0.65" stroke="#3C7DD9" stroke-width="2"/>
<circle cx="220" cy="230" r="14" fill="#2AA86B" stroke="#FFFFFF" stroke-width="3"/>
<circle cx="528" cy="300" r="13" fill="#3C7DD9" stroke="#FFFFFF" stroke-width="3"/>
<path d="M220 230 L528 300" stroke="#DC4040" stroke-width="4" fill="none" marker-end="url(#arrowR)"/>
<path d="M528 300 L575 214" stroke="#1AA6A6" stroke-width="3" fill="none" marker-end="url(#arrowT)"/>
<path d="M220 230 L220 300 L528 300" stroke="#E98A3A" stroke-width="3" stroke-dasharray="8 7" fill="none"/>
<defs>
<marker id="arrowR" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 Z" fill="#DC4040"/></marker>
<marker id="arrowT" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 Z" fill="#1AA6A6"/></marker>
</defs>
</svg>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vis/paper_hsi_local_probe_elements"))
    parser.add_argument("--scale", type=int, default=3, help="Internal antialiasing scale for PNG rendering.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scale < 1:
        raise ValueError("--scale must be >= 1")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    outputs = {}
    outputs["05a_depth_patch_grid_3x3.png"] = create_depth_window(out / "05a_depth_patch_grid_3x3.png", args.scale)
    outputs["05b_anchor_project_to_depth.png"] = create_anchor_projection(out / "05b_anchor_project_to_depth.png", args.scale)
    outputs["05c_local_scene_probe_composite.png"] = create_local_probe_composite(out / "05c_local_scene_probe_composite.png", args.scale)
    outputs["06a_body_scene_residual.png"] = create_body_scene_residual(out / "06a_body_scene_residual.png", args.scale)

    write_svg(out / "05a_depth_patch_grid_3x3.svg", "depth_window")
    write_svg(out / "05b_anchor_project_to_depth.svg", "local_probe")
    write_svg(out / "06a_body_scene_residual.svg", "residual")

    manifest = {
        "purpose": "Paper-figure elements for HSI local scene probing and body-scene residual geometry.",
        "source_component": "vggt_omega.models.heads.hsi_refinement_head.HSIRefinementHead",
        "real_logic": [
            "anchor_3d + intrinsics -> projected uv",
            "projected uv -> depth / 3x3 scene patch-token window",
            "(u, v, z_scene) -> scene xyz",
            "offset = scene_point - anchor",
            "distance = ||offset||",
            "z residual = scene_point.z - anchor.z",
            "depth gradients -> approximate normal",
        ],
        "files": list(outputs.keys())
        + [
            "05a_depth_patch_grid_3x3.svg",
            "05b_anchor_project_to_depth.svg",
            "06a_body_scene_residual.svg",
        ],
        "notes": [
            "PNG assets use transparent backgrounds.",
            "SVG assets are minimal editable geometry with no text labels.",
            "Add labels such as scene xyz, offset / distance, normal, and z residual in the final paper figure.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "files": manifest["files"]}, indent=2))


if __name__ == "__main__":
    main()
