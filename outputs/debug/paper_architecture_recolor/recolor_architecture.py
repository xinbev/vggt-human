from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(r"C:\Users\ROG\AppData\Local\Temp\codex-clipboard-84c26fbc-1b78-4394-889a-a4baecc40183.png")
OUT = ROOT / "outputs/vis/paper_architecture_recolor"

COLORS = {
    "ink": "#1F2937",
    "neutral": "#51647C",
    "neutral_light": "#EEF3F7",
    "scene": "#2A9D8F",
    "scene_light": "#E6F5F2",
    "camera": "#4C83E6",
    "camera_light": "#EAF1FF",
    "human": "#E76F51",
    "human_light": "#FCEBE6",
    "tsmr": "#D89B2B",
    "tsmr_light": "#FFF4DB",
    "output": "#4F8A5B",
    "output_light": "#EAF4E7",
    "panel": "#F8FAFC",
    "line": "#D4DAE1",
}

FONT = "C:/Windows/Fonts/arial.ttf"
BOLD = "C:/Windows/Fonts/arialbd.ttf"


def rgb(hex_color):
    value = hex_color.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def scaled_box(box, scale):
    return tuple(int(v * scale) for v in box)


def tint_light_pixels(image, box, color, scale, threshold=220):
    x0, y0, x1, y1 = scaled_box(box, scale)
    crop = np.asarray(image.crop((x0, y0, x1, y1))).copy()
    high = crop.max(axis=2)
    low = crop.min(axis=2)
    mask = (low >= threshold) & ((high - low) <= 35)
    crop[mask] = np.asarray(rgb(color), dtype=np.uint8)
    image.paste(Image.fromarray(crop), (x0, y0))


def outline(draw, box, color, scale, width=3, radius=18):
    draw.rounded_rectangle(
        scaled_box(box, scale),
        radius=int(radius * scale),
        outline=rgb(color),
        width=max(1, int(width * scale)),
    )


def fill_and_outline(image, draw, box, fill, stroke, scale, width=3, radius=18, threshold=220):
    tint_light_pixels(image, box, fill, scale, threshold=threshold)
    outline(draw, box, stroke, scale, width=width, radius=radius)


def draw_text(draw, xy, text, size, color, scale, bold=False, anchor=None):
    draw.text(
        (int(xy[0] * scale), int(xy[1] * scale)),
        text,
        font=ImageFont.truetype(BOLD if bold else FONT, int(size * scale)),
        fill=rgb(color),
        anchor=anchor,
    )


def render(scale=1):
    original = Image.open(SOURCE).convert("RGB")
    if scale != 1:
        original = original.resize((original.width * scale, original.height * scale), Image.Resampling.LANCZOS)
    image = original.copy()

    # Preserve all real/schematic raster elements so recoloring only affects layout surfaces.
    preserve_boxes = [
        (47, 66, 196, 410),
        (1464, 186, 1642, 325),
        (49, 542, 198, 642),
        (547, 534, 714, 648),
        (248, 716, 415, 834),
        (447, 728, 532, 812),
    ]
    preserved = [(scaled_box(box, scale), original.crop(scaled_box(box, scale))) for box in preserve_boxes]

    draw = ImageDraw.Draw(image)

    # Panel backgrounds establish overview / query / TSMR hierarchy.
    tint_light_pixels(image, (23, 23, 1645, 474), COLORS["panel"], scale, threshold=242)
    tint_light_pixels(image, (22, 480, 820, 871), "#F7FBFD", scale, threshold=242)
    tint_light_pixels(image, (833, 480, 1645, 871), "#FFFAF6", scale, threshold=242)
    outline(draw, (23, 23, 1645, 474), COLORS["ink"], scale, width=2, radius=31)
    outline(draw, (22, 480, 820, 871), COLORS["camera"], scale, width=3, radius=24)
    outline(draw, (833, 480, 1645, 871), COLORS["tsmr"], scale, width=3, radius=24)

    # Overview modules.
    fill_and_outline(image, draw, (35, 50, 208, 428), "#F7F9FB", COLORS["neutral"], scale, width=3, radius=30)
    fill_and_outline(image, draw, (227, 180, 392, 304), COLORS["human_light"], COLORS["human"], scale, width=3, radius=22)
    fill_and_outline(image, draw, (409, 68, 756, 429), "#F9FBFC", COLORS["neutral"], scale, width=3, radius=56)
    fill_and_outline(image, draw, (780, 103, 944, 417), COLORS["neutral_light"], COLORS["neutral"], scale, width=3, radius=27)
    fill_and_outline(image, draw, (973, 68, 1260, 429), "#F7F9FB", COLORS["neutral"], scale, width=3, radius=52)
    fill_and_outline(image, draw, (1009, 96, 1227, 178), COLORS["camera_light"], COLORS["camera"], scale, width=3, radius=15)
    fill_and_outline(image, draw, (1009, 208, 1227, 290), COLORS["human_light"], COLORS["human"], scale, width=3, radius=15)
    fill_and_outline(image, draw, (1009, 322, 1227, 404), COLORS["scene_light"], COLORS["scene"], scale, width=3, radius=15)
    fill_and_outline(image, draw, (1278, 184, 1450, 312), COLORS["tsmr_light"], COLORS["tsmr"], scale, width=4, radius=24)
    fill_and_outline(image, draw, (1455, 171, 1645, 374), COLORS["output_light"], COLORS["output"], scale, width=3, radius=20)

    # Human query construction branch.
    fill_and_outline(image, draw, (289, 543, 447, 639), COLORS["camera_light"], COLORS["camera"], scale, width=3, radius=15)
    fill_and_outline(image, draw, (44, 724, 204, 816), COLORS["human_light"], COLORS["human"], scale, width=3, radius=17)
    fill_and_outline(image, draw, (537, 711, 724, 813), COLORS["human_light"], COLORS["human"], scale, width=3, radius=18)

    # TSMR internals use semantic color rather than one undifferentiated green.
    fill_and_outline(image, draw, (846, 545, 1010, 642), COLORS["human_light"], COLORS["human"], scale, width=3, radius=18)
    fill_and_outline(image, draw, (846, 653, 1010, 749), COLORS["scene_light"], COLORS["scene"], scale, width=3, radius=18)
    fill_and_outline(image, draw, (846, 759, 1010, 855), COLORS["camera_light"], COLORS["camera"], scale, width=3, radius=18)
    fill_and_outline(image, draw, (1045, 492, 1427, 749), "#F7F4ED", COLORS["tsmr"], scale, width=3, radius=38)
    fill_and_outline(image, draw, (1059, 526, 1223, 623), COLORS["human_light"], COLORS["human"], scale, width=3, radius=18)
    fill_and_outline(image, draw, (1256, 526, 1412, 623), COLORS["scene_light"], COLORS["scene"], scale, width=3, radius=18)
    fill_and_outline(image, draw, (1150, 642, 1325, 738), COLORS["tsmr_light"], COLORS["tsmr"], scale, width=3, radius=18)
    fill_and_outline(image, draw, (1449, 501, 1609, 749), "#F5F8F5", COLORS["output"], scale, width=3, radius=28)
    fill_and_outline(image, draw, (1459, 520, 1595, 610), COLORS["human_light"], COLORS["human"], scale, width=3, radius=18)
    fill_and_outline(image, draw, (1459, 641, 1595, 739), COLORS["output_light"], COLORS["output"], scale, width=3, radius=18)
    fill_and_outline(image, draw, (1044, 776, 1422, 872), COLORS["neutral_light"], COLORS["neutral"], scale, width=3, radius=20)

    # Restore untouched visual assets after all background operations.
    for box, crop in preserved:
        image.paste(crop, box[:2])

    draw = ImageDraw.Draw(image)
    # Repaint section labels so the hierarchy uses the same semantic palette.
    draw.rounded_rectangle(scaled_box((42, 486, 397, 521), scale), radius=int(6 * scale), fill=rgb("#F7FBFD"))
    draw_text(draw, (52, 492), "(B) Human Query Construction", 26, COLORS["camera"], scale, bold=True)
    draw.rounded_rectangle(scaled_box((844, 486, 1040, 521), scale), radius=int(6 * scale), fill=rgb("#FFFAF6"))
    draw_text(draw, (853, 492), "(C) TSMR", 26, COLORS["tsmr"], scale, bold=True)
    draw_text(draw, (247, 31), "(A) Overview", 20, COLORS["neutral"], scale, bold=True)

    # Palette strip records both the primary stroke and its light module fill.
    palette_y = 912
    palette_h = 126
    canvas = Image.new("RGB", (image.width, int((palette_y + palette_h) * scale)), rgb("#F5F7F9"))
    canvas.paste(image, (0, 0))
    palette = ImageDraw.Draw(canvas)
    palette.line(
        (int(24 * scale), int(905 * scale), int(1639 * scale), int(905 * scale)),
        fill=rgb(COLORS["line"]), width=max(1, int(2 * scale)),
    )
    draw_text(palette, (34, 928), "COLOR PALETTE", 18, COLORS["ink"], scale, bold=True)
    draw_text(palette, (34, 956), "Primary / light fill", 13, "#697582", scale)

    swatches = [
        ("Backbone", COLORS["neutral"], COLORS["neutral_light"]),
        ("Scene", COLORS["scene"], COLORS["scene_light"]),
        ("Camera", COLORS["camera"], COLORS["camera_light"]),
        ("Human", COLORS["human"], COLORS["human_light"]),
        ("TSMR", COLORS["tsmr"], COLORS["tsmr_light"]),
        ("Output", COLORS["output"], COLORS["output_light"]),
    ]
    start_x = 250
    cell_w = 226
    for index, (name, primary, light) in enumerate(swatches):
        x = start_x + index * cell_w
        palette.rounded_rectangle(
            scaled_box((x, 928, x + 42, 970), scale), radius=int(7 * scale),
            fill=rgb(primary), outline=rgb(primary), width=max(1, int(scale)),
        )
        palette.rounded_rectangle(
            scaled_box((x + 48, 928, x + 90, 970), scale), radius=int(7 * scale),
            fill=rgb(light), outline=rgb(primary), width=max(1, int(2 * scale)),
        )
        draw_text(palette, (x + 100, 927), name, 15, COLORS["ink"], scale, bold=True)
        draw_text(palette, (x + 100, 950), f"{primary} / {light}", 11, "#697582", scale)

    return canvas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_copy = OUT / "source_architecture.png"
    Image.open(SOURCE).convert("RGB").save(source_copy)

    standard = render(scale=1)
    standard.save(OUT / "architecture_recolored_palette.png", quality=100)

    high_res = render(scale=2)
    high_res.save(OUT / "architecture_recolored_palette_2x.png", quality=100)

    print(OUT / "architecture_recolored_palette.png")
    print(OUT / "architecture_recolored_palette_2x.png")


if __name__ == "__main__":
    main()
