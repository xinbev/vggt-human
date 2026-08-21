from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "outputs/vis/paper_architecture_recolor/concept_A_navy_teal_coral.png"
OUT = ROOT / "outputs/vis/paper_architecture_recolor"
FONT = "C:/Windows/Fonts/arial.ttf"
BOLD = "C:/Windows/Fonts/arialbd.ttf"


def rgb(value):
    value = value.lstrip("#")
    return np.array([int(value[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def preserve_photos(source, result):
    boxes = [
        (27, 42, 215, 438),
        (1325, 126, 1515, 405),
        (48, 551, 198, 662),
        (487, 548, 650, 668),
        (229, 735, 395, 852),
    ]
    for box in boxes:
        result.paste(source.crop(box), box[:2])


def recolor(source, cool, warm, light, output_name):
    hsv = np.asarray(source.convert("HSV"))
    arr = np.asarray(source).astype(np.float32)
    hue = hsv[..., 0]
    sat = hsv[..., 1]
    val = hsv[..., 2]

    colored = sat > 38
    warm_mask = colored & ((hue < 35) | (hue > 235))
    cool_mask = colored & ~warm_mask
    light_mask = colored & (val > 205)

    for mask, target in ((cool_mask, rgb(cool)), (warm_mask, rgb(warm))):
        strength = (0.48 + 0.52 * (val[..., None] / 255.0))
        mapped = target * strength
        arr[mask] = mapped[mask]

    arr[light_mask] = rgb(light)
    result = Image.fromarray(np.uint8(np.clip(arr, 0, 255)))
    preserve_photos(source, result)
    result.save(OUT / output_name, quality=100)
    return result


def palette(image, items):
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 944, image.width, image.height), fill="white")
    draw.line((28, 945, image.width - 28, 945), fill="#D9DEE3", width=2)
    draw.text((40, 963), "PALETTE", font=ImageFont.truetype(BOLD, 18), fill="#20252B")
    x = 260
    for name, color in items:
        draw.rounded_rectangle((x, 960, x + 44, 1004), radius=7, fill=color)
        draw.text((x + 58, 957), name, font=ImageFont.truetype(BOLD, 16), fill="#20252B")
        draw.text((x + 58, 981), color, font=ImageFont.truetype(FONT, 13), fill="#66717B")
        x += 300


def main():
    source = Image.open(SOURCE).convert("RGB")

    swiss = recolor(
        source,
        cool="#20252B",
        warm="#E15D44",
        light="#EEF0F2",
        output_name="concept_B_swiss_coral.png",
    )
    palette(swiss, [("Charcoal", "#20252B"), ("Coral", "#E15D44"), ("Mist", "#EEF0F2")])
    swiss.save(OUT / "concept_B_swiss_coral.png", quality=100)

    technical = recolor(
        source,
        cool="#183B56",
        warm="#237D72",
        light="#E8F2F1",
        output_name="concept_C_technical_teal.png",
    )
    palette(technical, [("Graphite", "#183B56"), ("Teal", "#237D72"), ("Ice", "#E8F2F1")])
    technical.save(OUT / "concept_C_technical_teal.png", quality=100)

    print(OUT / "concept_B_swiss_coral.png")
    print(OUT / "concept_C_technical_teal.png")


if __name__ == "__main__":
    main()
