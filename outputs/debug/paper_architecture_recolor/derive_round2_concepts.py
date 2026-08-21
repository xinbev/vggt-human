from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
VIS = ROOT / "outputs/vis/paper_architecture_recolor"
FONT = "C:/Windows/Fonts/arial.ttf"
BOLD = "C:/Windows/Fonts/arialbd.ttf"


def color(value):
    value = value.lstrip("#")
    return np.array([int(value[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def transform(source_path, target_path, primary, accent, pale):
    source = Image.open(source_path).convert("RGB")
    hsv = np.asarray(source.convert("HSV"))
    arr = np.asarray(source).astype(np.float32)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    colored = sat > 32
    warm = colored & ((hue < 38) | (hue > 235))
    cool = colored & ~warm
    # Only pastel surface fills become pale; saturated labels, arrows, and tokens
    # keep the strong semantic color even when their value is high.
    very_light = colored & (val > 225) & (sat < 95)

    for mask, target in ((cool, color(primary)), (warm, color(accent))):
        factor = 0.52 + 0.48 * val[..., None] / 255.0
        mapped = target * factor
        arr[mask] = mapped[mask]
    arr[very_light] = color(pale)

    result = Image.fromarray(np.uint8(np.clip(arr, 0, 255)))
    # Keep the real-image anchors untouched.
    for box in ((27, 42, 215, 438), (1325, 126, 1515, 405), (47, 550, 198, 665), (468, 550, 652, 674), (208, 734, 397, 860)):
        result.paste(source.crop(box), box[:2])

    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 944, result.width, result.height), fill="white")
    draw.line((28, 946, result.width - 28, 946), fill="#D7DCE0", width=2)
    draw.text((48, 967), "PALETTE", font=ImageFont.truetype(BOLD, 18), fill="#1F2933")
    items = [("Structure", primary), ("TSMR", accent), ("Background", pale)]
    x = 280
    for name, value in items:
        draw.rounded_rectangle((x, 960, x + 44, 1004), radius=7, fill=value)
        draw.text((x + 58, 957), name, font=ImageFont.truetype(BOLD, 16), fill="#1F2933")
        draw.text((x + 58, 981), value, font=ImageFont.truetype(FONT, 13), fill="#65717B")
        x += 360
    result.save(target_path, quality=100)


def main():
    transform(
        VIS / "concept_E_timeline_tsmr.png",
        VIS / "concept_F_nature_black_vermilion.png",
        primary="#1F2933",
        accent="#D9483B",
        pale="#F3F4F4",
    )
    transform(
        VIS / "concept_D_radial_tsmr.png",
        VIS / "concept_G_graphite_jade.png",
        primary="#24323D",
        accent="#287D73",
        pale="#EDF3F1",
    )
    print(VIS / "concept_F_nature_black_vermilion.png")
    print(VIS / "concept_G_graphite_jade.png")


if __name__ == "__main__":
    main()
