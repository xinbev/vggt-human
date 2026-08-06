from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "outputs/vis/tsmr_paper_main_figure/tsmr_method_overview_v1.png"
TARGET = ROOT / "outputs/vis/tsmr_paper_main_figure/tsmr_method_overview_paper.png"

image = Image.open(SOURCE).convert("RGB")
draw = ImageDraw.Draw(image)

regular = "C:/Windows/Fonts/arial.ttf"
bold = "C:/Windows/Fonts/arialbd.ttf"


def font(size: int, is_bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(bold if is_bold else regular, size=size)


def centered_text(box, text, size, fill=(32, 35, 38), is_bold=False):
    x0, y0, x1, y1 = box
    fnt = font(size, is_bold)
    bbox = draw.textbbox((0, 0), text, font=fnt)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text(((x0 + x1 - width) / 2, (y0 + y1 - height) / 2 - bbox[1]), text, font=fnt, fill=fill)


# Correct the probe-token label while preserving the generated mesh and token graphics.
draw.rounded_rectangle((792, 145, 970, 207), radius=6, fill=(255, 255, 255))
centered_text((795, 149, 967, 203), "Body-anchored\ngeometric probe tokens", 17, is_bold=True)

# Rebuild the hypothesis readout so the walking example selects bilateral support.
draw.rounded_rectangle((1068, 194, 1215, 402), radius=8, fill=(255, 255, 255))
centered_text((1084, 199, 1211, 239), "Hypothesis\nprobabilities", 16, is_bold=True)
rows = [
    ("No-op", 0.04, (170, 170, 170)),
    ("Left", 0.03, (234, 104, 83)),
    ("Right", 0.03, (68, 140, 204)),
    ("Bilateral", 0.87, (239, 177, 62)),
    ("Body Support", 0.03, (77, 170, 112)),
]
for index, (label, probability, color) in enumerate(rows):
    y = 250 + index * 28
    label_size = 10 if label == "Body Support" else 12
    draw.text((1088, y), label, font=font(label_size, label == "Bilateral"), fill=(36, 38, 41))
    draw.rounded_rectangle((1146, y + 3, 1184, y + 16), radius=3, fill=(235, 237, 239))
    draw.rounded_rectangle((1146, y + 3, 1146 + max(2, int(38 * probability)), y + 16), radius=3, fill=color)
    draw.text((1188, y), f"{probability:.2f}", font=font(11, label == "Bilateral"), fill=(36, 38, 41))

# Replace the full generated badge strip to avoid retaining malformed labels.
draw.rectangle((1215, 205, 1252, 690), fill=(255, 255, 255))
draw.line((1233, 206, 1233, 688), fill=(26, 102, 203), width=4)
for y, label in ((275, "r=1"), (455, "r=2"), (635, "r=3")):
    draw.ellipse((1215, y - 19, 1253, y + 19), fill=(246, 252, 255), outline=(26, 102, 203), width=3)
    centered_text((1215, y - 19, 1253, y + 19), label, 13, fill=(26, 102, 203), is_bold=True)

# Correct the constrained residual caption inside the generated recurrent module.
draw.rounded_rectangle((838, 589, 1054, 652), radius=5, fill=(245, 243, 252))
centered_text((843, 593, 1049, 648), "Bounded residual along\nselected support direction", 15, is_bold=True)

# Clarify both floating examples without retaining generated garbled distance labels.
draw.rectangle((151, 421, 180, 479), fill=(255, 255, 255))
draw.line((165, 433, 165, 473), fill=(219, 70, 54), width=4)
draw.polygon(((158, 465), (172, 465), (165, 478)), fill=(219, 70, 54))
draw.rounded_rectangle((139, 472, 223, 506), radius=5, fill=(255, 255, 255))
centered_text((141, 475, 221, 503), "8 cm float", 15, fill=(219, 70, 54), is_bold=True)
draw.rectangle((1307, 398, 1340, 462), fill=(255, 255, 255))
draw.rectangle((1338, 398, 1374, 462), fill=(255, 255, 255))
draw.line((1323, 411, 1323, 452), fill=(219, 70, 54), width=4)
draw.polygon(((1316, 444), (1330, 444), (1323, 457)), fill=(219, 70, 54))
draw.rounded_rectangle((1278, 457, 1372, 515), radius=5, fill=(255, 255, 255))
centered_text((1282, 470, 1368, 503), "8 cm gap", 15, fill=(219, 70, 54), is_bold=True)

# Correct the final support mode in the output summary.
draw.rounded_rectangle((1306, 654, 1516, 705), radius=4, fill=(255, 255, 255))
draw.text((1320, 667), "Support Mode", font=font(15, True), fill=(36, 38, 41))
draw.text((1432, 667), "Bilateral", font=font(16, True), fill=(49, 148, 89))

TARGET.parent.mkdir(parents=True, exist_ok=True)
image.save(TARGET, quality=100)
print(TARGET)
