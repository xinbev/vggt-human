from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "outputs/vis/tsmr_paper_main_figure/tsmr_method_overview_v1.png"
TARGET = ROOT / "outputs/vis/tsmr_paper_main_figure/tsmr_method_overview_abc_paper.png"

W, H = 1800, 1200
canvas = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(canvas)
source = Image.open(SOURCE).convert("RGB")

FONT = "C:/Windows/Fonts/arial.ttf"
BOLD = "C:/Windows/Fonts/arialbd.ttf"


def f(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def rounded(box, fill, outline=(125, 130, 136), width=2, radius=14):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(xy, value, size=18, fill=(30, 32, 35), bold=False, anchor=None):
    draw.text(xy, value, font=f(size, bold), fill=fill, anchor=anchor)


def center(box, value, size=18, fill=(30, 32, 35), bold=False):
    x0, y0, x1, y1 = box
    text(((x0 + x1) / 2, (y0 + y1) / 2), value, size, fill, bold, "mm")


def arrow(start, end, color=(65, 70, 76), width=4, head=12):
    x0, y0 = start
    x1, y1 = end
    draw.line((x0, y0, x1, y1), fill=color, width=width)
    if abs(x1 - x0) >= abs(y1 - y0):
        sign = 1 if x1 >= x0 else -1
        draw.polygon(((x1, y1), (x1 - sign * head, y1 - head * 0.65), (x1 - sign * head, y1 + head * 0.65)), fill=color)
    else:
        sign = 1 if y1 >= y0 else -1
        draw.polygon(((x1, y1), (x1 - head * 0.65, y1 - sign * head), (x1 + head * 0.65, y1 - sign * head)), fill=color)


def paste_crop(src_box, dst_box):
    crop = source.crop(src_box)
    width = dst_box[2] - dst_box[0]
    height = dst_box[3] - dst_box[1]
    crop.thumbnail((width, height), Image.Resampling.LANCZOS)
    x = dst_box[0] + (width - crop.width) // 2
    y = dst_box[1] + (height - crop.height) // 2
    canvas.paste(crop, (x, y))


def tag(box, value, color):
    rounded(box, fill=color, outline=color, width=1, radius=8)
    center(box, value, 15, (255, 255, 255), True)


# Main panels.
rounded((18, 16, 1782, 438), (255, 255, 255), outline=(91, 95, 101), width=3, radius=14)
rounded((18, 460, 882, 1182), (250, 254, 255), outline=(99, 190, 215), width=5, radius=14)
rounded((902, 460, 1782, 1182), (255, 252, 249), outline=(233, 166, 124), width=5, radius=14)

text((34, 30), "(A) Overview", 28, bold=True)
text((36, 474), "(B) Support Manifold & Hypothesis Bank", 26, bold=True, fill=(27, 111, 137))
text((920, 474), "(C) Track-aware Recurrent Refinement", 26, bold=True, fill=(172, 87, 52))

# ---------------------------------------------------------------------------
# (A) Compact method bus.
# ---------------------------------------------------------------------------
nodes = [
    ((45, 82, 322, 397), (241, 250, 252), "Frozen Stage2 + Tracking"),
    ((372, 82, 672, 397), (233, 249, 252), "Local Support Manifold"),
    ((722, 82, 1002, 397), (255, 248, 230), "Analytic Hypothesis Bank"),
    ((1052, 82, 1435, 397), (239, 246, 255), "Track-aware Recurrent Selector"),
    ((1485, 82, 1752, 397), (240, 251, 244), "Grounded Output"),
]
for box, fill, title in nodes:
    rounded(box, fill, outline=(150, 155, 160), width=2, radius=12)
    center((box[0] + 8, box[1] + 10, box[2] - 8, box[1] + 53), title, 19, bold=True)

paste_crop((35, 90, 300, 245), (60, 140, 307, 275))
paste_crop((95, 275, 235, 560), (116, 275, 250, 382))
tag((54, 347, 106, 378), "ID 7", (31, 160, 165))
text((274, 348), "FROZEN", 13, (38, 121, 157), True, "mm")

paste_crop((420, 188, 568, 356), (396, 140, 648, 275))
paste_crop((430, 445, 570, 525), (438, 283, 625, 361))
center((390, 356, 655, 388), "Temporal fusion + normals", 15, fill=(27, 111, 137), bold=True)

candidate_labels = [("No-op", (125, 130, 136)), ("Left", (224, 93, 73)), ("Right", (54, 128, 197)), ("Bilateral", (219, 154, 36)), ("Body", (54, 150, 92))]
for i, (label, color) in enumerate(candidate_labels):
    y = 143 + i * 43
    rounded((752, y, 972, y + 32), (255, 255, 255), outline=color, width=2, radius=8)
    text((767, y + 16), label, 15, color, True, "lm")
    if label == "No-op":
        text((943, y + 16), "0", 15, color, True, "mm")
    else:
        arrow((922, y + 23), (950, y + 9), color, width=3, head=7)

paste_crop((817, 210, 940, 407), (1072, 151, 1153, 300))
for i, color in enumerate(((234, 104, 83), (239, 177, 62), (77, 170, 112), (68, 140, 204))):
    draw.rounded_rectangle((1140, 167 + i * 28, 1158, 185 + i * 28), radius=3, fill=color)
rounded((1175, 146, 1280, 302), (255, 255, 255), outline=(53, 133, 203), width=2, radius=9)
center((1182, 158, 1273, 203), "Transformer\nSelector", 15, bold=True, fill=(36, 101, 166))
for i in range(3):
    rounded((1194, 218 + i * 22, 1261, 234 + i * 22), (239, 246, 255), outline=(53, 133, 203), width=1, radius=4)
rounded((1300, 146, 1408, 222), (255, 255, 255), outline=(53, 133, 203), width=2, radius=9)
center((1304, 151, 1404, 184), "ID Memory", 15, bold=True, fill=(36, 101, 166))
center((1304, 184, 1404, 216), "t-1 -> t", 13, fill=(36, 101, 166))
rounded((1300, 239, 1408, 305), (255, 255, 255), outline=(53, 133, 203), width=2, radius=9)
center((1304, 244, 1404, 272), "Bilateral  0.87", 13, bold=True, fill=(36, 101, 166))
center((1304, 273, 1404, 300), "Abstain if uncertain", 11, fill=(36, 101, 166))
arrow((1159, 228), (1175, 228), color=(31, 105, 204), width=3, head=7)
arrow((1280, 228), (1297, 228), color=(31, 105, 204), width=3, head=7)
draw.arc((1077, 130, 1420, 365), 205, 520, fill=(31, 105, 204), width=5)
text((1247, 350), "Update -> Re-probe (x3)", 14, (31, 105, 204), True, "mm")

paste_crop((1268, 132, 1527, 568), (1506, 132, 1730, 338))
center((1502, 339, 1735, 388), "Translation refined\npose / shape frozen", 13, fill=(44, 133, 76), bold=True)

for x0, x1 in ((322, 372), (672, 722), (1002, 1052), (1435, 1485)):
    arrow((x0 + 6, 239), (x1 - 6, 239), width=5, head=13)

# ---------------------------------------------------------------------------
# (B) Support manifold and analytic hypothesis construction.
# ---------------------------------------------------------------------------
sub_y = 540
steps = [
    ((42, sub_y, 224, 780), "Inputs", (238, 249, 252)),
    ((255, sub_y, 418, 780), "Filter", (238, 249, 252)),
    ((449, sub_y, 612, 780), "Temporal Fusion", (238, 249, 252)),
    ((643, sub_y, 850, 780), "Support Surface", (238, 249, 252)),
]
for box, title, fill in steps:
    rounded(box, fill, outline=(99, 190, 215), width=2, radius=10)
    center((box[0], box[1] + 8, box[2], box[1] + 42), title, 18, bold=True, fill=(27, 111, 137))

paste_crop((35, 92, 300, 245), (58, 596, 208, 676))
text((65, 696), "Metric Pointmap", 14, bold=True)
text((65, 722), "Camera K", 14)
text((65, 744), "Person Mask", 12)
text((65, 762), "Neighbor Frames", 12)

text((276, 603), "Human exclusion", 15, bold=True)
text((276, 641), "Confidence filtering", 15, bold=True)
text((276, 679), "Depth discontinuity", 15)
text((276, 717), "Invalid point rejection", 15)

paste_crop((409, 190, 566, 275), (466, 601, 596, 688))
text((530, 711), "Align frames", 15, bold=True, anchor="mm")
text((530, 741), "Fuse local scene", 15, anchor="mm")

paste_crop((418, 281, 568, 525), (664, 591, 832, 712))
text((746, 731), "points | normals", 14, bold=True, anchor="mm")
text((746, 755), "roughness | confidence", 14, anchor="mm")

for x0, x1 in ((224, 255), (418, 449), (612, 643)):
    arrow((x0 + 5, 660), (x1 - 5, 660), color=(44, 139, 167), width=4, head=10)

rounded((42, 814, 405, 1138), (255, 255, 255), outline=(99, 190, 215), width=2, radius=10)
text((62, 834), "Body-anchored Geometric Probes", 20, (27, 111, 137), True)
paste_crop((445, 535, 568, 733), (112, 884, 280, 1077))
probe_tags = [("Heel", 304, 895), ("Toe", 304, 941), ("Ankle", 304, 987), ("Pelvis / Torso", 304, 1033)]
for label, x, y in probe_tags:
    tag((x - 7, y - 16, 390, y + 16), label, (44, 139, 167))

rounded((446, 814, 850, 1138), (255, 255, 255), outline=(222, 167, 60), width=2, radius=10)
text((466, 834), "Geometry-constrained Candidate Bank", 20, (169, 111, 15), True)
labels = ["No-op  Delta t = 0", "Left-foot support", "Right-foot support", "Bilateral support", "Body support / Abstain"]
colors = [(115, 120, 126), (224, 93, 73), (54, 128, 197), (219, 154, 36), (54, 150, 92)]
for i, (label, color) in enumerate(zip(labels, colors)):
    y = 892 + i * 47
    rounded((475, y, 821, y + 34), (252, 252, 252), outline=color, width=2, radius=8)
    text((491, y + 17), label, 15, color, True, "lm")
    if i > 0:
        arrow((770, y + 25), (803, y + 8), color=color, width=3, head=7)
arrow((405, 976), (446, 976), color=(44, 139, 167), width=5, head=12)
arrow((744, 780), (744, 814), color=(44, 139, 167), width=5, head=12)

# ---------------------------------------------------------------------------
# (C) Track-aware recurrent candidate selection and re-probing.
# ---------------------------------------------------------------------------
rounded((928, 535, 1130, 774), (255, 255, 255), outline=(233, 166, 124), width=2, radius=10)
text((945, 552), "Current State", 19, (172, 87, 52), True)
paste_crop((817, 210, 940, 407), (957, 595, 1045, 712))
tag((1050, 600, 1112, 630), "SMPL", (192, 104, 68))
tag((1050, 646, 1112, 676), "Probes", (192, 104, 68))
tag((1050, 692, 1112, 722), "ID t-1", (192, 104, 68))
text((1029, 751), "pose / shape frozen", 13, (100, 103, 108), True, "mm")

rounded((1170, 550, 1388, 760), (255, 244, 237), outline=(233, 166, 124), width=3, radius=12)
center((1180, 566, 1378, 608), "Transformer Selector", 20, (172, 87, 52), True)
for i in range(4):
    rounded((1210, 625 + i * 27, 1348, 645 + i * 27), (255, 255, 255), outline=(192, 104, 68), width=1, radius=5)
text((1279, 742), "shared weights", 13, (172, 87, 52), True, "mm")

rounded((1430, 535, 1755, 774), (255, 255, 255), outline=(233, 166, 124), width=2, radius=10)
text((1450, 552), "Selection & Safety", 19, (172, 87, 52), True)
prob_rows = [("No-op", .04), ("Left", .03), ("Right", .03), ("Bilateral", .87), ("Body", .03)]
for i, (label, prob) in enumerate(prob_rows):
    y = 601 + i * 29
    text((1450, y), label, 13, bold=label == "Bilateral")
    draw.rounded_rectangle((1530, y + 2, 1665, y + 17), radius=4, fill=(235, 236, 238))
    draw.rounded_rectangle((1530, y + 2, 1530 + max(3, int(135 * prob)), y + 17), radius=4, fill=(223, 148, 68))
    text((1680, y), f"{prob:.2f}", 12, bold=label == "Bilateral")
text((1450, 748), "High entropy -> Abstain", 14, (172, 87, 52), True)

arrow((1130, 655), (1170, 655), color=(192, 104, 68), width=5, head=12)
arrow((1388, 655), (1430, 655), color=(192, 104, 68), width=5, head=12)

rounded((960, 817, 1172, 1025), (255, 255, 255), outline=(233, 166, 124), width=2, radius=10)
text((976, 838), "Bounded Update", 19, (172, 87, 52), True)
center((975, 885, 1157, 930), "selected candidate\n+ small residual", 16, bold=True)
center((975, 953, 1157, 994), "Translation only", 16, fill=(172, 87, 52), bold=True)

rounded((1240, 817, 1452, 1025), (255, 255, 255), outline=(233, 166, 124), width=2, radius=10)
text((1257, 838), "Decode & Re-probe", 19, (172, 87, 52), True)
paste_crop((820, 235, 936, 400), (1290, 884, 1398, 984))

rounded((1520, 817, 1753, 1025), (244, 252, 247), outline=(94, 170, 112), width=2, radius=10)
text((1538, 838), "Grounded SMPL", 19, (44, 133, 76), True)
paste_crop((1412, 165, 1510, 560), (1580, 878, 1693, 991))

arrow((1592, 774), (1592, 817), color=(192, 104, 68), width=5, head=12)
arrow((1172, 922), (1240, 922), color=(192, 104, 68), width=5, head=12)
arrow((1452, 922), (1520, 922), color=(94, 170, 112), width=5, head=12)
draw.arc((1087, 735, 1510, 1080), 5, 198, fill=(192, 104, 68), width=5)
arrow((1100, 838), (1100, 790), color=(192, 104, 68), width=5, head=12)
text((1318, 1040), "Update SMPL -> Re-probe Scene", 15, (172, 87, 52), True, "mm")
for x, label in ((1205, "r=1"), (1320, "r=2"), (1435, "r=3")):
    draw.ellipse((x - 22, 1055, x + 22, 1099), fill=(255, 250, 246), outline=(192, 104, 68), width=3)
    center((x - 22, 1055, x + 22, 1099), label, 12, (172, 87, 52), True)

losses = ["Candidate CE", "Translation Loss", "Monotonic Loss", "Clean Invariance", "Temporal Consistency"]
for i, label in enumerate(losses):
    x0 = 928 + i * 165
    rounded((x0, 1121, x0 + 150, 1158), (255, 242, 234), outline=(233, 166, 124), width=1, radius=7)
    center((x0, 1121, x0 + 150, 1158), label, 13, (172, 87, 52), True)

TARGET.parent.mkdir(parents=True, exist_ok=True)
canvas.save(TARGET, quality=100)
print(TARGET)
