from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "outputs/vis/tsmr_paper_main_figure/tsmr_method_overview_v1.png"
TARGET = ROOT / "outputs/vis/tsmr_paper_main_figure/tsmr_full_pipeline_abc_paper.png"

W, H = 2400, 1600
canvas = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(canvas)
source = Image.open(SOURCE).convert("RGB")

FONT = "C:/Windows/Fonts/arial.ttf"
BOLD = "C:/Windows/Fonts/arialbd.ttf"

INK = (30, 34, 39)
MUTED = (93, 101, 110)
LINE = (78, 85, 93)
BLUE = (42, 126, 165)
BLUE_FILL = (238, 249, 253)
NAVY = (55, 104, 166)
NAVY_FILL = (239, 246, 255)
ORANGE = (196, 91, 51)
ORANGE_FILL = (255, 245, 238)
GOLD = (185, 124, 22)
GOLD_FILL = (255, 249, 232)
GREEN = (48, 142, 82)
GREEN_FILL = (240, 251, 244)
GRAY_FILL = (247, 248, 249)


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def rr(box, fill="white", outline=LINE, width=2, radius=12):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def txt(xy, value, size=18, fill=INK, bold=False, anchor=None):
    draw.text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)


def centered(box, value, size=18, fill=INK, bold=False, spacing=4):
    x0, y0, x1, y1 = box
    draw.multiline_text(
        ((x0 + x1) / 2, (y0 + y1) / 2),
        value,
        font=font(size, bold),
        fill=fill,
        anchor="mm",
        align="center",
        spacing=spacing,
    )


def arrow(start, end, color=LINE, width=4, head=12, dashed=False):
    x0, y0 = start
    x1, y1 = end
    if dashed:
        length = math.hypot(x1 - x0, y1 - y0)
        if length > 0:
            ux, uy = (x1 - x0) / length, (y1 - y0) / length
            pos = 0
            while pos < length - head:
                stop = min(pos + 11, length - head)
                draw.line((x0 + ux * pos, y0 + uy * pos, x0 + ux * stop, y0 + uy * stop), fill=color, width=width)
                pos += 19
    else:
        draw.line((x0, y0, x1, y1), fill=color, width=width)
    angle = math.atan2(y1 - y0, x1 - x0)
    p1 = (x1 - head * math.cos(angle - 0.55), y1 - head * math.sin(angle - 0.55))
    p2 = (x1 - head * math.cos(angle + 0.55), y1 - head * math.sin(angle + 0.55))
    draw.polygon(((x1, y1), p1, p2), fill=color)


def poly_arrow(points, color=LINE, width=4, head=12, dashed=False):
    for index in range(len(points) - 2):
        draw.line((*points[index], *points[index + 1]), fill=color, width=width)
    arrow(points[-2], points[-1], color=color, width=width, head=head, dashed=dashed)


def crop(src_box, dst_box, contain=True):
    image = source.crop(src_box)
    dw, dh = dst_box[2] - dst_box[0], dst_box[3] - dst_box[1]
    if contain:
        image.thumbnail((dw, dh), Image.Resampling.LANCZOS)
        x = dst_box[0] + (dw - image.width) // 2
        y = dst_box[1] + (dh - image.height) // 2
        canvas.paste(image, (x, y))
    else:
        canvas.paste(image.resize((dw, dh), Image.Resampling.LANCZOS), dst_box[:2])


def chip(box, value, color, fill="white", size=15):
    rr(box, fill=fill, outline=color, width=2, radius=7)
    centered(box, value, size=size, fill=color, bold=True)


def snowflake(x, y, size=15, color=NAVY):
    for angle in (0, math.pi / 3, 2 * math.pi / 3):
        dx, dy = math.cos(angle) * size, math.sin(angle) * size
        draw.line((x - dx, y - dy, x + dx, y + dy), fill=color, width=3)


def panel_title(x, y, tag, title, color=INK):
    txt((x, y), f"({tag})", 29, color, True)
    txt((x + 56, y), title, 29, color, True)


# ---------------------------------------------------------------------------
# Panels and legend.
# ---------------------------------------------------------------------------
rr((18, 16, 2382, 610), fill="white", outline=(78, 83, 90), width=3, radius=14)
rr((18, 632, 1187, 1578), fill=(250, 254, 255), outline=(88, 182, 211), width=5, radius=14)
rr((1212, 632, 2382, 1578), fill=(255, 252, 249), outline=(232, 151, 105), width=5, radius=14)

panel_title(38, 29, "A", "End-to-End Overview")
panel_title(38, 649, "B", "Support Manifold & Analytic Hypotheses", BLUE)
panel_title(1232, 649, "C", "Track-aware Recurrent Refinement", ORANGE)

snowflake(1900, 50, 13)
txt((1925, 50), "Frozen baseline", 16, NAVY, True, "lm")
rr((2078, 36, 2106, 64), ORANGE_FILL, ORANGE, 2, 5)
txt((2118, 50), "Trainable TSMR", 16, ORANGE, True, "lm")
arrow((2246, 50), (2294, 50), color=LINE, width=3, head=9, dashed=True)
txt((2304, 50), "ID memory", 14, MUTED, False, "lm")

# ---------------------------------------------------------------------------
# (A) Complete RGB-to-output architecture.
# ---------------------------------------------------------------------------
rgb = (42, 102, 274, 554)
obs = (310, 102, 540, 554)
backbone = (582, 82, 1332, 574)
aligned = (1374, 102, 1656, 554)
tsmr = (1698, 102, 2042, 554)
output = (2084, 102, 2358, 554)

rr(rgb, GRAY_FILL, outline=(150, 155, 161), width=2)
txt((62, 121), "RGB Video", 23, INK, True)
crop((35, 90, 300, 245), (60, 168, 256, 329), contain=False)
for y in (355, 386, 417):
    draw.rounded_rectangle((84, y, 232, y + 22), radius=4, fill=(227, 231, 234))
txt((158, 480), "I1 ... IT", 19, MUTED, True, "mm")

rr(obs, BLUE_FILL, outline=BLUE, width=2)
txt((330, 121), "Observation Frontend", 21, BLUE, True)
for i, (label, color) in enumerate((("Person boxes", (49, 154, 170)), ("Track IDs", (35, 132, 184)), ("Human masks", (76, 143, 104)))):
    chip((344, 193 + i * 68, 506, 235 + i * 68), label, color, size=16)
txt((425, 425), "YOLO / GT boxes", 15, MUTED, False, "mm")
txt((425, 451), "BoostTrack++ / GT IDs", 15, MUTED, False, "mm")
txt((425, 489), "side information", 14, BLUE, True, "mm")

rr(backbone, (248, 250, 252), outline=(109, 119, 131), width=3)
snowflake(610, 111, 13)
txt((635, 111), "VGGT-Omega Reconstruction Backbone", 22, NAVY, True, "lm")

rr((610, 153, 1304, 327), NAVY_FILL, outline=(103, 146, 200), width=2, radius=10)
txt((630, 169), "Scene Geometry Branch", 18, NAVY, True)
chip((636, 214, 804, 276), "VGGT\nAggregator", NAVY, fill="white", size=17)
chip((842, 199, 1006, 245), "Camera Head", NAVY, fill="white", size=16)
chip((842, 259, 1006, 305), "Dense Head", NAVY, fill="white", size=16)
arrow((804, 245), (836, 222), color=NAVY, width=3, head=9)
arrow((804, 245), (836, 282), color=NAVY, width=3, head=9)
rr((1044, 194, 1280, 309), fill="white", outline=NAVY, width=2, radius=9)
crop((420, 188, 568, 356), (1062, 210, 1164, 288))
txt((1170, 219), "Camera K, T", 15, NAVY, True)
txt((1170, 251), "Depth + conf.", 15, NAVY, True)
txt((1170, 283), "Metric pointmap", 15, NAVY, True)
arrow((1006, 222), (1038, 222), color=NAVY, width=3, head=9)
arrow((1006, 282), (1038, 282), color=NAVY, width=3, head=9)

rr((610, 351, 1304, 530), (247, 252, 250), outline=(91, 158, 119), width=2, radius=10)
txt((630, 367), "Human Reconstruction Branch", 18, (50, 127, 79), True)
chip((636, 414, 855, 489), "NLF SMPL\nProvider", (50, 127, 79), fill="white", size=18)
crop((817, 210, 940, 407), (888, 391, 1007, 511))
rr((1038, 400, 1280, 513), fill="white", outline=(50, 127, 79), width=2, radius=9)
txt((1056, 420), "Initial SMPL", 17, (50, 127, 79), True)
txt((1056, 454), "pose | shape", 15, MUTED)
txt((1056, 483), "camera-space transl.", 15, MUTED)
arrow((855, 451), (880, 451), color=(50, 127, 79), width=3, head=9)
arrow((1008, 451), (1033, 451), color=(50, 127, 79), width=3, head=9)

rr(aligned, (243, 249, 255), outline=NAVY, width=2)
snowflake(1400, 132, 12)
txt((1424, 132), "Frozen HSI", 20, NAVY, True, "lm")
chip((1402, 185, 1628, 247), "Stage1 Scene Affine", NAVY, fill="white", size=17)
chip((1402, 275, 1628, 354), "Stage2 Human-Scene\nAlignment", NAVY, fill="white", size=17)
txt((1515, 387), "translation only", 15, NAVY, True, "mm")
crop((95, 275, 235, 560), (1451, 411, 1578, 520))
txt((1515, 529), "Aligned scene + tracked SMPL", 14, MUTED, True, "mm")
arrow((1515, 247), (1515, 270), color=NAVY, width=3, head=9)

rr(tsmr, ORANGE_FILL, outline=ORANGE, width=4)
txt((1722, 121), "Proposed TSMR", 23, ORANGE, True)
txt((1722, 151), "Track-aware Support Manifold Refinement", 14, ORANGE, True)
steps = ["Support manifold", "Analytic hypotheses", "Track-aware selector", "Bounded update", "Re-probe x3"]
for i, label in enumerate(steps):
    y = 202 + i * 58
    rr((1736, y, 2004, y + 40), fill="white", outline=ORANGE, width=2, radius=7)
    txt((1756, y + 20), f"{i + 1}", 15, "white", True, "mm")
    draw.ellipse((1741, y + 5, 1771, y + 35), fill=ORANGE)
    txt((1785, y + 20), label, 16, ORANGE, True, "lm")
txt((1870, 505), "trainable", 15, ORANGE, True, "mm")

rr(output, GREEN_FILL, outline=GREEN, width=3)
txt((2105, 121), "Grounded 4D Output", 21, GREEN, True)
crop((1268, 132, 1527, 568), (2112, 166, 2330, 405))
chip((2121, 430, 2321, 472), "Tracked SMPL", GREEN, fill="white", size=16)
chip((2121, 485, 2321, 527), "Metric scene", GREEN, fill="white", size=16)

# Main forward arrows and cross-branch inputs.
arrow((274, 275), (304, 275), width=5, head=12)
poly_arrow(((540, 207), (558, 207), (576, 236)), color=NAVY, width=4, head=11)
txt((558, 187), "RGB", 13, NAVY, True, "mm")
poly_arrow(((540, 274), (566, 274), (566, 451), (603, 451)), color=(50, 127, 79), width=4, head=11)
poly_arrow(((1332, 246), (1352, 246), (1352, 225), (1368, 225)), color=NAVY, width=4, head=11)
poly_arrow(((1332, 451), (1352, 451), (1352, 330), (1368, 330)), color=(50, 127, 79), width=4, head=11)
arrow((1656, 328), (1692, 328), color=ORANGE, width=5, head=12)
arrow((2042, 328), (2078, 328), color=GREEN, width=5, head=12)
poly_arrow(((506, 282), (550, 282), (550, 545), (1860, 545), (1860, 524)), color=NAVY, width=3, head=10, dashed=True)
txt((1195, 563), "track identity t-1 -> t", 14, NAVY, True, "mm")

# ---------------------------------------------------------------------------
# (B) Confidence-aware support construction and candidate bank.
# ---------------------------------------------------------------------------
stage_boxes = [
    ((42, 714, 255, 985), "Inputs"),
    ((290, 714, 502, 985), "Human Exclusion"),
    ((537, 714, 750, 985), "Temporal Fusion"),
    ((785, 714, 1148, 985), "Support Manifold"),
]
for box, title in stage_boxes:
    rr(box, fill="white", outline=BLUE, width=2, radius=10)
    txt((box[0] + 18, box[1] + 17), title, 19, BLUE, True)

crop((35, 90, 300, 245), (62, 770, 235, 864), contain=False)
txt((149, 890), "Pointmap + K", 15, INK, True, "mm")
txt((149, 920), "masks + confidence", 14, MUTED, False, "mm")
txt((149, 948), "neighbor frames", 14, MUTED, False, "mm")

draw.ellipse((348, 780, 446, 878), fill=(236, 243, 246), outline=BLUE, width=3)
draw.line((355, 865, 438, 790), fill=ORANGE, width=8)
txt((396, 904), "Remove human pixels", 15, INK, True, "mm")
txt((396, 934), "Reject low confidence", 14, MUTED, False, "mm")
txt((396, 960), "Preserve scene only", 14, BLUE, True, "mm")

crop((409, 190, 566, 275), (558, 779, 730, 876))
txt((643, 905), "Warp t-1, t, t+1", 15, INK, True, "mm")
txt((643, 936), "Fuse local geometry", 14, MUTED, False, "mm")
txt((643, 961), "confidence weighted", 14, BLUE, True, "mm")

crop((418, 281, 568, 525), (808, 761, 945, 895))
crop((430, 445, 570, 525), (968, 768, 1120, 866))
txt((965, 908), "points | normals | roughness", 15, INK, True, "mm")
txt((965, 939), "local support surface", 14, BLUE, True, "mm")
txt((965, 965), "with confidence", 14, MUTED, False, "mm")

for x0, x1 in ((255, 290), (502, 537), (750, 785)):
    arrow((x0 + 4, 850), (x1 - 4, 850), color=BLUE, width=4, head=10)

rr((42, 1022, 492, 1518), fill="white", outline=BLUE, width=2, radius=10)
txt((62, 1043), "Body-anchored Probes", 21, BLUE, True)
crop((445, 535, 568, 733), (78, 1104, 273, 1398))
for i, (label, color) in enumerate((("Heel", (223, 80, 65)), ("Toe", (43, 122, 198)), ("Ankle", (198, 136, 33)), ("Pelvis", (65, 150, 91)), ("Torso", (117, 102, 181)))):
    chip((292, 1101 + i * 59, 458, 1140 + i * 59), label, color, size=16)
txt((267, 1424), "Nearest surface point", 14, MUTED, False, "mm")
txt((267, 1452), "signed distance + normal", 15, BLUE, True, "mm")
txt((267, 1480), "bilateral consistency", 14, MUTED, False, "mm")

rr((530, 1022, 1148, 1518), fill="white", outline=GOLD, width=2, radius=10)
txt((550, 1043), "Geometry-constrained Candidate Bank", 21, GOLD, True)
candidates = [
    ("No-op", "Delta t = 0", MUTED),
    ("Left-foot support", "+ hL nL", (218, 78, 58)),
    ("Right-foot support", "+ hR nR", (39, 116, 194)),
    ("Bilateral support", "+ hB nB", (210, 139, 19)),
    ("Body support / Abstain", "+ hK nK", GREEN),
]
for i, (name, delta, color) in enumerate(candidates):
    y = 1101 + i * 66
    rr((560, y, 1118, y + 48), fill=GRAY_FILL, outline=color, width=2, radius=8)
    txt((580, y + 24), name, 16, color, True, "lm")
    txt((1088, y + 24), delta, 15, color, True, "rm")
txt((839, 1457), "Analytic translation hypotheses", 15, GOLD, True, "mm")
txt((839, 1484), "computed before learning", 14, MUTED, False, "mm")
arrow((492, 1265), (524, 1265), color=BLUE, width=5, head=11)
arrow((965, 985), (965, 1016), color=BLUE, width=5, head=11)

# ---------------------------------------------------------------------------
# (C) Track-aware selection, bounded update, and recurrent re-probing.
# ---------------------------------------------------------------------------
rr((1238, 714, 1482, 1008), fill="white", outline=ORANGE, width=2, radius=10)
txt((1256, 733), "Probe Tokens", 20, ORANGE, True)
crop((817, 210, 940, 407), (1270, 786, 1372, 931))
for i, color in enumerate(((225, 83, 61), (231, 146, 47), (79, 156, 101), (62, 130, 199), (119, 102, 180))):
    draw.rounded_rectangle((1392, 791 + i * 35, 1420, 819 + i * 35), radius=4, fill=color)
txt((1360, 967), "SMPL + geometry", 14, MUTED, True, "mm")

rr((1520, 714, 1807, 1008), ORANGE_FILL, outline=ORANGE, width=3, radius=10)
txt((1541, 733), "Transformer Selector", 20, ORANGE, True)
for i in range(5):
    rr((1570, 798 + i * 36, 1755, 823 + i * 36), fill="white", outline=ORANGE, width=1, radius=5)
txt((1663, 974), "shared across r", 14, ORANGE, True, "mm")

rr((1845, 714, 2132, 1008), NAVY_FILL, outline=NAVY, width=2, radius=10)
txt((1865, 733), "ID Memory", 20, NAVY, True)
for i, label in enumerate(("t-3", "t-2", "t-1", "t")):
    rr((1870 + i * 60, 821, 1914 + i * 60, 885), fill="white", outline=NAVY, width=2, radius=6)
    centered((1870 + i * 60, 821, 1914 + i * 60, 885), label, 13, NAVY, True)
arrow((1892, 914), (2074, 914), color=NAVY, width=3, head=9, dashed=True)
txt((1988, 958), "track-conditioned state", 14, NAVY, True, "mm")

rr((2170, 714, 2356, 1008), fill="white", outline=ORANGE, width=2, radius=10)
txt((2188, 733), "Decision", 20, ORANGE, True)
for i, (label, value) in enumerate((("No-op", .05), ("Left", .08), ("Right", .04), ("Bilateral", .79), ("Body", .04))):
    y = 790 + i * 34
    txt((2190, y), label, 13, INK, label == "Bilateral")
    draw.rounded_rectangle((2260, y + 2, 2328, y + 16), radius=4, fill=(230, 232, 235))
    draw.rounded_rectangle((2260, y + 2, 2260 + max(3, int(68 * value)), y + 16), radius=4, fill=ORANGE)
txt((2263, 969), "uncertain -> abstain", 13, ORANGE, True, "mm")

arrow((1482, 858), (1514, 858), color=ORANGE, width=5, head=11)
arrow((1845, 858), (1813, 858), color=NAVY, width=3, head=10, dashed=True)
poly_arrow(((1807, 800), (1825, 800), (1825, 690), (2150, 690), (2150, 858), (2164, 858)), color=ORANGE, width=5, head=11)

rr((1260, 1051, 1516, 1328), fill="white", outline=ORANGE, width=2, radius=10)
txt((1278, 1071), "Bounded Update", 20, ORANGE, True)
centered((1280, 1132, 1496, 1198), "selected analytic candidate\n+ small learned residual", 16, INK, True)
centered((1280, 1220, 1496, 1262), "Translation only", 17, ORANGE, True)
txt((1388, 1291), "pose / shape fixed", 14, MUTED, True, "mm")

rr((1586, 1051, 1842, 1328), fill="white", outline=ORANGE, width=2, radius=10)
txt((1604, 1071), "Decode SMPL", 20, ORANGE, True)
crop((820, 235, 936, 400), (1650, 1131, 1779, 1265))
txt((1714, 1291), "updated human state", 14, MUTED, True, "mm")

rr((1912, 1051, 2168, 1328), fill="white", outline=ORANGE, width=2, radius=10)
txt((1930, 1071), "Re-probe Scene", 20, ORANGE, True)
crop((430, 445, 570, 525), (1960, 1130, 2120, 1220))
for x, label in ((1970, "r=1"), (2040, "r=2"), (2110, "r=3")):
    draw.ellipse((x - 25, 1250, x + 25, 1300), fill=ORANGE_FILL, outline=ORANGE, width=2)
    centered((x - 25, 1250, x + 25, 1300), label, 12, ORANGE, True)

rr((2238, 1051, 2356, 1328), GREEN_FILL, outline=GREEN, width=2, radius=10)
txt((2297, 1071), "Output", 19, GREEN, True, "mm")
crop((1412, 165, 1510, 560), (2257, 1122, 2337, 1248))
centered((2250, 1260, 2344, 1310), "Grounded\nSMPL", 15, GREEN, True)

poly_arrow(((2263, 1008), (2263, 1028), (1388, 1028), (1388, 1045)), color=ORANGE, width=5, head=11)
arrow((1516, 1189), (1580, 1189), color=ORANGE, width=5, head=11)
arrow((1842, 1189), (1906, 1189), color=ORANGE, width=5, head=11)
arrow((2168, 1189), (2232, 1189), color=GREEN, width=5, head=11)
poly_arrow(((2040, 1328), (2040, 1350), (1388, 1350), (1388, 1334)), color=ORANGE, width=4, head=10)
txt((1714, 1370), "shared-weight recurrent refinement", 15, ORANGE, True, "mm")

# Loss strip.
losses = ["Candidate CE", "Translation", "Monotonic", "Clean Invariance", "Temporal Consistency"]
for i, label in enumerate(losses):
    x0 = 1240 + i * 220
    rr((x0, 1430, x0 + 204, 1480), fill=ORANGE_FILL, outline=(230, 148, 101), width=1, radius=7)
    centered((x0, 1430, x0 + 204, 1480), label, 14, ORANGE, True)
txt((1239, 1510), "Training only:", 15, MUTED, True)
txt((1342, 1510), "supervise candidate selection, safe translation, monotonic improvement, and track consistency", 15, MUTED)

TARGET.parent.mkdir(parents=True, exist_ok=True)
canvas.save(TARGET, quality=100)
print(TARGET)
