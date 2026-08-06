from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "outputs/vis/tsmr_paper_main_figure/tsmr_method_overview_v1.png"
TARGET = ROOT / "outputs/vis/tsmr_paper_main_figure/tsmr_detail_logic_panel.png"

W, H = 2400, 1200
canvas = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(canvas)
source = Image.open(SOURCE).convert("RGB")

FONT = "C:/Windows/Fonts/arial.ttf"
BOLD = "C:/Windows/Fonts/arialbd.ttf"

INK = (29, 33, 38)
MUTED = (91, 99, 108)
LINE = (74, 81, 89)
BLUE = (39, 131, 170)
BLUE_FILL = (239, 250, 253)
NAVY = (51, 103, 168)
NAVY_FILL = (239, 246, 255)
GOLD = (184, 123, 19)
GOLD_FILL = (255, 249, 231)
ORANGE = (198, 88, 47)
ORANGE_FILL = (255, 244, 237)
GREEN = (48, 143, 81)
GREEN_FILL = (240, 251, 244)
GRAY = (245, 247, 248)


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def rr(box, fill="white", outline=LINE, width=2, radius=12):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(xy, value, size=18, fill=INK, bold=False, anchor=None):
    draw.text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)


def center(box, value, size=18, fill=INK, bold=False, spacing=4):
    x0, y0, x1, y1 = box
    draw.multiline_text(
        ((x0 + x1) / 2, (y0 + y1) / 2), value,
        font=font(size, bold), fill=fill, anchor="mm",
        align="center", spacing=spacing,
    )


def arrow(start, end, color=LINE, width=4, head=12, dashed=False):
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if dashed and length:
        ux, uy = dx / length, dy / length
        pos = 0
        while pos < length - head:
            stop = min(pos + 12, length - head)
            draw.line((x0 + ux * pos, y0 + uy * pos, x0 + ux * stop, y0 + uy * stop), fill=color, width=width)
            pos += 20
    else:
        draw.line((x0, y0, x1, y1), fill=color, width=width)
    angle = math.atan2(dy, dx)
    p1 = (x1 - head * math.cos(angle - .55), y1 - head * math.sin(angle - .55))
    p2 = (x1 - head * math.cos(angle + .55), y1 - head * math.sin(angle + .55))
    draw.polygon(((x1, y1), p1, p2), fill=color)


def routed(points, color=LINE, width=4, head=12, dashed=False):
    for a, b in zip(points[:-2], points[1:-1]):
        draw.line((*a, *b), fill=color, width=width)
    arrow(points[-2], points[-1], color=color, width=width, head=head, dashed=dashed)


def crop(src_box, dst_box, contain=True):
    image = source.crop(src_box)
    width, height = dst_box[2] - dst_box[0], dst_box[3] - dst_box[1]
    if contain:
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        x = dst_box[0] + (width - image.width) // 2
        y = dst_box[1] + (height - image.height) // 2
        canvas.paste(image, (x, y))
    else:
        canvas.paste(image.resize((width, height), Image.Resampling.LANCZOS), dst_box[:2])


def chip(box, label, color, size=15, fill="white"):
    rr(box, fill=fill, outline=color, width=2, radius=7)
    center(box, label, size=size, fill=color, bold=True)


def numbered_title(x, y, number, title, color):
    draw.ellipse((x, y, x + 38, y + 38), fill=color)
    center((x, y, x + 38, y + 38), str(number), 18, "white", True)
    text((x + 52, y + 19), title, 22, color, True, "lm")


# Outer panel and header.
rr((18, 16, 2382, 1182), fill="white", outline=(69, 75, 82), width=3, radius=15)
text((42, 37), "TSMR: Track-aware Support Manifold Refinement", 31, INK, True)
text((42, 80), "Geometry proposes safe translations; temporal identity selects when and how to apply them.", 18, MUTED)

# Input bus.
rr((42, 126, 2358, 270), fill=(248, 250, 251), outline=(160, 166, 172), width=2, radius=10)
text((62, 146), "Inputs from frozen RGB-to-HSI backbone", 18, NAVY, True)
input_cards = [
    ((70, 188, 340, 245), "Metric pointmap  P_t", NAVY),
    ((380, 188, 650, 245), "Camera  K_t, T_t", NAVY),
    ((690, 188, 960, 245), "Confidence + masks", BLUE),
    ((1000, 188, 1270, 245), "Aligned SMPL  M_t", GREEN),
    ((1310, 188, 1580, 245), "Track ID  k", NAVY),
    ((1620, 188, 1890, 245), "Neighbor frames", BLUE),
    ((1930, 188, 2325, 245), "Stage2 translation  t_t^(0)", GREEN),
]
for box, label, color in input_cards:
    chip(box, label, color, size=16)

arrow((500, 270), (500, 304), color=BLUE, width=4, head=10)
arrow((1450, 270), (1450, 304), color=NAVY, width=4, head=10)
arrow((2180, 270), (2180, 304), color=GREEN, width=4, head=10)

# Three principal method regions.
geom = (42, 310, 780, 924)
cand = (814, 310, 1286, 924)
learned = (1320, 310, 2358, 924)
rr(geom, BLUE_FILL, outline=BLUE, width=4, radius=13)
rr(cand, GOLD_FILL, outline=GOLD, width=4, radius=13)
rr(learned, ORANGE_FILL, outline=ORANGE, width=4, radius=13)
numbered_title(64, 329, 1, "Support Manifold & Probes", BLUE)
numbered_title(836, 329, 2, "Analytic Candidate Bank", GOLD)
numbered_title(1342, 329, 3, "Track-aware Recurrent Selection", ORANGE)

# Region 1: support manifold and probes.
rr((66, 401, 352, 645), fill="white", outline=BLUE, width=2, radius=9)
text((84, 418), "Scene-only Surface", 18, BLUE, True)
crop((420, 188, 568, 356), (91, 465, 214, 550))
crop((430, 445, 570, 525), (219, 456, 333, 535))
text((209, 578), "exclude human pixels", 14, INK, True, "mm")
text((209, 605), "filter confidence", 14, MUTED, False, "mm")
text((209, 628), "fuse t-1, t, t+1", 14, BLUE, True, "mm")

rr((384, 401, 754, 645), fill="white", outline=BLUE, width=2, radius=9)
text((404, 418), "Local Support Manifold  S_t", 18, BLUE, True)
crop((418, 281, 568, 525), (420, 459, 581, 577))
text((609, 473), "point", 14, INK, True)
text((609, 503), "normal", 14, INK, True)
text((609, 533), "roughness", 14, INK, True)
text((609, 563), "confidence", 14, INK, True)
text((569, 611), "S_t = {p, n, rho, c}", 15, BLUE, True, "mm")
arrow((352, 524), (378, 524), color=BLUE, width=4, head=10)

rr((66, 681, 754, 892), fill="white", outline=BLUE, width=2, radius=9)
text((84, 699), "Body-anchored Geometric Probes", 18, BLUE, True)
crop((445, 535, 568, 733), (92, 740, 244, 862))
probe_data = [
    ("Heel L/R", (221, 75, 59)),
    ("Toe L/R", (37, 117, 192)),
    ("Ankle L/R", (202, 137, 25)),
    ("Pelvis / Torso", (46, 143, 80)),
]
for i, (label, color) in enumerate(probe_data):
    chip((286 + (i % 2) * 205, 744 + (i // 2) * 47, 474 + (i % 2) * 205, 779 + (i // 2) * 47), label, color, size=14)
center((286, 838, 698, 878), "q_i = [signed distance, normal, confidence]", 14, BLUE, True)
arrow((569, 645), (569, 675), color=BLUE, width=4, head=10)

# Region 2: analytic candidate bank.
text((1050, 393), "C_t = {Delta t_0, Delta t_L, Delta t_R, Delta t_B, Delta t_K}", 15, GOLD, True, "mm")
candidates = [
    ("No-op", "Delta t_0 = 0", MUTED),
    ("Left support", "h_L n_L", (218, 78, 59)),
    ("Right support", "h_R n_R", (39, 115, 192)),
    ("Bilateral", "h_B n_B", (208, 138, 19)),
    ("Body / Abstain", "h_K n_K", GREEN),
]
for i, (name, formula, color) in enumerate(candidates):
    y = 433 + i * 76
    rr((844, y, 1256, y + 57), fill="white", outline=color, width=2, radius=8)
    text((864, y + 28), name, 16, color, True, "lm")
    text((1235, y + 28), formula, 15, color, True, "rm")

rr((844, 835, 1256, 891), fill=(255, 252, 241), outline=GOLD, width=1, radius=7)
center((844, 835, 1256, 891), "Closed-form geometry, no learned large jump", 14, GOLD, True)

arrow((780, 617), (808, 617), color=GOLD, width=5, head=11)

# Region 3: tokens, selector, safety, recurrent update.
rr((1344, 401, 1536, 648), fill="white", outline=ORANGE, width=2, radius=9)
text((1362, 418), "Probe Tokens", 18, ORANGE, True)
crop((817, 210, 940, 407), (1375, 466, 1465, 578))
for i, color in enumerate(((224, 80, 61), (229, 145, 42), (68, 153, 93), (53, 124, 193), (113, 96, 177))):
    draw.rounded_rectangle((1480, 469 + i * 30, 1504, 493 + i * 30), radius=4, fill=color)
text((1440, 615), "Q_t^(r)", 15, ORANGE, True, "mm")

rr((1570, 401, 1778, 648), fill=NAVY_FILL, outline=NAVY, width=2, radius=9)
text((1588, 418), "ID Memory", 18, NAVY, True)
for i, label in enumerate(("t-2", "t-1", "t")):
    rr((1592 + i * 56, 487, 1634 + i * 56, 548), fill="white", outline=NAVY, width=2, radius=5)
    center((1592 + i * 56, 487, 1634 + i * 56, 548), label, 12, NAVY, True)
arrow((1608, 578), (1740, 578), color=NAVY, width=3, head=9, dashed=True)
text((1674, 615), "m_k", 15, NAVY, True, "mm")

rr((1812, 401, 2044, 648), fill="white", outline=ORANGE, width=3, radius=9)
text((1928, 419), "Transformer", 18, ORANGE, True, "mm")
text((1928, 446), "Selector", 18, ORANGE, True, "mm")
for i in range(4):
    rr((1852, 491 + i * 31, 2004, 512 + i * 31), fill=ORANGE_FILL, outline=ORANGE, width=1, radius=4)
text((1928, 624), "shared for r=1..3", 13, ORANGE, True, "mm")

rr((2078, 401, 2334, 648), fill="white", outline=ORANGE, width=2, radius=9)
text((2096, 418), "Selection & Abstention", 17, ORANGE, True)
for i, (label, prob) in enumerate((("No-op", .04), ("Left", .05), ("Right", .03), ("Bilateral", .84), ("Body", .04))):
    y = 473 + i * 28
    text((2098, y), label, 12, INK, label == "Bilateral")
    draw.rounded_rectangle((2170, y + 1, 2290, y + 14), radius=4, fill=(230, 232, 235))
    draw.rounded_rectangle((2170, y + 1, 2170 + max(3, int(prob * 120)), y + 14), radius=4, fill=ORANGE)
    text((2314, y), f"{prob:.2f}", 11, INK, label == "Bilateral", "rm")
text((2206, 621), "high uncertainty -> no-op", 13, ORANGE, True, "mm")

routed(((1536, 452), (1550, 452), (1550, 383), (1794, 383), (1794, 452), (1806, 452)), color=ORANGE, width=5, head=11)
arrow((1778, 579), (1806, 579), color=NAVY, width=3, head=9, dashed=True)
arrow((2044, 524), (2072, 524), color=ORANGE, width=5, head=11)

# Lower recurrent sequence.
lower_boxes = [
    ((1360, 704, 1580, 876), "Select candidate", "Delta t_c", ORANGE),
    ((1630, 704, 1850, 876), "Bounded residual", "clip(delta t_r)", ORANGE),
    ((1900, 704, 2120, 876), "Translation update", "t^(r+1) = t^(r) + Delta t", ORANGE),
    ((2170, 704, 2334, 876), "Decode & re-probe", "M^(r+1), Q^(r+1)", GREEN),
]
for box, title, formula, color in lower_boxes:
    rr(box, fill="white" if color == ORANGE else GREEN_FILL, outline=color, width=2, radius=9)
    center((box[0] + 8, box[1] + 17, box[2] - 8, box[1] + 62), title, 16, color, True)
    center((box[0] + 10, box[1] + 82, box[2] - 10, box[3] - 15), formula, 14, INK, True)

routed(((2206, 648), (2206, 677), (1470, 677), (1470, 698)), color=ORANGE, width=5, head=11)
for x0, x1 in ((1580, 1630), (1850, 1900), (2120, 2170)):
    arrow((x0 + 5, 790), (x1 - 5, 790), color=ORANGE if x1 < 2170 else GREEN, width=5, head=11)

# Recurrent loop from re-probe back to tokens.
routed(((2252, 876), (2252, 904), (1440, 904), (1440, 654)), color=ORANGE, width=4, head=11)
for x, label in ((1800, "r=1"), (1880, "r=2"), (1960, "r=3")):
    draw.ellipse((x - 24, 883, x + 24, 931), fill=ORANGE_FILL, outline=ORANGE, width=2)
    center((x - 24, 883, x + 24, 931), label, 12, ORANGE, True)
text((2058, 909), "update -> re-probe", 14, ORANGE, True, "mm")

# Output and core design statement.
rr((42, 952, 830, 1139), fill=(248, 250, 251), outline=(146, 152, 159), width=2, radius=10)
text((62, 971), "Inference principle", 18, INK, True)
text((62, 1010), "1. Geometry proposes physically meaningful translations.", 16, INK)
text((62, 1044), "2. ID-aware temporal reasoning decides apply vs. abstain.", 16, INK)
text((62, 1078), "3. Only translation changes; Stage2 pose, shape, and scene stay fixed.", 16, INK)
text((62, 1112), "4. Re-probing verifies the new human-scene state at every step.", 16, INK)

rr((866, 952, 2034, 1139), fill=ORANGE_FILL, outline=(229, 148, 101), width=2, radius=10)
text((886, 971), "Training supervision", 18, ORANGE, True)
losses = ["Candidate CE", "Translation", "Monotonic", "Clean Invariance", "Temporal Consistency"]
for i, label in enumerate(losses):
    x0 = 890 + i * 220
    chip((x0, 1020, x0 + 202, 1070), label, ORANGE, size=14, fill="white")
center((886, 1090, 2015, 1125), "GT contact teachers supervise selection; clean frames teach no-op and protect accepted alignment.", 14, MUTED, False)

rr((2070, 952, 2358, 1139), fill=GREEN_FILL, outline=GREEN, width=3, radius=10)
text((2214, 971), "Output", 18, GREEN, True, "mm")
crop((1412, 165, 1510, 560), (2100, 1003, 2194, 1111))
center((2200, 1005, 2335, 1072), "Grounded\ntracked SMPL", 17, GREEN, True)
center((2200, 1075, 2335, 1123), "Delta t* | mode\nconfidence", 11, MUTED, False)

TARGET.parent.mkdir(parents=True, exist_ok=True)
canvas.save(TARGET, quality=100)
print(TARGET)
