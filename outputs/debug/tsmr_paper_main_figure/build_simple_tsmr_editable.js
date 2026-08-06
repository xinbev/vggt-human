const pptxgen = require("pptxgenjs");
const path = require("path");

const pptx = new pptxgen();
pptx.defineLayout({ name: "PAPER_FIG", width: 13.333, height: 7.5 });
pptx.layout = "PAPER_FIG";
pptx.author = "vggt-omega";
pptx.subject = "Editable TSMR paper method figure";
pptx.title = "Simple TSMR Method Figure";
pptx.company = "vggt-omega";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  lang: "en-US",
};

const ROOT = path.resolve(__dirname, "..", "..", "..");
const ASSET = path.join(ROOT, "outputs", "vis", "tsmr_paper_main_figure", "assets", "simple_tsmr");
const OUT = path.join(ROOT, "outputs", "vis", "tsmr_paper_main_figure", "tsmr_simple_editable.pptx");

const C = {
  ink: "22272C",
  muted: "68717A",
  line: "6F7881",
  blue: "2783AA",
  blueFill: "EFFAFC",
  gold: "B87B13",
  goldFill: "FFF9E7",
  orange: "C8582F",
  orangeFill: "FFF4ED",
  green: "309052",
  greenFill: "F0FBF4",
  grayFill: "F7F8F9",
  white: "FFFFFF",
};

const slide = pptx.addSlide();
slide.background = { color: C.white };

function shape(type, x, y, w, h, fill, line, radius = false) {
  slide.addShape(radius ? pptx.ShapeType.roundRect : type, {
    x, y, w, h,
    rectRadius: radius ? 0.08 : undefined,
    fill: { color: fill },
    line: { color: line, width: 1.4 },
  });
}

function label(text, x, y, w, h, size = 14, color = C.ink, bold = false, align = "left") {
  slide.addText(text, {
    x, y, w, h, fontFace: "Arial", fontSize: size, color,
    bold, align, valign: "mid", margin: 0, breakLine: false,
  });
}

function arrow(x, y, w, h, color = C.line, width = 2.2, dash = "solid") {
  slide.addShape(pptx.ShapeType.line, {
    x, y, w, h,
    line: { color, width, dashType: dash, endArrowType: "triangle" },
  });
}

function image(name, x, y, w, h, altText) {
  slide.addImage({
    path: path.join(ASSET, name), x, y, w, h,
    sizing: { type: "contain", w, h }, altText,
  });
}

function moduleTitle(number, title, x, y, color) {
  slide.addShape(pptx.ShapeType.ellipse, {
    x, y, w: 0.28, h: 0.28,
    fill: { color }, line: { color, transparency: 100 },
  });
  label(String(number), x, y, 0.28, 0.28, 11, C.white, true, "center");
  label(title, x + 0.38, y - 0.01, 2.8, 0.31, 16, color, true);
}

// Title.
label("TSMR: Track-aware Support Manifold Refinement", 0.35, 0.18, 8.3, 0.36, 24, C.ink, true);

// Input strip: every image here is a separate replaceable picture.
shape(pptx.ShapeType.rect, 0.35, 0.66, 12.63, 1.05, C.grayFill, "A7ADB3", true);
label("Frozen inputs", 0.55, 0.78, 1.12, 0.22, 11, "46698D", true);
image("tracked_rgb.png", 1.55, 0.78, 1.42, 0.65, "REPLACE: RGB track frames");
image("aligned_human_scene.png", 3.2, 0.77, 0.8, 0.68, "REPLACE: aligned SMPL over point cloud");
label("RGB track", 1.54, 1.46, 1.43, 0.15, 9, C.muted, false, "center");
label("Aligned SMPL", 3.03, 1.46, 1.14, 0.15, 9, C.muted, false, "center");
label("+", 4.18, 1.04, 0.3, 0.3, 18, C.line, true, "center");
shape(pptx.ShapeType.rect, 4.55, 0.88, 1.45, 0.48, C.white, "4E83B6", true);
label("Metric point cloud", 4.67, 0.99, 1.2, 0.23, 11, "4E83B6", true, "center");
shape(pptx.ShapeType.rect, 6.25, 0.88, 1.25, 0.48, C.white, "4E83B6", true);
label("Track ID", 6.45, 0.99, 0.85, 0.23, 11, "4E83B6", true, "center");
shape(pptx.ShapeType.rect, 7.75, 0.88, 1.55, 0.48, C.white, "4E83B6", true);
label("Human mask", 7.95, 0.99, 1.15, 0.23, 11, "4E83B6", true, "center");
shape(pptx.ShapeType.rect, 9.55, 0.88, 1.55, 0.48, C.white, "4E83B6", true);
label("Neighbor frames", 9.72, 0.99, 1.2, 0.23, 11, "4E83B6", true, "center");
shape(pptx.ShapeType.rect, 11.35, 0.88, 1.35, 0.48, C.white, C.green, true);
label("Stage2 trans", 11.52, 0.99, 1.02, 0.23, 11, C.green, true, "center");

// Three large, easily reproducible modules.
shape(pptx.ShapeType.rect, 0.35, 1.98, 3.62, 4.38, C.blueFill, C.blue, true);
shape(pptx.ShapeType.rect, 4.17, 1.98, 3.36, 4.38, C.goldFill, C.gold, true);
shape(pptx.ShapeType.rect, 7.73, 1.98, 5.25, 4.38, C.orangeFill, C.orange, true);
moduleTitle(1, "Build support", 0.58, 2.18, C.blue);
moduleTitle(2, "Propose candidates", 4.4, 2.18, C.gold);
moduleTitle(3, "Select, update, verify", 7.96, 2.18, C.orange);

// Module 1: only two replaceable real visuals.
image("aligned_human_scene.png", 0.65, 2.72, 1.35, 2.15, "REPLACE: real aligned human-scene render");
arrow(2.03, 3.75, 0.35, 0, C.blue);
image("support_surface.png", 2.4, 2.65, 1.25, 2.05, "REPLACE: real local support point cloud / surface");
label("Aligned human + scene", 0.57, 4.92, 1.52, 0.27, 11, C.ink, true, "center");
label("Local support surface", 2.27, 4.92, 1.52, 0.27, 11, C.ink, true, "center");
label("Remove human points", 0.68, 5.42, 1.36, 0.26, 11, C.blue, true, "center");
label("Fit ground / support", 2.36, 5.42, 1.34, 0.26, 11, C.blue, true, "center");
label("Output: support point, normal, confidence", 0.67, 5.85, 2.98, 0.28, 11, C.muted, false, "center");

// Module 2: one real probe render plus four simple candidate rows.
image("body_probes.png", 4.48, 2.72, 1.18, 2.05, "REPLACE: real SMPL body probes");
label("Probe feet / body", 4.45, 4.82, 1.25, 0.25, 11, C.ink, true, "center");
const candidates = ["No-op", "Left / right foot", "Both feet", "Body support"];
candidates.forEach((name, i) => {
  const y = 2.72 + i * 0.62;
  shape(pptx.ShapeType.rect, 5.88, y, 1.35, 0.42, C.white, i === 0 ? C.line : C.gold, true);
  label(name, 5.98, y + 0.08, 1.15, 0.21, name.length > 12 ? 9.5 : 10.5, i === 0 ? C.line : C.gold, true, "center");
});
label("Geometry gives a small candidate set", 4.52, 5.43, 2.66, 0.3, 11, C.gold, true, "center");
label("The network does not invent a large translation", 4.47, 5.82, 2.76, 0.3, 10.5, C.muted, false, "center");

// Module 3: ID history -> selector -> real before/after output.
image("tracked_rgb.png", 8.02, 2.75, 1.35, 0.9, "REPLACE: real frames of one tracked identity");
label("same ID over time", 8.03, 3.63, 1.34, 0.24, 10, "4E83B6", true, "center");
arrow(9.42, 3.18, 0.38, 0, C.orange);
shape(pptx.ShapeType.rect, 9.84, 2.76, 1.15, 0.86, C.white, C.orange, true);
label("Candidate\nselector", 10.0, 2.93, 0.83, 0.42, 12, C.orange, true, "center");
arrow(11.04, 3.18, 0.35, 0, C.orange);
image("grounded_before_after.png", 11.43, 2.5, 1.27, 1.95, "REPLACE: real before/after grounding result");
label("Before  ->  Grounded", 11.37, 4.47, 1.38, 0.25, 10.5, C.green, true, "center");

shape(pptx.ShapeType.rect, 8.1, 4.23, 1.38, 0.48, C.white, C.orange, true);
label("Choose / abstain", 8.25, 4.34, 1.08, 0.21, 10.5, C.orange, true, "center");
arrow(9.51, 4.47, 0.36, 0, C.orange);
shape(pptx.ShapeType.rect, 9.9, 4.23, 1.42, 0.48, C.white, C.orange, true);
label("Update translation", 10.03, 4.34, 1.16, 0.21, 10.5, C.orange, true, "center");
arrow(11.35, 4.47, 0.42, 0.64, C.green);
shape(pptx.ShapeType.rect, 11.72, 4.92, 0.98, 0.48, C.greenFill, C.green, true);
label("Re-probe", 11.86, 5.03, 0.7, 0.21, 10.5, C.green, true, "center");

// One loop only.
slide.addShape(pptx.ShapeType.line, {
  x: 8.79, y: 5.52, w: 3.42, h: 0,
  line: { color: C.orange, width: 2.2, dashType: "dash" },
});
slide.addShape(pptx.ShapeType.line, {
  x: 12.21, y: 5.39, w: 0, h: 0.13,
  line: { color: C.orange, width: 2.2, dashType: "dash" },
});
slide.addShape(pptx.ShapeType.line, {
  x: 8.79, y: 4.72, w: 0, h: 0.8,
  line: { color: C.orange, width: 2.2, dashType: "dash", beginArrowType: "triangle" },
});
label("re-probe and repeat x3", 9.72, 5.34, 1.6, 0.22, 10, C.orange, true, "center");
label("Only translation is changed", 8.2, 5.82, 2.05, 0.27, 11, C.orange, true, "center");
label("Pose, shape, and scene stay fixed", 10.35, 5.82, 2.24, 0.27, 11, C.muted, false, "center");

// Main arrows between modules.
arrow(3.98, 4.05, 0.17, 0, C.line, 2.8);
arrow(7.54, 4.05, 0.17, 0, C.line, 2.8);

// Minimal footer, suitable for a paper figure.
shape(pptx.ShapeType.rect, 0.35, 6.58, 12.63, 0.62, C.grayFill, "D2D6DA", true);
label("TRAIN", 0.57, 6.76, 0.65, 0.2, 10, C.orange, true);
label("candidate selection", 1.2, 6.75, 1.25, 0.22, 10, C.ink, true, "center");
label("translation", 2.7, 6.75, 0.9, 0.22, 10, C.ink, true, "center");
label("clean no-op", 3.85, 6.75, 0.95, 0.22, 10, C.ink, true, "center");
label("temporal consistency", 5.05, 6.75, 1.45, 0.22, 10, C.ink, true, "center");
label("INFERENCE", 8.08, 6.76, 0.85, 0.2, 10, C.green, true);
label("support mode  |  refined translation  |  confidence", 8.98, 6.75, 3.55, 0.22, 10.5, C.ink, true, "center");

pptx.writeFile({ fileName: OUT });
