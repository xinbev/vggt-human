const pptxgen = require("pptxgenjs");

const pptx = new pptxgen();
pptx.defineLayout({ name: "PANORAMA", width: 16, height: 4.5 });
pptx.layout = "PANORAMA";
pptx.author = "VGGT-Omega";
pptx.subject = "Editable reconstruction of a metric 4D reconstruction pipeline diagram";
pptx.title = "Metric 4D Reconstruction Pipeline";
pptx.company = "VGGT-Omega";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-US",
};
pptx.margin = 0;

const slide = pptx.addSlide();
slide.background = { color: "FFFFFF" };

const C = {
  navy: "0D2447",
  teal: "008B9A",
  tealLight: "EAF8F8",
  tealLine: "64C7CD",
  blue: "1976C9",
  blueLight: "EEF7FF",
  blueLine: "7BB6ED",
  coral: "EF725E",
  coralLight: "FFF0EC",
  gold: "F6AF44",
  goldLight: "FFF8EA",
  green: "28A776",
  ink: "23354D",
  muted: "617088",
  grid: "CAD9E8",
  pale: "F8FBFD",
  gray: "94A7B9",
  cloud: "B3C0CC",
};

const sh = pptx.ShapeType;
const font = "Aptos";

function box(x, y, w, h, fill = "FFFFFF", line = C.grid, radius = false, transparency = 0) {
  slide.addShape(radius ? sh.roundRect : sh.rect, {
    x, y, w, h,
    rectRadius: radius ? 0.04 : undefined,
    fill: { color: fill, transparency },
    line: { color: line, width: 0.8 },
  });
}

function text(value, x, y, w, h, options = {}) {
  slide.addText(value, {
    x, y, w, h,
    margin: 0,
    fontFace: font,
    fontSize: options.fontSize ?? 7.2,
    color: options.color ?? C.ink,
    bold: options.bold ?? false,
    italic: options.italic ?? false,
    align: options.align ?? "center",
    valign: options.valign ?? "mid",
    breakLine: false,
    fit: "shrink",
    ...options,
  });
}

function line(x1, y1, x2, y2, color = C.gray, width = 0.8, dash = "solid", arrowEnd = false, arrowBegin = false) {
  slide.addShape(sh.line, {
    x: x1,
    y: y1,
    w: x2 - x1,
    h: y2 - y1,
    line: {
      color,
      width,
      dash,
      beginArrowType: arrowBegin ? "triangle" : "none",
      endArrowType: arrowEnd ? "triangle" : "none",
    },
  });
}

function arrow(x1, y1, x2, y2, color = C.navy, width = 1.0) {
  line(x1, y1, x2, y2, color, width, "solid", true, false);
}

function dot(x, y, r, fill, lineColor = fill) {
  slide.addShape(sh.ellipse, {
    x: x - r, y: y - r, w: r * 2, h: r * 2,
    fill: { color: fill }, line: { color: lineColor, width: 0.35 },
  });
}

function check(x, y, color = C.green) {
  dot(x, y, 0.045, color);
  text("✓", x - 0.035, y - 0.048, 0.07, 0.09, { fontSize: 5.2, color: "FFFFFF", bold: true });
}

function cross(x, y) {
  dot(x, y, 0.045, C.coral);
  text("×", x - 0.035, y - 0.052, 0.07, 0.10, { fontSize: 5.4, color: "FFFFFF", bold: true });
}

function panel(x, y, w, h, accent, fill, title, number) {
  box(x, y, w, h, fill, accent, true);
  text(number, x + 0.16, y + 0.10, 0.28, 0.26, { fontSize: 13, color: accent, bold: true, align: "left" });
  text(title, x + 0.55, y + 0.11, w - 0.7, 0.26, { fontSize: 11.5, color: C.navy, bold: true, align: "left" });
}

function smallLabel(value, x, y, w, color = C.muted) {
  text(value, x, y, w, 0.13, { fontSize: 5.1, color, bold: true, align: "center" });
}

function person(x, y, s, color = C.coral, opacity = 0) {
  const lc = opacity ? "FFFFFF" : color;
  const fill = opacity ? { color, transparency: opacity } : { color };
  slide.addShape(sh.ellipse, { x: x + 0.29 * s, y, w: 0.16 * s, h: 0.16 * s, fill, line: { color: lc, transparency: opacity, width: 0.3 } });
  slide.addShape(sh.roundRect, { x: x + 0.22 * s, y: y + 0.16 * s, w: 0.30 * s, h: 0.37 * s, rectRadius: 0.02, fill, line: { color: lc, transparency: opacity, width: 0.3 } });
  line(x + 0.23 * s, y + 0.27 * s, x + 0.05 * s, y + 0.43 * s, color, 1.4 * s, "solid");
  line(x + 0.51 * s, y + 0.27 * s, x + 0.69 * s, y + 0.43 * s, color, 1.4 * s, "solid");
  line(x + 0.29 * s, y + 0.52 * s, x + 0.18 * s, y + 0.82 * s, color, 1.6 * s, "solid");
  line(x + 0.45 * s, y + 0.52 * s, x + 0.56 * s, y + 0.82 * s, color, 1.6 * s, "solid");
}

function photo(x, y, w, h, frame = C.gray) {
  box(x, y, w, h, "F3F5F4", frame, false);
  slide.addShape(sh.rect, { x: x + 0.02, y: y + 0.02, w: w - 0.04, h: h * 0.42, fill: { color: "D9D3CB" }, line: { color: "D9D3CB", transparency: 100 } });
  slide.addShape(sh.rect, { x: x + 0.02, y: y + h * 0.44, w: w - 0.04, h: h * 0.54, fill: { color: "B4BEC2" }, line: { color: "B4BEC2", transparency: 100 } });
  person(x + w * 0.21, y + h * 0.14, Math.min(w, h) * 0.78, "3C4855");
}

function camera(x, y, s, color = C.blue) {
  slide.addShape(sh.trapezoid, { x, y: y + 0.10 * s, w: 0.28 * s, h: 0.17 * s, fill: { color: "FFFFFF" }, line: { color, width: 0.75 } });
  slide.addShape(sh.ellipse, { x: x + 0.09 * s, y: y + 0.14 * s, w: 0.08 * s, h: 0.08 * s, fill: { color: C.blueLight }, line: { color, width: 0.65 } });
  line(x + 0.14 * s, y + 0.27 * s, x + 0.07 * s, y + 0.39 * s, color, 0.8);
  line(x + 0.14 * s, y + 0.27 * s, x + 0.22 * s, y + 0.39 * s, color, 0.8);
}

function featureMap(x, y, w, h, color = C.blue) {
  box(x, y, w, h, "F7FCFF", C.grid, false);
  const cols = 10;
  const rows = 5;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cx = x + 0.04 + c * ((w - 0.08) / cols);
      const cy = y + 0.04 + r * ((h - 0.08) / rows);
      const tone = (c + r * 2) % 5;
      const fill = tone === 0 ? color : tone === 1 ? "63BFE7" : tone === 2 ? "B2E3F4" : tone === 3 ? "CFEFE9" : "E6F6FB";
      slide.addShape(sh.rect, { x: cx, y: cy, w: (w - 0.13) / cols, h: (h - 0.13) / rows, fill: { color: fill }, line: { color: fill, transparency: 100 } });
    }
  }
}

function scenePatch(x, y, w, h) {
  box(x, y, w, h, "F6FAFD", C.grid, false);
  for (let i = 0; i < 9; i++) {
    for (let j = 0; j < 5; j++) {
      const px = x + 0.05 + i * (w - 0.1) / 9;
      const py = y + 0.05 + j * (h - 0.1) / 5;
      dot(px, py, 0.012 + ((i + j) % 2) * 0.004, i % 3 === 0 ? "72B7D7" : "D8E2E7");
    }
  }
}

function anchorStrip(x, y, w, h) {
  box(x, y, w, h, "FFFFFF", C.grid, false);
  const labels = ["t - 1", "t", "t + 1"];
  labels.forEach((lab, idx) => {
    const px = x + 0.07 + idx * (w - 0.22) / 3;
    smallLabel(lab, px - 0.02, y + 0.06, 0.31, C.muted);
    for (let k = 0; k < 3; k++) {
      const fill = k === 0 ? C.teal : k === 1 ? "63C4C5" : C.coral;
      slide.addShape(sh.roundRect, { x: px + k * 0.10, y: y + 0.22, w: 0.075, h: 0.075, rectRadius: 0.01, fill: { color: fill }, line: { color: fill, width: 0.2 } });
    }
    idx === 0 ? check(px + 0.13, y + 0.43) : idx === 1 ? check(px + 0.13, y + 0.43) : cross(px + 0.13, y + 0.43);
  });
}

function miniChart(x, y, w, h) {
  box(x, y, w, h, "FFFFFF", C.tealLine, true);
  smallLabel("Temporal consensus", x + 0.25, y + 0.05, w - 0.5, C.ink);
  const times = [0, 1, 2, 3, 4];
  const coords = times.map((t) => x + 0.18 + t * (w - 0.36) / 4);
  const series = [
    { color: C.gold, base: y + 0.38, vals: [0.04, 0.14, 0.09, 0.02, 0.09] },
    { color: C.blue, base: y + 0.49, vals: [0.10, 0.03, 0.06, 0.13, 0.10] },
    { color: C.teal, base: y + 0.60, vals: [0.03, 0.10, 0.04, 0.12, 0.03] },
  ];
  series.forEach((s) => {
    for (let i = 0; i < 4; i++) line(coords[i], s.base - s.vals[i], coords[i + 1], s.base - s.vals[i + 1], s.color, 0.7, "dash");
    coords.forEach((cx, idx) => dot(cx, s.base - s.vals[idx], 0.03, s.color));
  });
  coords.forEach((cx, idx) => smallLabel(["t - 1", "t", "t + 1", "t + 2", "" ][idx], cx - 0.17, y + h - 0.15, 0.34, C.muted));
}

function tokenGrid(x, y, w, h, accent) {
  box(x, y, w, h, "FFFFFF", C.grid, false);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      const active = (r + c) % 3 === 0;
      slide.addShape(sh.rect, {
        x: x + 0.06 + c * 0.10,
        y: y + 0.18 + r * 0.10,
        w: 0.055,
        h: 0.055,
        fill: { color: active ? accent : "F4F8FB" },
        line: { color: active ? accent : C.grid, width: 0.35 },
      });
    }
  }
}

function trackWindow(x, y, w, h) {
  box(x, y, w, h, "FFFFFF", C.grid, false);
  smallLabel("Track window", x + 0.2, y + 0.05, w - 0.4, C.blue);
  const xs = [x + 0.18, x + 0.56, x + 0.94, x + 1.32, x + 1.70];
  const ys = [y + 0.44, y + 0.70, y + 0.96];
  const colors = [C.blue, C.coral, C.green];
  colors.forEach((color, row) => {
    for (let i = 0; i < xs.length - 1; i++) line(xs[i], ys[row], xs[i + 1], ys[row] + (i % 2 ? -0.04 : 0.04), color, 0.75, "dash");
    xs.forEach((px, idx) => dot(px, ys[row] + (idx % 2 ? 0.04 : 0), 0.035, color));
  });
  ["t - 2", "t - 1", "t", "t + 1", "t + 2"].forEach((lab, idx) => smallLabel(lab, xs[idx] - 0.12, y + h - 0.17, 0.24, C.muted));
}

function pointCloud(x, y, w, h) {
  for (let i = 0; i < 260; i++) {
    const u = ((i * 83) % 997) / 997;
    const v = ((i * 197) % 991) / 991;
    const px = x + u * w;
    const py = y + v * h;
    const inVolume = px > x + 0.11 * w && px < x + 0.89 * w && py > y + 0.08 * h && py < y + 0.85 * h;
    if (!inVolume) continue;
    dot(px, py, 0.009 + (i % 3) * 0.004, i % 11 === 0 ? "9EB6CC" : C.cloud);
  }
  line(x + 0.18 * w, y + 0.18 * h, x + 0.73 * w, y + 0.10 * h, "C5D1DB", 0.55);
  line(x + 0.18 * w, y + 0.18 * h, x + 0.18 * w, y + 0.77 * h, "C5D1DB", 0.55);
  line(x + 0.73 * w, y + 0.10 * h, x + 0.89 * w, y + 0.55 * h, "C5D1DB", 0.55);
  line(x + 0.18 * w, y + 0.77 * h, x + 0.89 * w, y + 0.55 * h, "C5D1DB", 0.55);
  line(x + 0.73 * w, y + 0.10 * h, x + 0.18 * w, y + 0.77 * h, "D6E0E8", 0.4, "dash");
  person(x + 0.34 * w, y + 0.24 * h, 0.53, C.coral, 20);
  person(x + 0.60 * w, y + 0.41 * h, 0.42, C.coral, 20);
  camera(x + 0.06 * w, y + 0.72 * h, 0.55, C.blue);
  camera(x + 0.77 * w, y + 0.73 * h, 0.55, C.blue);
}

// Left input column.
text("Video and geometric priors", 0.08, 0.15, 2.67, 0.25, { fontSize: 11, bold: true, align: "left", color: C.navy });
for (let i = 0; i < 3; i++) {
  photo(0.05, 0.68 + i * 0.74, 0.58, 0.58);
  smallLabel(["t - 1", "t", "t + 1"][i], 0.06, 1.29 + i * 0.74, 0.54, C.muted);
}
text("⋮", 0.25, 2.93, 0.12, 0.22, { fontSize: 14, color: C.muted });

box(0.90, 0.71, 1.95, 1.30, "FFFFFF", C.gray, true);
text("Scene encoder", 1.02, 0.83, 1.55, 0.18, { fontSize: 8.3, bold: true, align: "left" });
for (let j = 0; j < 4; j++) photo(1.03 + j * 0.22, 1.13, 0.17, 0.22);
featureMap(1.92, 1.13, 0.34, 0.25);
camera(1.81, 1.50, 0.75, C.blue);
arrow(2.30, 1.24, 2.50, 1.24, C.muted, 0.7);
arrow(2.30, 1.50, 2.50, 1.50, C.muted, 0.7);
arrow(2.30, 1.76, 2.50, 1.76, C.muted, 0.7);
text("Depth\nCameras\nFeatures", 2.52, 1.11, 0.28, 0.78, { fontSize: 5.5, align: "left", breakLine: false, color: C.ink });

box(0.90, 2.15, 1.95, 1.54, "FFFFFF", C.gray, true);
text("Human estimator", 1.02, 2.27, 1.55, 0.18, { fontSize: 8.3, bold: true, align: "left" });
person(1.02, 2.57, 0.65, C.coral);
person(1.58, 2.60, 0.55, C.coral);
for (let j = 0; j < 9; j++) {
  const px = 2.15 + (j % 3) * 0.14;
  const py = 2.58 + Math.floor(j / 3) * 0.20;
  dot(px, py, 0.035, j % 3 === 0 ? C.blue : j % 3 === 1 ? C.green : C.gold);
  if (j > 2) line(px, py, px - 0.12, py - 0.17, C.gray, 0.45);
}
text("Pose / shape / translation", 1.07, 3.44, 1.55, 0.14, { fontSize: 5.7, bold: true });

arrow(0.63, 1.08, 0.90, 1.08, C.muted, 0.9);
arrow(0.63, 2.56, 0.90, 2.56, C.muted, 0.9);
text("Scene\nevidence", 2.95, 1.38, 0.34, 0.28, { fontSize: 5.3, color: C.teal, bold: true });
text("Human\nevidence", 2.93, 2.58, 0.36, 0.28, { fontSize: 5.3, color: C.coral, bold: true });

// Stage 01: Temporal Scale Calibration.
panel(3.20, 0.20, 5.00, 4.08, C.tealLine, "F4FCFC", "Temporal Scale Calibration", "01");
box(3.30, 0.67, 4.80, 1.62, "FFFFFF", C.tealLine, true);
person(3.42, 0.86, 0.63, C.coral);
line(3.86, 1.12, 4.12, 1.25, C.gold, 0.65, "dash");
line(3.86, 1.35, 4.12, 1.25, C.gold, 0.65, "dash");
smallLabel("Surface scale anchors", 3.32, 1.95, 0.82, C.muted);
smallLabel("Human-scene evidence", 4.25, 0.91, 1.11, C.ink);
anchorStrip(4.13, 1.04, 1.24, 0.65);
arrow(5.40, 1.34, 5.65, 1.34, C.teal, 0.8);
box(5.68, 1.05, 0.80, 0.68, "F8FCFE", C.tealLine, true);
text("Cross\nAttention", 5.77, 1.21, 0.62, 0.28, { fontSize: 6.2, bold: true });
smallLabel("Scene features", 5.74, 0.80, 0.70, C.ink);
for (let k = 0; k < 3; k++) arrow(5.87 + k * 0.14, 0.90, 5.87 + k * 0.14, 1.02, C.teal, 0.6);
arrow(6.48, 1.34, 6.64, 1.34, C.teal, 0.75);
box(6.67, 0.92, 0.79, 0.43, "FFFFFF", C.tealLine, true);
box(6.67, 1.52, 0.79, 0.43, "FFFFFF", C.tealLine, true);
text("Scale residual", 6.75, 0.99, 0.62, 0.11, { fontSize: 5.0, bold: true });
text("Depth bias", 6.79, 1.59, 0.55, 0.11, { fontSize: 5.0, bold: true });
box(7.00, 1.14, 0.17, 0.16, "BCE8E8", C.tealLine, true);
text("s", 7.04, 1.15, 0.08, 0.10, { fontSize: 7, bold: true, color: C.teal });
box(7.00, 1.73, 0.17, 0.16, "FFF0D9", "F2BE6E", true);
text("b", 7.04, 1.74, 0.08, 0.10, { fontSize: 7, bold: true, color: "C98221" });
dot(7.66, 1.43, 0.11, "FFFFFF", C.ink); text("×", 7.60, 1.35, 0.12, 0.14, { fontSize: 10, bold: true });
dot(7.95, 1.43, 0.11, "FFFFFF", C.ink); text("+", 7.89, 1.35, 0.12, 0.14, { fontSize: 10, bold: true });
arrow(7.47, 1.13, 7.64, 1.35, C.teal, 0.6); arrow(7.47, 1.73, 7.64, 1.50, C.teal, 0.6); arrow(7.77, 1.43, 7.84, 1.43, C.teal, 0.6);
line(4.19, 1.73, 4.19, 2.05, C.teal, 0.7); line(4.19, 2.05, 6.07, 2.05, C.teal, 0.7); line(6.07, 2.05, 6.07, 1.73, C.teal, 0.7, "dash", true);
smallLabel("Anchor-scene interaction", 4.96, 2.02, 1.20, C.teal);

miniChart(3.52, 2.59, 2.02, 1.06);
box(5.69, 2.54, 1.72, 0.89, "FFFFFF", C.tealLine, true);
smallLabel("Metric scene + cameras", 5.87, 2.62, 1.35, C.ink);
featureMap(5.83, 2.88, 0.54, 0.38);
camera(6.45, 2.93, 0.52, C.blue); camera(6.80, 2.85, 0.52, C.blue);
arrow(7.39, 3.02, 7.72, 3.02, C.teal, 0.75);
box(7.73, 2.87, 0.20, 0.19, "C8F1EE", C.tealLine, true); text("s", 7.80, 2.89, 0.06, 0.10, { fontSize: 7, color: C.teal, bold: true });
camera(7.71, 3.18, 0.45, C.blue);
box(7.73, 3.37, 0.20, 0.19, "FFF0D9", "F2BE6E", true); text("b", 7.80, 3.39, 0.06, 0.10, { fontSize: 7, color: "C98221", bold: true });
arrow(5.54, 3.12, 5.67, 3.12, C.teal, 0.7);
line(6.60, 3.44, 6.60, 3.78, C.teal, 0.7); line(6.60, 3.78, 4.00, 3.78, C.teal, 0.7); line(4.00, 3.78, 4.00, 3.47, C.teal, 0.7, "dash", true);
person(3.54, 3.74, 0.25, C.coral);
text("Fixed pose and shape", 3.86, 3.79, 1.25, 0.15, { fontSize: 5.7, color: C.coral, bold: true, align: "left" });
line(3.12, 2.76, 3.12, 3.99, C.coral, 1.1); arrow(3.12, 3.99, 8.33, 3.99, C.coral, 1.1);

// Stage 02: Implicit Contact Refinement.
panel(8.30, 0.20, 4.83, 4.08, C.blueLine, "F5FAFF", "Implicit Contact Refinement", "02");
box(8.40, 0.67, 4.63, 1.93, "FFFFFF", C.blueLine, true);
person(8.59, 0.83, 0.70, "6FAF9E");
smallLabel("Body regions", 8.48, 2.33, 0.85, C.muted);
smallLabel("Scene probes", 9.31, 0.80, 0.78, C.ink);
const probeXs = [9.03, 9.49, 9.95];
["local", "medium", "large"].forEach((lab, idx) => {
  tokenGrid(probeXs[idx], 0.99, 0.42, 0.56, idx === 0 ? C.blue : idx === 1 ? C.teal : "6F9FE3");
  smallLabel(lab, probeXs[idx], 0.90, 0.42, C.muted);
  line(probeXs[idx] + 0.21, 1.55, probeXs[idx] + 0.21, 1.94, C.gold, 0.45, "dash");
  scenePatch(probeXs[idx] - 0.02, 1.96, 0.46, 0.21);
});
arrow(9.30, 1.63, 10.18, 1.63, C.blue, 0.75);
box(10.22, 0.94, 0.84, 1.27, "FFFFFF", C.grid, true);
["Self", "Environment", "Other people"].forEach((lab, idx) => {
  if (idx > 0) line(10.26, 1.32 + idx * 0.37, 11.01, 1.32 + idx * 0.37, C.grid, 0.45);
  slide.addShape(sh.rect, { x: 10.32, y: 1.05 + idx * 0.37, w: 0.13, h: 0.18, fill: { color: idx === 0 ? "B5DFD4" : idx === 1 ? "E7ECEE" : "DFC9DB" }, line: { color: C.gray, width: 0.25 } });
  smallLabel(lab, 10.50, 1.06 + idx * 0.37, 0.33, C.muted);
  idx < 2 ? check(10.97, 1.14 + idx * 0.37) : cross(10.97, 1.14 + idx * 0.37);
});
arrow(11.06, 1.60, 11.23, 1.60, C.blue, 0.75);
box(11.26, 1.16, 0.51, 0.51, "FFFFFF", C.blueLine, true);
dot(11.51, 1.41, 0.055, C.blue); dot(11.36, 1.30, 0.04, C.teal); dot(11.66, 1.29, 0.04, C.gold); dot(11.66, 1.52, 0.04, C.coral);
line(11.51, 1.41, 11.36, 1.30, C.gray, 0.5); line(11.51, 1.41, 11.66, 1.29, C.gray, 0.5); line(11.51, 1.41, 11.66, 1.52, C.gray, 0.5);
smallLabel("Regional\ninteraction", 11.79, 0.94, 0.63, C.ink);
arrow(11.78, 1.42, 12.06, 1.42, C.blue, 0.75);
smallLabel("Translation\nvotes", 12.08, 0.96, 0.43, C.ink);
for (let i = 0; i < 4; i++) arrow(12.10, 1.54, 12.33, 1.33 + i * 0.13, [C.blue, C.gold, C.coral, C.teal][i], 0.6);
smallLabel("Reliability\ngate", 12.55, 0.96, 0.40, C.ink);
slide.addShape(sh.roundRect, { x: 12.62, y: 1.41, w: 0.20, h: 0.13, rectRadius: 0.02, fill: { color: "D1E8F8" }, line: { color: C.blueLine, width: 0.4 } });
arrow(12.83, 1.47, 13.00, 1.47, C.blue, 0.7);
person(12.70, 1.18, 0.46, C.coral, 0);

trackWindow(8.53, 2.83, 2.21, 1.02);
arrow(10.74, 3.34, 10.93, 3.34, C.blue, 0.75);
box(10.95, 3.08, 0.58, 0.52, "FFFFFF", C.gray, true);
text("Masked\ntemporal\nfusion", 11.05, 3.18, 0.38, 0.24, { fontSize: 5.7, bold: true });
arrow(11.54, 3.34, 11.72, 3.34, C.blue, 0.75);
box(11.75, 3.08, 0.56, 0.52, "FFFFFF", C.blueLine, true);
text("Gated\nupdate", 11.86, 3.19, 0.34, 0.18, { fontSize: 5.9, bold: true });
dot(12.03, 3.49, 0.08, "FFFFFF", C.ink); text("∕", 12.00, 3.43, 0.06, 0.10, { fontSize: 9, bold: true });
line(11.06, 2.61, 11.06, 2.85, C.blue, 0.7); line(11.06, 2.85, 12.03, 2.85, C.blue, 0.7); line(12.03, 2.85, 12.03, 3.07, C.blue, 0.7, "solid", true);
person(12.35, 3.02, 0.40, C.coral);
text("Refined human\ntrajectory", 12.25, 3.68, 0.73, 0.26, { fontSize: 5.5, bold: true, color: C.muted });

// Output panel.
arrow(13.14, 2.28, 13.45, 2.28, C.navy, 1.0);
text("Metric 4D reconstruction", 13.33, 0.18, 2.46, 0.25, { fontSize: 11, bold: true, align: "left", color: C.navy });
pointCloud(13.34, 0.55, 2.38, 2.42);

// Legend beneath point cloud.
for (let i = 0; i < 5; i++) line(13.39 + i * 0.07, 3.48 + (i % 2) * 0.08, 13.39 + i * 0.07, 3.72 - (i % 2) * 0.03, C.teal, 1.35);
text("Shared scale", 13.32, 3.90, 0.74, 0.14, { fontSize: 5.8, bold: true, color: C.muted });
for (let i = 0; i < 5; i++) {
  const px = 14.38 + i * 0.10;
  dot(px, 3.62 - i * 0.05, 0.03, "FFFFFF", C.blue);
  if (i) line(px - 0.10, 3.62 - (i - 1) * 0.05, px, 3.62 - i * 0.05, C.blue, 0.55);
}
text("Coherent motion", 14.17, 3.90, 0.97, 0.14, { fontSize: 5.8, bold: true, color: C.muted });
person(15.30, 3.39, 0.43, C.gold);
text("Consistent support", 15.15, 3.90, 0.85, 0.14, { fontSize: 5.8, bold: true, color: C.muted });

async function main() {
  await pptx.writeFile({ fileName: "outputs/vis/metric_4d_reconstruction_pipeline/metric_4d_reconstruction_pipeline.pptx" });
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
