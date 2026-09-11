const pptxgen = require('pptxgenjs');
const path = require('path');

const pptx = new pptxgen();
pptx.defineLayout({ name: 'PAPER_FIGURE', width: 13.5, height: 9.0 });
pptx.layout = 'PAPER_FIGURE';
pptx.author = 'VGGT-Omega';
pptx.subject = 'Editable paper architecture figure';
pptx.title = 'Prior-Guided Metric Scale Calibration';
pptx.company = 'VGGT-Omega';
pptx.lang = 'en-US';
pptx.theme = {
  headFontFace: 'Aptos Display',
  bodyFontFace: 'Aptos',
  lang: 'en-US',
};

const C = {
  paper: 'F7F6F2', ink: '2F3438', border: '596268',
  cream: 'E7DAAC', yellow: 'E9D36E', sage: 'AFCAB2', cyan: 'A9CFD8',
  blue: '99ADD4', periwinkle: '9BB8D8', lavender: 'CBBAD8', blush: 'D9ACA6',
  white: 'FFFFFF', gray: 'E8E8E2', paleBlue: 'EAF4F5', paleSage: 'EEF5EE',
};

const assets = path.resolve(__dirname, '../../outputs/vis/iclr_metric_prior_packaging/pptx_assets');
const outPath = path.resolve(__dirname, '../../outputs/vis/iclr_metric_prior_packaging/vggt_omega_architecture_editable_3slides.pptx');

let slide = pptx.addSlide();
slide.background = { color: C.paper };

function addText(text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x, y, w, h, margin: 0,
    fontFace: opts.fontFace || 'Aptos',
    fontSize: opts.fontSize || 11,
    color: opts.color || C.ink,
    bold: opts.bold || false,
    italic: opts.italic || false,
    align: opts.align || 'left',
    valign: opts.valign || 'mid',
    breakLine: false,
    fit: 'shrink',
    ...opts,
  });
}

function rr(x, y, w, h, fill, line = C.border, radius = 0.08, transparency = 0) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    rectRadius: radius,
    fill: { color: fill, transparency },
    line: { color: line, width: 0.8 },
  });
}

function rect(x, y, w, h, fill, line = C.border, transparency = 0) {
  slide.addShape(pptx.ShapeType.rect, {
    x, y, w, h,
    fill: { color: fill, transparency },
    line: { color: line, width: 0.65 },
  });
}

function line(x1, y1, x2, y2, color = C.ink, width = 1.1, dash = 'solid', endArrow = null) {
  const lineOptions = { color, width, dashType: dash };
  if (endArrow) lineOptions.endArrowType = endArrow;
  slide.addShape(pptx.ShapeType.line, { x: x1, y: y1, w: x2 - x1, h: y2 - y1, line: lineOptions });
}

function arrow(x1, y1, x2, y2, color = C.ink, width = 1.1) {
  line(x1, y1, x2, y2, color, width, 'solid', 'triangle');
}

function imageCard(imgName, label, x, y, w, h, fill = C.paleBlue) {
  rr(x, y, w, h, fill, C.border, 0.06, 20);
  addText(label, x + 0.08, y + 0.06, w - 0.16, 0.32, { fontSize: 10, bold: true, align: 'center' });
  slide.addImage({ path: path.join(assets, imgName), x: x + 0.11, y: y + 0.43, w: w - 0.22, h: h - 0.53, transparency: 0 });
}

function token(x, y, w, h, fill, label = '') {
  rr(x, y, w, h, fill, C.border, 0.05, 4);
  if (label) addText(label, x, y, w, h, { fontSize: 8.5, align: 'center', bold: true });
}

function tokenGrid(x, y, rows, cols, cellW, cellH, gapX, gapY, colors) {
  let idx = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      token(x + c * (cellW + gapX), y + r * (cellH + gapY), cellW, cellH, colors[idx % colors.length]);
      idx++;
    }
  }
}

// Title
addText('Prior-Guided Metric Scale Calibration', 0.6, 0.12, 12.3, 0.48, {
  fontFace: 'Aptos Display', fontSize: 25, bold: true, align: 'center',
});
addText('parameter-free metric prior + human-conditioned residual refinement', 1.1, 0.60, 11.3, 0.28, {
  fontSize: 12.5, color: '4F5960', align: 'center',
});

// Upper module
rr(0.18, 0.95, 13.14, 2.55, C.cream, C.border, 0.08, 80);
addText('Validity-Aware Metric Prior Tokenizer', 0.35, 1.04, 3.55, 0.28, { fontSize: 15, bold: true });
rr(3.93, 1.02, 1.86, 0.30, C.yellow, 'B49B43', 0.06, 18);
addText('analytic · parameter-free', 4.03, 1.04, 1.66, 0.25, { fontSize: 10, bold: true, align: 'center' });

imageCard('metric_smpl.png', 'Metric SMPL', 0.40, 1.40, 1.05, 1.35, C.paleBlue);
addText('+', 1.53, 1.91, 0.25, 0.35, { fontSize: 22, bold: true, align: 'center' });
imageCard('raw_depth.png', 'Raw depth', 1.82, 1.40, 1.05, 1.35, C.paleBlue);
arrow(2.91, 2.08, 3.18, 2.08);

rr(3.20, 1.36, 6.95, 1.55, C.white, '8A969A', 0.06, 32);
addText('Paired geometry tokens', 3.34, 1.43, 2.0, 0.24, { fontSize: 10.5, bold: true });
const pairColors = [C.cream, C.cyan, C.yellow, C.blue, C.blush, C.cream];
for (let i = 0; i < 6; i++) {
  const x = 3.36 + i * 1.05;
  rr(x, 1.70, 0.92, 0.34, C.white, C.border, 0.08, 0);
  rect(x + 0.01, 1.71, 0.44, 0.32, pairColors[i], pairColors[i], 10);
  rect(x + 0.46, 1.71, 0.45, 0.32, i % 2 ? C.lavender : C.paleBlue, i % 2 ? C.lavender : C.paleBlue, 8);
  addText('zₛₘₚₗ', x + 0.04, 1.75, 0.37, 0.20, { fontSize: 8, italic: true, align: 'center' });
  addText('zdepth', x + 0.50, 1.75, 0.35, 0.20, { fontSize: 7.5, italic: true, align: 'center' });
}
addText('Validity mask', 3.34, 2.10, 1.15, 0.22, { fontSize: 10.5, bold: true });
const mask = [1, 1, 0, 1, 1, 0];
for (let i = 0; i < 6; i++) {
  const x = 4.55 + i * 0.72;
  rect(x, 2.10, 0.68, 0.30, mask[i] ? (i % 2 ? C.cream : C.sage) : C.gray, 'AAB0AD', mask[i] ? 25 : 52);
  addText(String(mask[i]), x, 2.12, 0.68, 0.23, { fontSize: 9, align: 'center', color: mask[i] ? C.ink : '9EA4A2' });
}

addText('Scale hypothesis tokens {sᵢ}', 3.34, 2.52, 1.80, 0.22, { fontSize: 10, bold: true });
const validX = [4.65, 5.64, 7.10, 8.10];
const validColors = [C.yellow, C.cyan, C.blue, C.blush];
for (let i = 0; i < validX.length; i++) token(validX[i], 2.49, 0.72, 0.32, validColors[i]);
for (const x of [4.89, 5.88, 7.34, 8.34]) arrow(x, 2.40, x, 2.48, '6E7C80', 0.8);
arrow(8.92, 2.65, 9.28, 2.65);
rr(9.30, 2.38, 1.55, 0.56, C.lavender, C.border, 0.07, 4);
addText('Robust median\npool', 9.42, 2.44, 1.31, 0.43, { fontSize: 11, bold: true, align: 'center', breakLine: false });
arrow(10.90, 2.65, 11.20, 2.65);
slide.addShape(pptx.ShapeType.ellipse, {
  x: 11.24, y: 2.23, w: 1.08, h: 1.08,
  fill: { color: C.yellow, transparency: 5 }, line: { color: 'A68C3C', width: 0.9 },
});
addText('Metric prior\nC coarse', 11.36, 2.43, 0.84, 0.60, { fontSize: 11.5, bold: true, align: 'center' });

// Lower module
rr(0.18, 3.72, 13.14, 4.03, C.paleSage, C.border, 0.08, 32);
addText('Human-Conditioned Residual Calibration', 0.34, 3.83, 4.2, 0.30, { fontSize: 15, bold: true });

imageCard('raw_depth_lower.png', 'Raw VGGT\ndepth Dᵥggt', 0.38, 4.90, 1.05, 1.65, C.paleBlue);
arrow(1.44, 5.73, 1.70, 5.73);
rr(1.72, 5.23, 1.15, 1.00, C.cyan, C.border, 0.08, 20);
addText('Apply\nmetric prior', 1.84, 5.41, 0.91, 0.58, { fontSize: 12, bold: true, align: 'center' });
arrow(2.88, 5.73, 3.13, 5.73);
imageCard('coarse_depth.png', 'Coarse depth\nDcoarse', 3.15, 4.90, 1.10, 1.65, C.paleBlue);
arrow(4.27, 5.73, 4.50, 5.73);

rr(4.52, 4.45, 4.80, 2.65, C.paleSage, '7C9586', 0.08, 15);
addText('Anchor–Scene Interaction', 4.68, 4.55, 2.45, 0.28, { fontSize: 13.5, bold: true });
addText('24 body-anchor tokens', 4.70, 4.91, 1.60, 0.22, { fontSize: 9.5, bold: true });
tokenGrid(4.78, 5.18, 4, 6, 0.20, 0.17, 0.08, 0.08, [C.sage, C.cyan, C.lavender, C.blush, C.yellow]);
addText('Local scene probes', 6.22, 4.91, 1.40, 0.22, { fontSize: 9.5, bold: true });
for (let i = 0; i < 3; i++) slide.addImage({ path: path.join(assets, `probe${i + 1}.png`), x: 6.25 + i * 0.52, y: 5.19, w: 0.44, h: 0.44 });
addText('Human state {θ, β, τ}', 4.70, 6.27, 1.80, 0.22, { fontSize: 9.5, bold: true });
for (let i = 0; i < 4; i++) token(4.80 + i * 0.38, 6.54, 0.32, 0.25, [C.cream, C.cyan, C.lavender, C.yellow][i]);

// bracket joining interaction evidence
line(6.95, 5.02, 7.10, 5.02, C.ink, 1.1);
line(7.10, 5.02, 7.10, 6.72, C.ink, 1.1);
line(6.95, 6.72, 7.10, 6.72, C.ink, 1.1);
arrow(7.10, 5.86, 7.35, 5.86);
rr(7.38, 5.28, 0.95, 1.10, C.periwinkle, C.border, 0.08, 18);
addText('Attention', 7.46, 5.38, 0.79, 0.24, { fontSize: 10.5, bold: true, align: 'center' });
tokenGrid(7.57, 5.72, 3, 3, 0.16, 0.16, 0.06, 0.06, [C.cyan, C.blue, C.periwinkle]);
line(8.33, 5.83, 8.52, 5.83, C.ink, 1.0);
line(8.52, 5.27, 8.52, 6.43, C.ink, 1.0);
arrow(8.52, 5.34, 8.65, 5.34);
arrow(8.52, 6.33, 8.65, 6.33);
rr(8.66, 4.98, 0.52, 0.76, C.sage, C.border, 0.07, 15);
addText('Residual\nscale\nRₕₛᵢ', 8.70, 5.05, 0.44, 0.60, { fontSize: 9, bold: true, align: 'center' });
rr(8.66, 5.96, 0.52, 0.76, C.sage, C.border, 0.07, 15);
addText('Depth\nbias\nbₕₛᵢ', 8.70, 6.03, 0.44, 0.60, { fontSize: 9, bold: true, align: 'center' });

arrow(9.33, 5.73, 9.68, 5.73);
rr(9.70, 5.19, 2.10, 1.10, C.lavender, C.border, 0.08, 18);
addText('Metric Composition', 9.85, 5.31, 1.80, 0.26, { fontSize: 12, bold: true, align: 'center' });
addText('Dmetric = Rhsi · Dcoarse\n+ bhsi', 9.92, 5.67, 1.66, 0.45, { fontSize: 11, italic: true, align: 'center' });
arrow(11.82, 5.73, 12.05, 5.73);
imageCard('metric_depth.png', 'Metric depth\nDmetric', 12.07, 4.90, 1.03, 1.65, C.paleBlue);

// Ccoarse initialization connector (orthogonal)
line(11.78, 3.30, 11.78, 4.22, C.ink, 1.0);
line(11.78, 4.22, 2.30, 4.22, C.ink, 1.0);
arrow(2.30, 4.22, 2.30, 5.22, C.ink, 1.0);
rr(10.82, 3.98, 0.86, 0.25, C.paper, C.paper, 0.03, 0);
addText('initializes', 10.86, 4.00, 0.78, 0.20, { fontSize: 9.5, bold: true, align: 'center' });

// Main path brace
line(1.42, 7.12, 1.42, 7.27, C.ink, 0.8);
line(1.42, 7.27, 11.82, 7.27, C.ink, 0.8);
line(11.82, 7.27, 11.82, 7.12, C.ink, 0.8);
addText('coarse-to-fine residual correction', 4.95, 7.33, 3.35, 0.23, { fontSize: 10.5, bold: true, align: 'center' });

// Palette footer
rr(0.18, 7.95, 13.14, 0.84, C.white, 'B6BAB8', 0.06, 45);
addText('Figure palette', 0.43, 8.18, 1.55, 0.26, { fontSize: 12, bold: true });
const palette = [
  ['E7DAAC', C.cream], ['E9D36E', C.yellow], ['AFCAB2', C.sage], ['A9CFD8', C.cyan],
  ['99ADD4', C.blue], ['CBBAD8', C.lavender], ['D9ACA6', C.blush], ['2F3438', C.ink],
];
palette.forEach((item, i) => {
  const x = 2.15 + i * 1.36;
  rr(x, 8.11, 0.38, 0.32, item[1], item[1], 0.05, 0);
  addText(`#${item[0]}`, x - 0.17, 8.48, 0.72, 0.18, { fontSize: 8.5, bold: true, align: 'center' });
});

// Speaker note for handoff.
slide.addNotes('All labels, panels, tokens, masks, arrows, and formulas are editable PowerPoint shapes. Embedded raster crops are limited to SMPL/depth/probe thumbnails.');

// ---------------------------------------------------------------------------
// Slide 2: detailed spatial-temporal refinement
// ---------------------------------------------------------------------------
slide = pptx.addSlide();
slide.background = { color: C.paper };

addText('Detailed architecture', 0.18, 0.18, 2.5, 0.30, { fontSize: 15, bold: true });
rr(0.18, 0.62, 0.92, 0.34, 'DCECF8', '77A8CE', 0.12, 0);
addText('TRSTR', 0.36, 0.67, 0.58, 0.22, { fontSize: 12, bold: true, color: '25689E', align: 'center' });
line(1.10, 0.79, 1.55, 0.79, '77A8CE', 0.9, 'dash', 'triangle');

// Upper TRSTR group
rr(1.43, 0.78, 11.85, 3.88, 'EDF5FB', '6FA4CF', 0.08, 20);
addText('TRSTR Regional Translator', 4.90, 0.92, 4.20, 0.30, { fontSize: 16, bold: true, align: 'center' });

// Input card
rr(0.18, 1.43, 1.52, 2.47, C.white, '96B8CF', 0.07, 22);
addText('Metric SMPL', 0.40, 1.56, 1.08, 0.25, { fontSize: 12, bold: true, align: 'center' });
slide.addImage({ path: path.join(assets, 'detail_smpl.png'), x: 0.57, y: 1.87, w: 0.75, h: 1.25, transparency: 0 });
addText('Metric scene depth', 0.33, 3.10, 1.22, 0.24, { fontSize: 10.5, bold: true, align: 'center' });
slide.addImage({ path: path.join(assets, 'detail_depth.png'), x: 0.48, y: 3.38, w: 0.90, h: 0.42 });
arrow(1.71, 2.65, 1.90, 2.65);

// Region construction
rr(1.92, 1.43, 2.56, 2.47, C.white, '78A6C8', 0.07, 16);
addText('Region Construction', 2.13, 1.56, 2.12, 0.25, { fontSize: 12, bold: true });
addText('96 body regions', 2.10, 1.91, 1.12, 0.20, { fontSize: 10 });
slide.addImage({ path: path.join(assets, 'detail_segmented.png'), x: 2.27, y: 2.10, w: 0.74, h: 1.35 });
addText('multi-scale probes', 3.24, 1.91, 1.05, 0.20, { fontSize: 9.5, align: 'center' });
const probeLabels = ['3×3', '7×7', 'adaptive', 'annulus'];
const probeCols = [C.lavender, C.blue, C.sage, C.cream];
for (let i = 0; i < 4; i++) {
  tokenGrid(3.31, 2.16 + i * 0.40, i === 3 ? 1 : 3, i === 3 ? 1 : 3, i === 3 ? 0.30 : 0.09, i === 3 ? 0.30 : 0.09, 0.035, 0.035, [probeCols[i]]);
  if (i === 3) {
    slide.addShape(pptx.ShapeType.ellipse, { x: 3.31, y: 3.36, w: 0.30, h: 0.30, fill: { color: C.cream, transparency: 25 }, line: { color: 'B8A454', width: 0.8 } });
    slide.addShape(pptx.ShapeType.ellipse, { x: 3.39, y: 3.44, w: 0.14, h: 0.14, fill: { color: C.white, transparency: 10 }, line: { color: 'B8A454', width: 0.6 } });
  }
  addText(probeLabels[i], 3.69, 2.19 + i * 0.40, 0.63, 0.18, { fontSize: 9 });
}
arrow(4.49, 2.65, 4.66, 2.65);

// Interaction encoder
rr(4.68, 1.43, 2.80, 2.47, 'EEFAF7', '6EB6B6', 0.07, 17);
addText('Human–Scene Interaction\nEncoder', 4.87, 1.56, 2.15, 0.42, { fontSize: 11.5, bold: true });
addText('region geometry', 4.88, 2.10, 1.04, 0.18, { fontSize: 8.5, color: '536A67' });
for (let i = 0; i < 4; i++) token(4.98 + i * 0.31, 2.34, 0.23, 0.17, C.blue);
addText('local scene', 4.88, 2.62, 0.90, 0.18, { fontSize: 8.5, color: '536A67' });
for (let i = 0; i < 4; i++) token(4.98 + i * 0.31, 2.86, 0.23, 0.17, C.sage);
addText('owner mask', 4.88, 3.13, 0.90, 0.18, { fontSize: 8.5, color: '536A67' });
for (let i = 0; i < 4; i++) token(4.98 + i * 0.31, 3.37, 0.23, 0.17, C.cream);
arrow(6.17, 2.95, 6.36, 2.95);
rr(6.38, 2.15, 0.40, 1.55, C.paleBlue, C.border, 0.05, 14);
addText('concat', 6.44, 2.53, 0.28, 0.80, { fontSize: 9, bold: true, rotate: 90, align: 'center' });
arrow(6.79, 2.95, 6.95, 2.95);
rr(6.96, 2.38, 0.42, 0.42, C.lavender, C.border, 0.06, 10);
addText('MLP', 7.00, 2.48, 0.34, 0.18, { fontSize: 8.5, align: 'center' });
rr(6.89, 2.91, 0.53, 0.50, C.blue, C.border, 0.06, 18);
addText('interaction\ntoken', 6.94, 3.00, 0.43, 0.32, { fontSize: 8.5, bold: true, align: 'center' });
arrow(7.49, 2.65, 7.67, 2.65);

// Regional heads
rr(7.69, 1.43, 2.24, 2.47, C.white, '91A9C9', 0.07, 20);
addText('Regional Correction Heads', 7.86, 1.56, 1.77, 0.28, { fontSize: 11, bold: true, align: 'center' });
addText('interaction token', 7.82, 2.38, 0.88, 0.18, { fontSize: 8.5 });
for (let i = 0; i < 3; i++) token(8.32 + i * 0.22, 2.37, 0.17, 0.16, C.blue);
line(8.94, 2.45, 9.02, 2.45, C.ink, 0.9);
line(9.02, 2.03, 9.02, 3.36, C.ink, 0.9);
const headYs = [1.96, 2.56, 3.16];
const headColors = [C.blush, C.sage, C.blue];
const headText = ['vote Δτᵣ', 'gate gᵣ', 'uncertainty σᵣ'];
for (let i = 0; i < 3; i++) {
  arrow(9.02, headYs[i] + 0.22, 9.09, headYs[i] + 0.22);
  rr(9.10, headYs[i], 0.72, 0.43, headColors[i], C.border, 0.07, 10);
  addText(headText[i], 9.14, headYs[i] + 0.08, 0.64, 0.26, { fontSize: 8.3, bold: true, align: 'center' });
}
arrow(9.83, 2.65, 9.99, 2.65);

// Robust aggregation
rr(9.99, 1.43, 2.28, 2.47, C.white, '91A9C9', 0.07, 20);
addText('Robust Aggregation', 10.15, 1.56, 1.90, 0.27, { fontSize: 11.5, bold: true, align: 'center' });
for (let i = 0; i < 3; i++) token(10.14, 2.16 + i * 0.32, 0.34, 0.20, headColors[i]);
addText('weighted\nfusion', 10.68, 2.20, 0.88, 0.38, { fontSize: 9.5, bold: true, align: 'center' });
slide.addShape(pptx.ShapeType.funnel, { x: 10.75, y: 2.61, w: 0.58, h: 0.67, fill: { color: C.blue, transparency: 18 }, line: { color: '4B78A4', width: 0.9 } });
arrow(11.34, 2.93, 11.48, 2.93);
rr(11.50, 2.56, 0.60, 0.74, C.lavender, C.border, 0.06, 15);
addText('person\ngate', 11.56, 2.73, 0.48, 0.35, { fontSize: 9.5, bold: true, align: 'center' });
slide.addShape(pptx.ShapeType.ellipse, { x: 11.44, y: 0.94, w: 0.75, h: 0.75, fill: { color: C.white, transparency: 30 }, line: { color: C.ink, width: 0.9, dashType: 'dash' } });
addText('2×\nre-probe', 11.53, 1.07, 0.57, 0.46, { fontSize: 10.5, bold: true, align: 'center' });
line(11.82, 1.69, 11.82, 1.88, C.ink, 0.8, 'dash', 'triangle');
arrow(12.28, 2.65, 12.45, 2.65);
rr(12.47, 2.18, 0.70, 0.95, C.white, C.border, 0.07, 8);
addText('refined\ntranslation\nτ*', 12.54, 2.37, 0.56, 0.56, { fontSize: 10.5, bold: true, align: 'center' });

rr(1.92, 4.14, 9.82, 0.32, C.white, 'A6B8C5', 0.08, 42);
addText('▣  pose θ and shape β unchanged', 4.79, 4.18, 4.0, 0.20, { fontSize: 11.5, bold: true, color: '286BA0', align: 'center' });

// Lower temporal group
rr(0.18, 4.92, 0.96, 0.34, 'DDF4F1', '70BFB8', 0.12, 0);
addText('Temporal', 0.29, 4.97, 0.74, 0.22, { fontSize: 11.5, bold: true, color: '2B918A', align: 'center' });
line(1.14, 5.09, 1.55, 5.09, '70BFB8', 0.9, 'dash', 'triangle');
rr(1.43, 4.89, 11.85, 3.30, 'ECFAF9', '65B9B2', 0.08, 20);
addText('Track-wise Temporal Fusion', 4.95, 5.04, 4.15, 0.30, { fontSize: 16, bold: true, align: 'center' });

// Track window
rr(1.64, 5.40, 2.92, 2.37, C.white, '7ABCB9', 0.07, 25);
addText('Track Window', 1.85, 5.55, 1.50, 0.25, { fontSize: 12, bold: true });
addText('t−4    ···    t    ···    t+4', 2.45, 5.95, 1.55, 0.18, { fontSize: 9.5, align: 'center' });
const laneColors = ['78AEE6', 'EF9CA5', '6DC6B5'];
for (let r = 0; r < 3; r++) {
  const y = 6.30 + r * 0.47;
  addText(`ID ${r + 1}`, 1.79, y - 0.09, 0.38, 0.20, { fontSize: 9.5, bold: true });
  line(2.20, y, 4.28, y, laneColors[r], 1.3);
  for (let i = 0; i < 9; i++) {
    slide.addShape(pptx.ShapeType.ellipse, { x: 2.16 + i * 0.26, y: y - 0.10, w: 0.20, h: 0.20, fill: { color: i === 4 ? C.white : laneColors[r], transparency: i === 4 ? 0 : 28 }, line: { color: laneColors[r], width: 0.8 } });
  }
}
rr(3.20, 6.18, 0.34, 1.36, C.white, '447CA8', 0.03, 75);
arrow(4.57, 6.58, 4.77, 6.58);

// Motion proposal
rr(4.79, 5.40, 3.25, 2.37, C.white, '7ABCB9', 0.07, 25);
addText('Motion Proposal', 4.98, 5.55, 1.65, 0.25, { fontSize: 12, bold: true });
for (let r = 0; r < 3; r++) {
  for (let i = 0; i < 3; i++) token(5.00 + i * 0.30, 6.05 + r * 0.37, 0.20, 0.18, [C.blue, C.blush, C.sage][r]);
}
arrow(5.90, 6.48, 6.10, 6.48);
rr(6.12, 6.03, 0.86, 0.94, C.paleBlue, C.border, 0.07, 15);
addText('Masked\nTemporal\nMLP', 6.23, 6.19, 0.64, 0.58, { fontSize: 10, bold: true, align: 'center' });
addText('neighbors only', 5.06, 7.35, 1.32, 0.20, { fontSize: 9.5, bold: true, align: 'center' });
addText('×', 6.40, 7.08, 0.27, 0.28, { fontSize: 22, color: 'E85C67', bold: true, align: 'center' });
arrow(6.99, 6.48, 7.18, 6.48);
token(7.20, 6.33, 0.22, 0.18, C.lavender);
token(7.47, 6.33, 0.22, 0.18, C.lavender);
token(7.74, 6.33, 0.22, 0.18, C.lavender);
addText('Mₜ', 7.46, 6.08, 0.42, 0.20, { fontSize: 10.5, italic: true, align: 'center' });
arrow(8.05, 6.58, 8.25, 6.58);

// Conservative fusion
rr(8.27, 5.40, 3.03, 2.37, C.white, '7ABCB9', 0.07, 25);
addText('Conservative Fusion', 8.46, 5.55, 1.90, 0.25, { fontSize: 12, bold: true });
rr(8.47, 6.04, 0.55, 0.38, C.paleBlue, C.border, 0.06, 10);
addText('Oₜ', 8.61, 6.11, 0.27, 0.20, { fontSize: 11, italic: true, align: 'center' });
rr(8.47, 6.52, 0.55, 0.38, C.lavender, C.border, 0.06, 10);
addText('Mₜ', 8.61, 6.59, 0.27, 0.20, { fontSize: 11, italic: true, align: 'center' });
arrow(9.04, 6.46, 9.27, 6.46);
rr(9.29, 6.10, 0.64, 0.73, C.paleBlue, C.border, 0.06, 14);
addText('Gate\nMLP', 9.39, 6.27, 0.44, 0.36, { fontSize: 10.5, bold: true, align: 'center' });
arrow(9.94, 6.46, 10.13, 6.46);
rr(10.15, 6.18, 0.78, 0.52, C.cream, 'B99C4B', 0.07, 12);
addText('αₜ ≤ 0.5', 10.27, 6.31, 0.54, 0.23, { fontSize: 11, bold: true, align: 'center' });
line(10.54, 6.71, 10.54, 7.00, C.ink, 0.9, 'solid', 'triangle');
rr(8.68, 7.05, 2.10, 0.55, C.paleBlue, C.border, 0.07, 30);
addText('X*ₜ = Oₜ + αₜ(Mₜ − Oₜ)', 8.87, 7.20, 1.72, 0.23, { fontSize: 11, italic: true, bold: true, align: 'center' });
arrow(11.31, 6.58, 11.51, 6.58);

// Stable sequence
rr(11.53, 5.40, 1.55, 2.37, C.white, '7ABCB9', 0.07, 25);
addText('Stable sequence', 11.75, 5.55, 1.12, 0.25, { fontSize: 11.5, bold: true, align: 'center' });
for (let r = 0; r < 3; r++) {
  const y = 6.18 + r * 0.50;
  line(11.75, y, 12.86, y, laneColors[r], 1.2);
  for (let i = 0; i < 5; i++) slide.addShape(pptx.ShapeType.ellipse, { x: 11.72 + i * 0.27, y: y - 0.10, w: 0.20, h: 0.20, fill: { color: laneColors[r], transparency: 28 }, line: { color: laneColors[r], width: 0.8 } });
}

addText('(C) Spatial–Temporal Refinement', 0.18, 8.42, 3.65, 0.34, { fontSize: 16, bold: true });
slide.addNotes('Editable reconstruction of the detailed TRSTR and track-wise temporal refinement architecture.');

// ---------------------------------------------------------------------------
// Slide 3: overall architecture
// ---------------------------------------------------------------------------
slide = pptx.addSlide();
slide.background = { color: C.paper };
rr(0.30, 0.55, 12.90, 7.75, C.white, C.border, 0.08, 46);

// RGB sequence
imageCard('overall_frames.png', 'RGB Sequence', 0.47, 1.20, 1.00, 4.70, C.white);
arrow(1.49, 3.47, 1.73, 3.47);

// Parallel encoding block
rr(1.75, 1.00, 3.00, 5.20, C.white, '9488AF', 0.08, 18);
addText('Parallel Human–Scene Encoding', 1.96, 1.14, 2.58, 0.26, { fontSize: 12.5, bold: true, align: 'center' });
rr(1.93, 1.55, 2.64, 1.82, C.lavender, '9989B2', 0.07, 55);
addText('VGGT Scene Encoder', 2.12, 1.67, 1.72, 0.25, { fontSize: 11.5, bold: true });
tokenGrid(2.12, 2.08, 4, 3, 0.17, 0.17, 0.06, 0.06, [C.lavender, C.blue]);
for (let i = 0; i < 5; i++) token(2.92 + i * 0.26, 2.45, 0.18, 0.15, C.lavender);
line(4.28, 2.00, 4.28, 3.08, C.ink, 0.8);
for (const [yy, label] of [[2.00, 'Camera K'], [2.53, 'Raw depth Dᵥggt'], [3.06, 'Scene tokens Fscene']]) {
  arrow(4.03, yy, 4.27, yy);
  addText(label, 4.33, yy - 0.10, 0.92, 0.20, { fontSize: 8.2, bold: true });
}

rr(1.93, 3.64, 2.64, 2.32, C.blush, 'AF7F7C', 0.07, 62);
addText('NLF Human Estimator', 2.36, 3.77, 1.80, 0.25, { fontSize: 11.5, bold: true, align: 'center' });
slide.addImage({ path: path.join(assets, 'overall_nlf.png'), x: 2.10, y: 4.18, w: 0.90, h: 1.35 });
arrow(3.00, 4.86, 3.18, 4.86);
slide.addImage({ path: path.join(assets, 'overall_smpl.png'), x: 3.19, y: 4.17, w: 0.56, h: 1.36 });
for (const [yy, label] of [[4.22, 'pose θ'], [4.59, 'shape β'], [4.96, 'translation τ'], [5.33, 'confidence c']]) {
  addText(label, 3.84, yy, 0.67, 0.18, { fontSize: 8.6, bold: true, align: 'center' });
}
line(3.67, 3.39, 3.67, 3.62, C.ink, 0.8, 'dash', 'triangle');
addText('intrinsics', 3.75, 3.41, 0.65, 0.18, { fontSize: 8.5, bold: true });
arrow(4.76, 3.47, 4.98, 3.47);

// Evidence block
rr(5.00, 1.18, 1.78, 4.85, C.white, '8C9EAA', 0.08, 16);
addText('Metric Human–Scene\nEvidence', 5.19, 1.35, 1.40, 0.42, { fontSize: 11.5, bold: true, align: 'center' });
addText('Scene evidence\n{Dᵥggt, Fscene}', 5.22, 2.02, 1.35, 0.36, { fontSize: 8.5, bold: true, align: 'center' });
for (let i = 0; i < 6; i++) token(5.30 + i * 0.20, 2.48, 0.14, 0.15, C.lavender);
arrow(5.91, 2.70, 5.91, 2.92);
addText('Camera evidence {K}', 5.22, 2.95, 1.35, 0.22, { fontSize: 8.5, bold: true, align: 'center' });
for (let i = 0; i < 6; i++) token(5.30 + i * 0.20, 3.30, 0.14, 0.15, C.cyan);
arrow(5.91, 3.52, 5.91, 3.74);
addText('Human evidence\n{θ, β, τ, c}', 5.22, 3.77, 1.35, 0.36, { fontSize: 8.5, bold: true, align: 'center' });
for (let i = 0; i < 6; i++) token(5.30 + i * 0.20, 4.24, 0.14, 0.15, C.blush);
line(5.34, 4.58, 5.34, 4.83, C.ink, 0.8);
line(5.34, 4.83, 6.47, 4.83, C.ink, 0.8);
line(6.47, 4.58, 6.47, 4.83, C.ink, 0.8);
arrow(5.91, 4.83, 5.91, 5.04);
for (let i = 0; i < 7; i++) token(5.25 + i * 0.20, 5.18, 0.14, 0.15, [C.lavender, C.blue, C.cyan, C.sage, C.blush, C.yellow, C.cream][i]);
arrow(6.80, 3.47, 7.04, 3.47);

// HSI scale
rr(7.06, 1.38, 1.25, 4.43, C.cream, 'BBA85C', 0.08, 48);
addText('HSI Scale\nCalibration', 7.22, 1.55, 0.92, 0.42, { fontSize: 11.5, bold: true, align: 'center' });
rr(7.20, 2.15, 0.96, 1.05, C.cream, 'C8B772', 0.06, 18);
addText('Coarse evidence\n{Ccoarse, Fscene}', 7.28, 2.27, 0.80, 0.32, { fontSize: 8, bold: true, align: 'center' });
slide.addShape(pptx.ShapeType.arc, { x: 7.42, y: 2.62, w: 0.50, h: 0.38, adjustPoint: 0.20, fill: { color: C.white, transparency: 100 }, line: { color: '807348', width: 1.0 } });
line(7.67, 2.83, 7.83, 2.71, '807348', 1.2);
arrow(7.68, 3.22, 7.68, 3.43);
rr(7.20, 3.45, 0.96, 1.15, C.white, 'C8B772', 0.06, 36);
addText('Residual affine\nRhsi, bhsi', 7.31, 3.56, 0.74, 0.34, { fontSize: 8.5, bold: true, align: 'center' });
tokenGrid(7.31, 4.02, 3, 3, 0.12, 0.12, 0.04, 0.04, [C.cream, C.yellow]);
for (let i = 0; i < 3; i++) token(7.83, 4.03 + i * 0.18, 0.12, 0.12, C.cream);
addText('Metric depth\nDmetric', 8.32, 2.67, 0.62, 0.35, { fontSize: 8.5, bold: true, align: 'center' });
addText('Metric\nSMPL', 8.32, 4.07, 0.62, 0.35, { fontSize: 8.5, bold: true, align: 'center' });
arrow(8.32, 3.47, 8.58, 3.47);

// TRSTR block
rr(8.60, 1.42, 1.86, 4.35, C.periwinkle, '6E91BB', 0.08, 60);
addText('TRSTR Spatial Refinement', 8.77, 1.60, 1.52, 0.28, { fontSize: 10.5, bold: true, align: 'center' });
addText('Segmented\nSMPL', 8.78, 2.06, 0.67, 0.35, { fontSize: 8.5, bold: true, align: 'center' });
slide.addImage({ path: path.join(assets, 'detail_segmented.png'), x: 8.89, y: 2.41, w: 0.51, h: 1.10 });
addText('96-region\ntoken matrix', 9.55, 2.07, 0.70, 0.35, { fontSize: 8.5, bold: true, align: 'center' });
tokenGrid(9.64, 2.52, 4, 4, 0.13, 0.13, 0.04, 0.04, [C.blue, C.periwinkle]);
addText('Multi-scale\nprobes', 8.72, 3.80, 0.73, 0.35, { fontSize: 8.5, bold: true, align: 'center' });
for (let i = 0; i < 4; i++) tokenGrid(8.73 + i * 0.20, 4.27, 2, 2, 0.05, 0.05, 0.02, 0.02, [[C.blue, C.sage, C.cream, C.lavender][i]]);
addText('Weighted\nvoting', 9.55, 3.81, 0.71, 0.35, { fontSize: 8.5, bold: true, align: 'center' });
slide.addShape(pptx.ShapeType.funnel, { x: 9.75, y: 4.28, w: 0.42, h: 0.55, fill: { color: C.blue, transparency: 15 }, line: { color: '4B78A4', width: 0.8 } });
addText('▣ θ, β fixed', 9.05, 5.18, 0.98, 0.22, { fontSize: 8.5, bold: true, align: 'center' });
arrow(10.47, 3.47, 10.70, 3.47);

// Temporal block
rr(10.72, 1.72, 1.28, 3.75, C.paleBlue, '5AAAB1', 0.08, 30);
addText('Track-wise\nTemporal Stabilizer', 10.85, 1.94, 1.02, 0.42, { fontSize: 10.5, bold: true, align: 'center' });
addText('9-frame window', 10.91, 2.60, 0.89, 0.20, { fontSize: 8.5, align: 'center' });
line(10.88, 3.05, 11.80, 3.05, C.ink, 0.8);
for (let i = 0; i < 9; i++) slide.addShape(pptx.ShapeType.ellipse, { x: 10.84 + i * 0.12, y: 2.97, w: 0.14, h: 0.14, fill: { color: i === 4 ? C.blue : C.white, transparency: i === 4 ? 5 : 0 }, line: { color: '56819A', width: 0.7 } });
addText('Oₜ', 11.31, 3.18, 0.25, 0.19, { fontSize: 9, italic: true, align: 'center' });
arrow(11.38, 3.42, 11.38, 3.63);
addText('Neighbor-only proposal\nMₜ', 10.85, 3.68, 1.05, 0.38, { fontSize: 8.3, bold: true, align: 'center' });
for (let i = 0; i < 5; i++) slide.addShape(pptx.ShapeType.ellipse, { x: 10.92 + i * 0.20, y: 4.11, w: 0.14, h: 0.14, fill: { color: C.white }, line: { color: '78B5C0', width: 0.7, dashType: 'dash' } });
arrow(11.38, 4.30, 11.38, 4.49);
addText('Gating\nαₜ ≤ 0.5', 10.99, 4.53, 0.78, 0.36, { fontSize: 9, bold: true, align: 'center' });
arrow(12.01, 3.47, 12.24, 3.47);

// Output
slide.addImage({ path: path.join(assets, 'overall_output.png'), x: 12.25, y: 2.52, w: 0.72, h: 1.12 });
addText('Stable Metric\nHuman–Scene\nSequence', 12.14, 3.80, 0.95, 0.72, { fontSize: 10.5, bold: true, align: 'center' });
addText('(A) Overall Architecture', 0.47, 7.65, 2.35, 0.32, { fontSize: 15, bold: true });
slide.addNotes('Editable reconstruction of the overall RGB–VGGT–NLF–HSI–TRSTR–temporal architecture.');

pptx.writeFile({ fileName: outPath });
