const pptxgen = require('pptxgenjs');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'Codex';
pptx.subject = 'Editable ICLR Spatial-Temporal Refinement inset';
pptx.title = 'Spatial–Temporal Refinement';
pptx.company = 'VGGT-Omega';
pptx.lang = 'en-US';
pptx.theme = {
  headFontFace: 'Arial',
  bodyFontFace: 'Arial',
  lang: 'en-US'
};
pptx.defineLayout({ name: 'CUSTOM_WIDE', width: 13.333, height: 7.5 });
pptx.layout = 'CUSTOM_WIDE';
pptx.margin = 0;
pptx.layout = 'CUSTOM_WIDE';

const C = {
  paper: 'F7F6F2', ink: '26313A', muted: '6C7880', line: '8B9AA5',
  blue: '99ADD4', blueLight: 'EAF0FA', cyan: 'A9CFD8', cyanLight: 'ECFAFC',
  lavender: 'CBBAD8', lavenderLight: 'F2EDF7', sage: 'AFCAB2', sageLight: 'EEF6EE',
  blush: 'D9ACA6', blushLight: 'FBEEEE', cream: 'E7DAAC', creamLight: 'FCF8E7',
  yellow: 'E9D36E', white: 'FFFFFF', green: '6FA982', red: 'C96767'
};

const slide = pptx.addSlide();
slide.background = { color: C.paper };

// The source figure is visually taller than a default 16:9 canvas.  Scale all
// editable elements vertically so the reconstructed diagram fills the slide
// instead of leaving a large unused footer area.
const verticalScale = 1.22;
const originalAddShape = slide.addShape.bind(slide);
const originalAddText = slide.addText.bind(slide);
function scaleOptions(opts) {
  const next = { ...opts };
  if (typeof next.y === 'number') next.y *= verticalScale;
  if (typeof next.h === 'number') next.h *= verticalScale;
  return next;
}
slide.addShape = (shape, opts) => originalAddShape(shape, scaleOptions(opts));
slide.addText = (text, opts) => originalAddText(text, scaleOptions(opts));

function addText(text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x, y, w, h, margin: 0,
    fontFace: opts.fontFace || 'Arial',
    fontSize: opts.fontSize || 11,
    color: opts.color || C.ink,
    bold: opts.bold || false,
    align: opts.align || 'left',
    valign: opts.valign || 'mid',
    breakLine: false,
    fit: 'shrink',
    ...opts
  });
}

function round(x, y, w, h, fill, line = C.line, radius = 0.08, opts = {}) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    rectRadius: radius,
    fill: { color: fill, transparency: opts.transparency || 0 },
    line: { color: line, width: opts.width || 0.8, dash: opts.dash },
  });
}

function rect(x, y, w, h, fill, line = C.line, opts = {}) {
  slide.addShape(pptx.ShapeType.rect, {
    x, y, w, h,
    fill: { color: fill, transparency: opts.transparency || 0 },
    line: { color: line, width: opts.width || 0.7, dash: opts.dash },
  });
}

function line(x1, y1, x2, y2, opts = {}) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color: opts.color || C.ink, width: opts.width || 1, dash: opts.dash, beginArrowType: opts.beginArrowType, endArrowType: opts.endArrowType }
  });
}

function arrow(x1, y1, x2, y2, opts = {}) {
  line(x1, y1, x2, y2, { ...opts, endArrowType: 'triangle' });
}

function token(x, y, color, w = 0.16, h = 0.16, lineColor = C.line) {
  round(x, y, w, h, color, lineColor, 0.025, { width: 0.5 });
}

function tokenStrip(x, y, colors, w = 0.16, gap = 0.04) {
  colors.forEach((color, i) => token(x + i * (w + gap), y, color, w, w));
}

function miniMesh(x, y, scale = 1, segmented = false) {
  const lc = C.ink;
  slide.addShape(pptx.ShapeType.ellipse, { x: x + 0.28 * scale, y, w: 0.16 * scale, h: 0.18 * scale, fill: { color: 'FFFFFF' }, line: { color: lc, width: 0.7 } });
  const bodyFill = segmented ? C.sageLight : 'FFFFFF';
  round(x + 0.2 * scale, y + 0.18 * scale, 0.32 * scale, 0.52 * scale, bodyFill, lc, 0.04, { width: 0.7 });
  line(x + 0.22 * scale, y + 0.28 * scale, x + 0.04 * scale, y + 0.56 * scale, { color: lc, width: 0.8 });
  line(x + 0.50 * scale, y + 0.28 * scale, x + 0.68 * scale, y + 0.56 * scale, { color: lc, width: 0.8 });
  line(x + 0.29 * scale, y + 0.7 * scale, x + 0.18 * scale, y + 1.05 * scale, { color: lc, width: 0.8 });
  line(x + 0.43 * scale, y + 0.7 * scale, x + 0.54 * scale, y + 1.05 * scale, { color: lc, width: 0.8 });
  if (segmented) {
    [[0.23,0.30,C.blush],[0.38,0.30,C.blue],[0.23,0.48,C.cream],[0.38,0.48,C.lavender],[0.22,0.67,C.sage],[0.39,0.67,C.blush]].forEach(([dx,dy,c]) => {
      round(x + dx * scale, y + dy * scale, 0.12 * scale, 0.14 * scale, c, C.line, 0.02, { width: 0.35 });
    });
  }
}

function depthTile(x, y, w, h) {
  rect(x, y, w, h, 'E2E2DF', 'A7A7A1', { width: 0.5 });
  const bands = ['D6D6D0', 'BFBFBB', 'A8A8A4', '8C8D8A'];
  bands.forEach((c, i) => rect(x + 0.03 + i * w / 4, y + 0.03, w / 4 - 0.025, h - 0.06, c, c, { width: 0 }));
  slide.addShape(pptx.ShapeType.ellipse, { x: x + w * 0.53, y: y + h * 0.15, w: w * 0.2, h: h * 0.56, fill: { color: '5D6060', transparency: 20 }, line: { color: '5D6060', transparency: 100 } });
}

function sourceTag(x, y, w, label, fill, targetX, targetY) {
  round(x, y, w, 0.28, fill, C.line, 0.06, { width: 0.7 });
  addText(label, x, y + 0.015, w, 0.23, { fontSize: 10, bold: true, align: 'center' });
  line(x + w / 2, y + 0.28, targetX, targetY, { color: C.line, width: 0.7, dash: 'dash' });
}

// Header + source expansion callouts
addText('Detailed architecture', 0.16, 0.08, 2.1, 0.22, { fontSize: 8.8, bold: true, color: C.muted });
sourceTag(0.16, 0.36, 1.0, 'TRSTR', C.blueLight, 0.70, 0.86);
sourceTag(9.36, 0.36, 1.25, 'Temporal', C.cyanLight, 9.96, 0.86);

// Main group containers
round(0.12, 0.82, 8.05, 4.35, C.blueLight, C.blue, 0.1, { width: 1.0 });
round(8.32, 0.82, 4.89, 4.35, C.cyanLight, '4EA6B3', 0.1, { width: 1.0 });
addText('(B) TRSTR Internal Dataflow', 0.26, 0.94, 3.2, 0.3, { fontSize: 14, bold: true });
addText('(C) Temporal Stabilizer Internal Dataflow', 8.46, 0.94, 4.3, 0.3, { fontSize: 13.2, bold: true });

// TRSTR column headers
const colY = 1.42;
const cols = [
  [0.25, 0.78, 'Inputs'], [1.08, 1.33, 'Frozen Geometry'], [2.46, 1.36, 'RegionalSceneProbe'],
  [3.87, 1.27, 'Region Encoder'], [5.19, 1.68, 'Prediction Heads'], [6.92, 1.10, 'Shared Iterative\nUpdate']
];
cols.forEach(([x,w,label]) => { round(x, colY, w, 0.30, 'DCE8F7', C.blue, 0.05, { width: 0.6 }); addText(label, x + 0.03, colY + 0.035, w - 0.06, 0.22, { fontSize: 7.3, bold: true, align: 'center' }); });

// Inputs
addText('pose6d\n[B,S,Q,144]', 0.32, 1.86, 0.6, 0.34, { fontSize: 7.4, align: 'center' });
tokenStrip(0.42, 2.18, [C.blush,C.blush,C.blush,C.blush]);
addText('betas\n[B,S,Q,10]', 0.32, 2.48, 0.6, 0.33, { fontSize: 7.4, align: 'center' });
tokenStrip(0.42, 2.80, [C.sage,C.sage,C.sage,C.sage]);
addText('transl\n[B,S,Q,3]', 0.32, 3.10, 0.6, 0.33, { fontSize: 7.4, align: 'center' });
tokenStrip(0.42, 3.42, [C.cream,C.cream,C.cream]);
addText('D_metric\n[F,H,W]', 0.32, 3.75, 0.6, 0.30, { fontSize: 7.1, align: 'center' });
tokenStrip(0.42, 4.06, [C.blue,C.blue,C.blue,C.blue]);
addText('K\n[F,3,3]', 0.32, 4.38, 0.6, 0.28, { fontSize: 7.1, align: 'center' });
tokenStrip(0.42, 4.67, [C.lavender,C.lavender,C.lavender]);

// Frozen Geometry
round(1.22, 2.17, 0.67, 0.44, C.white, C.ink, 0.05, { width: 0.8 });
addText('SMPL\nDecode', 1.22, 2.23, 0.67, 0.28, { fontSize: 8.2, bold: true, align: 'center' });
arrow(1.89, 2.39, 2.08, 2.39, { width: 0.8 });
rect(2.09, 2.18, 0.35, 0.35, 'D8E8EF', C.line, { width: 0.5 });
addText('V\n[F,Q,6890,3]', 2.03, 2.59, 0.48, 0.28, { fontSize: 6.7, align: 'center' });
round(1.23, 3.30, 0.72, 0.40, C.white, C.ink, 0.05, { width: 0.8 });
addText('Region Bank', 1.25, 3.39, 0.68, 0.16, { fontSize: 7.5, bold: true, align: 'center' });
line(1.60, 2.61, 1.60, 3.30, { width: 0.8 });
rect(2.08, 3.15, 0.35, 0.35, 'D6E8D6', C.line, { width: 0.5 });
addText('centers\n[F,Q,96,3]', 1.98, 3.54, 0.55, 0.27, { fontSize: 6.4, align: 'center' });
rect(2.08, 4.08, 0.35, 0.35, 'E4D9EF', C.line, { width: 0.5 });
addText('reps\n[F,Q,96,8,3]', 1.93, 4.46, 0.65, 0.27, { fontSize: 6.0, align: 'center' });
arrow(1.95, 3.50, 2.08, 3.32, { width: 0.7 });
arrow(1.95, 3.50, 2.08, 4.25, { width: 0.7 });

// RegionalSceneProbe
round(2.55, 1.86, 1.18, 0.74, C.white, C.line, 0.05, { width: 0.7 });
addText('Project by K → Depth\nprobe', 2.59, 1.98, 1.10, 0.28, { fontSize: 7.4, align: 'center' });
depthTile(2.70, 2.26, 0.28, 0.22);
miniMesh(3.22, 2.14, 0.32, false);
line(3.00, 2.37, 3.24, 2.35, { color: C.line, width: 0.5, dash: 'dash' });
round(2.55, 2.77, 1.18, 0.62, C.white, C.line, 0.05, { width: 0.7 });
addText('Owner Map', 2.67, 2.86, 0.50, 0.15, { fontSize: 7.5, bold: true, align: 'center' });
token(2.72, 3.08, C.blue); addText('self', 2.91, 3.04, 0.22, 0.14, { fontSize: 6.2 });
token(3.15, 3.08, C.blush); addText('other', 3.35, 3.04, 0.30, 0.14, { fontSize: 6.2 });
token(2.72, 3.22, C.sage); addText('scene', 2.91, 3.18, 0.30, 0.14, { fontSize: 6.2 });
round(2.55, 3.59, 1.18, 0.72, C.white, C.line, 0.05, { width: 0.7 });
addText('Multi-Scale Sampling', 2.63, 3.68, 1.02, 0.15, { fontSize: 7.1, bold: true, align: 'center' });
['3×3','7×7','adaptive','annulus'].forEach((t, i) => { const xx = 2.68 + (i % 2) * 0.50; const yy = 3.94 + Math.floor(i / 2) * 0.19; round(xx, yy, 0.40, 0.15, [C.lavenderLight,C.blueLight,C.sageLight,C.creamLight][i], C.line, 0.03, { width: 0.4 }); addText(t, xx, yy + 0.01, 0.40, 0.12, { fontSize: 5.7, align: 'center' }); });
arrow(3.15, 4.31, 3.15, 4.55, { width: 0.7 });
addText('Attention Pooling', 2.74, 4.51, 0.96, 0.15, { fontSize: 7, bold: true, align: 'center' });
rect(3.05, 4.73, 0.27, 0.27, 'D0E2E8', C.line, { width: 0.5 });
addText('probe\n[FQ,96,8,16]', 2.78, 5.02, 0.75, 0.28, { fontSize: 6.3, align: 'center' });

// Region Encoder
['geometry','proj. uv','probe\ntokens','valid ratios','region emb.'].forEach((label, i) => {
  const colors = [C.sageLight,C.cyanLight,C.lavenderLight,C.creamLight,C.blushLight];
  round(3.99, 1.90 + i * 0.35, 0.67, 0.23, colors[i], C.line, 0.035, { width: 0.5 });
  addText(label, 4.00, 1.95 + i * 0.35, 0.65, 0.13, { fontSize: 6.6, align: 'center' });
});
slide.addShape(pptx.ShapeType.ellipse, { x: 4.78, y: 2.65, w: 0.24, h: 0.24, fill: { color: C.white }, line: { color: C.ink, width: 0.7 } });
addText('C', 4.80, 2.69, 0.20, 0.12, { fontSize: 7.2, bold: true, align: 'center' });
addText('xᵣ\n[FQ,96,178]', 4.63, 2.94, 0.50, 0.30, { fontSize: 6.2, align: 'center' });
round(4.11, 3.70, 0.74, 0.47, C.white, C.ink, 0.05, { width: 0.8 });
addText('MLP\n178 → 256 →\n256', 4.12, 3.76, 0.72, 0.32, { fontSize: 7.2, bold: true, align: 'center' });
arrow(4.54, 3.23, 4.54, 3.70, { width: 0.7 });
rect(4.31, 4.45, 0.27, 0.27, 'BFD8E3', C.line, { width: 0.5 });
addText('hᵣ\n[FQ,96,256]', 4.12, 4.74, 0.66, 0.27, { fontSize: 6.4, align: 'center' });
round(4.00, 4.98, 0.82, 0.30, C.creamLight, C.cream, 0.05, { width: 0.6 });
addText('person gate', 4.04, 5.06, 0.74, 0.13, { fontSize: 7.0, bold: true, align: 'center' });

// Prediction Heads
[['vote head\n→ Δτᵣ [96,3]', C.blushLight], ['gate head\n→ gᵣ [96,1]', C.sageLight], ['logvar head\n→ σᵣ [96,1]', C.lavenderLight]].forEach(([label,fill], i) => {
  round(5.38, 2.20 + i * 0.62, 0.78, 0.42, fill, C.line, 0.05, { width: 0.7 });
  addText(label, 5.40, 2.27 + i * 0.62, 0.74, 0.26, { fontSize: 6.7, bold: true, align: 'center' });
});
arrow(4.58, 4.55, 5.38, 3.25, { width: 0.7 });
addText('wᵣ = validᵣ · gᵣ ·\nexp(−σᵣ)', 5.28, 4.20, 0.98, 0.28, { fontSize: 6.5, align: 'center' });
slide.addShape(pptx.ShapeType.ellipse, { x: 6.25, y: 4.36, w: 0.32, h: 0.32, fill: { color: C.white }, line: { color: C.ink, width: 0.8 } });
addText('Σ', 6.28, 4.42, 0.26, 0.15, { fontSize: 11, bold: true, align: 'center' });
arrow(6.05, 3.40, 6.40, 4.36, { width: 0.7 });
rect(6.25, 4.82, 0.34, 0.20, 'BDD7E7', C.line, { width: 0.5 });
addText('Δτ [FQ,3]', 6.04, 5.04, 0.75, 0.16, { fontSize: 6.6, align: 'center' });

// Shared iterative update
addText('k = 1,2', 7.16, 1.84, 0.52, 0.17, { fontSize: 8.5, bold: true, align: 'center' });
round(7.15, 2.17, 0.62, 1.14, C.white, C.ink, 0.05, { width: 0.7, dash: 'dash' });
addText('τᵏ⁺¹ = τᵏ +\ng_person Δτ', 7.19, 2.48, 0.54, 0.35, { fontSize: 7.2, bold: true, align: 'center' });
arrow(7.47, 3.31, 7.47, 3.86, { width: 0.8 });
rect(7.32, 3.90, 0.30, 0.30, 'A9D2E4', C.line, { width: 0.5 });
addText('τ*\n[B,S,Q,3]', 7.13, 4.24, 0.66, 0.28, { fontSize: 6.6, align: 'center' });
line(7.15, 2.17, 6.65, 2.17, { color: C.line, width: 0.7, dash: 'dash' });
line(6.65, 2.17, 6.65, 1.66, { color: C.line, width: 0.7, dash: 'dash' });
line(6.65, 1.66, 1.58, 1.66, { color: C.line, width: 0.7, dash: 'dash' });
line(1.58, 1.66, 1.58, 2.16, { color: C.line, width: 0.7, endArrowType: 'triangle' });
round(1.48, 4.86, 5.6, 0.27, 'DCE8F7', C.blue, 0.04, { width: 0.5 });
addText('θ, β unchanged', 1.48, 4.92, 5.6, 0.13, { fontSize: 8, bold: true, align: 'center' });

// Temporal panel
round(8.55, 1.89, 0.62, 0.42, C.white, C.line, 0.05, { width: 0.7 });
addText('O\n[B,9,3]', 8.56, 1.97, 0.60, 0.22, { fontSize: 7.0, bold: true, align: 'center' });
tokenStrip(8.63, 2.29, [C.blue,C.blue,C.blue,C.blue]);
round(8.55, 2.75, 0.62, 0.42, C.white, C.line, 0.05, { width: 0.7 });
addText('valid\n[B,9]', 8.56, 2.83, 0.60, 0.22, { fontSize: 7.0, bold: true, align: 'center' });
tokenStrip(8.63, 3.16, [C.sage,C.sage,C.sage,C.sage]);
arrow(9.18, 2.18, 9.56, 2.18, { width: 0.8 });
round(9.59, 2.07, 1.27, 0.64, C.white, C.ink, 0.06, { width: 0.8 });
addText('Center Mask +\nNeighbor Pack', 9.65, 2.22, 1.15, 0.26, { fontSize: 8.3, bold: true, align: 'center' });
['t−2','t−1','t','t+1','t+2'].forEach((t, i) => { const xx=9.72+i*0.20; slide.addShape(pptx.ShapeType.ellipse,{x:xx,y:2.80,w:0.13,h:0.13,fill:{color:i===2?C.blush:C.blueLight},line:{color:C.line,width:0.45}}); addText(t,xx-0.035,2.96,0.20,0.12,{fontSize:5.4,align:'center'}); });
addText('×', 10.10, 2.78, 0.15, 0.17, { fontSize: 12, color: C.red, bold: true, align: 'center' });
arrow(10.86, 2.38, 11.10, 2.38, { width: 0.8 });
addText('relative neighbors\n[B,12]', 10.94, 2.52, 0.75, 0.22, { fontSize: 6.6, align: 'center' });
tokenStrip(11.11, 2.80, [C.lavender,C.lavender,C.lavender,C.lavender]);
round(11.25, 1.78, 1.18, 0.64, C.white, C.ink, 0.06, { width: 0.8 });
addText('Proposal MLP\n12 → 128 → 128 → 3', 11.32, 1.90, 1.05, 0.28, { fontSize: 7.6, bold: true, align: 'center' });
addText('tanh × 0.25 m', 11.40, 2.27, 0.90, 0.13, { fontSize: 6.2, align: 'center' });
arrow(11.68, 2.74, 11.82, 2.42, { width: 0.8 });
addText('Mₜ\n[B,3]', 12.52, 1.91, 0.36, 0.30, { fontSize: 7, bold: true, align: 'center' });
tokenStrip(12.54, 2.23, [C.blush,C.blush,C.blush]);

// Gate feature branch
round(8.56, 3.84, 1.77, 0.70, C.white, C.line, 0.05, { width: 0.7 });
[['Oₜ − Mₜ',C.cyanLight], ['|Oₜ − Mₜ|',C.sageLight], ['neighbor velocity',C.creamLight]].forEach(([t,c],i)=>{round(8.72,3.96+i*0.18,0.88,0.14,c,C.line,0.03,{width:0.4});addText(t,8.74,3.98+i*0.18,0.84,0.10,{fontSize:5.9,align:'center'});});
slide.addShape(pptx.ShapeType.ellipse,{x:10.05,y:4.08,w:0.21,h:0.21,fill:{color:C.white},line:{color:C.ink,width:0.7}});
addText('C',10.07,4.12,0.17,0.10,{fontSize:6.5,bold:true,align:'center'});
addText('gate feature\n[B,9]', 10.27, 3.95, 0.55, 0.28, { fontSize: 6.4, align: 'center' });
tokenStrip(10.35, 4.29, [C.lavender,C.lavender,C.lavender,C.lavender]);
arrow(10.32, 4.18, 10.93, 4.18, { width: 0.8 });
round(10.96, 3.76, 1.05, 0.61, C.white, C.ink, 0.06, { width: 0.8 });
addText('Gate MLP\n9 → 64 → 64 → 1', 11.03, 3.89, 0.91, 0.26, { fontSize: 7.1, bold: true, align: 'center' });
addText('sigmoid × 0.5',11.10,4.22,0.77,0.12,{fontSize:6.2,align:'center'});
arrow(12.01,4.08,12.32,4.08,{width:0.8});
addText('αₜ\n[B,1]', 12.37,3.84,0.35,0.27,{fontSize:7,bold:true,align:'center'});
token(12.46,4.16,C.lavender,0.16,0.16);

// fusion
slide.addShape(pptx.ShapeType.ellipse,{x:11.80,y:2.90,w:0.30,h:0.30,fill:{color:C.white},line:{color:C.ink,width:0.8}});
addText('⊙',11.83,2.94,0.24,0.14,{fontSize:10,bold:true,align:'center'});
line(11.98,2.50,11.98,2.90,{width:0.8});
line(12.56,4.20,12.56,3.05,{width:0.8});
line(12.56,3.05,12.10,3.05,{width:0.8});
arrow(12.10,3.05,12.26,3.05,{width:0.8});
round(12.15,3.07,0.88,0.40,C.white,C.ink,0.05,{width:0.7});
addText('X* = Oₜ + αₜ(Mₜ − Oₜ)',12.19,3.19,0.80,0.14,{fontSize:6.5,bold:true,align:'center'});
arrow(12.60,3.48,12.60,3.75,{width:0.7});
addText('X*\n[B,9,3]',12.59,3.78,0.40,0.26,{fontSize:7,bold:true,align:'center'});
tokenStrip(12.54,4.10,[C.blue,C.blue,C.blue,C.blue]);

round(8.56,4.72,1.82,0.28,C.white,'4EA6B3',0.05,{width:0.6});
addText('✓  invalid context → no-op',8.64,4.80,1.66,0.13,{fontSize:7.2,bold:true,align:'center'});
round(10.63,4.72,2.25,0.28,C.white,'4EA6B3',0.05,{width:0.6});
addText('✓  tracks processed independently',10.70,4.80,2.10,0.13,{fontSize:7.2,bold:true,align:'center'});

// Footer
addText('(C) Spatial–Temporal Refinement', 0.16, 5.40, 3.7, 0.28, { fontSize: 12, bold: true });

pptx.writeFile({ fileName: 'outputs/vis/iclr_reference_structure/spatial_temporal_refinement_editable.pptx' });
