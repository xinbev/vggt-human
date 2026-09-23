const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');

// Build a single-slide deck that preserves the supplied overview figure at native aspect ratio.
const imagePath = process.argv[2];
const outputPath = process.argv[3];

if (!imagePath || !outputPath) {
  console.error('Usage: node create_vggt_overview_ppt.js <image-path> <output-pptx>');
  process.exit(2);
}
if (!fs.existsSync(imagePath)) {
  console.error(`Input image does not exist: ${imagePath}`);
  process.exit(2);
}

const pptx = new pptxgen();
// 1600x455 source image => 3.5165:1 aspect ratio. A custom slide size keeps the figure sharp
// and avoids adding large letterbox bands around the research diagram.
pptx.defineLayout({ name: 'VGGT_OVERVIEW', width: 16, height: 4.55 });
pptx.layout = 'VGGT_OVERVIEW';
pptx.author = 'Codex';
pptx.company = 'OpenAI';
pptx.subject = 'VGGT / vggt-human overview';
pptx.title = 'Video and geometric priors — Metric 4D reconstruction';
pptx.lang = 'zh-CN';
pptx.theme = {
  headFontFace: 'Aptos Display',
  bodyFontFace: 'Aptos',
  lang: 'zh-CN',
};

const slide = pptx.addSlide();
slide.background = { color: 'FFFFFF' };
slide.addImage({
  path: path.resolve(imagePath),
  x: 0,
  y: 0,
  w: 16,
  h: 4.55,
  altText: 'Video and geometric priors pipeline for metric 4D reconstruction',
});

const outDir = path.dirname(path.resolve(outputPath));
fs.mkdirSync(outDir, { recursive: true });
pptx.writeFile({ fileName: path.resolve(outputPath) });

