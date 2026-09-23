const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');

const imagePath = process.argv[2];
const cropDir = process.argv[3];
const outputPath = process.argv[4];

if (!imagePath || !cropDir || !outputPath) {
  console.error('Usage: node create_vggt_editable_components_ppt.js <source-image> <crop-dir> <output-pptx>');
  process.exit(2);
}

const pptx = new pptxgen();
pptx.defineLayout({ name: 'VGGT_OVERVIEW', width: 16, height: 4.55 });
pptx.layout = 'VGGT_OVERVIEW';
pptx.author = 'Codex';
pptx.company = 'OpenAI';
pptx.subject = 'Editable VGGT / vggt-human overview components';
pptx.title = 'Video and geometric priors — editable components';
pptx.lang = 'zh-CN';
pptx.theme = { headFontFace: 'Arial', bodyFontFace: 'Arial', lang: 'zh-CN' };

const addComponent = (slide, name, x, y, w, h) => {
  const componentPath = path.join(cropDir, `${name}.png`);
  if (!fs.existsSync(componentPath)) throw new Error(`Missing crop: ${componentPath}`);
  slide.addImage({
    path: componentPath,
    x, y, w, h,
    objectName: `component_${name}`,
    altText: `Editable component: ${name}`,
  });
};

const addTitle = (slide, name, text, x, w, fontSize = 15.5) => {
  slide.addText(text, {
    x, y: 0.06, w, h: 0.32,
    fontFace: 'Arial', fontSize, bold: true, color: '102A43',
    margin: 0, breakLine: false, fit: 'shrink',
    objectName: `editable_title_${name}`,
  });
};

// Slide 1: the figure is assembled from six independent image components and four editable text boxes.
const editable = pptx.addSlide();
editable.background = { color: 'FFFFFF' };
addComponent(editable, 'left_body', 0, 0.48, 3.2, 4.07);
addComponent(editable, 'stage01_upper', 3.2, 0.62, 5.1, 1.88);
addComponent(editable, 'stage01_lower', 3.2, 2.50, 5.1, 2.05);
addComponent(editable, 'stage02_upper', 8.3, 0.62, 4.9, 1.88);
addComponent(editable, 'stage02_lower', 8.3, 2.50, 4.9, 2.05);
addComponent(editable, 'right_body', 13.2, 0.62, 2.8, 3.93);
addTitle(editable, 'left', 'Video and geometric priors', 0.08, 3.0, 15.5);
addTitle(editable, 'stage01', '01  Temporal Scale Calibration', 3.34, 4.75, 15.5);
addTitle(editable, 'stage02', '02  Implicit Contact Refinement', 8.44, 4.5, 15.5);
addTitle(editable, 'right', 'Metric 4D reconstruction', 13.34, 2.55, 15.5);

// Slide 2: unmodified reference image for quick comparison.
const reference = pptx.addSlide();
reference.background = { color: 'FFFFFF' };
reference.addImage({
  path: path.resolve(imagePath), x: 0, y: 0, w: 16, h: 4.55,
  objectName: 'reference_full_figure',
  altText: 'Original VGGT overview figure',
});

const outDir = path.dirname(path.resolve(outputPath));
fs.mkdirSync(outDir, { recursive: true });
pptx.writeFile({ fileName: path.resolve(outputPath) });
