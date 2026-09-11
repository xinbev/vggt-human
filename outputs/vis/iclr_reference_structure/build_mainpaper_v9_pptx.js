const pptxgen = require('pptxgenjs');
const pptx = new pptxgen();
pptx.defineLayout({ name: 'CUSTOM_WIDE', width: 13.333, height: 7.5 });
pptx.layout = 'CUSTOM_WIDE';
pptx.author = 'Codex';
pptx.title = 'Spatial–Temporal Refinement — Editable Main-paper Figure';
pptx.subject = 'Editable PPT reconstruction of the main-paper architecture inset';
pptx.lang = 'en-US';

const C = {
  paper: 'F7F6F2', ink: '26313A', muted: '69757C', line: '8A99A5',
  blue: '99ADD4', blueLt: 'EAF1FB', cyan: 'A9CFD8', cyanLt: 'ECFAFC',
  lavender: 'CBBAD8', lavenderLt: 'F1EDF7', sage: 'AFCAB2', sageLt: 'EEF6EE',
  blush: 'D9ACA6', blushLt: 'FBEDEC', cream: 'E7DAAC', creamLt: 'FCF8E7',
  yellow: 'E9D36E', white: 'FFFFFF', red: 'C76B6B'
};
const slide = pptx.addSlide();
slide.background = { color: C.paper };

function txt(t,x,y,w,h,o={}) { slide.addText(t,{x,y,w,h,margin:0,fontFace:'Arial',fontSize:o.fontSize||11,color:o.color||C.ink,bold:o.bold||false,align:o.align||'left',valign:o.valign||'mid',fit:'shrink',...o}); }
function box(x,y,w,h,fill,line=C.line,o={}) { slide.addShape(pptx.ShapeType.roundRect,{x,y,w,h,rectRadius:0.08,fill:{color:fill,transparency:o.transparency||0},line:{color:line,width:o.width||0.8,dash:o.dash}}); }
function rect(x,y,w,h,fill,line=C.line,o={}) { slide.addShape(pptx.ShapeType.rect,{x,y,w,h,fill:{color:fill,transparency:o.transparency||0},line:{color:line,width:o.width||0.5}}); }
function ln(x1,y1,x2,y2,o={}) { slide.addShape(pptx.ShapeType.line,{x:x1,y:y1,w:x2-x1,h:y2-y1,line:{color:o.color||C.ink,width:o.width||1,dash:o.dash,endArrowType:o.arrow?'triangle':undefined}}); }
function arrow(x1,y1,x2,y2,o={}) { ln(x1,y1,x2,y2,{...o,arrow:true}); }
function dot(x,y,c,r=0.11) { slide.addShape(pptx.ShapeType.ellipse,{x,y,w:r,h:r,fill:{color:c},line:{color:C.line,width:0.45}}); }
function chip(x,y,w,label,fill) { box(x,y,w,0.30,fill,C.line,{width:0.55}); txt(label,x+0.02,y+0.07,w-0.04,0.15,{fontSize:7.3,bold:true,align:'center'}); }
function tokenRow(x,y,colors) { colors.forEach((c,i)=>box(x+i*0.16,y,0.12,0.12,c,C.line,{width:0.35})); }
function mesh(x,y,s=1,segmented=false) {
  slide.addShape(pptx.ShapeType.ellipse,{x:x+.18*s,y,w:.13*s,h:.14*s,fill:{color:C.white},line:{color:C.ink,width:.65}});
  box(x+.12*s,y+.14*s,.25*s,.44*s,segmented?C.sageLt:C.white,C.ink,{width:.65});
  ln(x+.14*s,y+.24*s,x-.01*s,y+.47*s,{width:.65}); ln(x+.35*s,y+.24*s,x+.50*s,y+.47*s,{width:.65});
  ln(x+.19*s,y+.58*s,x+.10*s,y+.88*s,{width:.65}); ln(x+.30*s,y+.58*s,x+.40*s,y+.88*s,{width:.65});
  if(segmented){[[.15,.24,C.blush],[.27,.24,C.blue],[.15,.39,C.cream],[.27,.39,C.lavender],[.15,.55,C.sage]].forEach(a=>rect(x+a[0]*s,y+a[1]*s,.09*s,.10*s,a[2],C.line,{width:.25}));}
}
function depth(x,y,w,h){rect(x,y,w,h,'D8D8D4','A0A09A',{width:.4});['CCCCCA','AFAFAC','8E8F8C'].forEach((c,i)=>rect(x+.02+i*w/3,y+.02,w/3-.02,h-.04,c,c,{width:0}));slide.addShape(pptx.ShapeType.ellipse,{x:x+w*.51,y:y+h*.13,w:w*.20,h:h*.63,fill:{color:'5E6261',transparency:28},line:{color:'5E6261',transparency:100}});}

// Header and source callouts
txt('Detailed architecture',0.17,0.12,1.6,.18,{fontSize:8.8,bold:true,color:C.muted});
box(.18,.38,.82,.28,C.blueLt,C.blue,{width:.7}); txt('TRSTR',.18,.45,.82,.13,{fontSize:9,bold:true,align:'center'});
box(11.78,.38,1.04,.28,C.cyanLt,'4AA7B5',{width:.7}); txt('Temporal',11.78,.45,1.04,.13,{fontSize:9,bold:true,align:'center'});
ln(.59,.66,.59,.90,{color:C.line,width:.7,dash:'dash'}); ln(12.30,.66,12.30,4.50,{color:C.line,width:.7,dash:'dash'});

// TRSTR outer panel
box(.16,.90,13.01,3.25,C.blueLt,C.blue,{width:1.0});
txt('TRSTR Regional Translator',4.60,1.02,4.05,.30,{fontSize:15,bold:true,align:'center'});

// Input card
box(.34,1.38,1.27,1.97,C.white,C.line,{width:.7});
txt('Metric SMPL',.48,1.54,.95,.18,{fontSize:9.5,bold:true,align:'center'}); mesh(.77,1.82,.78,false);
txt('Metric scene\ndepth',.47,2.52,.98,.31,{fontSize:8.8,bold:true,align:'center'}); depth(.63,2.84,.63,.37);
arrow(1.62,2.32,1.84,2.32,{width:.95});

// Stage 1
box(1.84,1.38,2.02,1.97,C.white,C.line,{width:.8});
txt('Region Construction',1.98,1.53,1.72,.20,{fontSize:10.7,bold:true,align:'center'});
txt('96 body regions',2.03,1.85,.85,.14,{fontSize:8.2,align:'center'}); mesh(2.37,2.02,.95,true);
txt('multi-scale probes',2.95,1.85,.78,.14,{fontSize:7.5,align:'center'});
[['3×3',C.lavenderLt],['7×7',C.blueLt],['adaptive',C.sageLt],['annulus',C.creamLt]].forEach((v,i)=>{const yy=2.06+i*.28;box(3.02,yy,.24,.20,v[1],C.line,{width:.4});txt(v[0],3.34,yy+.04,.48,.11,{fontSize:7.5,bold:true});});
arrow(3.86,2.32,4.12,2.32,{width:.95});

// Stage 2 interaction encoder
box(4.12,1.38,2.78,1.97,C.sageLt,'6FA982',{width:.8});
txt('Human–Scene Interaction\nEncoder',4.32,1.52,2.36,.38,{fontSize:10.6,bold:true,align:'center'});
[['region geometry',C.sageLt],['local scene',C.cyanLt],['owner mask',C.creamLt]].forEach((v,i)=>{txt(v[0],4.43,2.06+i*.33,1.10,.13,{fontSize:7.8,color:i===2?'9B8054':C.muted});tokenRow(4.57,2.21+i*.33,[v[1],v[1],v[1]]);});
slide.addShape(pptx.ShapeType.ellipse,{x:5.53,y:2.35,w:.25,h:.25,fill:{color:C.white},line:{color:C.ink,width:.7}});txt('C',5.56,2.40,.18,.10,{fontSize:7,bold:true,align:'center'});
arrow(5.78,2.48,6.05,2.48,{width:.75});
box(6.07,2.12,.62,.75,C.lavenderLt,C.line,{width:.65});txt('MLP',6.13,2.25,.50,.14,{fontSize:8.3,bold:true,align:'center'});txt('interaction\ntoken',6.10,2.52,.57,.22,{fontSize:7.3,align:'center'});
arrow(6.91,2.32,7.10,2.32,{width:.95});

// Stage 3 heads
box(7.10,1.38,2.10,1.97,C.white,C.line,{width:.8});
txt('Regional Correction Heads',7.24,1.53,1.78,.20,{fontSize:10,bold:true,align:'center'});
txt('interaction token',7.28,2.05,.68,.13,{fontSize:7.0,color:C.muted});tokenRow(7.30,2.23,[C.blueLt,C.blueLt,C.blueLt]);
arrow(7.95,2.30,8.20,2.30,{width:.65});
chip(8.24,1.98,.72,'vote Δτᵣ',C.blushLt);chip(8.24,2.40,.72,'gate gᵣ',C.sageLt);chip(8.24,2.82,.72,'uncertainty σᵣ',C.blueLt);
arrow(9.20,2.32,9.42,2.32,{width:.95});

// Stage 4 aggregation
box(9.42,1.38,2.22,1.97,C.white,C.line,{width:.8});
txt('Robust Aggregation',9.63,1.53,1.80,.20,{fontSize:10.5,bold:true,align:'center'});
txt('weighted\nfusion',9.88,2.14,.48,.28,{fontSize:8.5,bold:true,align:'center'});
slide.addShape(pptx.ShapeType.chevron,{x:10.02,y:2.46,w:.32,h:.50,rotate:90,fill:{color:C.blueLt},line:{color:C.ink,width:.65}});
arrow(10.35,2.72,10.64,2.72,{width:.75});
box(10.66,2.48,.67,.42,C.lavenderLt,C.line,{width:.65});txt('person\ngate',10.70,2.57,.59,.18,{fontSize:7.5,bold:true,align:'center'});
slide.addShape(pptx.ShapeType.arc,{x:11.00,y:1.37,w:.65,h:.58,adjustPoint:.3,line:{color:C.ink,width:.8,beginArrowType:'triangle',endArrowType:'triangle'}});
txt('2×\nre-probe',11.08,1.43,.45,.28,{fontSize:8.2,bold:true,align:'center'});
arrow(11.64,2.32,11.88,2.32,{width:.95});
box(11.90,2.04,1.00,.68,C.white,C.line,{width:.75});txt('refined\ntranslation\nτ*',12.00,2.15,.80,.35,{fontSize:8.7,bold:true,align:'center'});
txt('translation only',12.00,2.97,.80,.15,{fontSize:7.2,bold:true,align:'center'});
box(1.84,3.54,9.72,.30,'DCE8F7',C.blue,{width:.45});txt('pose θ and shape β unchanged',1.84,3.61,9.72,.13,{fontSize:8.7,bold:true,align:'center'});

// Temporal outer panel
box(.16,4.45,13.01,2.55,C.cyanLt,'4AA7B5',{width:1.0});
txt('Track-wise Temporal Fusion',4.62,4.58,4.08,.28,{fontSize:15,bold:true,align:'center'});

// Window card
box(.42,4.95,2.68,1.70,C.white,'4AA7B5',{width:.75});
txt('Track Window',.62,5.10,1.4,.20,{fontSize:10.5,bold:true});
txt('t−4   …   t   …   t+4',1.18,5.38,1.80,.14,{fontSize:7.5,align:'center'});
['ID 1','ID 2','ID 3'].forEach((id,row)=>{txt(id,.58,5.66+row*.38,.40,.15,{fontSize:8.5,bold:true});const c=[C.blue,C.blush,C.cyan][row];for(let i=0;i<7;i++){dot(1.10+i*.34,5.62+row*.38,c,.17);if(i<6)ln(1.27+i*.34,5.71+row*.38,1.44+i*.34,5.71+row*.38,{color:c,width:.7});}});
box(1.98,5.47,.28,1.18,'FFFFFF','4AA7B5',{width:.55,dash:'dash'});
arrow(3.10,5.80,3.32,5.80,{width:.95});

// Motion proposal
box(3.32,4.95,3.14,1.70,C.white,'4AA7B5',{width:.75});
txt('Motion Proposal',3.52,5.10,1.65,.20,{fontSize:10.5,bold:true});
tokenRow(3.59,5.56,[C.blueLt,C.blueLt,C.blueLt]);tokenRow(3.59,5.89,[C.blushLt,C.blushLt,C.blushLt]);tokenRow(3.59,6.22,[C.sageLt,C.sageLt,C.sageLt]);
arrow(4.50,5.92,4.84,5.92,{width:.8});
box(4.86,5.46,1.05,.68,C.blueLt,C.line,{width:.7});txt('Masked\nTemporal MLP',4.97,5.62,.83,.29,{fontSize:8.6,bold:true,align:'center'});
txt('×',5.22,6.15,.25,.20,{fontSize:16,bold:true,color:C.red,align:'center'});
txt('neighbors only',3.66,6.40,1.34,.13,{fontSize:7.5,bold:true,align:'center'});
arrow(5.92,5.80,6.20,5.80,{width:.8});
tokenRow(6.20,5.73,[C.lavender,C.lavender,C.lavender]);txt('Mₜ',6.22,5.54,.40,.13,{fontSize:8.3,bold:true,align:'center'});
arrow(6.47,5.80,6.70,5.80,{width:.95});

// Fusion
box(6.70,4.95,3.05,1.70,C.white,'4AA7B5',{width:.75});
txt('Conservative Fusion',6.94,5.10,1.9,.20,{fontSize:10.5,bold:true});
chip(6.92,5.45,.40,'Oₜ',C.cyanLt);chip(6.92,5.97,.40,'Mₜ',C.lavenderLt);
arrow(7.32,5.60,7.62,5.60,{width:.75});arrow(7.32,6.12,7.62,6.12,{width:.75});
box(7.64,5.57,.68,.54,C.white,C.line,{width:.65});txt('Gate\nMLP',7.73,5.68,.50,.22,{fontSize:8.2,bold:true,align:'center'});
arrow(8.33,5.84,8.55,5.84,{width:.75});
box(8.56,5.60,.65,.34,C.creamLt,C.cream,{width:.6});txt('αₜ ≤ 0.5',8.58,5.70,.61,.13,{fontSize:7.3,bold:true,align:'center'});
arrow(8.88,5.96,8.88,6.10,{width:.7});
box(7.10,6.13,2.16,.32,C.white,C.line,{width:.55});txt('X* = Oₜ + αₜ(Mₜ − Oₜ)',7.20,6.22,1.96,.12,{fontSize:7.3,bold:true,align:'center'});
arrow(9.75,5.80,9.98,5.80,{width:.95});

// stable card
box(9.98,4.95,2.82,1.70,C.white,'4AA7B5',{width:.75});
txt('Stable sequence',10.20,5.10,1.50,.20,{fontSize:10.2,bold:true});
[[C.blue,5.62],[C.blush,5.96],[C.cyan,6.30]].forEach(([c,y])=>{for(let i=0;i<5;i++){dot(10.20+i*.37,y,c,.16);if(i<4)ln(10.36+i*.37,y+.08,10.57+i*.37,y+.08,{color:c,width:.75});}});
txt('(C) Spatial–Temporal Refinement',.17,7.12,3.8,.22,{fontSize:12.5,bold:true});

pptx.writeFile({ fileName: 'outputs/vis/iclr_reference_structure/spatial_temporal_refinement_mainpaper_editable.pptx' });
