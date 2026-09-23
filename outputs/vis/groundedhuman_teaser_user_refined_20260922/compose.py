"""Compose the user's teaser assets; no new inference or synthetic evidence."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64
import html
import math

OUT = Path(__file__).resolve().parent
SOURCE = Path('C:/Users/ROG/AppData/Local/Temp/codex-clipboard-d90a79ac-3330-48fe-a550-51247d07f03f.png')
W, H, S = 1860, 1080, 2
INK, MUTED = '#293341', '#65717D'
PINK, BLUE, TEAL, AMBER = '#B44E8B', '#4775BF', '#24999E', '#D39825'
PALE, RULE = '#FAF5F8', '#E5E9EE'
src = Image.open(SOURCE).convert('RGB')
assets = {}
for name, box in {
    'anchor_attention': (98, 118, 232, 252),
    'body_projection': (240, 100, 506, 319),
    'regional_samples': (260, 488, 508, 696),
    'human_scene': (573, 206, 1123, 585),
}.items():
    assets[name] = src.crop(box)
    if name == 'body_projection':
        # Remove fragments of the old caption/pill caught at crop margins.
        cleanup = ImageDraw.Draw(assets[name])
        cleanup.rectangle((0,210,140,219),fill='white')
        cleanup.rectangle((0,160,5,199),fill='white')
    assets[name].save(OUT / f'{name}.png')

im = Image.new('RGB', (W*S, H*S), 'white')
d = ImageDraw.Draw(im)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
       '<rect width="100%" height="100%" fill="white"/>',
       '<g font-family="Arial, DejaVu Sans, sans-serif">']

def font(n, bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/arial'+('bd' if bold else '')+'.ttf', round(n*S))

def txt(x, y, t, n=22, c=INK, bold=False, align='left'):
    f = font(n, bold)
    length = d.textlength(t, font=f)/S
    if align == 'center': x -= length/2
    if align == 'right': x -= length
    # Baseline-aligned text gives identical positioning in raster and vector.
    d.text((round(x*S), round(y*S)), t, font=f, fill=c, anchor='ls')
    svg.append(f'<text x="{x:.2f}" y="{y}" font-size="{n}" fill="{c}" font-weight="{700 if bold else 400}">{html.escape(t)}</text>')

def line(points, c=RULE, w=1.5, dashed=False):
    q = [(round(x*S), round(y*S)) for x, y in points]
    if dashed:
        for (x0,y0),(x1,y1) in zip(q,q[1:]):
            distance=math.hypot(x1-x0,y1-y0)
            for start in range(0,round(distance),14*S):
                end=min(start+7*S,distance)
                d.line([(x0+(x1-x0)*start/distance,y0+(y1-y0)*start/distance),
                        (x0+(x1-x0)*end/distance,y0+(y1-y0)*end/distance)],fill=c,width=round(w*S))
    else: d.line(q, fill=c, width=round(w*S))
    dash = ' stroke-dasharray="7 7"' if dashed else ''
    svg.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="{c}" stroke-width="{w}"{dash}/>')

def rect(x,y,w,h,fill=None,stroke=None,sw=1):
    d.rectangle((round(x*S),round(y*S),round((x+w)*S),round((y+h)*S)),fill=fill,outline=stroke,width=round(sw*S))
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill or "none"}" stroke="{stroke or "none"}" stroke-width="{sw}"/>')

def circle(x,y,r,fill=None,stroke=None,sw=1):
    d.ellipse((round((x-r)*S),round((y-r)*S),round((x+r)*S),round((y+r)*S)),fill=fill,outline=stroke,width=round(sw*S))
    svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill or "none"}" stroke="{stroke or "none"}" stroke-width="{sw}"/>')

def pic(name,x,y,w):
    a=assets[name];h=w*a.height/a.width
    im.paste(a.resize((round(w*S),round(h*S)),Image.Resampling.LANCZOS),(round(x*S),round(y*S)))
    buffer=BytesIO();a.save(buffer,format='PNG')
    b64=base64.b64encode(buffer.getvalue()).decode()
    svg.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="data:image/png;base64,{b64}"/>')
    return h

def badge(x,y,t,c):
    circle(x,y,16,'white',c,1.5)
    txt(x,y+6,t,20,c,True,'center')

def window_icon(x,y,kind,c):
    # Symbolic support legend, independent of the recorded sample crop.
    rect(x,y,34,34,stroke='#D2D9DF')
    if kind=='small': rect(x+12,y+12,10,10,stroke=c,sw=2)
    elif kind=='medium': rect(x+7,y+7,20,20,stroke=c,sw=2)
    elif kind=='adaptive': rect(x+3,y+3,28,28,stroke=c,sw=2)
    else:
        rect(x,y,34,34,fill='#EEF4F4',stroke=c,sw=2)
        rect(x+6,y+6,22,22,fill='white',stroke=c)
    circle(x+17,y+17,1.5,c)

# 1. The user's anchor/projection vignette stays above the regional crop.
txt(44,60,'(a) Human-anchored scale alignment',30,INK,True)
txt(44,98,'Anchor–Scene Cross Attention',23,BLUE)
line([(44,120),(721,120)],RULE,1)
pic('anchor_attention',44,154,202)
pic('body_projection',307,142,374)
txt(44,387,'3 × 3 scene tokens',21,BLUE,True)
txt(44,414,'Body-anchor query',20,AMBER)
txt(319,475,'24 anatomical anchors',20,PINK,True)
# Camera projection rays remain distinct from feature-attention links.
txt(621,475,'Projection',16,MUTED)
line([(252,244),(292,244)],'#A6B7CB',1.5,True)

rect(44,454,242,42,fill='#F4F7FC')
txt(58,481,'Local 3D scene probe',20,BLUE)
txt(44,523,'Nearest 3D point in a 9 × 9 depth-pixel search',18,MUTED)
line([(44,544),(66,544)],PINK,3)
txt(78,551,'Scene scale + depth-bias correction',23,PINK,True)

# 2. Actual sampled pixel crop + compact, explicitly square support legend.
txt(44,613,'(b) Region-guided placement refinement',30,INK,True)
txt(44,651,'Regional Translation Prediction',23,TEAL)
line([(44,672),(721,672)],RULE,1)
pic('regional_samples',44,700,296)
txt(376,720,'Four image-space supports',21,INK,True)
rows=[('small','3 × 3','Fine local geometry',BLUE),
      ('medium','7 × 7','Wider local geometry','#539877'),
      ('adaptive','Adaptive','Projected region extent',AMBER),
      ('annulus','Outer annulus','Surrounding context','#9466AA')]
for i,(kind,label,desc,col) in enumerate(rows):
    y=740+i*53
    window_icon(376,y,kind,col)
    txt(423,y+15,label,20,INK,True)
    txt(423,y+37,desc,17,MUTED)
circle(50,974,4.5,TEAL);txt(64,981,'Self-surface',18,MUTED)
circle(209,974,4.5,AMBER);txt(223,981,'Environment',18,MUTED)
line([(44,1014),(66,1014)],TEAL,3)
txt(78,1021,'Reliability-weighted votes → one body translation',23,TEAL,True)

# 3. Keep the tilted scene and pink body as the dominant hero.
line([(772,42),(772,1035)],'#EDF0F3',1)
txt(842,60,'Human–scene alignment',32,INK,True)
txt(842,102,'A shared metric reference. A refined body position.',24,MUTED)
pic('human_scene',827,268,992)

# These are conceptual outcome annotations, not measured scale bars/vectors.
badge(872,211,'a',PINK)
txt(902,219,'Calibrated scene geometry',24,PINK,True)
line([(863,248),(1775,248)],'#C5A8BA',1.5)
line([(863,240),(863,260)],'#C5A8BA',1.5)
line([(1775,240),(1775,260)],'#C5A8BA',1.5)

# Visual locator for the pelvis region; not an exact projected model anchor.
px=827+(790-573)/550*992
py=268+(505-206)/550*992
circle(px,py,19,None,TEAL,1.7)
line([(px-18,py+8),(1007,970),(899,970)],TEAL,1.5)
badge(872,970,'b',TEAL)
txt(904,1009,'Whole-body translation',24,TEAL,True)
txt(904,1040,'Body pose and shape preserved',20,MUTED)
txt(1757,1009,'Multi-scale evidence',22,INK,True,'right')
txt(1757,1040,'Self-surface + environment',20,MUTED,False,'right')

svg.append('</g></svg>')
im.save(OUT/'teaser_refined.png',dpi=(300,300))
(OUT/'teaser_refined.svg').write_text('\n'.join(svg),encoding='utf-8')
# A small comparison sheet for reviewing structure, not the submission figure.
comparison=Image.new('RGB',(1860,700),'#F4F5F7')
cd=ImageDraw.Draw(comparison)
cf=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',24)
cd.text((35,25),'YOUR COMPOSITION',font=cf,fill=INK)
cd.text((790,25),'REFINED SCIENTIFIC NARRATIVE',font=cf,fill=INK)
original=src.copy();original.thumbnail((720,580),Image.Resampling.LANCZOS)
comparison.paste(original,(30,88))
preview=im.copy();preview.thumbnail((1050,610),Image.Resampling.LANCZOS)
comparison.paste(preview,(785,88))
comparison.save(OUT/'comparison.png')
print(OUT/'teaser_refined.png')
