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


# B: a visual query specification, not a new transformer architecture.
def polygon(points, fill):
    d.polygon([(round(x*S),round(y*S)) for x,y in points],fill=fill)
    svg.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in points)}" fill="{fill}"/>')

def support(x,y,kind,col):
    # Idealized support glyphs. Dots represent sampled locations, not attention scores.
    rect(x,y,58,58,fill='#FAFBFC',stroke='#D8E0E7')
    extent={'3':9,'7':19,'adp':25,'ann':28}[kind]
    half={'3':1,'7':3,'adp':4,'ann':4}[kind]
    pitch=8 if kind=='3' else 6
    for i in range(-half,half+1):
        for j in range(-half,half+1):
            radius=max(abs(i*pitch),abs(j*pitch))
            if kind!='ann' or radius>=24:
                circle(x+29+i*pitch,y+29+j*pitch,1.35,col)
    rect(x+29-extent,y+29-extent,extent*2,extent*2,stroke=col,sw=1.5)
    if kind=='ann':rect(x+7,y+7,44,44,stroke=col,sw=1)
    circle(x+29,y+29,2.6,PINK)

txt(44,613,'(b) Body Region Context Query',30,INK,True)
txt(44,649,'Query the surrounding geometry to refine body placement',21,TEAL)
line([(44,668),(721,668)],RULE,1)

# One source region and its four square sampling domains.
pic('regional_samples',44,706,220)
txt(44,696,'Body region r',20,PINK,True)
txt(324,696,'Multi-scale scene context',20,INK,True)
columns=[(340,'3 × 3','3',BLUE),(441,'7 × 7','7','#539877'),
         (542,'Adaptive','adp',AMBER),(643,'Annulus','ann','#9466AA')]
for x,label,kind,col in columns:
    support(x,710,kind,col)
    txt(x+29,792,label,17,INK,False,'center')
    # A token slot is symbolic. No fill intensity encodes measured confidence.
    rect(x+1,809,56,16,fill='#D8EFEF',stroke=TEAL)
    rect(x+1,835,56,16,fill='#F9EACC',stroke=AMBER)
txt(332,823,'Self',15,TEAL,True,'right')
txt(332,849,'Env.',15,AMBER,True,'right')

# Tinted ribbons borrow the reference's query-construction language.
# Identity and geometry both come from the region; context and validity from sampling.
polygon([(52,891),(107,891),(238,917),(132,917)],'#F6DFED')
polygon([(190,891),(258,891),(381,917),(244,917)],'#E7EAF1')
polygon([(342,857),(700,857),(565,917),(387,917)],'#E4F2EF')
polygon([(705,857),(721,857),(709,917),(571,917)],'#E6E9EE')
txt(514,883,'Pool by scale & type',15,TEAL,False,'center')

# Four constituents of the assembled per-region descriptor.
rect(44,917,677,84,fill='#F4F5F7')
txt(58,951,'Query',19,INK,True)
txt(63,980,'qᵣ',23,PINK,True)
fields=[(132,106,'Identity','Body-part ID','Embedding','#FCECF5',PINK),
        (244,137,'Geometry','Position + extent','3D / projected 2D','#EFF1F6','#677793'),
        (387,178,'Context','4 scales × 2 types','Self + environment','#EEF8F6',TEAL),
        (571,138,'Validity','Support ratios','Occlusion cues','#F0F1F4',MUTED)]
for x,w,title,desc,detail,bg,col in fields:
    rect(x,925,w,68,fill=bg)
    txt(x+w/2,945,title,18,col,True,'center')
    txt(x+w/2,966,desc,15,INK,False,'center')
    txt(x+w/2,984,detail,14,MUTED,False,'center')

# Keep the decoder implicit: the existing network predicts and weights regional proposals.
txt(44,1035,'Regional proposals',21,TEAL,True)
line([(241,1028),(281,1028)],TEAL,2)
polygon([(281,1028),(274,1024),(274,1032)],TEAL)
txt(297,1035,'One whole-body translation',21,TEAL,True)
txt(249,1062,'Reliability-weighted aggregation',17,MUTED)

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
im.save(OUT/'teaser_body_region_query.png',dpi=(300,300))
(OUT/'teaser_body_region_query.svg').write_text('\n'.join(svg),encoding='utf-8')

im.crop((20*S,580*S,747*S,1080*S)).save(OUT/'body_region_context_query.png',dpi=(300,300))
detail_svg='\n'.join(svg).replace('width="1860" height="1080" viewBox="0 0 1860 1080"',
    'width="1454" height="1000" viewBox="20 580 727 500"',1)
(OUT/'body_region_context_query.svg').write_text(detail_svg,encoding='utf-8')
print(OUT/'body_region_context_query.png')
