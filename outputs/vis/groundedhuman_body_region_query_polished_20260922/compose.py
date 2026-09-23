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



# B: all support glyphs and token slots below are conceptual, not measured scores.
def polygon(points, fill):
    d.polygon([(round(x*S),round(y*S)) for x,y in points],fill=fill)
    svg.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in points)}" fill="{fill}"/>')

def rounded(x,y,w,h,r,fill,stroke=None,sw=1):
    d.rounded_rectangle((round(x*S),round(y*S),round((x+w)*S),round((y+h)*S)),
                        radius=round(r*S),fill=fill,outline=stroke,width=round(sw*S))
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{sw}"/>')

def arrow(x1,y1,x2,y2,col=TEAL,width=1.5):
    line([(x1,y1),(x2,y2)],col,width)
    a=math.atan2(y2-y1,x2-x1)
    polygon([(x2,y2),(x2-6*math.cos(a)+3*math.sin(a),y2-6*math.sin(a)-3*math.cos(a)),
             (x2-6*math.cos(a)-3*math.sin(a),y2-6*math.sin(a)+3*math.cos(a))],col)

def support(x,y,kind,col):
    rounded(x,y,60,60,3,'#FBFCFD','#DFE5E9')
    # Common pixel pitch across glyphs. Adaptive radius is an illustrative example.
    half={'3':1,'7':3,'adp':4,'ann':6}[kind]
    pitch=4.2
    edge=(half+.5)*pitch
    if kind=='ann':
        rect(x+30-edge,y+30-edge,edge*2,edge*2,fill='#F2EBF8')
        rect(x+11.1,y+11.1,37.8,37.8,fill='#FBFCFD')
    for i in range(-half,half+1):
        for j in range(-half,half+1):
            if kind!='ann' or max(abs(i),abs(j))>4:
                circle(x+30+i*pitch,y+30+j*pitch,1.15,col)
    rect(x+30-edge,y+30-edge,edge*2,edge*2,stroke=col,sw=1.2)
    if kind=='ann':rect(x+11.1,y+11.1,37.8,37.8,stroke=col,sw=1)
    # Region center, distinct from the surrounding sampled pixel dots.
    circle(x+30,y+30,3.1,'white')
    circle(x+30,y+30,2.15,PINK)

txt(44,613,'(b) Body Region Context Query',30,INK,True)
txt(44,648,'Region-conditioned geometry for body placement',22,TEAL)
line([(44,667),(721,667)],RULE,1)

txt(44,697,'Body region r',20,PINK,True)
txt(324,697,'Multi-scale scene context',20,INK,True)
pic('regional_samples',44,710,210)
rect(44,710,210,210*208/248,stroke='#E4D7E0',sw=.8)
circle(49,899,3.2,TEAL);txt(59,904,'Self',16,TEAL)
circle(115,899,3.2,AMBER);txt(125,904,'Environment',16,AMBER)

cols=[(342,'3 × 3','3',BLUE),(442,'7 × 7','7','#539877'),
      (542,'Adaptive','adp',AMBER),(642,'Outer ring','ann','#9466AA')]
for x,label,kind,col in cols:
    support(x,714,kind,col)
    txt(x+30,796,label,18,INK,False,'center')
    # Eight token slots: no intensity / length encodes confidence.
    for y,fg,bg in [(811,TEAL,'#E3F2F1'),(840,AMBER,'#FAF0DC')]:
        rounded(x+3,y,54,18,2,bg,fg,.9)
        for k in range(1,4):line([(x+3+k*13.5,y+4),(x+3+k*13.5,y+14)],fg,.6)
txt(332,825,'Self',16,TEAL,True,'right')
txt(332,854,'Env.',16,AMBER,True,'right')

# Visual grouping: all four scales and both channels contribute to Context.
line([(345,866),(345,871),(699,871),(699,866)],'#83BCB9',1)
txt(523,892,'4 scales × 2 evidence types',17,TEAL,False,'center')

# The four channels descend into one structured descriptor, as in the reference.
polygon([(51,910),(107,910),(239,932),(134,932)],'#F4DFED')
polygon([(186,910),(249,910),(381,932),(245,932)],'#E8ECF3')
polygon([(345,900),(699,900),(565,932),(387,932)],'#E0F0ED')
polygon([(706,864),(717,864),(709,932),(571,932)],'#EBEDF1')

rounded(44,932,677,78,5,'#F7F8FA','#E6E9ED',.9)
txt(84,958,'Query',17,INK,True,'center')
txt(84,990,'qᵣ',28,PINK,True,'center')
# Brackets indicate concatenation, not arithmetic addition of incompatible inputs.
line([(130,941),(124,941),(124,1001),(130,1001)],'#AAB3BE',1.3)
line([(711,941),(715,941),(715,1001),(711,1001)],'#AAB3BE',1.3)
fields=[(134,105,'Identity','Body-part ID','#FCEDF5',PINK),
        (245,136,'Geometry','Position · extent','#EFF2F7','#677793'),
        (387,178,'Context','8 pooled tokens','#EAF6F3',TEAL),
        (571,138,'Validity','Support · overlap','#F0F2F5',MUTED)]
for x,w,title,desc,bg,col in fields:
    rounded(x,940,w,62,3,bg)
    line([(x+9,942),(x+w-9,942)],col,1.8)
    txt(x+w/2,968,title,19,col,True,'center')
    txt(x+w/2,991,desc,16,INK,False,'center')

# A small downward cue makes the query-to-prediction connection explicit.
arrow(152,1011,152,1023,'#94AAA9',1)
txt(44,1047,'Regional proposals',21,TEAL,True)
arrow(248,1040,286,1040)
txt(300,1047,'One whole-body translation',21,TEAL,True)
txt(417,1072,'Reliability-weighted aggregation',17,MUTED,False,'center')

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
im.save(OUT/'teaser_body_region_query_polished.png',dpi=(300,300))
(OUT/'teaser_body_region_query_polished.svg').write_text('\n'.join(svg),encoding='utf-8')

im.crop((20*S,580*S,747*S,1080*S)).save(OUT/'body_region_context_query_polished.png',dpi=(300,300))
detail_svg='\n'.join(svg).replace('width="1860" height="1080" viewBox="0 0 1860 1080"',
    'width="1454" height="1000" viewBox="20 580 727 500"',1)
(OUT/'body_region_context_query_polished.svg').write_text(detail_svg,encoding='utf-8')
print(OUT/'body_region_context_query_polished.png')
