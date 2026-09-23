"""Polish the approved B composition without changing its scientific narrative."""
from pathlib import Path
import runpy

OUT=Path(__file__).resolve().parent
base=(OUT.parent/'groundedhuman_body_region_query_20260922/compose.py').read_text(encoding='utf-8')
b=r'''
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
'''
start=base.index('# B: a visual query')
end=base.index('# 3. Keep the tilted scene')
result=base[:start]+b+'\n'+base[end:]
# Export standalone view with slight padding below its aggregation label.
result=result.replace("'teaser_body_region_query.png'","'teaser_body_region_query_polished.png'")
result=result.replace("'teaser_body_region_query.svg'","'teaser_body_region_query_polished.svg'")
result=result.replace("'body_region_context_query.png'","'body_region_context_query_polished.png'")
result=result.replace("'body_region_context_query.svg'","'body_region_context_query_polished.svg'")
(OUT/'compose.py').write_text(result,encoding='utf-8')
runpy.run_path(str(OUT/'compose.py'),run_name='__main__')
