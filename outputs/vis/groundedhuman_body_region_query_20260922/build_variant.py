"""Build a query-composition teaser revision using the existing screenshot assets."""
from pathlib import Path
import runpy

OUT=Path(__file__).resolve().parent
base=(OUT.parent/'groundedhuman_teaser_user_refined_20260922/compose.py').read_text(encoding='utf-8')
b=r'''
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
'''
start=base.index('# 2. Actual sampled pixel')
end=base.index('# 3. Keep the tilted scene')
result=base[:start]+b+'\n'+base[end:]
result=result.replace("'teaser_refined.png'","'teaser_body_region_query.png'")
result=result.replace("'teaser_refined.svg'","'teaser_body_region_query.svg'")
result=result[:result.index('# A small comparison sheet')]
result += '''
im.crop((20*S,580*S,747*S,1080*S)).save(OUT/'body_region_context_query.png',dpi=(300,300))
detail_svg='\\n'.join(svg).replace('width="1860" height="1080" viewBox="0 0 1860 1080"',
    'width="1454" height="1000" viewBox="20 580 727 500"',1)
(OUT/'body_region_context_query.svg').write_text(detail_svg,encoding='utf-8')
print(OUT/'body_region_context_query.png')
'''
(OUT/'compose.py').write_text(result,encoding='utf-8')
runpy.run_path(str(OUT/'compose.py'),run_name='__main__')
