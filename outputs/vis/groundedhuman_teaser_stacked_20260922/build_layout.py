from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO

OUT=Path(__file__).resolve().parent
ASSETS=OUT.parent/'groundedhuman_teaser_layout_20260922'
W,H,S=1800,1000,2
im=Image.new('RGB',(W*S,H*S),'white');d=ImageDraw.Draw(im)
ink='#183A45';muted='#667D83';coral='#CD776C';teal='#28959A';amber='#D79C28';purple='#9466AA'
def font(n,b=False,serif=False):return ImageFont.truetype('C:/Windows/Fonts/'+('times' if serif else 'arial')+('bd' if b else '')+'.ttf',n*S)
def txt(x,y,t,n=22,c=ink,b=False,serif=False):d.text((x*S,y*S),t,font=font(n,b,serif),fill=c)
def line(p,c=muted,w=1):d.line([(int(x*S),int(y*S)) for x,y in p],fill=c,width=w*S)
def ellipse(box,c=coral,w=2):d.ellipse(tuple(int(v*S) for v in box),outline=c,width=w*S)
def crop_asset(name,box):
 x,y,w,h=box;a=Image.open(ASSETS/(name+'.png')).convert('RGB');a=a.resize((w*S,h*S),Image.Resampling.LANCZOS);im.paste(a,(x*S,y*S))

# The two explanations stay in a left column; hero takes 65% of width.
txt(38,27,'(a) Human-anchored metric alignment',25,b=True)
line([(38,70),(512,70)],'#DEC2BC',2)
crop_asset('anchor20',(38,97,246,246))
txt(306,110,'Body anchor',21,c=coral,b=True)
txt(306,149,'3 x 3 local',21)
txt(306,178,'scene tokens',21)
txt(306,225,'Cross-attention',19,c=muted)
txt(306,254,'weights',19,c=muted)
txt(38,367,'Anchor-scene cross attention',21,b=True)
txt(38,401,'Human metric prior + local scene evidence',18,c=muted)
txt(38,437,'D',28,serif=True);txt(59,430,'m',16,serif=True)
txt(80,437,'= s D',28,serif=True);txt(148,430,'r',16,serif=True);txt(165,437,'+ b',28,serif=True)
txt(290,444,'Shared metric reference',17,c=coral)

txt(38,535,'(b) Regional placement refinement',25,b=True)
line([(38,578),(512,578)],'#B7D4D3',2)
crop_asset('hands54',(38,602,246,246))
for label,y,col in [('3 x 3',620,'#2885DB'),('7 x 7',664,'#3F9D66'),('Adaptive',708,amber),('Outer annulus',752,purple)]:
 line([(303,y+11),(317,y+11)],col,3);txt(328,y,label,20,c=col)
txt(38,866,'Regional translation prediction',21,b=True)
txt(38,900,'Multi-scale evidence -> reliable regional votes',18,c=muted)
for x,label,col in [(38,'Self-surface',teal),(210,'Environment',amber)]:
 d.ellipse((x*S,944*S,(x+9)*S,953*S),fill=col);txt(x+18,936,label,17,c=muted)

txt(608,29,'Grounded human-scene reconstruction',31,b=True)
txt(608,76,'A shared scale. A jointly refined body position.',22,c=muted)
crop_asset('scene',(608,163,1148,764))
# Modest scientific annotations on the same scene. Leader endpoints are visual locators.
hand=(608+(559-382)/365*1148,163+(155-21)/243*764)
ellipse((hand[0]-29,hand[1]-23,hand[0]+29,hand[1]+23),coral,2)
line([(515,223),(558,223),(588,488),(hand[0]-26,hand[1]-12)],'#D9A69D',1)
line([(515,724),(570,724),(hand[0]-23,hand[1]+16)],teal,2)

# Scale calibration and rigid placement are conceptual callouts, not fabricated measurements.
txt(1450,133,'Scene metric scale  s',19,c=muted)
line([(1449,160),(1719,160)],muted,1)
line([(1449,152),(1449,168)],muted,1);line([(1719,152),(1719,168)],muted,1)
txt(1000,946,'Fixed body pose and shape',20,c=ink,b=True)
txt(1430,946,'One whole-body translation',20,c=teal,b=True)
im.save(OUT/'layout_existing_assets.png')
print(OUT/'layout_existing_assets.png')
