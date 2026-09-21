from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO

OUT=Path(__file__).resolve().parent
SOURCE=Path(r'C:/Users/ROG/AppData/Local/Temp/codex-clipboard-c9d38c76-cf01-4a6b-8b90-e99ab23afd45.png')
src=Image.open(SOURCE).convert('RGB')
crops={'scene':(382,21,747,264),'anchor20':(810,330,1079,599),'hands54':(442,652,685,895)}
assets={}
for key,box in crops.items():
    assets[key]=src.crop(box)
    assets[key].save(OUT/(key+'.png'))

S=2;W,H=1900,770
canvas=Image.new('RGB',(W*S,H*S),'white');d=ImageDraw.Draw(canvas)
navy='#193C49';muted='#6B7D83';coral='#D9796C';teal='#23989D';amber='#DCA234'
def f(n,bold=False):return ImageFont.truetype('C:/Windows/Fonts/arial'+('bd' if bold else '')+'.ttf',n*S)
def text(x,y,t,n=22,color=navy,bold=False):d.text((x*S,y*S),t,font=f(n,bold),fill=color)
def line(points,color,width=1):d.line([(int(x*S),int(y*S)) for x,y in points],fill=color,width=width*S)
def ellipse(box,color,width=2):d.ellipse(tuple(int(v*S) for v in box),outline=color,width=width*S)
def paste(key,box):
    x,y,w,h=box
    canvas.paste(assets[key].resize((w*S,h*S),Image.Resampling.LANCZOS),(x*S,y*S))
text(48,35,'GroundedHuman',34,bold=True)
text(48,82,'Human-anchored metric alignment and regional placement refinement',23,muted)

# One dominant scene plus two smaller evidence magnifications; no module boxes.
hero=(48,156,890,593)
paste('scene',hero)
# Sample locations are approximate visual leaders on the contact-sheet crop.
hand=(48+(559-382)/365*890,156+(155-21)/243*593)
ellipse((hand[0]-26,hand[1]-22,hand[0]+26,hand[1]+22),coral,3)
line([(hand[0]+25,hand[1]-12),(977,274),(1008,300)],coral,2)
line([(hand[0]+26,hand[1]+13),(955,714),(1396,714),(1396,581),(1460,581)],teal,2)

text(1008,167,'Anchor-scene attention',26,bold=True)
text(1008,205,'Local scene context for scale calibration',18,muted)
paste('anchor20',(1008,252,320,320))
text(1008,590,'Body anchor + 3 x 3 scene tokens',19,coral)
text(1008,622,'Line strength: attention weight',17,muted)

text(1426,167,'Multi-scale regional evidence',26,bold=True)
text(1426,205,'Self-surface and surrounding environment',18,muted)
paste('hands54',(1426,252,320,320))
# Exact nesting present in the supplied crop, labels identify existing supports.
for label,col,start_y,ty in [
    ('3 x 3','#2885DB',390,465),('7 x 7','#3F9D66',373,418),
    ('Adaptive',amber,307,338),('Outer annulus','#9664B8',279,270)]:
    line([(1687,start_y),(1759,ty+8),(1775,ty+8)],col,1)
    text(1781,ty,label,15,col)
text(1426,590,'One region, four sampling supports',19,teal)
text(1426,622,'Regional proposals refine whole-body position',17,muted)
for x,color,label in [(1429,teal,'Self-surface'),(1591,amber,'Environment')]:
    d.ellipse((x*S,662*S,(x+9)*S,671*S),fill=color)
    text(x+17,654,label,17,muted)
# Red/teal leader lines are explanatory crop locators, not algorithmic forces.
canvas.save(OUT/'teaser_composition.png')

# Self-contained SVG keeps asset placement, leaders, and main text editable.
def b64(key):
    bio=BytesIO();assets[key].save(bio,format='PNG')
    return base64.b64encode(bio.getvalue()).decode()
def svgimg(key,x,y,w,h):return f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="data:image/png;base64,{b64(key)}"/>'
def svgtxt(x,y,value,size=22,color=navy,weight=400):return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}">{value}</text>'
pieces=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="1900" height="770" fill="white"/><g font-family="Arial, sans-serif">',
        svgtxt(48,67,'GroundedHuman',34,navy,700),
        svgtxt(48,105,'Human-anchored metric alignment and regional placement refinement',23,muted),
        svgimg('scene',*hero),
        f'<ellipse cx="{hand[0]}" cy="{hand[1]}" rx="26" ry="22" fill="none" stroke="{coral}" stroke-width="3"/>',
        f'<polyline points="{hand[0]+25},{hand[1]-12} 977,274 1008,300" fill="none" stroke="{coral}" stroke-width="2"/>',
        f'<polyline points="{hand[0]+26},{hand[1]+13} 955,714 1396,714 1396,581 1460,581" fill="none" stroke="{teal}" stroke-width="2"/>',
        svgtxt(1008,192,'Anchor-scene attention',26,navy,700),svgtxt(1008,223,'Local scene context for scale calibration',18,muted),
        svgimg('anchor20',1008,252,320,320),svgtxt(1008,608,'Body anchor + 3 x 3 scene tokens',19,coral),
        svgtxt(1008,640,'Line strength: attention weight',17,muted),
        svgtxt(1426,192,'Multi-scale regional evidence',26,navy,700),svgtxt(1426,223,'Self-surface and surrounding environment',18,muted),
        svgimg('hands54',1426,252,320,320),svgtxt(1426,608,'One region, four sampling supports',19,teal),
        svgtxt(1426,640,'Regional proposals refine whole-body position',17,muted),
        svgtxt(1426,678,'Self-surface',17,teal),svgtxt(1590,678,'Environment',17,amber)]
for label,col,start_y,ty in [('3 x 3','#2885DB',390,465),('7 x 7','#3F9D66',373,418),('Adaptive',amber,307,338),('Outer annulus','#9664B8',279,270)]:
    pieces.append(f'<polyline points="1687,{start_y} 1759,{ty+8} 1775,{ty+8}" fill="none" stroke="{col}"/>')
    pieces.append(svgtxt(1781,ty+15,label,15,col))
pieces.append('</g></svg>')
(OUT/'teaser_composition.svg').write_text('\n'.join(pieces),encoding='utf-8')
(OUT/'README.md').write_text('''# 基于用户上传素材总览的 teaser 构图

这是截图裁切排版，不是重新生成的人体、采样或实验结果。使用主场景、anchor20 和 hands r54。
右侧放大图属于同一左侧 teaser 主视觉；未设计论文右侧定量图。
连接线只是示意放大图来源，基于截图目视定位，不是精确投影或 translation vector。
保留截图中的真实采样图样；标题只是方法用途说明，不表示这个样本的校准/接触改善已获验证。
当前没有单独的位移/权重文件，因此未绘制位移箭头或人为放置 base mesh 残影。
最终使用完整输出文件替换内嵌截图，保持 SVG 坐标和版式，即可得到高清版本。
''',encoding='utf-8')
print(OUT/'teaser_composition.png')
