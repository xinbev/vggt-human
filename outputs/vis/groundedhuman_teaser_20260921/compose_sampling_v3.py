from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
import math

ROOT = Path(__file__).resolve().parent
S = 2
im = Image.open(ROOT / 'teaser_sampling_v2.png').convert('RGB').resize((3072, 2048))
d = ImageDraw.Draw(im)
navy = '#183D4C'
cyan = '#08AFBC'
amber = '#E39817'
blue = '#288AD5'
green = '#419565'
magenta = '#AD549B'
def font(n, bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/arial' + ('bd' if bold else '') + '.ttf', int(n*S))
def rect(box, **kw):
    d.rectangle(tuple(round(v*S) for v in box), **kw)
def line(points, fill, width=1):
    d.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=round(width*S))
def dot(x,y,r,fill,outline=None):
    d.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=fill,outline=outline,width=S)
def text(x,y,t,size=17,fill=navy,bold=False):
    d.text((x*S,y*S),t,font=font(size,bold),fill=fill)

# Replace the generated right detail with a clean NBAI render and exact image-space supports.
rect((1011,173,1535,739),fill='white')
photo = ImageOps.fit(Image.open(ROOT/'foot_sampling_background.png').convert('RGB'),(370*S,430*S))
im.paste(photo,(1020*S,215*S))
d = ImageDraw.Draw(im)
cx, cy, step = 1202, 461, 21
radii = [1,3,4,6]
edges = [(r+.5)*step for r in radii]

# Square annulus: r_adaptive < Chebyshev distance <= r_adaptive + 2.
shade=Image.new('RGBA',im.size,(0,0,0,0)); sd=ImageDraw.Draw(shade)
outer,inner=edges[-1],edges[-2]
sd.rectangle(((cx-outer)*S,(cy-outer)*S,(cx+outer)*S,(cy+outer)*S),fill=(173,84,155,27))
sd.rectangle(((cx-inner)*S,(cy-inner)*S,(cx+inner)*S,(cy+inner)*S),fill=(0,0,0,0))
im=Image.alpha_composite(im.convert('RGBA'),shade).convert('RGB'); d=ImageDraw.Draw(im)
for k in range(-6,8):
    off=(k-.5)*step
    line([(cx-outer,cy+off),(cx+outer,cy+off)],'#C7CCCE',.5)
    line([(cx+off,cy-outer),(cx+off,cy+outer)],'#C7CCCE',.5)

for iy in range(-6,7):
    for ix in range(-6,7):
        x,y=cx+ix*step,cy+iy*step
        rgb=photo.getpixel((round((x-1020)*S),round((y-215)*S)))
        # This is an illustrative surface mask on synthetic imagery, not measured model ownership.
        is_self=rgb[1]-rgb[0]>12 and rgb[2]-rgb[0]>10
        dot(x,y,2.6,cyan if is_self else amber, 'white')

for radius,color in zip(radii,[blue,green,amber,magenta]):
    e=(radius+.5)*step
    rect((cx-e,cy-e,cx+e,cy+e),outline=color,width=4)
# Both the inner and outer annulus boundaries are explicit; the interior is unshaded.
e=edges[2]
rect((cx-e-2,cy-e-2,cx+e+2,cy+e+2),outline=magenta,width=2)

# A small projected body region represented by eight points.
region=[(1182,403),(1200,405),(1220,421),(1243,450),(1255,489),(1235,513),(1207,506),(1181,467)]
line(region+[region[0]],'white',1)
for x,y in region: dot(x,y,2.1,'white',navy)
dot(cx,cy,3.7,navy,'white')

labels=[('Outer annulus',magenta,outer,300),('Adaptive',amber,edges[2],352),('7 x 7',green,edges[1],409),('3 x 3',blue,edges[0],466)]
for label,col,e,ty in labels:
    sx,sy=cx+e,cy-e
    line([(sx,sy),(1382,ty+8),(1394,ty+8)],col,1.2)
    text(1400,ty,label,16,col)
text(1064,651,'Image-space sampling',17)
dot(1064,696,5,cyan);text(1078,685,'Self-surface',17)
dot(1230,696,5,amber);text(1244,685,'Environment',17)

# Render crisp, consistent method labels and outcome statement.
rect((0,100,450,151),fill='white')
text(25,112,'Anchor-scene cross attention',22,bold=True)
rect((1000,100,1535,151),fill='white')
text(1030,112,'Regional translation prediction',22,bold=True)
rect((475,775,1110,834),fill='white')
text(527,796,'Regional evidence, one whole-body translation',20)

# Regional arrows denote proposals for a shared rigid translation, not limb deformation.
for x,y,dx,dy,col in [(543,412,28,-3,'#8760A9'),(694,527,25,-7,'#9E82B5'),(729,702,30,-4,'#8760A9')]:
    line([(x,y),(x+dx,y+dy)],col,2)
    ang=math.atan2(dy,dx)
    tip=(x+dx,y+dy)
    left=(tip[0]-7*math.cos(ang-.45),tip[1]-7*math.sin(ang-.45))
    right=(tip[0]-7*math.cos(ang+.45),tip[1]-7*math.sin(ang+.45))
    d.polygon([(round(a*S),round(b*S)) for a,b in [tip,left,right]],fill=col)
text(815,738,'Regional votes',17,'#8760A9')
line([(810,748),(776,735),(756,704)],'#AD98BC',1)

im.crop((0,95*S,1536*S,846*S)).save(ROOT/'teaser_sampling_v3.png')
# A larger stand-alone sampling detail helps inspect the sampling design.
im.crop((1014*S,206*S,1533*S,716*S)).save(ROOT/'sampling_detail_v3.png')
print('Saved teaser_sampling_v3.png and sampling_detail_v3.png')
