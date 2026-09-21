"""CPU geometry and drawing helpers for measured GroundedHuman teaser assets.

No model/runtime imports: numeric and image export can be checked without ckpts.
Pixel coordinates refer to the processed image/depth plane, never the source PNG.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TEAL = (34, 163, 170)
AMBER = (234, 160, 38)
CORAL = (225, 106, 99)
PURPLE = (150, 100, 184)
INK = (40, 60, 72)
SUPPORT_COLORS = [(40, 133, 219), (63, 157, 102), AMBER, PURPLE]


def json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def project(xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz)
    with np.errstate(invalid="ignore", divide="ignore"):
        return xyz[..., :2] / xyz[..., 2:3] * np.array([k[0, 0], k[1, 1]]) + k[:2, 2]


def unproject(xy: np.ndarray, z: np.ndarray, k: np.ndarray, image_hw, depth_hw) -> np.ndarray:
    uv = xy * np.array([image_hw[1] / depth_hw[1], image_hw[0] / depth_hw[0]])
    x = (uv[..., 0] - k[0, 2]) * z / max(float(k[0, 0]), 1e-5)
    y = (uv[..., 1] - k[1, 2]) * z / max(float(k[1, 1]), 1e-5)
    return np.stack([x, y, z], axis=-1)


def regional_samples(center, reps, depth, human_depth, owner, k, image_hw, query,
                     fixed_sizes=(3, 7), radius_max=8, annulus_width=2, tolerance=.18):
    """Replay RegionalSceneProbe's integer sampling/ownership, not a new sampler.

    Keep raw and clamped pixels separately: padding samples must remain invalid.
    Support denominators include out-of-frame cells, matching model valid_ratios.
    """
    h, w = depth.shape
    scale = np.array([w / image_hw[1], h / image_hw[0]])
    center_uv = project(np.asarray(center), k) * scale
    reps_uv = project(np.asarray(reps), k) * scale
    if not np.isfinite(center_uv).all() or not np.isfinite(reps_uv).all():
        raise ValueError("Cannot export a region with non-finite projections")
    radius = int(np.clip(np.ceil(np.abs(reps_uv - center_uv).max() + 1), 1, radius_max))
    maximum = max(max(s // 2 for s in fixed_sizes), radius_max + annulus_width)
    oy, ox = np.mgrid[-maximum:maximum+1, -maximum:maximum+1]
    offsets = np.stack([ox.ravel(), oy.ravel()], axis=-1)
    raw_xy = np.rint(center_uv).astype(np.int64) + offsets
    image_valid = (raw_xy[:, 0] >= 0) & (raw_xy[:, 0] < w) & (raw_xy[:, 1] >= 0) & (raw_xy[:, 1] < h)
    xy = np.clip(raw_xy, [0, 0], [w-1, h-1])
    z = depth[xy[:, 1], xy[:, 0]]
    valid = image_valid & np.isfinite(z) & (z > 1e-5)
    hz = human_depth[xy[:, 1], xy[:, 0]]
    sampled_owner = owner[xy[:, 1], xy[:, 0]]
    any_human = np.isfinite(hz) & (np.abs(z - hz) <= tolerance)
    self_mask = valid & any_human & (sampled_owner == query)
    other_mask = valid & any_human & (sampled_owner != query)
    env_mask = valid & ~any_human
    chebyshev = np.abs(offsets).max(axis=-1)
    supports = [chebyshev <= s // 2 for s in fixed_sizes]
    supports += [chebyshev <= radius, (chebyshev > radius) & (chebyshev <= radius + annulus_width)]
    supports = np.stack(supports)
    channels = np.stack([self_mask, env_mask])
    masks = supports[:, None, :] & channels[None, :, :]
    ratios = masks.sum(-1) / np.maximum(supports.sum(-1)[:, None], 1)
    return dict(center_uv=center_uv, representatives_uv=reps_uv, adaptive_radius=np.array(radius),
                raw_xy=raw_xy, sample_xy=xy, offsets=offsets, valid=valid, image_valid=image_valid,
                self_surface=self_mask, environment=env_mask, other_human=other_mask,
                supports=supports, channel_masks=masks, valid_ratios=ratios.reshape(-1),
                scene_xyz=unproject(xy, z, k, image_hw, depth.shape), depth=z)


def attention_pool_weights(logits, masks):
    # Same masked softmax/renormalization as _masked_attention_pool, including empty support.
    masked = np.where(masks, np.asarray(logits)[None, None, :], -1e4)
    weights = np.exp(masked - masked.max(-1, keepdims=True)) * masks
    return weights / np.maximum(weights.sum(-1, keepdims=True), 1e-6)


def local_geometry_samples(anchor, depth, k, image_hw, window):
    """HSI's depth-pixel search candidates, distinct from its feature-token grid."""
    h,w=depth.shape
    center_uv=project(anchor,k)*np.array([w/image_hw[1],h/image_hw[0]])
    radius=window//2
    oy,ox=np.mgrid[-radius:radius+1,-radius:radius+1]
    offsets=np.stack([ox.ravel(),oy.ravel()],-1)
    raw_xy=np.rint(center_uv).astype(np.int64)+offsets
    valid=(raw_xy[:,0]>=0)&(raw_xy[:,0]<w)&(raw_xy[:,1]>=0)&(raw_xy[:,1]<h)
    xy=np.clip(raw_xy,[0,0],[w-1,h-1]);z=depth[xy[:,1],xy[:,0]]
    valid &= np.isfinite(z)&(z>1e-6)
    xyz=unproject(xy,z,k,image_hw,depth.shape)
    distance=np.linalg.norm(xyz-anchor,axis=-1)
    distance=np.where(valid,distance,np.inf)
    nearest=int(np.argmin(distance)) if valid.any() else -1
    return dict(center_uv=center_uv,raw_xy=raw_xy,sample_xy=xy,offsets=offsets,valid=valid,
                scene_xyz=xyz,distance_m=distance,nearest_index=np.array(nearest))


def token_neighborhood(anchors, k, image_hw, depth_hw, window=3):
    """Exact HSI token-grid gather indices; repeated border tokens are retained."""
    h, w = depth_hw
    gh, gw = h // 16, w // 16  # Mirrors current HSI _gather_local_scene_tokens.
    if min(gh, gw) < 1:
        raise ValueError("Depth plane too small for the HSI token grid")
    uv = project(anchors, k)
    center = np.floor(uv / np.array([image_hw[1], image_hw[0]]) * [gw, gh]).astype(np.int64)
    center = np.clip(center, [0, 0], [gw-1, gh-1])
    radius=window//2
    oy, ox = np.mgrid[-radius:radius+1, -radius:radius+1]
    offsets = np.stack([ox.ravel(), oy.ravel()], -1)
    cells = np.clip(center[:, None] + offsets, [0, 0], [gw-1, gh-1])
    pixel_centers = (cells + .5) * np.array([image_hw[1]/gw, image_hw[0]/gh])
    return cells, pixel_centers, (gh, gw)


def write_ply(path, vertices, colors, faces=None):
    vertices = np.asarray(vertices).reshape(-1, 3)
    faces = np.zeros((0, 3), np.int64) if faces is None else np.asarray(faces).reshape(-1, 3)
    colors = np.broadcast_to(np.asarray(colors, np.uint8), vertices.shape)
    if not np.isfinite(vertices).all():
        raise ValueError(f"Non-finite PLY coordinates: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for v, c in zip(vertices, colors):
            f.write(f"{v[0]:.7g} {v[1]:.7g} {v[2]:.7g} {c[0]} {c[1]} {c[2]}\n")
        for a, b, c in faces:
            f.write(f"3 {a} {b} {c}\n")


def font(size):
    for name in ("DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def mesh_layer(vertices, faces, k, image_wh, color=TEAL, scale=2):
    """CPU z-buffer with perspective-correct depth, flat normal shading, RGBA.

    Camera-space x right, y down, z forward. No OpenGL dependency or axis flips.
    """
    width, height = (int(image_wh[0]*scale), int(image_wh[1]*scale))
    uv = project(vertices, k) * scale
    pixels = np.zeros((height, width, 4), np.uint8)
    zbuf = np.full((height, width), np.inf, np.float32)
    for face in np.asarray(faces):
        v = vertices[face]
        p = uv[face]
        if not np.isfinite(p).all() or not np.isfinite(v).all() or (v[:, 2] <= 1e-5).any():
            continue
        lo = np.maximum(np.floor(p.min(0)).astype(int), [0, 0])
        hi = np.minimum(np.ceil(p.max(0)).astype(int), [width-1, height-1])
        if (hi < lo).any():
            continue
        x0, y0 = lo; x1, y1 = hi
        yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
        x, y = xx+.5, yy+.5
        den = (p[1,1]-p[2,1])*(p[0,0]-p[2,0]) + (p[2,0]-p[1,0])*(p[0,1]-p[2,1])
        if abs(den) < 1e-8:
            continue
        a = ((p[1,1]-p[2,1])*(x-p[2,0]) + (p[2,0]-p[1,0])*(y-p[2,1]))/den
        b = ((p[2,1]-p[0,1])*(x-p[2,0]) + (p[0,0]-p[2,0])*(y-p[2,1]))/den
        c = 1-a-b
        inv = a/v[0,2]+b/v[1,2]+c/v[2,2]
        depth = 1/np.maximum(inv, 1e-12)
        keep = (a >= -1e-5) & (b >= -1e-5) & (c >= -1e-5) & (depth < zbuf[y0:y1+1, x0:x1+1])
        normal = np.cross(v[1]-v[0], v[2]-v[0])
        normal /= max(np.linalg.norm(normal), 1e-8)
        shade = .48 + .52*abs(float(normal @ np.array([.25, -.45, -.857])))
        rgba = np.r_[np.clip(np.array(color)*shade, 0, 255).astype(np.uint8), 255]
        pixels[y0:y1+1, x0:x1+1][keep] = rgba
        zbuf[y0:y1+1, x0:x1+1][keep] = depth[keep]
    return Image.fromarray(pixels), zbuf


def draw_arrow(draw, start, end, color, width=3):
    start, end = np.asarray(start), np.asarray(end)
    delta = end-start
    length = np.linalg.norm(delta)
    if not np.isfinite(length) or length < .5:
        return
    u = delta/length; n = np.array([-u[1], u[0]])
    head = min(12, length*.4)
    draw.line([tuple(start), tuple(end)], fill=color, width=width)
    draw.polygon([tuple(end), tuple(end-u*head+n*head*.45), tuple(end-u*head-n*head*.45)], fill=color)


def draw_samples_crop(rgb, samples, depth_hw, path, title, support_index=None, size=720):
    """Export clean background, transparent overlay and annotated sampling view."""
    depth_rgb = rgb.resize((depth_hw[1], depth_hw[0]), Image.Resampling.LANCZOS)
    center = np.rint(samples['center_uv']).astype(int)
    active = samples['supports'].any(0)
    radius = int(np.abs(samples['offsets'][active]).max()) + 2
    # Half-pixel bounds map the sampled integer pixel center to the center of its displayed cell.
    bounds = (center[0]-radius-.5, center[1]-radius-.5,
              center[0]+radius+.5, center[1]+radius+.5)
    bg = depth_rgb.transform((size, size), Image.Transform.EXTENT, bounds,
                             resample=Image.Resampling.BILINEAR, fillcolor=(245,245,245))
    overlay = Image.new("RGBA", bg.size)
    d = ImageDraw.Draw(overlay)
    pitch = size/(2*radius+1)
    def xy(point):
        return tuple((np.asarray(point)-np.array(bounds[:2]))*pitch)
    chosen = active if support_index is None else samples['supports'][support_index]
    for index in np.flatnonzero(chosen):
        x, y = xy(samples['raw_xy'][index]); r = max(3, pitch*.14)
        if samples['self_surface'][index]: color = TEAL
        elif samples['environment'][index]: color = AMBER
        elif samples['other_human'][index]: color = PURPLE
        else: color = (140,140,140)
        if not samples['valid'][index]:
            d.line((x-r,y-r,x+r,y+r), fill=(*color,180), width=2)
            d.line((x-r,y+r,x+r,y-r), fill=(*color,180), width=2)
        else:
            d.ellipse((x-r,y-r,x+r,y+r),fill=(*color,255),outline='white',width=1)
    for si, support in enumerate(samples['supports']):
        if support_index is not None and si != support_index:
            continue
        off = samples['offsets'][support]
        edge = np.abs(off).max()+.5
        d.rectangle([xy(center-edge),xy(center+edge)],outline=(*SUPPORT_COLORS[si],255),width=3)
        if si == len(samples['supports'])-1:
            edge = int(samples['adaptive_radius'])+.5
            d.rectangle([xy(center-edge),xy(center+edge)],outline=(*SUPPORT_COLORS[si],255),width=2)
    for rep in samples['representatives_uv']:
        x,y=xy(rep);d.ellipse((x-3,y-3,x+3,y+3),fill='white',outline=INK,width=1)
    x,y=xy(samples['center_uv']);d.line((x-7,y,x+7,y),fill=CORAL,width=2);d.line((x,y-7,x,y+7),fill=CORAL,width=2)
    composite=Image.alpha_composite(bg.convert('RGBA'),overlay)
    labeled=Image.new('RGB',(size,size+78),'white');labeled.paste(composite,(0,40))
    ld=ImageDraw.Draw(labeled);ld.text((12,8),title,font=font(20),fill=INK)
    ld.text((12,size+47),'teal: self  |  amber: environment  |  purple: other  |  x: invalid',font=font(15),fill=INK)
    path.parent.mkdir(parents=True,exist_ok=True)
    labeled.save(path.with_suffix('.png'))
    bg.save(path.with_name(path.name+'_background').with_suffix('.png'))
    overlay.save(path.with_name(path.name+'_overlay').with_suffix('.png'))
    json_write(path.with_suffix('.json'),{'crop_depth_xyxy':list(bounds),'pixel_scale':pitch,
              'title':title,'adaptive_radius':int(samples['adaptive_radius'])})


def contact_sheet(items, path, columns=3):
    tile_w, tile_h = 480, 390
    sheet = Image.new('RGB',(columns*tile_w,((len(items)+columns-1)//columns)*tile_h),'white')
    draw = ImageDraw.Draw(sheet)
    for idx,(label,image) in enumerate(items):
        image=image.convert('RGBA');image.thumbnail((tile_w-20,tile_h-50))
        x=(idx%columns)*tile_w;y=(idx//columns)*tile_h
        sheet.paste(image,(x+(tile_w-image.width)//2,y+40),image)
        draw.text((x+12,y+8),label,font=font(18),fill=INK)
    sheet.save(path)
