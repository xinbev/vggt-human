# HSI Anchor Projection PLY Elements

This document records the real-data PLY layer set for visualizing the HSI
anchor projection step in the paper architecture figure.

## Goal

Create reusable 3D assets for this HSI logic:

```text
base SMPL parameters
  -> frozen SMPLLayer
  -> 24 body anchors in camera space
  -> project anchors with intrinsics from pose_enc
  -> sample HSI-adjusted VGGT depth at projected locations
  -> unproject depth samples back to camera-space scene points
```

The assets are intended for screenshots in Blender, MeshLab, Open3D, or
CloudCompare. They contain no text labels, so labels and formulas can remain
editable in the final vector figure.

## Code Mapping

The visualization corresponds to the current HSI implementation in:

```text
vggt_omega/models/heads/hsi_refinement_head.py
```

Important logic:

- 24 anchor construction: `_anchors_cam`
- 3D-to-2D projection: `_project_points`
- input-pixel to depth-grid scaling: `_scale_points_to_depth`
- depth unprojection: `_unproject_pixels`
- local scene-token sampling: `_gather_local_scene_tokens`

This script reuses the real inference/probe helpers from:

```text
scripts/vis/create_hsi_local_probe_real_elements.py
```

It does not change model behavior.

## Server Command

Local script path:

```text
scripts/vis/create_hsi_anchor_projection_ply_elements.py
```

Server wrapper:

```text
scripts/vis/create_hsi_anchor_projection_ply_elements.sh
```

Server path after sync:

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human/scripts/vis/create_hsi_anchor_projection_ply_elements.sh
```

Run from the Linux project root:

```bash
bash scripts/vis/create_hsi_anchor_projection_ply_elements.sh
```

Default input image:

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human/assets/image/f2/f2.jpg
```

Default output:

```text
outputs/vis/paper_hsi_anchor_projection_ply_elements/
```

Useful overrides:

```bash
IMAGE=/path/to/frame.jpg \
PERSON_SELECT=all \
TOP_K=2 \
AUTO_TOP_K=2 \
DEPTH_SOURCE=hsi \
SMPL_STAGE=base \
DEPTH_UPSAMPLE=2 \
DEPTH_STRIDE=4 \
bash scripts/vis/create_hsi_anchor_projection_ply_elements.sh
```

Use `PERSON_SELECT=rightmost` if only the right-side person is needed. The
default `PERSON_SELECT=all` with `TOP_K=2` exports two detected people
separately.

Use `SMPL_STAGE=base` to visualize the exact base-SMPL anchor construction used
as HSI input. Use `SMPL_STAGE=refined` when checking whether the HSI-refined
human aligns with the HSI-adjusted depth surface.

## Exported PLY Files

Global layers:

```text
00_camera_frustum.ply
```

Classic CV camera frustum: a small camera body at the camera origin and four
rays forming the image-plane pyramid.

```text
00_depth_surface_hsi_teal.ply
```

Environment depth surface from HSI-adjusted depth by default. The surface uses
a teal/blue-green depth colormap, not RGB texture colors.

Per selected person/query:

```text
01_person*_q*_smpl_only.ply
```

Base SMPL mesh only.

```text
01_person*_q*_24anchors_only.ply
```

The 24 HSI body anchors only. Colors separate ordinary body anchors, hand-center
anchors, and the full-body center anchor.

```text
01_person*_q*_smpl_24anchors.ply
```

Base SMPL mesh plus the 24 anchors. This is the recommended asset for the
"base SMPL -> 24 anchors" panel.

```text
02_person*_q*_projection_links_yellow_points_only.ply
```

Only anchor-to-depth projection links and bright yellow depth correspondence
points. Use this as an overlay layer if you want to compose the camera, person,
and depth surface manually.

```text
02_person*_q*_camera_person_depth_projection.ply
```

Camera frustum, teal depth surface, base SMPL, 24 anchors, projection links, and
yellow projected depth points for one person.

Combined layer:

```text
03_hsi_anchor_projection_collection.ply
```

All selected people plus the shared camera and shared depth surface in the same
camera-space coordinate system.

Metadata:

```text
manifest.json
```

Records input image, checkpoint, depth source, selected queries, valid projected
anchor count, and the anchor schema.

```text
projection_diagnostics.json
```

Reports per-query distances between SMPL anchors and depth samples for
`base_raw`, `base_hsi`, and, when available, `refined_hsi`. This is the first
file to inspect when yellow projected points look detached from the person.

## Coordinate Notes

The original image is resized to the model input size before inference. Anchor
projection produces coordinates in model-input pixels. Those coordinates are
then scaled into the actual dense-depth grid before sampling.

With the usual `image_size=518` and `patch_size=16`, the native dense depth is
often `512x512`: the patch grid is `floor(518 / 16) = 32`, and the dense decoder
returns `32 * 16 = 512`. `DEPTH_UPSAMPLE` only densifies the exported PLY surface
for visualization; it does not add new model-predicted depth detail.

`DEPTH_SOURCE=hsi` applies the HSI scene scale and depth bias:

```text
D_hsi = s_hsi * D_vggt + b_hsi
```

Use `DEPTH_SOURCE=raw` only when comparing against uncorrected VGGT depth.

Important: the actual HSI tokenization step samples raw VGGT depth before the
HSI scene scale/bias exists. Therefore:

- `base_raw` is the faithful internal HSI probe state.
- `base_hsi` mixes base SMPL anchors with HSI-adjusted depth and can visually
  separate yellow points from the base person.
- `refined_hsi` is the meaningful alignment check for HSI-refined SMPL against
  HSI-adjusted depth.

## Local Verification

Windows local verification is limited to static checks because local execution
does not have the full server checkpoints and CUDA/SAM2 runtime. Static check:

```bash
python -m py_compile scripts/vis/create_hsi_anchor_projection_ply_elements.py
```
