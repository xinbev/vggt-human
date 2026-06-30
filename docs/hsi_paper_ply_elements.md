# HSI Paper PLY Elements

This note records the geometry-only PLY assets used as screenshot elements for
the HSI module architecture figure.

## Goal

Generate deterministic 3D visual elements for the paper figure without using
model checkpoints or data. The PLY files are intended for screenshots in
MeshLab, Blender, Open3D, or CloudCompare. Labels, formulas, and final arrows
should be added later in SVG/PDF so the paper text remains editable.

## Code

Local script:

```text
scripts/vis/create_hsi_paper_ply_elements.py
```

Server wrapper:

```text
scripts/vis/create_hsi_paper_ply_elements.sh
```

Server path after sync:

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human/scripts/vis/create_hsi_paper_ply_elements.sh
```

Run on server:

```text
bash scripts/vis/create_hsi_paper_ply_elements.sh
```

Default output:

```text
outputs/vis/paper_hsi_ply_elements/
```

## Exported PLY Files

```text
hsi_body_anchors.ply
```

Orange simplified SMPL skeleton, 24 green body-anchored HSI anchors, and token
chips. Use this for the "24 Body Anchors" or "HSI Tokens" inset.

```text
hsi_local_scene_probe.ply
```

Selected body anchor, projection into depth, 3x3 local scene window, scene xyz,
offset/distance arrow, normal arrow, and z-residual marker. This is the most
important PLY for explaining local scene probing.

```text
hsi_transformer_tokens.ply
```

Body-token self-attention and local scene cross-attention as token geometry.
Use as a 3D inset inside or beside the HSI Transformer block.

```text
hsi_scene_affine.ply
```

Multiple human query tokens aggregated into a frame-level scale/bias node,
then connected to calibrated depth. Use for the human-scale scene calibration
inset.

```text
hsi_full_paper_elements.ply
```

Combined overview with base human, scene/depth, token/probe relation,
transformer token geometry, refined human, and calibrated depth.

## Mapping To Real HSI Logic

The visual elements follow the current project implementation in:

```text
vggt_omega/models/heads/hsi_refinement_head.py
```

The PLY assets reflect:

```text
Base SMPL + VGGT depth/camera/tokens
  -> 24 body anchors
  -> projected/local scene probing
  -> HSI tokens
  -> body-token self-attention + local scene cross-attention
  -> human residual heads + scene scale/bias
```

Use the formula in the final vector figure:

```text
D_hsi = s_hsi * D_vggt + b_hsi
```

## Notes

- The files contain no text by design.
- The assets are concept rewrites for visualization, not imported reference
  code and not runtime model logic.
- Optional temporal memory is omitted from the PLY set because it should remain
  secondary in the main paper figure.
