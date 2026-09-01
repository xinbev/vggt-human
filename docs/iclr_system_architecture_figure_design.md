# ICLR System Architecture Figure Design

## Scope

This figure summarizes the current inference-oriented research pipeline:

```text
RGB sequence
  -> VGGT scene branch (camera intrinsics, depth, scene tokens)
  -> NLF human branch conditioned on VGGT intrinsics (metric SMPL)
  -> analytic coarse scene scale + learned HSI residual scale/bias
  -> TRSTR regional translation-only spatial refinement
  -> optional track-wise post-alignment temporal stabilization
```

The figure is based on the current model forward order and the accepted HSI
scale / TRSTR design records. Historical HSI translation, contact, grounding,
and V4 branches are intentionally excluded.

## Technical Contracts Shown

- VGGT produces camera intrinsics `K`, raw depth, and scene tokens from RGB.
- NLF consumes RGB and VGGT-derived `K`, and returns pose, shape, translation,
  confidence, and boxes for each detected person.
- HSI scale first obtains an absolute metric gauge from projected SMPL/depth
  anchor ratios, then predicts residual scale and bias:

  ```text
  D_metric = (C_coarse * R_hsi) D_vggt + b_hsi
  ```

- TRSTR keeps pose, global orientation, and shape read-only. It decodes the
  SMPL surface, pools 96 body regions, probes metric depth with multi-scale
  patches, predicts gated uncertainty-weighted regional translation votes,
  and re-probes for two spatial iterations.
- The final temporal block is post-alignment and track-wise. It must not mix
  identities. The figure presents it as an optional conservative refinement,
  not as part of TRSTR's currently disabled temporal gate.

## Visual Hierarchy

1. A compact RGB frame strip on the left.
2. Parallel VGGT scene and NLF human branches in the upper-middle area.
3. A central HSI metric-scale fusion block where both branches meet.
4. A larger TRSTR block with a body-region inset and multi-scale depth probes.
5. A rightmost temporal block and final stable human-in-scene output.
6. A palette strip across the bottom with exact hexadecimal labels.

## Color System

Primary user palette:

```text
#F8E8A4  warm yellow
#D8D0F0  soft lavender
#98B898  sage green
#C8A0A0  dusty rose
#A8ACB8  blue gray
#D0D0C7  warm gray
```

Supporting neutrals may use off-white and dark charcoal while preserving the
muted academic tone.

