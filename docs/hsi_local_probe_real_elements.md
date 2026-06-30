# Real HSI Local Probe Paper Elements

This document records the real-data version of the HSI local scene probing and
body-scene residual visualization elements.

## Goal

Generate reusable figure assets for architecture-diagram steps 5 and 6 from a
real image and real model inference, rather than deterministic schematic data.

Default server image:

`/home/zhw/lab_users/xyb/home/projects/vggt-human/assets/image/f2/f2.jpg`

## What Is Real

The script runs the project model with:

- VGGT camera output: `pose_enc`
- VGGT dense output: `depth`
- SMPLHead base SMPL output: `pred_pose_6d`, `pred_betas`, `pred_transl_cam`
- HSI output: `hsi_anchor_depth_residual`, `hsi_scene_scale`,
  `hsi_scene_depth_bias`, `hsi_refine_gate` when present
- HSI internal geometry recomputed with the same helper logic as
  `vggt_omega.models.heads.hsi_refinement_head.HSIRefinementHead`

For arbitrary images, the wrapper enables project YOLO+SAM2 query priors by
default. The SAM2 masks are converted to query patch masks so pooling is based
on person masks instead of a whole bounding-box block.

## Outputs

Server output directory by default:

`/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/paper_hsi_local_probe_real_elements`

For each selected person/query, the script writes:

- `05a_real_depth_patch_window_person*_q*_a*.png`
  Real VGGT depth heatmap with the HSI 3x3 local patch-token window highlighted.
- `05b_real_anchor_project_to_depth_person*_q*_a*.png`
  Real RGB anchor projection into the real depth map.
- `05c_real_local_scene_probe_person*_q*_a*.png`
  Combined RGB/depth local probe element.
- `06a_real_body_scene_residual_person*_q*_a*.png`
  Real body-anchor to scene-point residual in camera side view.
- `real_probe_values_person*_q*_a*.json`
  Numeric intermediate values: query id, anchor id, projected uv, scene xyz,
  anchor xyz, offset, distance, normal, and depth residual.
- `manifest.json`
  Full provenance and file list.

The PNGs are visual elements; the numeric JSON is the source of truth for the
actual HSI quantities used to draw them.

## Server Command

From the Linux project root:

```bash
bash scripts/vis/create_hsi_local_probe_real_elements.sh
```

Useful overrides:

```bash
IMAGE=/path/to/frame.jpg \
TOP_K=2 \
AUTO_TOP_K=2 \
DETECTOR_IMAGE_SIZE=640 \
PERSON_INDEX=-1 \
ANCHOR_INDEX=-1 \
bash scripts/vis/create_hsi_local_probe_real_elements.sh
```

`PERSON_INDEX=-1` exports all selected people up to `TOP_K`.

`ANCHOR_INDEX=-1` automatically picks the visible HSI anchor with the largest
absolute depth residual for that person. Set `ANCHOR_INDEX=...` to force a
specific HSI token/anchor.

## Local Verification

Windows local verification is limited to static checks because the local machine
does not have the server checkpoints and full CUDA/SAM2 runtime. The script was
checked with:

```bash
python -m py_compile scripts/vis/create_hsi_local_probe_real_elements.py
```
