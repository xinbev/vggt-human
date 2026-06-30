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

- `05_06_real_hsi_foot_scene_person*_q*_a*.ply`
  Main paper-figure geometry asset. It contains the HSI-corrected VGGT depth
  surface mesh, the selected base SMPL mesh, a foot-sole SMPL vertex sphere,
  the probed scene point sphere, and an arrow from the human foot point to the
  scene point.
- `environment_hsi_depth_person*_q*_a*.ply`
  Environment only. It is built from HSI-adjusted depth by default and uses RGB
  colors sampled from the resized input image, not depth colormap colors.
- `smpl_only_person*_q*_a*.ply`
  Selected base SMPL mesh only.
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

The PLY is the recommended main asset for screenshots. The PNGs are auxiliary
debug/inset assets; the numeric JSON is the source of truth for the actual HSI
quantities used to draw them.

## Server Command

From the Linux project root:

```bash
bash scripts/vis/create_hsi_local_probe_real_elements.sh
```

Useful overrides:

```bash
IMAGE=/path/to/frame.jpg \
TOP_K=1 \
AUTO_TOP_K=2 \
DETECTOR_IMAGE_SIZE=640 \
PERSON_SELECT=rightmost \
ANCHOR_MODE=foot \
PLY_DEPTH_SOURCE=hsi \
PERSON_INDEX=-1 \
ANCHOR_INDEX=-1 \
bash scripts/vis/create_hsi_local_probe_real_elements.sh
```

`PERSON_SELECT=rightmost` picks the right-side person from the detected people.
This is the default for the provided two-person `f2.jpg` image.

`ANCHOR_MODE=foot` picks among the SMPL/HSI foot-side anchors and chooses the
visible foot anchor closest to the probed depth surface. Set `ANCHOR_INDEX=...`
to force a specific HSI token/anchor.

`PLY_DEPTH_SOURCE=hsi` applies `hsi_scene_scale` and `hsi_scene_depth_bias` to
the VGGT depth before creating the environment surface and probing the scene
point. Set `PLY_DEPTH_SOURCE=raw` only for debugging the uncorrected VGGT depth.

For PLY screenshots, the red human point is not the abstract HSI joint token. It
is selected from SMPL foot-sole surface vertices, so the point lies on the
visible SMPL mesh. The HSI foot anchor is still recorded in JSON for reference.

Coordinate note: the original image is resized to the model input size before
VGGT inference. Camera projection produces coordinates in model-input pixels;
those coordinates are then scaled into the actual dense-depth grid. The
environment PLY colors are sampled from the RGB image after the same resize path
to keep color/depth alignment.

The 2D boxes used for auxiliary PNGs and query priors come from SAM2 mask
bounding boxes, not the model-predicted `pred_boxes`, so they should cover the
person similarly to the earlier patch-pooling素材.

## Local Verification

Windows local verification is limited to static checks because the local machine
does not have the server checkpoints and full CUDA/SAM2 runtime. The script was
checked with:

```bash
python -m py_compile scripts/vis/create_hsi_local_probe_real_elements.py
```
