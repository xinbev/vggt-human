# HSI Stage2 Human-Scene Translation Align

## Goal

Stage1 has already learned a reliable metric scene scale for VGGT depth. This
stage keeps VGGT, NLF, and Stage1 scene affine frozen, then trains a small
translation-only alignment head that pulls NLF SMPL roots toward the scaled
human point cloud.

The target problem is not global scale anymore. It is the small residual 3D
misalignment between:

- scaled VGGT human/foreground point cloud
- NLF SMPL mesh in camera coordinates

## Module

New module:

```text
vggt_omega/models/heads/hsi_human_scene_align_head.py
```

Class:

```text
HSIHumanSceneAlignHead
```

It runs after:

```text
HSIRefinementHead -> apply_hsi_scene_affine_mode
```

so it sees the final selected `hsi_scene_scale / hsi_scene_depth_bias`.

## Inputs

- `depth`
- `hsi_scene_scale / hsi_scene_depth_bias`
- `pose_enc` or `hsi_intrinsics_override`
- current SMPL base:
  - `hsi_refined_pred_poses` if present, otherwise `pred_poses`
  - `hsi_refined_pred_betas` if present, otherwise `pred_betas`
  - `hsi_refined_pred_transl_cam` if present, otherwise `pred_transl_cam`
  - `pred_confs`

## Output Contract

The head always writes diagnostics:

```text
hsi_align_base_pred_transl_cam
hsi_align_refined_pred_transl_cam
hsi_align_delta_transl_cam
hsi_align_delta_coeff
hsi_align_gate
hsi_align_valid_ratio
hsi_align_base_point_l1
hsi_align_refined_point_l1
hsi_align_point_l1_delta
loss_hsi_align_point
loss_hsi_align_delta_reg
loss_hsi_align_no_worse
```

With `model.hsi_align_overwrite_refined=true`, it also overwrites:

```text
hsi_refined_pred_transl_cam
```

so existing HSI losses and visualizers continue to use the aligned translation.

## Geometry

The module samples SMPL joints plus deterministic farthest-point SMPL vertices.
Those points are projected to the processed image plane, sampled against the
metric depth map:

```text
metric_depth = depth * hsi_scene_scale + hsi_scene_depth_bias
```

For each SMPL point, a local depth window is searched and the nearest 3D point
is used as its scene correspondence. The head pools residual statistics:

```text
scene_point - smpl_point
```

and predicts translation in a camera basis:

```text
delta_trans =
  ray        * delta_ray
+ tangent_x  * delta_x
+ tangent_y  * delta_y
```

This allows full 3D correction while keeping tangent motion bounded.

## Training Config

Config:

```text
configs/train_smpl_hsi_nlf_stage2_human_scene_align.yaml
```

Server script:

```text
scripts/train/train_smpl_hsi_nlf_stage2_human_scene_align.sh
```

Default checkpoint initialization:

```text
outputs/train/stage1_scale_linear_b20_gpu7/checkpoint_latest.pt
```

Default train policy:

- freeze VGGT aggregator/camera/depth
- freeze NLF
- freeze HSI scene affine and old HSI backbone
- train only `hsi_human_scene_align_head`
- save only:
  - `hsi_refinement_head.`
  - `hsi_human_scene_align_head.`

## Gate Criteria

Run a short gate first:

```bash
MAX_STEPS_PER_EPOCH=200 \
CUDA_VISIBLE_DEVICES_VALUE=7 \
BATCH_SIZE=16 \
bash scripts/train/train_smpl_hsi_nlf_stage2_human_scene_align.sh
```

Healthy signs:

```text
alignRef < alignBase
alignDT < 0
hsiDT <= 0
alignValid is not near 0
alignDelta stays small, normally centimeters to low decimeters
```

If `alignRef` does not improve, inspect the projected SMPL/depth overlay before
running a long job.
