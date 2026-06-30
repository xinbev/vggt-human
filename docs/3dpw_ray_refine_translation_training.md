# 3DPW Ray/Refine Translation Training

## Goal

This is the stronger 3DPW SMPL-base translation experiment. It keeps the old
direct-regression baseline intact and adds a separate config for camera-ray
translation decoding, translation refinement, and translation-heavy losses.

## Training Path

Use:

```bash
bash scripts/train/train_smpl_base_3dpw_ray_refine.sh
```

Default config:

```text
configs/train_smpl_base_3dpw_ray_refine.yaml
```

Default output:

```text
outputs/train/smpl_base_3dpw_ray_refine/
```

The old baseline remains:

```text
configs/train_smpl_base_3dpw.yaml
scripts/train/train_smpl_base_3dpw.sh
outputs/train/smpl_base_3dpw/
```

## Camera Choice

The model forward path still uses the VGGT camera head output for ray
construction. This matches inference, where GT camera intrinsics are not
available.

GT intrinsics from the 3DPW batch are used only by projected supervision losses
when:

```yaml
loss:
  projection_camera_source: gt
```

The legacy flag:

```yaml
loss:
  use_vggt_camera_projection: true
```

is still required because existing code uses it as the on/off switch for
projected bbox losses. The actual camera source is controlled by
`projection_camera_source`.

## Strong Translation Losses

The ray/refine config enables:

```yaml
model:
  smpl_translation_output_mode: ray_offset_depth
  smpl_enable_translation_refine: true

loss:
  transl_cam_weight: 6.0
  joints3d_weight: 16.0
  projected_joints2d_weight: 0.10
  transl_refine_ray_depth_weight: 1.50
  transl_refine_tangent_weight: 0.75
  transl_hard_topk_weight: 1.0
  transl_hard_severe_weight: 1.0
```

## Evaluation

Use the same evaluation script, but pass the ray/refine config for checkpoints
trained with this experiment:

```bash
CHECKPOINT=outputs/train/smpl_base_3dpw_ray_refine/checkpoint_top01.pt \
TRAIN_CONFIG=configs/train_smpl_base_3dpw_ray_refine.yaml \
bash scripts/eval/evaluate_3dpw_smpl_base_metrics.sh
```

The evaluator still reports:

```text
pa_mpjpe_mm
mpjpe_mm
pve_mm
```

It now also reports translation-sensitive metrics:

```text
transl_l2_mm
transl_xy_l2_mm
transl_z_abs_mm
cam_mpjpe_no_align_mm
cam_pve_no_align_mm
```

`transl_l2_mm` is the direct root translation error. `cam_mpjpe_no_align_mm`
and `cam_pve_no_align_mm` do not pelvis-align the bodies, so they expose global
translation errors that PA-MPJPE/MPJPE/PVE can hide.
