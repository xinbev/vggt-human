# SMPL Translation Ray Refine Full Design

## Goal

Repair the base SMPL `pred_transl_cam` before HSI/video refinement.

This is a foundation-stage fix. HSI should refine scene scale, contact, and
local temporal consistency after the base human root translation is already
reasonable. It should not be asked to rescue a severely misplaced root.

## Depth Boundary

Two different Z values must not be mixed:

1. Raw VGGT/depth-map Z:
   - Not used as base translation supervision.
   - Not sampled as a hard root-depth target.
   - Can be changed later by HSI scene affine, so it is not a stable teacher for
     the base SMPL head.
2. Dataset SMPL camera translation Z:
   - Used as part of `gt_transl_cam`.
   - This is the camera-space root/global SMPL annotation from the dataset.
   - It is valid supervision for the SMPL head on RGB+camera+SMPL datasets.

So the training target is camera-space SMPL translation, not scene depth.

## Reference Boundary

- Reference project: `.reference/SAT-HMR-smpl/`
- Referenced modules/ideas:
  - `heads/smpl_regression_head.py`: iterative SMPL pose/shape/camera heads from
    decoder/query hidden states.
  - `geometry/camera_projection.py`: camera-aware body projection and depth
    parameterization idea.
- Implementation type: project-local concept rewrite.
- Baseline: preserved. The new branch is disabled unless
  `model.smpl_enable_translation_refine=true`.
- No `.reference` module is imported by project code.

The design keeps the current VGGT aggregator/camera path intact and only adds a
config-gated translation branch under the existing SMPL head.

## Model Design

`CameraRayTranslationRefiner` lives in:

```text
vggt_omega/models/heads/smpl_head.py
```

Inputs:

```text
SMPL hidden token        [B, S, Q, C]
base pred_transl_cam     [B, S, Q, 3]
pred_pose_6d             [B, S, Q, 144]
pred_betas               [B, S, Q, 10]
reference/pred boxes     [B, S, Q, 4] normalized cxcywh
VGGT camera intrinsics K [B, S, 3, 3]
```

For each query, bbox center and camera intrinsics define a ray:

```text
ray = normalize(K^-1 [u, v, 1])
```

The base translation is decomposed into:

```text
ray_depth
tangent_x
tangent_y
```

The network predicts bounded residuals:

```text
delta_ray_m
delta_tangent_x_m
delta_tangent_y_m
delta_log_depth
delta_box_log_depth
box_prior_weight
```

It also computes a camera/focal/bbox-height prior:

```text
z_box_prior = fy * human_height_prior_m / bbox_height_px
```

This prior is a feature and a learnable residual anchor. It is not a depth-map
teacher.

The final layer is zero-initialized, so enabling the branch starts as a strict
identity map:

```text
pred_transl_cam_refined == base_pred_transl_cam
```

The log-depth residual is implemented relative to the signed base ray-depth
anchor, not by clamping the final prediction to positive Z. This preserves the
old checkpoint output exactly at initialization while still using positive
depth values for logarithmic features.

Outputs include both the original and refined translations:

```text
base_pred_transl_cam
pred_transl_cam
pred_transl_cam_delta
pred_transl_ray_dir
pred_transl_tangent_x
pred_transl_tangent_y
base_pred_transl_ray_depth
base_pred_transl_tangent
pred_transl_ray_depth
pred_transl_tangent
pred_transl_box_depth_prior
pred_transl_box_prior_weight
```

## Loss Design

Primary supervision:

```text
loss_transl_cam
loss_joints3d
```

Auxiliary supervision:

```text
loss_projected_joints2d
loss_projected_bbox
loss_projected_giou
loss_transl_refine_ray_depth
loss_transl_refine_tangent
loss_transl_refine_delta_reg
```

The ray-depth and tangent losses use `gt_transl_cam` projected into the same
camera-ray basis. They do not use raw depth maps.

Important diagnostics:

```text
metric_base_transl_l1
metric_refined_transl_l1
metric_transl_refine_l1_delta
metric_base_transl_ray_depth_l1
metric_refined_transl_ray_depth_l1
metric_transl_ray_depth_l1_delta
metric_base_transl_tangent_l1
metric_refined_transl_tangent_l1
metric_transl_tangent_l1_delta
metric_transl_box_prior_weight_abs
```

Negative deltas mean the refined branch improves over the original base output.

## Training Stages

Main full script:

```bash
bash scripts/train/train_smpl_translation_ray_refine_full_from0121.sh
```

Default stages:

```text
T0: evaluate original 0121 base translation
T1: freeze VGGT/camera/original SMPL heads, train only CameraRayTranslationRefiner
T2: still freeze VGGT/camera/pose/betas, train transl_cam_heads + refiner
T3: merge translation repair keys back into the full 0121 HSI checkpoint
```

Default output:

```text
outputs/train/smpl_translation_ray_refine_full_from0121/
  stage1_ray_refiner/checkpoint_latest.pt
  stage2_transl_heads_ray_refiner/checkpoint_latest.pt
  merged_hsi_translation/checkpoint_latest.pt

outputs/eval/smpl_translation_ray_refine_full_from0121/
  t0_init_0121/smpl_translation_metrics.json
  t0_init_0121/smpl_translation_person_metrics.csv
  t1_ray_refiner/smpl_translation_metrics.json
  t1_ray_refiner/smpl_translation_person_metrics.csv
  t2_final_translation/smpl_translation_metrics.json
  t2_final_translation/smpl_translation_person_metrics.csv
```

The JSON contains global means plus worst-case excerpts. The CSV contains one
row per matched person with sequence/frame/query/track identifiers and base vs
refined translation, ray-depth, tangent, MPJPE, and PVE errors. Use it to find
remaining bad frames before spending time on PLY inspection.

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES_VALUE=0 STAGE1_EXTRA_EPOCHS=8 STAGE2_EXTRA_EPOCHS=6 \
bash scripts/train/train_smpl_translation_ray_refine_full_from0121.sh
```

Skip T2 if T1 is already enough:

```bash
RUN_STAGE2=0 bash scripts/train/train_smpl_translation_ray_refine_full_from0121.sh
```

Run only merge from an existing translation checkpoint:

```bash
RUN_T0_EVAL=0 RUN_STAGE1=0 RUN_STAGE2=0 RUN_STAGE_EVAL=0 RUN_MERGE=1 \
STAGE1_OUTPUT_DIR=/path/to/existing/stage1_or_final_dir \
bash scripts/train/train_smpl_translation_ray_refine_full_from0121.sh
```

The older single-stage entry remains available:

```bash
bash scripts/train/train_smpl_translation_ray_refine_from0121.sh
```

## HSI Reconnect

The base translation repair is trained with HSI disabled. Therefore its
checkpoint does not contain HSI head weights.

To reconnect HSI, use the merged checkpoint produced by T3:

```text
outputs/train/smpl_translation_ray_refine_full_from0121/merged_hsi_translation/checkpoint_latest.pt
```

The merge utility is:

```bash
python scripts/diagnostics/merge_translation_refiner_into_hsi.py \
  --hsi-checkpoint outputs/train/smpl_hsi_refine_20q/checkpoint_epoch_0121.pt \
  --translation-checkpoint outputs/train/smpl_translation_ray_refine_full_from0121/stage2_transl_heads_ray_refiner/checkpoint_latest.pt \
  --output outputs/train/smpl_translation_ray_refine_full_from0121/merged_hsi_translation/checkpoint_latest.pt \
  --include-translation-heads
```

Optional HSI low-LR reconnect:

```bash
bash scripts/train/train_smpl_hsi_after_translation_ray_refine.sh
```

Evaluate the merged/T4 HSI result with translation refiner enabled:

```bash
bash scripts/eval/eval_smpl_hsi_after_translation_ray_refine.sh
```

## Server Smoke Checks

After git sync on the training server, run:

```bash
bash -n scripts/train/train_smpl_translation_ray_refine_full_from0121.sh \
  scripts/train/train_smpl_translation_ray_refine_from0121.sh \
  scripts/train/train_smpl_hsi_after_translation_ray_refine.sh \
  scripts/eval/eval_smpl_translation_ray_refine_from0121.sh \
  scripts/eval/eval_smpl_hsi_after_translation_ray_refine.sh
```

Then run a short translation smoke if desired:

```bash
STAGE1_EXTRA_EPOCHS=1 STAGE2_EXTRA_EPOCHS=0 EVAL_MAX_SAMPLES=8 \
bash scripts/train/train_smpl_translation_ray_refine_full_from0121.sh
```

## Acceptance Criteria

Base translation repair passes if:

```text
refined_transl_l2_m < base_transl_l2_m
refined_ray_depth_l1_m < base_ray_depth_l1_m
refined_tangent_l2_m < base_tangent_l2_m
refined_joints_mpjpe_m <= base_joints_mpjpe_m
refined_vertices_pve_m <= base_vertices_pve_m
```

HSI reconnect passes if:

```text
HSI MPJPE/PVE/translation improve or at least do not regress from the merged base
hsi_scene_scale does not saturate near exp(3) = 20.0855
depth metrics remain stable under clip_median scene affine
```

## Main Risks

1. Bbox-height prior can be biased by truncation or unusual body scale. It is
   bounded and learnable, but metrics must track `box_prior_weight_abs`.
2. If training and evaluation use different refiner max-delta hyperparameters,
   checkpoint behavior changes. The full scripts pin:

```text
max_ray_delta_m=1.20
max_tangent_delta_m=0.60
max_log_depth_delta=0.85
max_box_prior_weight=1.00
```

3. T2 changes the original `transl_cam_heads`; merge should include
   `--include-translation-heads`.
4. RGB+camera+SMPL datasets can train this design without depth. BEDLAM depth is
   only needed later when HSI scene affine is enabled.
