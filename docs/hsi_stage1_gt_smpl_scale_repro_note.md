# HSI Stage1 GT-SMPL Scale Teacher Reproduction Note

This note records the successful Stage1 setup for training HSI to correct the
environment/depth scale. It is intended as a handoff: a future agent should be
able to reproduce the stage from this document alone.

## Goal

Train only the HSI scene affine branch so that VGGT raw depth is converted to a
metric scene scale using visible GT SMPL as the scale teacher.

The effective inference chain remains:

```text
processed RGB sequence -> VGGT camera/depth -> NLF SMPL base -> HSI scene scale/bias
```

Stage1 does not optimize human pose, betas, or translation residuals. It only
learns:

```text
hsi_scene_scale
hsi_scene_depth_bias
```

The intended output depth is:

```text
hsi_depth = raw_vggt_depth * hsi_scene_scale + hsi_scene_depth_bias
```

## Key Geometry Contract

- VGGT forward and NLF inference use the processed/padded image plane.
- NLF receives VGGT-predicted intrinsics decoded from `pose_enc` with runtime
  `images.shape[-2:]`.
- Stage1 teacher uses BEDLAM GT SMPL and BEDLAM dataset camera `K_scal3r`,
  because GT SMPL belongs to the dataset camera coordinate system.
- Teacher only uses visible SMPL vertices and ignores far/deceptive depth.
- Stage1 freezes VGGT, NLF, and all non-scene-affine HSI branches.

For visualization/export after scale correction, use the UniSH-style rule:

```text
if depth/point cloud is scaled by s, camera translation must also be scaled by s
```

In the viewer this is implemented as:

```text
t_hsi = s * t_raw
```

Rotations and intrinsics are not scaled.

## Data Preparation

BEDLAM processed root on server:

```text
/home/zhw/xyb_space/bedlam/processed_bedlam
```

Required sidecar boxes root:

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/preprocess/bedlam_boxes
```

Generate sidecars from SMPL projection plus depth visibility:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

BEDLAM_ROOT=/home/zhw/xyb_space/bedlam/processed_bedlam \
OUTPUT_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/preprocess/bedlam_boxes \
SPLITS=Training \
MAX_HUMANS=20 \
USE_SMPL_PROJECTION=true \
USE_DEPTH_VISIBILITY=true \
PROJECTION_SOURCE=vertices \
DEPTH_VISIBILITY_MODE=abs \
DEPTH_VISIBILITY_TOLERANCE_M=0.20 \
VISIBLE_ONLY=true \
MIN_VISIBLE_PROJECTED_POINTS=20 \
MIN_VISIBLE_PROJECTED_RATIO=0.001 \
MIN_BOX_AREA=100 \
bash scripts/preprocess/prepare_bedlam_boxes.sh
```

The sidecars should only keep visible persons. Fully occluded GT people should
not become training slots.

## Training Script

Main script:

```text
scripts/train/train_smpl_hsi_nlf_stage1_gt_smpl_scale.sh
```

Convenience launcher used for the successful run:

```text
scripts/train/train_smpl_hsi_nlf_stage1_gt_smpl_scale_gpu7_fast.sh
```

Important defaults in the successful launcher:

```text
CUDA_VISIBLE_DEVICES_VALUE=7
BATCH_SIZE=20
NUM_WORKERS=16
NLF_INTERNAL_BATCH_SIZE=192
NUM_VIEWS=2
EPOCHS=3
LR=5e-6
DEPTH_MAX_M=20.0
HSI_SMPL_SCALE_TEACHER_MAX_Z_M=20.0
HSI_SMPL_SCALE_TEACHER_LOG_LOSS=false
```

Stage1 train-only settings:

```text
TRAIN_HSI_SCENE_AFFINE_ONLY=true
FREEZE_HSI_SCENE_AFFINE=false
HSI_SCENE_AFFINE_MODE=per_frame

HSI_POSE_WEIGHT=0
HSI_BETAS_WEIGHT=0
HSI_TRANSL_WEIGHT=0
HSI_JOINTS3D_WEIGHT=0
HSI_VERTICES_WEIGHT=0
HSI_PROJECTED_J2D_WEIGHT=0
ANCHOR_DEPTH_WEIGHT=0
ANCHOR_SCENE_XYZ_WEIGHT=0
DEPTH_TEACHER_WEIGHT=0
```

Checkpoint settings:

```text
SAVE_SCOPE=hsi
SAVE_TOP_K=3
SAVE_OPTIMIZER=false
SAVE_EPOCH_CHECKPOINT=false
```

This keeps only HSI weights. VGGT baseline weights are not saved in the Stage1
checkpoint.

## Successful Run Command

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

DATA_ROOT=/home/zhw/xyb_space \
BEDLAM_ROOT=/home/zhw/xyb_space/bedlam/processed_bedlam \
PREPROCESSED_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/preprocess/bedlam_boxes \
bash scripts/train/train_smpl_hsi_nlf_stage1_gt_smpl_scale_gpu7_fast.sh
```

Successful output directory:

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/stage1_scale_linear_b20_gpu7
```

Best/latest checkpoint:

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/stage1_scale_linear_b20_gpu7/checkpoint_latest.pt
```

Top checkpoint from the completed successful run:

```text
checkpoint_top_train_epoch_0003_loss_total_0.171740.pt
```

## Expected Metrics

Successful final epoch summary:

```text
epoch=3
total=0.171740
smplScale=0.171740
smplPts=898.56
scaleT=17.2155
scaleP=17.2168
scaleL1=0.425412
scaleRel=0.0248062
scaleLog=0.0247663
```

Representative final-step metrics:

```text
total=0.04896
smplScale=0.04896
smplPts=941.8
scaleT=15.77
scaleP=15.87
scaleL1=0.2469
scaleRel=0.01549
scaleLog=0.01548
```

Metric meanings:

- `smplPts`: visible GT SMPL vertices used by the teacher. Should usually be
  comfortably above `128`.
- `scaleT`: robust teacher scale estimated from visible GT SMPL and VGGT raw depth.
- `scaleP`: HSI predicted scale.
- `scaleL1`: absolute scale error, `|scaleP - scaleT|`.
- `scaleRel`: relative scale error.
- `scaleLog`: log-scale error.

Good acceptance range for this stage:

```text
scaleRel < 0.03 is good
scaleL1 around 0.3-0.5 is good for scale around 15-20
smplPts > 128 consistently
```

## Visualization And Validation

Full walking sequence viewer:

```text
scripts/vis/serve_stage1_scale_walking_vggt_nlf_viewer_full.sh
```

Smoke:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

DATA_ROOT=/home/zhw/xyb_space \
SMOKE_ONLY=true \
bash scripts/vis/serve_stage1_scale_walking_vggt_nlf_viewer_full.sh
```

Interactive Viser:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

DATA_ROOT=/home/zhw/xyb_space \
PORT=8080 \
bash scripts/vis/serve_stage1_scale_walking_vggt_nlf_viewer_full.sh
```

Default walking frames:

```text
/home/zhw/lab_users/xyb/home/projects/Human3R-master/outputs/walking/color
```

Default checkpoint:

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/stage1_scale_linear_b20_gpu7/checkpoint_latest.pt
```

Viewer output:

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/stage1_scale_walking_vggt_nlf_viewer_full/run_summary.json
```

Important viewer controls:

- `Depth Source`: `hsi_depth`, `raw_depth`, or `both`.
- `Camera Source`: `auto`, `hsi_scaled`, `raw_vggt`, or `both`.
- `Point Size`: point cloud display size.
- `Camera Size`: camera frustum display size.
- `Show Camera Trajectory`: displays camera centers over time.
- `SMPL Downsample` and `Camera Downsample`: reduce clutter on long sequences.

For HSI scaled visualization, the viewer should use:

```text
Depth Source = hsi_depth
Camera Source = auto or hsi_scaled
```

The summary contains:

```text
camera_motion.raw_vggt
camera_motion.hsi_scaled
depth_alignment_overall
depth_alignment_by_frame
```

Use `camera_motion.hsi_scaled` when judging the camera trajectory after depth
scale correction.

## Code Touchpoints

Training:

```text
scripts/train/train_smpl_hsi_nlf_stage1_gt_smpl_scale.sh
scripts/train/train_smpl_hsi_nlf_stage1_gt_smpl_scale_gpu7_fast.sh
scripts/train/train_smpl_hsi_nlf_provider.sh
scripts/train/train_smpl.py
vggt_omega/training/hungarian_losses.py
vggt_omega/models/heads/hsi_refinement_head.py
```

NLF provider:

```text
vggt_omega/integrations/nlf_smpl_provider.py
```

Visualization:

```text
scripts/vis/serve_nlf_hsi_vggt_sequence_viewer.py
scripts/vis/serve_stage1_scale_walking_vggt_nlf_viewer.sh
scripts/vis/serve_stage1_scale_walking_vggt_nlf_viewer_full.sh
```

## What Is Frozen

Stage1 should not train:

```text
VGGT aggregator/backbone
VGGT camera head
VGGT dense/depth head
NLF
HSI pose/betas/translation residual heads
ID/tracking branches
```

Stage1 trains HSI scene affine prediction only. The optimizer uses only
parameters with `requires_grad=True`, and checkpoints are saved with
`checkpoint.save_scope=hsi`.

## Successful Interpretation

The successful Stage1 behavior is:

```text
VGGT raw depth is scale-ambiguous.
Visible GT SMPL provides a robust metric scale teacher.
HSI learns scene scale around 15-20 on the current BEDLAM setup.
Predicted scale closely tracks teacher scale.
Human translation is not changed in Stage1.
The next stage can train human translation residuals against the scaled scene.
```

For any future viewer/export using the scaled scene, scale camera translation
together with depth/points, following UniSH:

```text
raw point/depth scale:      P_raw, C_raw
HSI scaled point/depth:     P_hsi = s * P_raw + bias_on_depth
HSI scaled camera center:   C_hsi = s * C_raw
```

This keeps the scaled scene, camera trajectory, and SMPL meshes in one coherent
3D coordinate system.
