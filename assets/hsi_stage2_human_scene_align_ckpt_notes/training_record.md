# HSI Stage2 Human-Scene Align Checkpoint Training Record

## Target Checkpoint

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt
```

This checkpoint is the `_full` output of the HSI Stage2 human-scene translation
alignment experiment. It is not a full-model training run from scratch. It
starts from the Stage1 metric scene-scale checkpoint and trains only the small
human-scene translation alignment head.

## Local Evidence

The local repository does not contain the actual `.pt` checkpoint file for this
run. It contains the server-returned training artifacts:

```text
outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/resolved_config.json
outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_topk_index.json
```

These files were introduced by git commit:

```text
a0505d382dd14180d17054c6b54ba09867bbe5c1  trans training returned
```

That commit added the resolved config and top-k checkpoint index for this run,
so the `resolved_config.json` is the most reliable local record of the actual
effective training configuration.

## Training Entry

Training script:

```text
scripts/train/train_smpl_hsi_nlf_stage2_human_scene_align.sh
```

Python trainer called by the script:

```text
scripts/train/train_smpl.py
```

Base config:

```text
configs/train_smpl_hsi_nlf_stage2_human_scene_align.yaml
```

The script default output directory is:

```text
outputs/train/smpl_hsi_nlf_stage2_human_scene_align
```

The target checkpoint came from the same script with `OUTPUT_DIR` overridden to:

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full
```

## Initialization

The run resumed model weights from:

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/stage1_scale_linear_b20_gpu7/checkpoint_latest.pt
```

Checkpoint behavior:

```text
checkpoint.resume_optimizer: false
checkpoint.reset_epoch: true
checkpoint.resume_strict: false
checkpoint.load_vggt_baseline: true
```

Interpretation:

- Load Stage1 model weights.
- Do not load the optimizer state.
- Reset epoch/global_step to 0.
- Continue as a fresh 3-epoch Stage2 training run.

## Trained Component

Main module:

```text
vggt_omega/models/heads/hsi_human_scene_align_head.py
```

Class:

```text
HSIHumanSceneAlignHead
```

The model config sets:

```text
model.enable_hsi_human_scene_align: true
model.train_hsi_human_scene_align_only: true
model.hsi_align_overwrite_refined: true
```

With `train_hsi_human_scene_align_only=true`, the training code freezes:

```text
aggregator
camera_head
dense_head
smpl_head
nlf_smpl_provider
hsi_refinement_head
```

and unfreezes:

```text
hsi_human_scene_align_head
```

So the run trains only the human-scene alignment head while keeping VGGT, NLF,
and the Stage1 HSI scene-affine path frozen.

## What The Head Does

The head keeps SMPL pose and betas fixed and predicts a small camera-space root
translation correction.

High-level flow:

1. Take current SMPL translation from `hsi_refined_pred_transl_cam` when
   available, otherwise from `pred_transl_cam`.
2. Sample SMPL joints plus deterministic farthest-point SMPL vertices.
3. Convert VGGT depth to metric depth using:

```text
metric_depth = depth * hsi_scene_scale + hsi_scene_depth_bias
```

4. Project SMPL points into the image and find local nearest scene-depth
   correspondences.
5. Pool residual statistics:

```text
scene_point - smpl_point
```

6. Predict translation coefficients along camera ray, tangent-x, and tangent-y
   basis vectors.
7. Apply a learned gate.
8. Add the gated delta to the base translation.
9. Because `hsi_align_overwrite_refined=true`, overwrite:

```text
hsi_refined_pred_transl_cam
```

so existing HSI losses and visualizers consume the aligned translation.

## Effective Run Configuration

Important values from:

```text
outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/resolved_config.json
```

Data:

```text
dataset: bedlam
train_split: Training
val_split: ""
sequence_length: 2
stride: 1
image_size: 518
image_resolution: 512
resize_mode: balanced
max_humans: 20
require_boxes: true
require_smpl: true
require_depth: true
num_workers: 12
```

Model:

```text
smpl_provider: nlf
nlf_use_detector: false
nlf_require_boxes: true
nlf_internal_batch_size: 128
num_smpl_queries: 20
freeze_aggregator: true
freeze_camera_head: true
freeze_dense_head: true
freeze_aggregator_forward: true
freeze_hsi_scene_affine: true
freeze_hsi_backbone: true
train_hsi_human_scene_align_only: true
hsi_scene_affine_mode: clip_median
hsi_align_hidden_dim: 256
hsi_align_num_sample_vertices: 96
hsi_align_local_window: 7
hsi_align_max_ray_delta_m: 0.6
hsi_align_max_tangent_delta_m: 0.25
hsi_align_use_delta_gate: true
hsi_align_overwrite_refined: true
```

Optimization:

```text
batch_size: 24
epochs: 3
lr: 1e-5
weight_decay: 0.05
grad_clip_norm: 1.0
log_interval: 20
save_interval: 1
val_interval: 1
```

Loss weights:

```text
hsi_transl_cam_weight: 2.0
hsi_joints3d_weight: 1.0
hsi_vertices_weight: 0.2
hsi_projected_joints2d_weight: 0.01
hsi_align_point_weight: 6.0
hsi_align_delta_reg_weight: 0.05
hsi_align_no_worse_weight: 2.0
hsi_delta_reg_weight: 0.15
hsi_no_worse_weight: 5.0
hsi_no_worse_margin_m: 0.005
```

All direct base SMPL losses were disabled:

```text
pose_weight: 0.0
betas_weight: 0.0
transl_cam_weight: 0.0
joints3d_weight: 0.0
projected_joints2d_weight: 0.0
bbox_weight: 0.0
giou_weight: 0.0
```

## Checkpoint Save Policy

Resolved checkpoint config:

```text
save_scope: hsi
save_prefixes:
  - hsi_refinement_head.
  - hsi_human_scene_align_head.
save_optimizer: false
save_epoch_checkpoint: false
save_latest: true
save_final: false
save_top_k: 3
save_top_k_from_train: true
topk_create_stable_copies: false
monitor: loss_total
monitor_mode: min
```

Because `save_scope=hsi`, the saved `.pt` file contains only model state_dict
entries with these prefixes:

```text
hsi_refinement_head.
hsi_human_scene_align_head.
```

It does not contain optimizer state.

Because `val_split` is empty, top-k checkpoints were selected from train metrics
instead of validation metrics.

## Top-K Result

From:

```text
outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_topk_index.json
```

Top checkpoints:

```text
rank 1: epoch 3, train loss_total 4.120545776768735
        checkpoint_top_train_epoch_0003_loss_total_4.120546.pt

rank 2: epoch 2, train loss_total 4.162178138180783
        checkpoint_top_train_epoch_0002_loss_total_4.162178.pt

rank 3: epoch 1, train loss_total 4.320682050146555
        checkpoint_top_train_epoch_0001_loss_total_4.320682.pt
```

Since `save_latest=true`, the target file:

```text
checkpoint_latest.pt
```

should be the latest checkpoint saved at the end of epoch 3. It should
correspond to the same training epoch as rank-1 top checkpoint, although the
actual `.pt` payload was not present locally to byte-compare.

## Reconstructed Server Command

The exact shell history was not found locally. This command reconstructs the run
from the resolved config and the training script overrides:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

OUTPUT_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full \
STAGE1_CKPT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/stage1_scale_linear_b20_gpu7/checkpoint_latest.pt \
BATCH_SIZE=24 \
EPOCHS=3 \
LR=1e-5 \
HSI_ALIGN_POINT_WEIGHT=6.0 \
HSI_TRANSL_WEIGHT=2.0 \
HSI_JOINTS3D_WEIGHT=1.0 \
HSI_VERTICES_WEIGHT=0.2 \
HSI_ALIGN_MAX_RAY_DELTA_M=0.6 \
HSI_ALIGN_MAX_TANGENT_DELTA_M=0.25 \
bash scripts/train/train_smpl_hsi_nlf_stage2_human_scene_align.sh
```

`CUDA_VISIBLE_DEVICES_VALUE` was not recorded in `resolved_config.json`; use the
server GPU assignment available at run time.

## Known Limitations Of This Record

- The local repository does not include the actual
  `checkpoint_latest.pt` payload.
- Local Windows environment has no full training/inference environment and no
  complete server checkpoints, so tensor-level inspection was not performed.
- The reconstructed command is based on `resolved_config.json`, script defaults,
  and script override behavior. It is more reliable than the base YAML defaults,
  but it is not a captured shell history line.

