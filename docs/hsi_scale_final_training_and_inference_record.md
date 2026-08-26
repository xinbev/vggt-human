# HSI Scale Final Training And Inference Record

## Status

This document freezes the currently accepted HSI environment-scale solution as
of 2026-08-26.

Accepted result:

```text
traditional SMPL/depth coarse scale
-> learned HSI residual scale/bias
-> Stage2 human-scene translation alignment
```

The final coarse-residual model completed five epochs and passed interactive
Viser inspection on the Human3R walking sequence with no extra viewer scale
calibration required. The walking check is a real inference visual acceptance
test, not a numerical GT benchmark.

## Problem History

The original scene-affine model was trained near metric GT depth. It learned
synthetic local scale corrections well, but real VGGT depth on the walking
sequence required a much larger absolute correction. Direct HSI inference
therefore under-corrected the environment, even though training loss was low.

The key observations were:

```text
BEDLAM test sequence analytic coarse scale: about 7.25
old direct HSI model scale: about 2.43
remaining multiplier after direct HSI: about 2.92
manual Viser log10 multiplier 0.45: 10^0.45 about 2.82
```

This showed that the visual observation and analytic SMPL/depth ratio agreed.
The issue was the train/inference scale distribution, not a broken GT
projection path.

An analytic-only test then showed:

```text
raw VGGT anchor depth error median:       2.255 m
direct HSI anchor depth error median:     1.754 m
analytic coarse anchor error median:      0.109 m
analytic coarse required residual scale:  1.0004
```

Traditional coarse estimation was therefore retained as the absolute metric
gauge. HSI was redefined as a residual corrector after coarse alignment.

## Final Training Pipeline

The training pipeline is:

```text
clean GT depth
-> strong non-Gaussian absolute disturbance S_extra
-> traditional coarse estimator from GT SMPL projection and disturbed depth
-> controlled coarse-estimation error E
-> coarse-corrected depth enters HSI
-> HSI predicts residual scale and bias
-> final depth is supervised by clean GT depth
```

Equations:

```text
D_disturbed = D_gt / S_extra
C_used = C_algorithm * E
D_coarse = D_disturbed * C_used
R_teacher = S_extra / C_used
D_final = D_coarse * R_pred + B_pred
```

Roles:

```text
S_extra:   simulates the large absolute VGGT scale ambiguity
C_algorithm: traditional robust SMPL/depth coarse estimate
E:         simulates imperfect coarse estimation
R_pred:    learned HSI residual scale
B_pred:    learned HSI residual depth bias
```

Only these parameters are trained:

```text
hsi_refinement_head.scale_delta.*
hsi_refinement_head.bias_delta.*
```

VGGT, camera/depth heads, HSI backbone, SMPL delta branches, and the Stage2
human-scene align head are not trained in this run.

## Non-Gaussian Disturbance

Absolute scale disturbance uses stratified log-uniform buckets:

```text
10%: exactly 1
20%: log-uniform [0.25, 2]
50%: log-uniform [2, 12]
20%: log-uniform [12, 20]
```

This deliberately allocates most samples to the `2-20` range observed in real
VGGT scale correction. It avoids a Gaussian distribution concentrating most
training samples around one.

Coarse-estimation error uses:

```text
30%: exactly 1
40%: log-uniform [0.67, 1.50]
25%: log-uniform [0.40, 2.50]
5%:  log-uniform [0.25, 4.00]
```

The 30% identity bucket is essential. It teaches HSI to return residual scale
one when traditional coarse alignment is already correct.

## Traditional Coarse Estimator

The estimator:

```text
1. decodes metric SMPL vertices;
2. projects vertices with camera intrinsics;
3. samples depth at projected pixels;
4. keeps the nearest SMPL anchor per pixel;
5. forms z_smpl / z_depth ratios;
6. rejects invalid and out-of-range ratios;
7. uses the robust median as coarse scale.
```

Training defaults:

```text
anchor stride:                8
minimum anchor pixels:        32
coarse scale range:           [0.05, 25]
maximum relative MAD:         0.50
```

Coarse-failed frames are excluded from both the SMPL scale teacher and dense
depth teacher. Fallback scale one is never treated as a training target. If an
entire batch has no valid supervision, the optimizer step is skipped.

## Box-Free Training Contract

This accepted training path does not use historical HMR box sidecars:

```text
data.require_boxes=false
data.box_free_gt_slots=true
model.smpl_query_box_prior=false
model.smpl_use_aggregator_queries=false
matching.mode=gt_slots
```

GT SMPL people are filtered online using GT intrinsics and clean GT depth
visibility. Empty or unsupervised batches are skipped explicitly and reported.

## Initialization And Optimization

Source checkpoint:

```text
outputs/train/smpl_hsi_gt_depth_scale_scene_affine_boxfree/
checkpoint_top_train_epoch_0005_loss_total_0.010738.pt
```

Final run:

```text
outputs/train/smpl_hsi_coarse_residual_stratified_v3
```

Optimization:

```text
epochs:       5
batch size:   20
sequence:     2 frames
learning rate: 5e-6
weight decay: 0.05
grad clip:    1.0
optimizer:    fresh AdamW over scale_delta and bias_delta only
```

The source model weights are continued, but the old optimizer state and epoch
counter are not restored.

## Final Training Result

Top-1 checkpoint:

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human/
outputs/train/smpl_hsi_coarse_residual_stratified_v3/
checkpoint_top_train_epoch_0005_loss_total_0.009242.pt
```

Final epoch metrics:

```text
epoch:                         5
global step:                   22805
loss_total:                    0.00924242
SMPL scale log L1:             0.0308226
SMPL scale relative L1:        0.0307604
pipeline final log L1:         0.0307276
coarse valid rate:             0.999775
coarse estimate mean:          6.22551
coarse used mean:              6.65785
residual teacher mean:         1.05547
residual prediction mean:      1.05412
absolute GT scale mean:        6.18732
final predicted scale mean:    6.19829
mean coarse anchor count:      2676.85
skipped batch rate:            0
dropped no-visible rate:       0
scale gradient norm:           17.0070
bias gradient norm:            0.28230
```

The final absolute prediction differs from the teacher mean by approximately:

```text
6.19829 - 6.18732 = 0.01097
relative mean offset about 0.18%
```

The per-sample residual/absolute log error is about 3.1%, which is the more
meaningful accuracy measure than the mean offset.

## Checkpoint Policy

The accepted run used:

```text
save_epoch_checkpoint=false
save_latest=true
save_final=false
save_top_k=1
save_top_k_from_train=true
topk_create_stable_copies=false
monitor=loss_total
```

Storage is bounded to one overwriting latest checkpoint and one top-1 file.

## W&B Lessons

Only useful scalar metrics should be uploaded. W&B Media charts and Tables are
disabled in the training process because W&B 0.28.2 triggered a segmentation
fault when serializing a custom chart at global step 200.

Stable comparison scalar namespaces:

```text
scale_compare/residual_teacher
scale_compare/residual_pred
scale_compare/absolute_GT
scale_compare/traditional_coarse
scale_compare/final_pred
scale_compare/final_log_l1
```

Overlay these scalar keys in W&B web line panels. Do not construct
`wandb.plot.line_series` or `wandb.Table` inside the trainer for this run.

## Final Inference Pipeline

Accepted real inference is:

```text
RGB -> VGGT camera/depth/features
RGB + VGGT intrinsics -> NLF detector -> metric SMPL
NLF SMPL and VGGT depth -> analytic coarse scale
coarse-corrected VGGT depth -> v3 HSI residual scale/bias
coarse depth + HSI residual -> Stage2 human-scene translation align
display-only ID overlay -> Viser
```

Checkpoint composition:

```text
main Stage2 checkpoint:
  outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt

HSI refinement overlay:
  outputs/train/smpl_hsi_coarse_residual_stratified_v3/
  checkpoint_top_train_epoch_0005_loss_total_0.009242.pt
```

The Stage2 checkpoint supplies `hsi_human_scene_align_head`. The v3 checkpoint
overlays `hsi_refinement_head.*` only.

Final environment depth:

```text
effective scale = coarse scale * HSI residual scale
D_final = D_vggt * effective scale + HSI residual bias
```

The inference wrapper forces:

```text
NLF detector mode, no sidecars
smpl_use_aggregator_queries=false
hsi_scene_affine_mode=per_frame
coarse fallback=sequence log-median
viewer visual multiplier=1.0
```

Coarse-failed inference frames are filled with the log-space median of valid
coarse scales in the selected sequence. They do not fall back to one.

## Final Walking Viser Acceptance

Accepted input sequence:

```text
/home/zhw/lab_users/xyb/home/projects/Human3R-master/outputs/walking/color
```

Interactive Viser inspection passed using the v3 residual checkpoint and the
coarse-residual cascade. No additional manual visual scale correction is part
of the accepted pipeline; keep the log10 Visual Scale Multiplier at `0`, which
means a multiplier of `1.0`.

Launch command:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

FRAMES_DIR=/home/zhw/lab_users/xyb/home/projects/Human3R-master/outputs/walking/color \
STAGE2_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full \
CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt \
SCALE_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt \
OUTPUT_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/stage2_walking_coarse_residual_v3 \
CUDA_VISIBLE_DEVICES_VALUE=7 \
PORT=8080 \
MAX_FRAMES=20 \
SMOKE_ONLY=false \
bash scripts/vis/serve_stage2_walking_coarse_scale_hsi_cascade.sh
```

Summary output:

```text
outputs/vis/stage2_walking_coarse_residual_v3/run_summary.json
```

The terminal and summary expose per-frame:

```text
coarse scale
HSI residual scale
effective scale
residual bias
anchor count
coarse applied/fallback status
```

## Reproduction Commands

One-step training gate:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
bash scripts/smoke/run_hsi_coarse_residual_one_step.sh
```

Training launcher:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

CUDA_VISIBLE_DEVICES_VALUE=7 \
OUTPUT_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_coarse_residual_stratified_v3 \
WANDB_RUN_NAME=smpl_hsi_coarse_residual_stratified_v3 \
bash scripts/train/train_smpl_hsi_coarse_residual_stratified.sh
```

## Accepted Engineering Decisions

Keep these decisions unless a future experiment explicitly replaces them:

```text
1. Use analytic SMPL/depth coarse scale for the absolute metric gauge.
2. Train HSI as a residual corrector after coarse alignment.
3. Use stratified log-uniform scale buckets, not Gaussian perturbation.
4. Include identity residual samples so correct coarse depth maps to scale one.
5. Exclude coarse failures from all scale/depth losses.
6. Use box-free GT slots and online visibility for training.
7. Use NLF detector without sidecars for real inference.
8. Use per-frame HSI residual scale in the accepted viewer.
9. Fill failed inference coarse frames with sequence log-median.
10. Keep W&B logging scalar-only for process stability.
11. Keep checkpoint storage bounded to latest plus top-1.
12. Require final real-sequence Viser inspection in addition to training loss.
```

## Remaining Limits

- The walking acceptance is visual and does not contain independent metric GT.
- Analytic coarse scale depends on NLF translation accuracy and sufficient
  projected human anchors.
- Sequence-median fallback assumes scale is approximately stable over the
  selected clip.
- A future benchmark should evaluate multiple real sequences with metric depth
  or known scene dimensions before declaring dataset-independent accuracy.
