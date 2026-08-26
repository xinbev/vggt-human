# Stage2 Walking Coarse-Scale HSI Cascade Viewer

## Pipeline

The dedicated viewer runs:

```text
RGB -> VGGT camera/depth/features
RGB + VGGT K -> NLF detector -> metric SMPL
NLF SMPL z / VGGT depth z -> analytic coarse scale
coarse depth -> current box-free HSI scale model -> residual scale/bias
coarse depth + HSI residual -> Stage2 human-scene align
post-HSI base-SMPL tracking overlay -> Viser
```

Checkpoint composition:

```text
main checkpoint:
  outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt

HSI refinement overlay:
  outputs/train/smpl_hsi_coarse_residual_stratified_v3/
  checkpoint_top_train_epoch_0005_loss_total_0.009242.pt
```

The overlay replaces `hsi_refinement_head.*` only. The Stage2
`hsi_human_scene_align_head` remains loaded from the main checkpoint.

The viewer explicitly uses `hsi_scene_affine_mode=per_frame`, matching the
coarse-residual training configuration. This avoids the old Stage2
`clip_median` policy collapsing every residual prediction to one sequence-wide
constant.

Frames where coarse estimation fails do not fall back to scale `1`. The
walking wrapper fills them with the log-space median of valid coarse scales in
the selected sequence (`COARSE_FALLBACK=sequence_median`) and records the
fallback in the terminal and `run_summary.json`.

The first pass estimates the coarse scale. The second pass supplies coarse
depth through `hsi_depth_override`, allowing HSI to predict a residual and the
Stage2 align head to consume the same corrected depth. Before visualization,
the scene affine is composed as:

```text
effective scale = coarse scale * residual HSI scale
effective bias = residual HSI bias
```

The terminal prints every frame as:

```text
[coarse-hsi] frame=0000 coarse=... hsi_residual=... effective=... bias=... anchors=... applied=... reason=...
```

and prints sequence min/median/max values under `[coarse-hsi-summary]`.
`effective` is exactly `coarse * hsi_residual`. The displayed depth is:

```text
raw VGGT depth * effective + bias
```

## Server Command

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
bash scripts/vis/serve_stage2_walking_coarse_scale_hsi_cascade.sh
```

Run `SMOKE_ONLY=true` first when desired. The summary is written to:

```text
outputs/vis/stage2_walking_coarse_residual_v3/run_summary.json
```

It records per-frame coarse, residual, and effective scales plus the robust
coarse-estimator diagnostics. The general viewer and previous Stage2 walking
script remain unchanged unless `SCENE_SCALE_PREALIGN=smpl_median` is selected.
