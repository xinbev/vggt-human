# Bonn video-depth benchmark

This benchmark evaluates the five Bonn sequences used by UniSH Table 1:
`balloon2`, `crowd2`, `crowd3`, `person_tracking2`, and `synchronous`. It uses
frames 30–139 (110 frames) per sequence, converts depth PNG values by dividing
by 5000, ignores invalid GT and GT depth at or above 70 m, and reports Abs Rel
and `delta < 1.25`.

Prediction layout:

```text
<pred-root>/<sequence>/*.npy
```

Nested layouts are also accepted, for example
`<pred-root>/rgbd_bonn_balloon2/depth/*.npy`. The evaluator only consumes
floating-point prediction files; Bonn RGB/depth files alone are not model
predictions and cannot produce the metrics.

Run one prediction stage:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
DATASET_ROOT=/home/zhw/xyb_space/rgbd_bonn_dataset \
PRED_ROOT=/home/zhw/xyb_space/vggt_bonn_predictions/pure_vggt \
bash benchmarks/bonn_depth/run.sh
```

Compare pure VGGT with VGGT + traditional coarse scale + HSI scale:

```bash
DATASET_ROOT=/home/zhw/xyb_space/rgbd_bonn_dataset \
PURE_VGGT_ROOT=/home/zhw/xyb_space/vggt_bonn_predictions/pure_vggt \
TRADITIONAL_HSI_ROOT=/home/zhw/xyb_space/vggt_bonn_predictions/vggt_traditional_hsi_scale \
bash benchmarks/bonn_depth/run_stages.sh
```

To generate the two prediction trees first, use `infer_stages.sh`:

```bash
DATASET_ROOT=/home/zhw/xyb_space/rgbd_bonn_dataset \
PRED_ROOT=/home/zhw/xyb_space/vggt_bonn_predictions \
CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/vggt_omega_1b_512.pt \
STAGE2_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt \
SCALE_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt \
bash benchmarks/bonn_depth/infer_stages.sh
```

Or run inference and comparison in one step with `run_full.sh`:

```bash
DATASET_ROOT=/home/zhw/xyb_space/rgbd_bonn_dataset \
PRED_ROOT=/home/zhw/xyb_space/vggt_bonn_predictions \
STAGE2_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt \
SCALE_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt \
bash benchmarks/bonn_depth/run_full.sh
```

Use `ALIGNMENT=metric` for a supplemental absolute-scale diagnostic. Results
are written below `outputs/eval/` by default.

## Human3R Figure 9b-style absolute-depth curve data

The long-sequence curve is a different protocol from Table 1. Following the
released Human3R/TTT3R preparation, requested lengths are
`N=50,100,...,500`, every window begins at original frame 30, and each sequence
uses `min(N, available_after_frame_30)` frames. Thus `N` is a requested maximum,
not the actual length of every sequence. In the five-sequence Bonn subset, the
short sequences are truncated rather than repeated. The evaluator records both
the requested and actual lengths.

Each requested prefix is run independently as one complete model input and no
test-time GT scale or shift is applied. Do not set a positive chunk size for
this experiment: chunking changes model context and does not measure cumulative
input length.

Generate only the proposed `vggt_traditional_hsi_scale` curve:

```bash
DATASET_ROOT=/home/zhw/xyb_space/rgbd_bonn_dataset \
PRED_ROOT=/home/zhw/xyb_space/vggt_bonn_curve_predictions \
STAGE=vggt_traditional_hsi_scale \
CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/vggt_omega_1b_512.pt \
STAGE2_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt \
SCALE_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt \
DEVICE=cuda \
bash benchmarks/bonn_depth/infer_curve.sh
```

Then calculate the five points and sequence-level 95% confidence intervals:

```bash
DATASET_ROOT=/home/zhw/xyb_space/rgbd_bonn_dataset \
PRED_ROOT=/home/zhw/xyb_space/vggt_bonn_curve_predictions \
STAGE=vggt_traditional_hsi_scale \
bash benchmarks/bonn_depth/run_curve.sh
```

The plot-ready data are written to
`outputs/eval/bonn_depth_curve/vggt_traditional_hsi_scale_curve_points.csv`.
