# Bonn video-depth evaluation (UniSH Table 1)

This project-native evaluator follows the protocol used by UniSH through its
Pi3 predecessor: five Bonn sequences (`balloon2`, `crowd2`, `crowd3`,
`person_tracking2`, `synchronous`), sorted depth frames 30–139 (110 frames per
sequence), depth PNG values divided by 5000, valid ground truth in `(0, 70 m)`,
one positive scale fitted per sequence, and valid-pixel-weighted averages of
Abs Rel and `delta < 1.25`.

Predictions must be arranged as:

```text
<pred-root>/balloon2/*.npy
<pred-root>/crowd2/*.npy
<pred-root>/crowd3/*.npy
<pred-root>/person_tracking2/*.npy
<pred-root>/synchronous/*.npy
```

Each directory may contain exactly 110 files, or predictions for the complete
original sequence. Files are sorted and, for a complete sequence, the evaluator
selects the same indices 30–139 as the ground truth. Each prediction is an
`H x W` or `H x W x 1` floating-point depth map; it is resized to the GT
resolution when necessary.

On the Linux server, run:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
DATASET_ROOT=/home/zhw/xyb_space/rgbd_bonn_dataset \
PRED_ROOT=/path/to/bonn_predictions \
bash benchmarks/bonn_depth/run.sh
```

Results are written to `outputs/eval/bonn_depth/bonn_metrics.json` and
`outputs/eval/bonn_depth/bonn_metrics.csv`.

The default `ALIGNMENT=scale` is the Table 1 protocol. To inspect whether the
HSI branch really improves absolute metric scale (without letting evaluation
fit it away), also run `ALIGNMENT=metric`; this supplemental result uses no
post-hoc scale correction.

## Two-stage comparison

For the requested ablation, save two prediction trees using the same 110 RGB
frames:

```text
outputs/eval/bonn_predictions/pure_vggt/<sequence>/*.npy
outputs/eval/bonn_predictions/vggt_traditional_hsi_scale/<sequence>/*.npy
```

`pure_vggt` is the raw `predictions["depth"]`. The second stage is the depth
after the analytic SMPL/depth coarse scale and the HSI residual affine
correction (effective scale and residual bias). Do not apply an additional
per-frame GT calibration during inference; the evaluator applies only the
single scale alignment required by the Bonn video-depth protocol to both
stages.

Compare both stages with:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
DATASET_ROOT=/home/zhw/xyb_space/rgbd_bonn_dataset \
PURE_VGGT_ROOT=/home/zhw/xyb_space/vggt_bonn_predictions/pure_vggt \
TRADITIONAL_HSI_ROOT=/home/zhw/xyb_space/vggt_bonn_predictions/vggt_traditional_hsi_scale \
bash benchmarks/bonn_depth/run_stages.sh
```

The comparison is saved under `outputs/eval/bonn_depth_stages/`.
For the metric-scale diagnostic, use the same command with
`ALIGNMENT=metric` and a separate `OUTPUT_DIR`.
