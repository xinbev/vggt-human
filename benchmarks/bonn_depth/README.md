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

Use `ALIGNMENT=metric` for a supplemental absolute-scale diagnostic. Results
are written below `outputs/eval/` by default.
