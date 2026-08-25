# HSI GT-Depth Scale Log10-Wide Continuation

## Goal

Continue the completed box-free scale model with a wider synthetic scale
distribution. The source checkpoint is:

```text
outputs/train/smpl_hsi_gt_depth_scale_scene_affine_boxfree/
checkpoint_top_train_epoch_0005_loss_total_0.010738.pt
```

The continuation writes to a separate directory and does not overwrite the
source run.

## Perturbation

```text
log10(depth perturb scale) ~ Normal(0, 0.30^2)
target scene scale = 1 / depth perturb scale
```

Coverage:

```text
68%:   0.50-2.00
95%:   0.25-3.98
99.7%: 0.126-7.94
```

A multiplier of `2.82` has `log10(2.82) ~= 0.45`, only 1.5 standard
deviations from the mean in this experiment.

The HSI architecture still predicts natural log-scale and applies `exp`.
Changing the perturbation sampling base does not change checkpoint semantics;
it only widens the supervised scale distribution.

## Diagnostics

The run logs:

```text
metric_hsi_smpl_scale_teacher_log10_std
metric_hsi_smpl_scale_teacher_pred_log10_std
metric_hsi_smpl_scale_teacher_log_correlation
metric_hsi_smpl_scale_teacher_identity_log_l1
metric_hsi_smpl_scale_teacher_identity_improvement
```

Healthy behavior means predicted log10 standard deviation approaches teacher
standard deviation, correlation rises toward `1`, and identity improvement is
positive. These metrics distinguish real per-sample learning from a constant
prediction near one.

## Server Commands

One-step gate:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
bash scripts/smoke/run_hsi_gt_depth_scale_log10_wide_one_step.sh
```

Full continuation:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

CUDA_VISIBLE_DEVICES_VALUE=7 \
bash scripts/train/train_smpl_hsi_gt_depth_scale_scene_affine_log10_wide.sh
```

Output:

```text
outputs/train/smpl_hsi_gt_depth_scale_scene_affine_log10_wide
```

Default learning rate is `5e-6`, epochs are `5`, W&B is enabled, and the
checkpoint policy remains one overwriting latest plus one top-1 checkpoint.
