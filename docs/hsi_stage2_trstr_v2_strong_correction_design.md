# TRSTR V2 Strong-Correction Training Design

## Objective

TRSTR v2 extends the spatial translation correction range to cover real cases
requiring approximately `0.7 m`, with additional X/Y robustness up to `2.0 m`,
while retaining v1's centimeter-level refinement and clean-input no-op
behavior. Pose, global orientation, and betas remain read-only. Temporal
refinement remains a later independent stage.

## Independence

```text
v1 source:
  outputs/train/smpl_hsi_stage2_trstr_v3_scale_spatial

v2 config:
  configs/train_smpl_hsi_stage2_trstr_v2_strong.yaml

v2 launcher:
  scripts/train/train_smpl_hsi_stage2_trstr_v2_strong.sh

v2 output:
  outputs/train/smpl_hsi_stage2_trstr_v2_strong

v2 W&B group:
  hsi_stage2_trstr_v2_strong
```

V2 loads the complete frozen v1 `hsi_trstr_head.*`, resets epoch/global step,
uses a fresh AdamW optimizer, and writes only to the v2 output directory. The
v3 HSI scale head is overlaid and frozen exactly as in v1.

## Coupled GT Construction

V2 preserves the physical dataset GT and creates separate model inputs and an
alignment pseudo-target:

```text
physical_transl = dataset GT transl_cam
clean_depth = dataset GT metric depth
sole_anchor = the left/right sole center best supported by clean depth in 3x3
scaled_depth = clean_depth * sampled_environment_scale
alignment_target = physical_transl + (scale - 1) * sole_anchor
base_input = physical_transl + sampled_NLF_translation_noise
supervised_delta = alignment_target - base_input
```

Pose, global orientation, and betas remain physical GT and read-only. The
dataset `gt_transl_cam` is never overwritten. `trstr_target_transl_cam` is a
separate tensor consumed only by the TRSTR loss.

Frame-level case probabilities are:

```text
15% clean:      clean depth + physical translation
40% scale-only: scaled depth + physical translation
25% NLF-only:   clean depth + perturbed translation
20% mixed:      scaled depth + perturbed translation
```

The environment residual scale is sampled in log space around one with
`std=0.08` and clipped to `[0.85, 1.15]`. One scale is shared by all people in
the frame. Consequently the same 5% residual creates a larger translation
target for a farther person. The sampled scale is distance-aware clamped when
needed so the farthest valid sole target in the frame does not exceed `2.0 m`;
the same adjusted scale is used for both depth and pseudo-target.

## NLF Perturbation Distribution

The ray/depth direction remains moderate for NLF-only and mixed cases:

| Component | Probability | Std | Clip |
| --- | ---: | ---: | ---: |
| fine ray | 30% | 0.06 m | +/-0.15 m |
| medium ray | 40% | 0.16 m | +/-0.35 m |
| strong ray | 30% | 0.30 m | +/-0.60 m |

The image-plane X/Y tangent vector uses a separate Gaussian mixture:

| Component | Probability | Per-axis Std | Vector-norm Clip |
| --- | ---: | ---: | ---: |
| fine XY | 20% | 0.12 m | 0.35 m |
| medium XY | 35% | 0.55 m | 1.20 m |
| strong XY | 45% | 1.10 m | 2.00 m |

The `2.00 m` limit applies to the combined two-dimensional tangent vector, not
independently to each axis. Scale-only samples contain no NLF translation
noise; NLF-only samples keep depth at physical metric scale.

## Correction Capacity

Per shared-weight iteration:

```text
ray vote bound: 1.00 m
tangent vote bound per axis: 1.25 m
person aggregate per-component bound: 1.25 m
iterations: 2
```

V2 uses staged remaining-error vote targets. The first iteration is supervised
on half of the current remaining correction; the second is supervised on the
actual residual. Two person updates cover `2.5 m` per component, so a 2 m
image-plane correction is legal without requiring either iteration to produce
the full displacement alone.

## Optimization

```text
initialization: v1 top-1
optimizer: fresh AdamW
learning rate: 5e-6
epochs: 7
batch size: 2
delta regularization weight: 0.001
temporal losses: 0
checkpoint monitor: refined camera-XY translation L2 p90
storage: latest + top-1 only
```

The lower learning rate protects v1 behavior. Delta regularization is reduced
from `0.01` to `0.001` so legitimate large corrections are not suppressed.

## Required Monitoring

Primary W&B signals:

```text
metric_hsi_trstr_base_l2_p90_m
metric_hsi_trstr_refined_l2_p90_m
metric_hsi_trstr_strong_coverage
metric_hsi_trstr_strong_base_l2_mean_m
metric_hsi_trstr_strong_refined_l2_mean_m
metric_hsi_trstr_strong_improvement_rate
metric_hsi_trstr_xy_base_l2_p90_m
metric_hsi_trstr_xy_refined_l2_p90_m
metric_hsi_trstr_strong_xy_coverage
metric_hsi_trstr_strong_xy_improvement_rate
metric_hsi_trstr_clean_displacement_l1
metric_hsi_trstr_region_valid_ratio
metric_hsi_trstr_monotonic_violation_rate
metric_trstr_alignment_scale_mean
metric_trstr_alignment_scale_std
metric_trstr_alignment_case_clean_rate
metric_trstr_alignment_case_scale_rate
metric_trstr_alignment_case_nlf_rate
metric_trstr_alignment_case_mixed_rate
metric_trstr_alignment_target_delta_p90_m
metric_trstr_alignment_nlf_delta_p90_m
```

Strong general samples are defined as base translation L2 error at least
`0.5 m`; strong XY samples use camera-XY error at least `1.0 m`. Success
requires XY p90 and strong-XY refined error to decrease, strong-XY improvement
rate to rise, clean displacement to remain near zero, and the second iteration
not to increase error.

## Server Gates

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
bash scripts/smoke/run_hsi_stage2_trstr_v2_strong_one_step.sh
```

The one-step gate forces the `mixed` case so it cannot pass by randomly drawing
a clean frame. It exercises scaled depth, sole-anchor pseudo-target, NLF
translation noise, forward, backward, v1 prefix loading, and checkpoint save in
one run. Formal training restores the configured 15/40/25/20 case mixture.

Only after the one-step checkpoint/gradient gate passes:

```bash
bash scripts/train/train_smpl_hsi_stage2_trstr_v2_strong.sh
```

One-step output:

```text
outputs/debug/hsi_stage2_trstr_v2_strong_one_step
```

Formal output:

```text
outputs/train/smpl_hsi_stage2_trstr_v2_strong
```
