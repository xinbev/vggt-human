# Base SMPL Translation Ray Refine Closeout Handoff

This document closes the side branch that repaired base SMPL camera-space
translation. It is written for the main-line agent that will continue HSI,
temporal, and video work.

## Executive Decision

The side branch succeeded.

Use the new HSI checkpoint as the next main-line starting point:

```text
outputs/train/smpl_hsi_after_translation_ray_refine/checkpoint_latest.pt
```

Do not continue spending time on base translation architecture unless a later
non-GT-box or temporal evaluation exposes a new failure. The remaining work
should move back to main-line validation: box-prior robustness, temporal
stability, and video/HSI behavior.

## What Was Fixed

The original problem was that base SMPL `pred_transl_cam` was not explicitly
designed around camera geometry. Bad single-frame root translation could enter
HSI, forcing HSI to rescue a person from a poor initial 3D position.

The implemented fix is a camera-ray residual refiner:

```text
bbox center + VGGT camera intrinsics -> camera ray + tangent basis
base SMPL transl_cam -> decomposed into ray depth + tangent offsets
network predicts bounded ray/tangent/log-depth residuals
final refined translation is written back to pred_transl_cam
```

Important depth boundary:

- Raw VGGT depth / depth-map Z is not used as base translation supervision.
- HSI may later rescale depth, so raw HSI-stage Z is not a stable target.
- Supervision is dataset SMPL camera translation, i.e. `gt_transl_cam`.

## Key Code Paths

Model implementation:

```text
vggt_omega/models/heads/smpl_head.py
```

Key behavior:

- `CameraRayTranslationRefiner` predicts the residual.
- When enabled, refined translation replaces `pred_transl_cam`.
- The pre-refiner value is preserved as `base_pred_transl_cam`.

Relevant flow:

```text
base_pred_transl_cam = pred_transl_cam
translation_refine_outputs = self.translation_refiner(...)
pred_transl_cam = translation_refine_outputs["pred_transl_cam"]
outputs["base_pred_transl_cam"] = base_pred_transl_cam
```

HSI uses the refined value:

```text
vggt_omega/models/heads/hsi_refinement_head.py
transl = smpl_outputs["pred_transl_cam"].float()
```

Therefore, if `model.smpl_enable_translation_refine=true`, HSI receives the
repaired translation automatically.

## Training Chain

Primary script:

```bash
bash scripts/train/train_smpl_translation_ray_refine_full_from0121.sh
```

Stages:

```text
T0: evaluate original 0121 HSI checkpoint
T1: train only camera-ray translation refiner, no raw depth, no HSI
T2: train transl_cam_heads + camera-ray refiner, no raw depth, no HSI
T3: merge translation repair keys back into full HSI checkpoint
```

HSI reconnect script:

```bash
bash scripts/train/train_smpl_hsi_after_translation_ray_refine.sh
```

HSI reconnect eval:

```bash
bash scripts/eval/eval_smpl_hsi_after_translation_ray_refine.sh
```

## Important Checkpoints

Old stable HSI baseline:

```text
outputs/train/smpl_hsi_refine_20q/checkpoint_epoch_0121.pt
```

Translation-only final checkpoint:

```text
outputs/train/smpl_translation_ray_refine_full_from0121/stage2_transl_heads_ray_refiner/checkpoint_latest.pt
```

Merged HSI + translation repair checkpoint:

```text
outputs/train/smpl_translation_ray_refine_full_from0121/merged_hsi_translation/checkpoint_latest.pt
```

New recommended main-line checkpoint:

```text
outputs/train/smpl_hsi_after_translation_ray_refine/checkpoint_latest.pt
```

The local Git sync may contain only `resolved_config.json` and eval JSON files,
not the large `.pt` checkpoints. On the server, continue from the checkpoint
paths above.

## Base Translation Evaluation

Evaluation files:

```text
outputs/eval/smpl_translation_ray_refine_full_from0121/
```

Settings:

```text
num_samples = 200
num_matched = 1189
use_gt_box_prior = true
sequence_length = 2
max_humans = 20
```

Metrics are in meters below, with millimeter notes for readability.

| Stage | transl L2 | Z L1 | XY L2 | MPJPE | PVE |
|---|---:|---:|---:|---:|---:|
| T0 original 0121 base | 0.06544 | 0.04406 | 0.04094 | 0.07453 | 0.07888 |
| T1 ray refiner only | 0.05205 | 0.03298 | 0.03452 | 0.06287 | 0.06765 |
| T2 transl heads + refiner | 0.04956 | 0.03117 | 0.03333 | 0.06184 | 0.06698 |

Relative to original 0121 base, T2 refined improves:

```text
transl L2: 65.44mm -> 49.56mm, +24.26%
Z L1:      44.06mm -> 31.17mm, +29.27%
XY L2:     40.94mm -> 33.33mm, +18.59%
MPJPE:     74.53mm -> 61.84mm, +17.03%
PVE:       78.88mm -> 66.98mm, +15.08%
```

Interpretation:

- T1 is the cleanest proof that the ray refiner is useful because the original
  SMPL translation heads are frozen.
- T2 gives the best final refined output.
- T2's pre-refiner `base_pred_transl_cam` is worse than old 0121. This is not a
  failure if all downstream code uses `pred_transl_cam`, because `pred_transl_cam`
  is the refined output when the refiner is enabled.
- It is unsafe to consume T2's `base_pred_transl_cam` as a final result.

## HSI Reconnect Evaluation

Evaluation file:

```text
outputs/eval/smpl_hsi_after_translation_ray_refine/hsi_refine_metrics.json
```

Settings:

```text
num_samples = 200
conf_threshold = 0.1
use_gt_box_prior = true
matched humans = 2.9725 / gt 2.9725
```

Old 0121 HSI:

```text
MPJPE:       0.06160 m
PVE:         0.06732 m
Translation: 0.04886 m
Projected:   4.92777 px
Depth median L1: 0.25903 m
```

New HSI after translation repair:

```text
MPJPE:       0.04879 m
PVE:         0.05373 m
Translation: 0.03645 m
Projected:   4.40295 px
Depth median L1: 0.22121 m
```

Relative to old 0121 HSI, the new HSI checkpoint improves approximately:

```text
MPJPE:       +20.8%
PVE:         +20.2%
Translation: +25.4%
Projected:   +10.6%
Depth median:+14.6%
```

Within the new HSI eval, HSI improves over its repaired base:

```text
3D joints MPJPE:   base=0.061839 -> hsi=0.048789, +21.10%
Vertices PVE:      base=0.066982 -> hsi=0.053728, +19.79%
Translation L2:    base=0.049561 -> hsi=0.036451, +26.45%
Projected joints:  base=5.943151 -> hsi=4.402954, +25.92%
Depth median L1:   base=5.826001 -> hsi=0.221214, +96.20%
Near depth median: base=5.739618 -> hsi=0.214317, +96.27%
Human ROI median:  base=5.063351 -> hsi=0.208229, +95.89%
```

Guard/contact signals:

```text
hsi_worse_than_base_ratio_2cm = 0.01430
hsi_joint_error_delta_m       = -0.01305
foot float:       0.19145 -> 0.16581
foot penetration: 0.08595 -> 0.08471
sole float:       0.29532 -> 0.25969
sole penetration: 0.07856 -> 0.06939
sole-plane float: 0.05174 -> 0.03911
sole-plane pen:   0.01970 -> 0.01817
```

Scene affine distribution is stable enough for this test:

```text
hsi_scene_scale count=400 mean=8.07982 median=7.74317 min=6.69822 max=11.15635
hsi_scene_depth_bias count=400 mean=0.09450 median=0.08867 min=-0.32867 max=0.49214
```

## Required Main-Line Config Invariants

For any continuation from the new checkpoint, keep:

```yaml
model:
  enable_camera: true
  enable_depth: true
  enable_smpl: true
  enable_hsi_refine: true
  smpl_enable_translation_refine: true
  smpl_translation_refine_max_ray_delta_m: 1.20
  smpl_translation_refine_max_tangent_delta_m: 0.60
  smpl_translation_refine_max_log_depth_delta: 0.85
  smpl_translation_refine_max_box_prior_weight: 1.00
```

If `smpl_enable_translation_refine=false`, the model falls back to the bare
translation head path and loses this side-branch fix.

Do not treat `base_pred_transl_cam` as the final translation for T2-derived
checkpoints. The final translation is `pred_transl_cam`.

## Recommended Next Main-Line Steps

1. Promote the new checkpoint to the current stable main-line baseline:

```text
outputs/train/smpl_hsi_after_translation_ray_refine/checkpoint_latest.pt
```

2. Run non-GT or noisy-box-prior evaluation.

The current strong result used `--use-gt-box-prior`. Main-line should test
robustness without oracle boxes, or with noisy boxes, before claiming full
generalization.

3. Run temporal diagnostics.

The ray refiner is single-frame. It improved HSI metrics, but the main line
still needs to verify translation velocity, joint velocity, acceleration, and
visual video stability.

4. Resume main-line HSI/video work from this checkpoint rather than from:

```text
outputs/train/smpl_hsi_refine_20q/checkpoint_epoch_0121.pt
```

5. Keep T1 as a fallback.

If later temporal/noisy-box tests show T2-specific instability, the safer
fallback is the T1 ray-refiner-only translation checkpoint merged into HSI,
because T1 improves translation without changing the original translation
heads.

## Status

Side branch status:

```text
CLOSED / SUCCESSFUL
```

Return to main line.

The base translation repair is now a validated component, not an open research
blocker. The remaining work is integration validation and main-line temporal
robustness.
