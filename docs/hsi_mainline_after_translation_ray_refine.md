# HSI Main Line After Translation Ray Refine

This is the current recommended main-line route after the successful base SMPL
camera-ray translation repair.

## Promoted Baseline

Use this checkpoint as the new main-line single-frame HSI baseline:

```text
outputs/train/smpl_hsi_after_translation_ray_refine/checkpoint_latest.pt
```

This replaces the older experimental starting point:

```text
outputs/train/smpl_hsi_refine_20q/checkpoint_epoch_0121.pt
```

The old 0121 checkpoint remains useful as a historical baseline, but future
HSI/video work should start from the translation-repaired checkpoint unless a
specific ablation requires otherwise.

## Required Model Switch

Any model loading a translation-repaired checkpoint must build the SMPL head
with the camera-ray translation refiner enabled:

```yaml
model:
  smpl_enable_translation_refine: true
  smpl_translation_refine_max_ray_delta_m: 1.20
  smpl_translation_refine_max_tangent_delta_m: 0.60
  smpl_translation_refine_max_log_depth_delta: 0.85
  smpl_translation_refine_max_box_prior_weight: 1.00
```

The final base translation is `pred_transl_cam`.

Do not use `base_pred_transl_cam` as the final translation for T2-derived
checkpoints; it is only the pre-refiner diagnostic value.

## Validated Side-Branch Result

The side branch reports the following GT-box-prior HSI improvement over old
0121 HSI:

```text
MPJPE:       0.06160m -> 0.04879m
PVE:         0.06732m -> 0.05373m
Translation: 0.04886m -> 0.03645m
Projected:   4.92777px -> 4.40295px
Depth median L1: 0.25903m -> 0.22121m
```

So base translation is no longer considered the current blocker. The remaining
main-line work is robustness and temporal/video validation.

## Main-Line Temporal Training

Run the temporal scene-affine + human momentum + no-worse route from the new
translation-repaired HSI checkpoint:

```bash
CUDA_VISIBLE_DEVICES_VALUE=6 \
NUM_VIEWS=12 \
STAGE1_EXTRA_EPOCHS=3 \
STAGE2_EXTRA_EPOCHS=4 \
TEMPORAL_NO_WORSE_WEIGHT=20.0 \
TEMPORAL_NO_WORSE_MARGIN_M=0.002 \
TEMPORAL_NO_WORSE_ACCEL_MARGIN_M=0.003 \
bash scripts/train/train_smpl_hsi_scene_then_temporal_noworse_after_translation_ray_refine.sh
```

If the stage1 scene-affine checkpoint already exists and should be reused:

```bash
STAGE1_EXTRA_EPOCHS=0 \
bash scripts/train/train_smpl_hsi_scene_then_temporal_noworse_after_translation_ray_refine.sh
```

Expected output:

```text
outputs/train/smpl_hsi_temporal_after_translation_ray_refine/
  stage1_scene_affine/checkpoint_latest.pt
  stage2_human_momentum_no_worse/checkpoint_latest.pt
```

## Main-Line Temporal Evaluation

```bash
bash scripts/eval/eval_smpl_hsi_scene_then_temporal_noworse_after_translation_ray_refine.sh
```

Expected output:

```text
outputs/eval/hsi_temporal_after_translation_ray_refine_noworse/hsi_temporal_metrics.json
```

Primary checks:

```text
hsi_scene_scale_range == 0
hsi_scene_bias_range_m == 0
hsi temporal velocity/acceleration improves over base
hsi_temporal_no_worse_margin_ratio remains low
```

## Main-Line Clip Visualization

```bash
bash scripts/vis/vis_smpl_hsi_scene_then_temporal_noworse_after_translation_ray_refine_clip.sh
```

Expected output:

```text
outputs/vis/hsi_temporal_after_translation_ray_refine_noworse_clip/
```

## Next Required Validation

Before claiming the route is ready for large-scale training, run:

1. Non-GT-box or noisy-box-prior evaluation.
2. Temporal evaluation from the new translation-repaired checkpoint.
3. Visual clip checks around known bad frames.
4. A comparison against the previous no-worse temporal checkpoint from 0121.

The translation repair is now part of the recommended main line, but it was
validated primarily under GT box prior. Box-prior robustness is the next risk.

