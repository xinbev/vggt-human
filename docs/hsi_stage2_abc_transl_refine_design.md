# HSI Stage2 ABC Translation Refinement

This note records the Stage2 training design that follows the successful
Stage1 GT-SMPL scale teacher.

## Goal

Train HSI to actively correct SMPL root translation after the scene/depth scale
has been learned.

Stage1 remains responsible for:

```text
hsi_scene_scale / hsi_scene_depth_bias
```

Stage2 focuses on:

```text
hsi_refined_pred_transl_cam
```

The key training input for Stage A is a deliberately corrupted GT SMPL base:

```text
pred_transl_cam = gt_transl_cam * ray_depth_noise
```

This scales the whole camera-space translation vector. It mainly creates a depth
error while preserving `x/z` and `y/z`, so the 2D projection center remains
stable.

## Stages

- Stage A: `gt_perturbed` SMPL provider, GT `K_scal3r` for HSI scene probing,
  VGGT depth, frozen Stage1 scene affine, train translation residual.
- Stage B: mixed GT-perturbed and NLF base. GT samples use GT K for HSI; NLF
  samples use VGGT K. This bridges controlled denoising to real inference.
- Stage C: NLF base and VGGT K only, with light pose/beta losses and temporal
  momentum enabled.

## Main Script

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

DATA_ROOT=/home/zhw/xyb_space \
BEDLAM_ROOT=/home/zhw/xyb_space/bedlam/processed_bedlam \
PREPROCESSED_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/preprocess/bedlam_boxes \
STAGE1_CKPT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/stage1_scale_linear_b20_gpu7/checkpoint_latest.pt \
CUDA_VISIBLE_DEVICES_VALUE=7 \
bash scripts/train/train_smpl_hsi_nlf_stage2_abc_transl_refine.sh
```

For a quick Stage A gate before full training:

```bash
RUN_STAGES=A \
MAX_STEPS_PER_EPOCH=200 \
CUDA_VISIBLE_DEVICES_VALUE=7 \
BATCH_SIZE_A=20 \
bash scripts/train/train_smpl_hsi_nlf_stage2_abc_transl_refine.sh
```

## Metrics

The most important progress keys are:

```text
hsiBaseT = metric_hsi_base_transl_l1
hsiRefT  = metric_hsi_refined_transl_l1
hsiDT    = metric_hsi_transl_l1_delta
```

Stage2 is working when:

```text
hsiRefT < hsiBaseT
hsiDT < 0
metric_hsi_joint_error_delta < 0
```

If Stage A cannot make `hsiDT` negative in a 200-step smoke, do not run the full
ABC schedule. Use the visualization script first.

## Diagnostics

Interface smoke:

```bash
bash scripts/smoke/check_hsi_stage2_transl_perturb_interface.sh
```

Single-frame visual diagnostics:

```bash
STAGE2_CKPT=/path/to/stage/checkpoint_latest.pt \
CUDA_VISIBLE_DEVICES_VALUE=7 \
bash scripts/vis/vis_hsi_stage2_transl_refine_debug.sh
```

All outputs are written under `outputs/`.
