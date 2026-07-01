# SMPL Base Training Pipeline: HF BEDLAM -> 3DPW

## Goal

Train the current box/proj-verts based SMPL base model in two stages:

```text
Stage A: HF BEDLAM pretrain
Stage B: 3DPW train fine-tune
Stage C: 3DPW validation top checkpoint check
Stage D: 3DPW test evaluation
```

This is the first mainline baseline before adding SAM2 mask-guided query pooling
or person crop/ROI features.

## One-Command Entry

Run on the server:

```bash
DEVICE=cuda bash scripts/train/train_smpl_base_hf_bedlam_then_3dpw.sh
```

To use a specific physical GPU, prefer:

```bash
GPU_ID=5 DEVICE=cuda bash scripts/train/train_smpl_base_hf_bedlam_then_3dpw.sh
```

`DEVICE=cuda:5` is also accepted by the wrapper; it is normalized to
`CUDA_VISIBLE_DEVICES=5` plus internal `DEVICE=cuda`, avoiding PyTorch visible
ordinal mismatches.

The script uses:

```text
configs/train_smpl_base_hf_bedlam_ray_refine.yaml
configs/train_smpl_base_3dpw_ray_refine.yaml
```

Default outputs:

```text
outputs/train/stageA_hf_bedlam_smpl_base_ray_refine
outputs/train/stageB_3dpw_smpl_base_ray_refine_from_hf_bedlam
outputs/eval/stageC_3dpw_validation_from_hf_bedlam
outputs/eval/stageD_3dpw_test_from_hf_bedlam
```

## 80G Defaults

The script defaults to:

```text
HF_BATCH_SIZE=2
THREEDPW_BATCH_SIZE=2
NUM_WORKERS=8
HF_EPOCHS=15
THREEDPW_EPOCHS=5
```

If CUDA OOM happens, rerun with:

```bash
HF_BATCH_SIZE=1 THREEDPW_BATCH_SIZE=1 \
DEVICE=cuda bash scripts/train/train_smpl_base_hf_bedlam_then_3dpw.sh
```

## Resume / Skip Controls

Use an existing HF checkpoint and only run 3DPW fine-tune:

```bash
RUN_HF_TRAIN=0 \
HF_CKPT=outputs/train/stageA_hf_bedlam_smpl_base_ray_refine/checkpoint_latest.pt \
DEVICE=cuda bash scripts/train/train_smpl_base_hf_bedlam_then_3dpw.sh
```

Only evaluate an existing 3DPW checkpoint:

```bash
RUN_CHECKS=0 RUN_HF_TRAIN=0 RUN_3DPW_FINETUNE=0 RUN_EVAL=1 \
HF_CKPT=outputs/train/stageA_hf_bedlam_smpl_base_ray_refine/checkpoint_latest.pt \
THREEDPW_CKPT=outputs/train/stageB_3dpw_smpl_base_ray_refine_from_hf_bedlam/checkpoint_top01.pt \
bash scripts/train/train_smpl_base_hf_bedlam_then_3dpw.sh
```

Small debug run:

```bash
MAX_NPZ_FILES=1 MAX_FRAMES=200 HF_EPOCHS=1 THREEDPW_EPOCHS=1 \
DEVICE=cuda bash scripts/train/train_smpl_base_hf_bedlam_then_3dpw.sh
```

## Notes

- HF BEDLAM boxes come from `proj_verts` first, then `gtkps`, then
  `center/scale`.
- HF BEDLAM translation uses `trans_cam + cam_ext[:3,3]`.
- SAM2 mask-guided query pooling is intentionally not included in this first
  mainline baseline; add it later as a controlled ablation.
