# HF BEDLAM SAM2 Pretrain -> 3DPW SAM2 Fine-Tune

## Goal

上一轮 3DPW-only SAM2 mask-intersection fine-tune 已经证明：

```text
box ROI-pool -> SAM2 mask_intersection pooling
```

能稳定改善 pose/beta。当前阶段把 SAM2 pooling 扩到 HF BEDLAM pretrain，再接 3DPW 长训 fine-tune，争取当前系统下的最好 pose/beta 指标。

## New Pipeline

```text
HF BEDLAM RGB + GT boxes
  -> SAM2 patch-mask cache
  -> HF BEDLAM mask_intersection pose/beta pretrain
  -> 3DPW mask_intersection pose/beta fine-tune
  -> 3DPW validation/test eval
```

这轮仍然先不把 translation 作为训练目标：

```yaml
model:
  freeze_smpl_translation: true

loss:
  transl_cam_weight: 0.0
  joints3d_weight: 0.0
  projected_joints2d_weight: 0.0
  local_joints3d_weight: 20.0
  local_vertices_weight: 12.0
```

## Added Files

HF BEDLAM SAM2 预处理：

```text
scripts/preprocess/prepare_hf_bedlam_sam2_patch_masks.py
scripts/preprocess/prepare_hf_bedlam_sam2_patch_masks.sh
```

HF BEDLAM SAM2 训练配置：

```text
configs/train_smpl_base_hf_bedlam_sam2_mask_pose_beta_extreme.yaml
```

总脚本：

```text
scripts/train/train_smpl_base_hf_bedlam_sam2_then_3dpw_extreme.sh
```

输出：

```text
outputs/preprocess/hf_bedlam_sam2_patch_masks/train.pkl
outputs/preprocess/3dpw_sam2_patch_masks/{train,validation,test}.pkl
outputs/train/stageF_hf_bedlam_sam2_mask_pose_beta_extreme
outputs/train/stageG_3dpw_sam2_mask_pose_beta_extreme_from_hf_bedlam
outputs/eval/stageG_3dpw_validation_sam2_mask_from_hf_bedlam
outputs/eval/stageG_3dpw_test_sam2_mask_from_hf_bedlam
```

## Server Commands

进入服务器项目：

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
```

### Smoke

先只处理少量 HF BEDLAM + 3DPW 帧，确认 SAM2、loader、训练和评测都能通：

```bash
HF_SAM2_MAX_OUTPUT_FRAMES=50 THREEDPW_SAM2_MAX_FRAMES=50 HF_EPOCHS=1 THREEDPW_EPOCHS=1 DEVICE=cuda:5 \
bash scripts/train/train_smpl_base_hf_bedlam_sam2_then_3dpw_extreme.sh
```

### Full Run

如果刚跑过 smoke，正式跑必须覆盖小 cache：

```bash
HF_SAM2_OVERWRITE=1 THREEDPW_SAM2_OVERWRITE=1 \
HF_BATCH_SIZE=40 THREEDPW_BATCH_SIZE=12 \
HF_EPOCHS=12 THREEDPW_EPOCHS=40 NUM_WORKERS=28 DEVICE=cuda:5 \
bash scripts/train/train_smpl_base_hf_bedlam_sam2_then_3dpw_extreme.sh
```

### Reuse Existing SAM2 Cache

如果 HF BEDLAM 和 3DPW SAM2 cache 已经全量生成过：

```bash
RUN_HF_PREPROCESS=0 RUN_3DPW_PREPROCESS=0 \
HF_BATCH_SIZE=40 THREEDPW_BATCH_SIZE=12 \
HF_EPOCHS=12 THREEDPW_EPOCHS=40 NUM_WORKERS=28 DEVICE=cuda:5 \
bash scripts/train/train_smpl_base_hf_bedlam_sam2_then_3dpw_extreme.sh
```

### Resume Only 3DPW From Existing HF SAM2 Checkpoint

```bash
RUN_HF_PREPROCESS=0 RUN_3DPW_PREPROCESS=0 RUN_HF_TRAIN=0 RUN_3DPW_FINETUNE=1 RUN_EVAL=1 \
HF_CKPT=outputs/train/stageF_hf_bedlam_sam2_mask_pose_beta_extreme/checkpoint_latest.pt \
THREEDPW_EPOCHS=40 DEVICE=cuda:5 \
bash scripts/train/train_smpl_base_hf_bedlam_sam2_then_3dpw_extreme.sh
```

## What To Watch

核心指标仍然是：

```text
oracle_pred_pose_gt_transl_mpjpe_mm
oracle_pred_pose_gt_transl_pve_mm
PA-MPJPE
MPJPE
PVE
```

因为当前阶段没有训练 translation，所以 `transl_l2_mm / transl_z_abs_mm` 只作为记录，不作为这轮是否成功的主要判断。
