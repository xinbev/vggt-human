# 3DPW SAM2 Mask-Intersection Pose/Beta Fine-Tune

## Goal

这轮实验先不解决 translation，只看更干净的人体 patch pooling 能不能继续降低 pose/beta 相关指标：

```text
ROI-pool checkpoint
  -> SAM2 person mask patch cache
  -> mask_intersection pooling
  -> pose/beta fine-tune
  -> 3DPW validation/test eval
```

## What Is Changed

### SAM2 Patch Mask Cache

新增预处理：

```text
scripts/preprocess/prepare_3dpw_sam2_patch_masks.py
scripts/preprocess/prepare_3dpw_sam2_patch_masks.sh
```

输出：

```text
outputs/preprocess/3dpw_sam2_patch_masks/train.pkl
outputs/preprocess/3dpw_sam2_patch_masks/validation.pkl
outputs/preprocess/3dpw_sam2_patch_masks/test.pkl
```

每个 cache 存的是每帧每个人的 SAM2 mask 转成的 patch mask，不复制图片、不改原始 3DPW。

### Dataset

`ThreeDPWDataset` 支持读取：

```yaml
data:
  sam2_patch_masks_root_key: datasets.threedpw_sam2_patch_masks_root
  require_sam2_patch_masks: true
```

输出给模型：

```text
smpl_query_patch_masks:       [S,Q,P]
smpl_query_patch_masks_valid: [S,Q]
```

### Aggregator

训练配置使用：

```yaml
model:
  smpl_query_patch_pool: true
  smpl_query_patch_pool_mode: mask_intersection
```

实际 pooling mask：

```text
final_mask = bbox_patch_mask & sam2_patch_mask
```

如果 SAM2 mask 太小，会按 Aggregator 已有逻辑 fallback 到 bbox patch mask。

### Translation Is Disabled For Training

这轮不是 trans 实验：

```yaml
model:
  freeze_smpl_translation: true

loss:
  transl_cam_weight: 0.0
  joints3d_weight: 0.0
  projected_joints2d_weight: 0.0
  projected_bbox_weight: 0.0
  projected_giou_weight: 0.0
```

新增 local SMPL losses，只约束人体局部形状，不吃 camera translation：

```yaml
loss:
  local_joints3d_weight: 20.0
  local_vertices_weight: 12.0
```

## One-Command Script

本地：

```text
C:\Users\ROG\PycharmProjects\vggt-omega\scripts\train\train_smpl_base_3dpw_sam2_mask_pose_beta_extreme.sh
```

服务器：

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human/scripts/train/train_smpl_base_3dpw_sam2_mask_pose_beta_extreme.sh
```

Smoke：

```bash
SAM2_MAX_FRAMES=50 EPOCHS=1 DEVICE=cuda:5 \
bash scripts/train/train_smpl_base_3dpw_sam2_mask_pose_beta_extreme.sh
```

正式跑。如果之前跑过 smoke，记得 `SAM2_OVERWRITE=1`，否则会复用只有 50 帧的 cache：

```bash
SAM2_OVERWRITE=1 BATCH_SIZE=16 EPOCHS=30 NUM_WORKERS=28 DEVICE=cuda:5 \
bash scripts/train/train_smpl_base_3dpw_sam2_mask_pose_beta_extreme.sh
```

如果 SAM2 cache 已经全量生成过：

```bash
RUN_PREPROCESS=0 BATCH_SIZE=16 EPOCHS=30 NUM_WORKERS=28 DEVICE=cuda:5 \
bash scripts/train/train_smpl_base_3dpw_sam2_mask_pose_beta_extreme.sh
```

## Outputs

训练：

```text
outputs/train/stageE_3dpw_sam2_mask_pose_beta_extreme_from_roi_pool
```

评测：

```text
outputs/eval/stageE_3dpw_validation_sam2_mask_pose_beta_extreme
outputs/eval/stageE_3dpw_test_sam2_mask_pose_beta_extreme
```

重点看：

```text
PA-MPJPE
MPJPE
PVE
oracle_pred_pose_gt_transl_mpjpe_mm
oracle_pred_pose_gt_transl_pve_mm
```

其中 `oracle_pred_pose_gt_transl_*` 是这轮最核心指标，因为它把 translation 换成 GT，更纯地反映 pose/beta 是否变好。
