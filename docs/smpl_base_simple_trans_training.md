# SMPL Base ROI-Pool Simple Translation Training

## Goal

当前 ROI-pool 训练已经证明 pose/beta 明显变好，但 base translation 的深度误差仍然很大。这个实验不再继续堆复杂 `ray_offset_depth + translation_refiner + hard-tail`，而是把 base translation 改成一个稳定、简单、可解释的初始化：

```text
ROI/box patch pooled human query token
  + bbox center
  + camera intrinsics
  -> camera ray
  -> bbox-height depth prior
  -> token predicts log-depth scale + tangent offset
  -> pred_transl_cam
```

这一步的目标不是让 base translation 成为最终最优结果，而是给后续 track-aware HSI 一个更稳的初始位置。

## Model Changes

新增 translation mode：

```yaml
model:
  smpl_translation_output_mode: simple_ray_depth
```

新增模块：

```text
vggt_omega/models/heads/smpl_head.py
  SimpleRayDepthTranslationDecoder
```

它和旧模式的区别：

- 保留 ROI-pool query 输入。
- 保留 bbox center + camera intrinsics 的射线几何。
- 使用 bbox height 得到正深度 prior。
- 只预测 `log_depth_delta` 和两个 tangent offset。
- 不使用 pose/beta 作为 translation decoder 输入。
- 不启用 `CameraRayTranslationRefiner`。
- 不启用 hard-tail translation loss。

## Checkpoint Loading Rule

从旧 ROI-pool checkpoint 热启动时，继承：

- Aggregator/VGGT backbone
- ROI-pool query 相关权重
- SMPL pose heads
- SMPL beta heads
- bbox/conf heads

跳过：

```yaml
checkpoint:
  resume_skip_prefixes:
    - smpl_head.regression_head.translation_decode_heads.
    - smpl_head.regression_head.translation_refiner.
    - smpl_head.temporal_translation_refiner.
```

原因是旧 `ray_offset_depth` decoder 和新 `simple_ray_depth` decoder 名字相近但输入语义不同，不能直接继承。

## Training Plan

默认训练只打开 simple translation decoder：

```yaml
model:
  freeze_smpl_head: true
  train_smpl_translation_decode_heads: true
  train_smpl_query_token: false
  train_smpl_box_prior_embed: false
  train_smpl_patch_pool_embed: false
```

这样 pose/beta 保持上次 ROI-pool 成功结果，第一轮只检查新 translation 方案本身是否有效。

训练阶段：

```text
Stage 0: HF BEDLAM data smoke check
Stage A: HF BEDLAM simple-trans train
Stage B: 3DPW simple-trans fine-tune
Stage C: 3DPW validation eval
Stage D: 3DPW test eval
```

## Server Command

本地脚本：

```text
C:\Users\ROG\PycharmProjects\vggt-omega\scripts\train\train_smpl_base_roi_pool_simple_trans_hf_bedlam_then_3dpw.sh
```

服务器对应：

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human/scripts/train/train_smpl_base_roi_pool_simple_trans_hf_bedlam_then_3dpw.sh
```

推荐先跑 smoke：

```bash
CHECK_MAX_NPZ_FILES=1 CHECK_MAX_FRAMES=50 HF_EPOCHS=1 THREEDPW_EPOCHS=1 DEVICE=cuda:5 \
bash scripts/train/train_smpl_base_roi_pool_simple_trans_hf_bedlam_then_3dpw.sh
```

正式跑：

```bash
HF_BATCH_SIZE=48 THREEDPW_BATCH_SIZE=16 NUM_WORKERS=28 DEVICE=cuda:5 \
bash scripts/train/train_smpl_base_roi_pool_simple_trans_hf_bedlam_then_3dpw.sh
```

如果只想从已有 ROI-pool checkpoint 直接跑 3DPW simple-trans：

```bash
RUN_HF_TRAIN=0 RUN_3DPW_FINETUNE=1 RUN_EVAL=1 DEVICE=cuda:5 \
bash scripts/train/train_smpl_base_roi_pool_simple_trans_hf_bedlam_then_3dpw.sh
```

## Outputs

训练输出：

```text
outputs/train/stageA_hf_bedlam_smpl_base_roi_pool_simple_trans
outputs/train/stageB_3dpw_smpl_base_roi_pool_simple_trans_from_hf_bedlam
```

评测输出：

```text
outputs/eval/stageC_3dpw_validation_roi_pool_simple_trans_from_hf_bedlam
outputs/eval/stageD_3dpw_test_roi_pool_simple_trans_from_hf_bedlam
```

重点看：

- `transl_l2_mm`
- `transl_xy_l2_mm`
- `transl_z_abs_mm`
- `oracle_pred_pose_gt_transl_mpjpe_mm`
- `PA-MPJPE / MPJPE / PVE`

判断方式：

- 如果 `oracle_pred_pose_gt_transl_mpjpe_mm` 基本接近 ROI-pool 结果，说明 pose/beta 没被破坏。
- 如果 `transl_z_abs_mm` 明显下降，说明 simple depth branch 有效。
- 如果 `transl` 下降但 `MPJPE/PVE` 仍高，后续应进入 track-aware HSI translation refinement。
