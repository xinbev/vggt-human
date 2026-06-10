# SMPL 到 HSI 全训练链路整理

本文档整理当前仓库从 VGGT baseline 开始，逐步训练到 `smpl_hsi_refine_20q` 的完整链路。目标是把当前小数据集探索中已经跑通并验证有效的路径固定下来，后续切换到大数据集时可以按同一条链路做完整训练、评估和可视化。

当前结论：`outputs/train/smpl_hsi_refine_20q/checkpoint_epoch_0121.pt` 是已验证有效的 HSI checkpoint；`checkpoint_epoch_0122.pt` 开始出现 HSI scene scale 饱和，不能作为有效结果继续使用。

重要更正：本仓库的 `optim.epochs` 是“绝对目标 epoch”，不是“本阶段额外训练 epoch”。例如从 `epoch=40` 的 checkpoint resume，设置 `optim.epochs=60`，实际只会训练 20 个 epoch，也就是 41 到 60。只有脚本中显式使用 `EXTRA_EPOCHS` 并计算 `TOTAL_EPOCHS=INIT_EPOCH+EXTRA_EPOCHS` 的阶段，才表示“额外训练 N 轮”。

因此当前 `checkpoint_epoch_0121.pt` 不是 HSI 训练了 121 轮，而是整条链路累计绝对 epoch 到 121。按当前主线推算，ROI-Pool 3D refine 到 `epoch=100`，HSI 从 `epoch=100` 开始，因此：

```text
checkpoint_epoch_0121.pt = HSI 阶段额外训练约 21 轮
checkpoint_epoch_0122.pt = HSI 阶段额外训练约 22 轮，并开始 scale 饱和
```

## 当前有效结论

基于 `checkpoint_epoch_0121.pt`，在 200 个样本上用 GT box prior 做 base SMPL vs HSI refined 对比，结果如下：

| 指标 | Base | HSI refined | 提升 |
|---|---:|---:|---:|
| 3D joints MPJPE | 0.07453 m | 0.06160 m | 17.34% |
| Vertices PVE | 0.07888 m | 0.06732 m | 14.65% |
| Translation L2 | 0.06544 m | 0.04886 m | 25.33% |
| Projected joints | 6.74 px | 4.93 px | 26.90% |
| Depth L1 median | 5.826 m | 0.259 m | 95.55% |

`checkpoint_epoch_0121.pt` 的 HSI scene scale 分布稳定：

```text
scale mean   = 8.13
scale median = 7.81
scale min    = 6.43
scale max    = 11.34
```

坏的 `checkpoint_epoch_0122.pt` 出现：

```text
hsi_scene_scale = 20.0855
```

这正好是当前实现里的 `exp(3)` 上限，说明 HSI scene affine 分支发生饱和。后续大训练不能盲目使用 latest checkpoint，必须保存并选择 best checkpoint。

## 路径和依赖

核心训练入口：

```text
scripts/train/train_smpl.py
```

路径配置：

```text
configs/path.yaml
```

关键路径：

```text
checkpoints.vggt_baseline
assets.smpl_model_dir
datasets.bedlam_root
datasets.bedlam_boxes_root
```

服务器当前默认路径写在各个 `sh` 脚本中：

```text
REPO_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human
BEDLAM_ROOT=/home/zhw/xyb_space/bedlam/processed_bedlam
PREPROCESSED_ROOT=${REPO_ROOT}/outputs/preprocess/bedlam_boxes
VGGT_CKPT=/home/zhw/lab_users/xyb/home/projects/vggt-omega/checkpoints/vggt_omega_1b_512.pt
SMPL_MODEL_DIR=/home/zhw/xyb_space/SAT-HMR/weights/smpl_data/
```

后续切换大数据集时，优先改这些路径，不要改核心模型代码。

## 数据准备

训练数据需要 BEDLAM 风格布局：

```text
<bedlam_root>/Training/<sequence>/rgb/*.png
<bedlam_root>/Training/<sequence>/smpl/*.pkl
<bedlam_root>/Training/<sequence>/depth/*.npy   # HSI 阶段需要
<bedlam_root>/Training/<sequence>/cam/*.npz     # 有则使用
```

bbox sidecar 数据位于：

```text
outputs/preprocess/bedlam_boxes
```

当前数据加载器通过 `boxes_root_key: datasets.bedlam_boxes_root` 读取这个目录。所有 SMPL query box prior、GT prior 可视化和 noisy prior 评估都依赖它。

当前已有预处理脚本：

```text
scripts/preprocess/prepare_bedlam_boxes.py
```

大数据集正式训练前需要确保：

```text
outputs/preprocess/bedlam_boxes/Training/<sequence>/smpl_boxes/*.pkl
```

数量和训练 RGB/SMPL 帧匹配。

## 模型数据流

主模型是 `VGGTOmega`：

```text
images
  -> Aggregator
      -> cached tokens / patch tokens / SMPL query tokens
  -> CameraHead
      -> pose_enc
  -> DenseHead
      -> depth, depth_conf
  -> AggregatorSMPLHead
      -> pred_poses, pred_betas, pred_transl_cam, pred_confs, pred_boxes
  -> HSIRefinementHead
      -> hsi_refined_pred_poses
      -> hsi_refined_pred_betas
      -> hsi_refined_pred_transl_cam
      -> hsi_scene_scale
      -> hsi_scene_depth_bias
      -> hsi_contact_logits
```

非 HSI 阶段通常不启用 depth：

```yaml
model.enable_depth: false
data.require_depth: false
```

HSI 阶段必须启用 camera 和 depth：

```yaml
model.enable_camera: true
model.enable_depth: true
model.enable_hsi_refine: true
data.require_depth: true
```

## 推荐完整训练链路

下面是从空 SMPL head 训练到 HSI 的完整链路。每一步都写成 `sh` 入口，服务器上应从仓库根目录执行。

## Epoch 语义和真实阶段长度

`scripts/train/train_smpl.py` 的核心循环是：

```python
start_epoch, global_step = resume_training_checkpoint(...)
for epoch in range(start_epoch, epochs):
    ...
    save_checkpoint(..., epoch + 1, ...)
```

所以阶段真实训练轮数是：

```text
真实训练轮数 = optim.epochs - resume_checkpoint.epoch
```

如果 `resume_checkpoint.epoch >= optim.epochs`，这个阶段会直接 0-step 结束。

当前脚本按默认 checkpoint 依赖推算出的真实链路如下：

| 阶段 | 脚本 | 输入 epoch | 目标 epoch | 真实新增 epoch | 输出 epoch |
|---|---|---:|---:|---:|---:|
| Stage 0 | `train_smpl_hungarian.sh` | 0 | 10 | 10 | 10 |
| Stage 1 | `train_smpl_projected_bbox.sh` | 10 | 15 | 5 | 15 |
| Stage 2 | `train_smpl_conf_quality.sh` | 15 | 20 | 5 | 20 |
| Stage 3 | `train_smpl_dab_box_prior.sh` | 20 | 40 | 20 | 40 |
| Stage 4 | `train_smpl_dab_joint_refine.sh` | 40 | 60 | 20 | 60 |
| Stage 5 | `train_smpl_dab_roi_pool_3d_refine.sh` | 60 | 100 | 40 | 100 |
| Stage 6 | `train_smpl_hsi_refine.sh` | 100 | 140 | 40 | 140 |

### Stage 0: Hungarian 20Q Bootstrap

脚本：

```bash
bash scripts/train/train_smpl_hungarian.sh
```

配置：

```text
configs/train_smpl.yaml
```

输出：

```text
outputs/train/smpl_hungarian_20q/checkpoint_latest.pt
```

作用：

```text
从 VGGT baseline 初始化，冻结 aggregator，训练基础 SMPL query head。
```

关键设置：

```text
EPOCHS=10        # 目标 epoch；从 0 开始，真实新增 10 轮
LR=1e-4
MAX_HUMANS=20
NUM_VIEWS=2
model.freeze_aggregator=true
model.train_smpl_query_token=true
model.predict_boxes=true
```

### Stage 1: Projected BBox

脚本：

```bash
bash scripts/train/train_smpl_projected_bbox.sh
```

输入：

```text
outputs/train/smpl_hungarian_20q/checkpoint_latest.pt
```

输出：

```text
outputs/train/smpl_projected_bbox_20q/checkpoint_latest.pt
```

作用：

```text
加入 VGGT camera projection 下的 projected bbox 监督，让预测框和 SMPL 2D 投影更一致。
```

关键设置：

```text
EPOCHS=15        # 目标 epoch；通常从 10 resume，真实新增 5 轮
LR=5e-5
model.enable_camera=true
model.freeze_camera_head=true
loss.projected_bbox_weight=1.0
loss.projected_giou_weight=1.0
checkpoint.resume_optimizer=false
```

### Stage 2: Confidence Quality

脚本：

```bash
bash scripts/train/train_smpl_conf_quality.sh
```

输入：

```text
outputs/train/smpl_projected_bbox_20q/checkpoint_latest.pt
```

输出：

```text
outputs/train/smpl_conf_quality_20q/checkpoint_latest.pt
```

作用：

```text
优化 confidence，使正负 query 分数更可分，降低重复/错误高置信预测。
```

关键设置：

```text
EPOCHS=20        # 目标 epoch；通常从 15 resume，真实新增 5 轮
LR=2e-5
loss.conf_loss_type=focal
loss.conf_target_type=matched_iou
loss.duplicate_conf_weight=1.0
```

### Stage 3: DAB Box Prior Query

脚本：

```bash
bash scripts/train/train_smpl_dab_box_prior.sh
```

输入：

```text
outputs/train/smpl_conf_quality_20q/checkpoint_latest.pt
```

输出：

```text
outputs/train/smpl_dab_box_prior_20q/checkpoint_latest.pt
```

作用：

```text
引入 GT/preprocessed box prior 作为 SMPL query 条件，使 query 和目标人体框绑定。
```

关键设置：

```text
EPOCHS=40        # 目标 epoch；通常从 20 resume，真实新增 20 轮
LR=2e-5
model.smpl_query_box_prior=true
model.smpl_bbox_mode=reference_residual
model.smpl_return_aux=true
data.require_boxes=true
```

说明：

```text
这是后续 DAB 系列主线的起点。大数据集训练时，box sidecar 的质量会直接影响这个阶段。
```

### Stage 4: DAB Joint Refine

脚本：

```bash
bash scripts/train/train_smpl_dab_joint_refine.sh
```

输入：

```text
outputs/train/smpl_dab_box_prior_20q/checkpoint_latest.pt
```

输出：

```text
outputs/train/smpl_dab_joint_refine_20q/checkpoint_latest.pt
```

作用：

```text
强化 SMPL pose/betas/translation、3D joints 和 projected joints 监督。
```

关键设置：

```text
EPOCHS=60        # 目标 epoch；通常从 40 resume，真实新增 20 轮
LR=1e-5
training_prior.center_noise=0.03
training_prior.size_noise=0.07
training_prior.drop_prob=0.03
loss.joints3d_weight=2.0
loss.projected_joints2d_weight=1.0
```

验证脚本：

```bash
bash scripts/eval/check_smpl_dab_joint_refine.sh
```

这个脚本会同时评估 clean GT box prior 和 noisy prior。

### Stage 5: DAB ROI-Pool 3D Refine

脚本：

```bash
bash scripts/train/train_smpl_dab_roi_pool_3d_refine.sh
```

输入：

```text
outputs/train/smpl_dab_joint_refine_20q/checkpoint_latest.pt
```

输出：

```text
outputs/train/smpl_dab_roi_pool_3d_refine_20q/checkpoint_latest.pt
```

作用：

```text
在 box prior 基础上增加 patch pool，给每个 SMPL query 汇聚对应人体局部 patch token。
这是 HSI 之前的当前最佳基础人体模型。
```

关键设置：

```text
EXTRA_EPOCHS=40  # 额外训练轮数；通常从 60 到 100
LR=1e-5
model.train_smpl_box_prior_embed=true
model.train_smpl_patch_pool_embed=true
model.smpl_query_patch_pool=true
model.smpl_query_patch_pool_expand=0.12
loss.pose_weight=6.0
loss.betas_weight=0.8
loss.transl_cam_weight=2.0
loss.joints3d_weight=12.0
loss.projected_joints2d_weight=0.25
```

验证脚本：

```bash
bash scripts/eval/check_smpl_dab_roi_pool_3d_refine.sh
```

这个阶段的输出是 HSI 的初始化 checkpoint。

### Stage 6: HSI GRAFT-style Refine

脚本：

```bash
bash scripts/train/train_smpl_hsi_refine.sh
```

输入：

```text
outputs/train/smpl_dab_roi_pool_3d_refine_20q/checkpoint_latest.pt
```

输出：

```text
outputs/train/smpl_hsi_refine_20q/checkpoint_latest.pt
```

作用：

```text
启用 HSIRefinementHead，用 24 个人体-场景交互 tokens 做 refinement。
同时输出 HSI refined SMPL 参数和 scene affine depth:

depth_hsi = hsi_scene_scale * depth_vggt + hsi_scene_depth_bias
```

关键设置：

```text
EXTRA_EPOCHS=40  # 额外训练轮数；通常从 100 到 140
LR=5e-6
data.require_depth=true
model.enable_depth=true
model.enable_hsi_refine=true
model.freeze_dense_head=true
model.hsi_hidden_dim=512
model.hsi_num_layers=5
model.hsi_num_heads=8
model.hsi_num_iters=3
model.hsi_scene_window=3
loss.hsi_pose_weight=6.0
loss.hsi_betas_weight=0.8
loss.hsi_transl_cam_weight=3.0
loss.hsi_joints3d_weight=16.0
loss.hsi_projected_joints2d_weight=0.35
loss.hsi_depth_teacher_weight=0.20
loss.hsi_anchor_depth_weight=0.10
loss.hsi_contact_weight=0.05
```

当前重要经验：

```text
小数据集上 checkpoint_epoch_0121.pt 有效。
checkpoint_epoch_0122.pt 已经发生 HSI scene scale 饱和。
因此 HSI 阶段不能只看 checkpoint_latest.pt，必须保存并选择 best。
```

当前有效评估：

```bash
bash scripts/eval/eval_smpl_hsi_refine_0121_metrics.sh
```

当前可视化：

```bash
SMPL_CKPT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_refine_20q/checkpoint_epoch_0121.pt \
OUTPUT_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/smpl_hsi_refine_epoch0121_gt_prior_aligned \
bash scripts/vis/vis_smpl_hsi_refine_vggt_camera.sh
```

## 当前不推荐作为主线的分支

这些脚本曾用于探索，但不是当前通往 HSI 的推荐大训练主线：

| 脚本 | 状态 | 原因 |
|---|---|---|
| `scripts/train/train_smpl_bbox_refine.sh` | 探索分支 | 从 projected bbox 继续做 bbox refine，但当前主线已转向 DAB box prior |
| `scripts/train/train_smpl_gt_box_prior.sh` | 探索分支 | 与 DAB box prior 方向重叠，当前主线使用 `train_smpl_dab_box_prior.sh` |
| `scripts/train/train_smpl_dab_smpl_refine.sh` | 探索分支 | 从 DAB box prior 做 SMPL refine，但当前更优路线是 `joint_refine -> roi_pool_3d_refine` |
| `scripts/train/resume_smpl_hsi_refine_latest_20ep.sh` | 谨慎使用 | 曾从 `0121` 继续训练到 `0122+`，触发 HSI scale 饱和；大训练前需要改造后再用 |

## 冻结和训练模块

### 非 HSI 阶段

通常冻结：

```text
aggregator 主体
camera_head
```

允许训练：

```text
smpl_head
aggregator.smpl_query_token
后期的 smpl_box_prior_embed
后期的 smpl_patch_pool_embed
```

关键配置：

```yaml
model.freeze_aggregator: true
model.freeze_camera_head: true
model.train_smpl_query_token: true
model.train_smpl_box_prior_embed: true      # ROI/HSI 阶段
model.train_smpl_patch_pool_embed: true     # ROI/HSI 阶段
```

### HSI 阶段

冻结：

```text
aggregator 主体
camera_head
dense_head
```

允许训练：

```text
SMPL head 后段
SMPL query token
box prior embedding
patch pool embedding
HSIRefinementHead
```

关键配置：

```yaml
model.freeze_aggregator: true
model.freeze_camera_head: true
model.freeze_dense_head: true
model.enable_hsi_refine: true
```

注意：当前训练代码不会保存/恢复 DataLoader RNG 和 sampler 状态。中断后继续训练不是严格 bitwise continuation，尤其 HSI 阶段可能改变训练轨迹。

## 验证与验收

### ROI-Pool 3D Refine 验收

执行：

```bash
bash scripts/eval/check_smpl_dab_roi_pool_3d_refine.sh
```

检查：

```text
outputs/eval/smpl_dab_roi_pool_3d_refine_20q_gt_prior/smpl_box_metrics.json
outputs/eval/smpl_dab_roi_pool_3d_refine_20q_noisy_prior_c005_s010/smpl_box_metrics.json
outputs/vis/smpl_dab_roi_pool_3d_refine_20q_gt_prior
outputs/vis/smpl_dab_roi_pool_3d_refine_20q_noisy_prior_c005_s010
```

关注：

```text
clean prior 下预测人数是否匹配 GT
noisy prior 下是否仍稳定
3D joints / projected joints 是否合理
SMPL mesh 是否和 GT mesh 接近
```

### HSI 验收

执行：

```bash
bash scripts/eval/eval_smpl_hsi_refine_0121_metrics.sh
```

关注：

```text
base_joints_mpjpe_m vs hsi_joints_mpjpe_m
base_vertices_pve_m vs hsi_vertices_pve_m
base_transl_l2_m vs hsi_transl_l2_m
base_projected_joints_l2_px vs hsi_projected_joints_l2_px
raw_depth_l1_median_m vs hsi_depth_l1_median_m
hsi_scene_scale mean/min/max/median
```

可接受的 HSI scale 范围应远离当前上限：

```text
健康示例: scale median 约 7.8，max 约 11.3
危险示例: scale 接近 20.0855
```

如果出现：

```text
hsi_scene_scale ≈ 20.0855
```

说明 scale head 已饱和，不能继续使用该 checkpoint。

诊断 `0121 -> 0122` 跳变：

```bash
bash scripts/diagnostics/compare_hsi_ckpt_0121_0122.sh
```

该诊断已确认：

```text
aggregator.frozen delta_norm=0
camera_head.frozen delta_norm=0
dense_head.frozen delta_norm=0
```

所以坏结果不是 VGGT/camera/dense 被误训练，而是 HSI scene affine 分支输出饱和。

## 大数据集训练建议

切换大数据集前建议做以下调整。

### 1. 保留完整主线但加入 validation

当前探索脚本普遍覆盖了：

```bash
--override "data.val_split="
```

这会关闭 validation。大训练时建议恢复验证集，例如：

```bash
--override "data.val_split=Test"
```

或准备单独 validation split。不要只依赖 `checkpoint_latest.pt`。

### 2. 给 HSI 阶段加 best checkpoint 选择

当前 HSI 不能只按 epoch 结束。至少要跟踪：

```text
hsi_joints_mpjpe_m
hsi_vertices_pve_m
hsi_transl_l2_m
hsi_projected_joints_l2_px
hsi_depth_l1_median_m
hsi_scene_scale max/median
```

保存 best 的原则：

```text
人体指标下降
depth median 指标下降
hsi_scene_scale 不接近 20.0855
```

### 3. HSI 继续训练要保护 scale/bias

当前小数据集上 `0121` 好、`0122` 坏，说明 HSI scene affine 分支非常敏感。大训练前建议修改：

```text
加入 loss_hsi_scale_prior
限制 hsi_scene_scale 上限或改为更温和参数化
限制 hsi_scene_depth_bias 范围
可选：后期冻结 scale_delta / bias_delta，只继续训练人体 refinement
```

### 4. 可视化导出要过滤 depth outlier

当前 VGGT raw depth 有远处 outlier。坏 HSI scale 会把场景撑到上万米，导致 PLY 里人“消失”。可视化导出时建议增加：

```text
depth percentile clipping
最大深度阈值
场景 bbox 统计
```

这不改变训练，只避免可视化被少量极端点破坏。

### 5. 保存策略

当前 `save_interval=1`，大数据集上 checkpoint 很大。建议：

```text
保留 checkpoint_latest.pt
保留 checkpoint_best.pt
按间隔保留 checkpoint_epoch_xxxx.pt
定期删除非 best 的旧 checkpoint
```

注意：磁盘满会导致 `torch.save` 写坏 checkpoint。恢复训练前应先验证 checkpoint 可读。

## 一键执行顺序

从零训练当前 HSI 主线：

```bash
bash scripts/train/train_smpl_hungarian.sh
bash scripts/train/train_smpl_projected_bbox.sh
bash scripts/train/train_smpl_conf_quality.sh
bash scripts/train/train_smpl_dab_box_prior.sh
bash scripts/train/train_smpl_dab_joint_refine.sh
bash scripts/train/train_smpl_dab_roi_pool_3d_refine.sh
bash scripts/train/train_smpl_hsi_refine.sh
```

阶段性检查：

```bash
bash scripts/eval/check_smpl_dab_joint_refine.sh
bash scripts/eval/check_smpl_dab_roi_pool_3d_refine.sh
bash scripts/eval/eval_smpl_hsi_refine_0121_metrics.sh
```

HSI 可视化：

```bash
SMPL_CKPT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_refine_20q/checkpoint_epoch_0121.pt \
OUTPUT_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/smpl_hsi_refine_epoch0121_gt_prior_aligned \
bash scripts/vis/vis_smpl_hsi_refine_vggt_camera.sh
```

## 当前风险清单

1. `checkpoint_latest.pt` 不一定是最佳 checkpoint。当前 HSI latest 已经可能坏于 `0121`。
2. HSI scale head 会饱和到 `exp(3)=20.0855`。
3. 训练代码不恢复 RNG/sampler 状态，中断续训不是严格连续。
4. 大数据集上不能关闭 validation。
5. depth mean 指标容易被 VGGT outlier 主导，优先看 median/percentile。
6. 当前训练全部 batch size 为 1；大数据集上如果调整 batch size，需要重新检查 lr 和 loss scale。
7. 早期脚本里路径硬编码较多，换机器/数据集前应统一检查 `REPO_ROOT`、`BEDLAM_ROOT`、`PREPROCESSED_ROOT`、`VGGT_CKPT`、`SMPL_MODEL_DIR`。

## 推荐的大训练改造优先级

在直接大训练前，建议先做三个工程改造：

1. 给 HSI eval 增加 best checkpoint 自动选择。
2. 给 HSI scale/bias 增加稳定约束，避免 `0122` 式饱和。
3. 把数据路径和 epoch/lr/max_samples 等参数从脚本顶部统一环境变量化，避免每次手改脚本。

这三项完成后，再把小数据集换成大数据集跑完整链路。
