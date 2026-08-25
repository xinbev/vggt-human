# VGGT-Omega Current Pipeline Status

本文档把当前仓库中分散在主线、人物识别/ID 跟踪支线、SMPL translation 支线里的信息收束成一张总图。目标是让不同 agent 后续继续工作时，不再只依赖各自对话上下文。

更新时间：2026-06-24

## 一句话总览

当前推荐主线已经从旧的 `outputs/train/smpl_hsi_refine_20q/checkpoint_epoch_0121.pt` 前进到 translation-repaired HSI：

```text
outputs/train/smpl_hsi_after_translation_ray_refine/checkpoint_latest.pt
```

base SMPL translation 的平均问题已被 camera-ray translation refiner 明显改善，并合入 HSI 主线；但仍有长尾单帧 root/translation 错位。人物检测/ID 跟踪线已经完成 YOLO + BoostTrack + sidecar + clip tensor 接入，主体可用，但短轨、gap、真实 GT-ID 指标仍是喂给 HSI temporal memory 前必须控制的风险。

## 当前主流程图

```text
BEDLAM / video frames
  -> person boxes
     A. GT/preprocessed BEDLAM boxes for controlled training/eval
     B. YOLO TorchScript + BoostTrack sidecar for real video/person ID
  -> VGGTOmega
     Aggregator
       -> CameraHead: pose_enc / camera geometry
       -> DenseHead: depth / depth_conf
       -> SMPLHead: base pose, betas, pred_transl_cam, boxes, conf
          -> optional CameraRayTranslationRefiner
          -> optional TemporalTranslationRefiner
       -> HSIRefinementHead
          -> hsi_refined pose, betas, transl
          -> hsi_scene_scale / hsi_scene_depth_bias
          -> optional temporal momentum keyed by person_id
  -> eval / diagnostics / PLY / video visualization
```

核心代码入口：

```text
vggt_omega/models/vggt_omega.py
vggt_omega/models/heads/smpl_head.py
vggt_omega/models/heads/hsi_refinement_head.py
vggt_omega/tracking/
vggt_omega/training/hungarian_losses.py
scripts/train/train_smpl.py
```

## 0. Baseline 和训练链路

### 已落地

原始小数据集主线已经跑通过：

```text
Stage 0: train_smpl_hungarian.sh
Stage 1: train_smpl_projected_bbox.sh
Stage 2: train_smpl_conf_quality.sh
Stage 3: train_smpl_dab_box_prior.sh
Stage 4: train_smpl_dab_joint_refine.sh
Stage 5: train_smpl_dab_roi_pool_3d_refine.sh
Stage 6: train_smpl_hsi_refine.sh
```

旧有效 HSI checkpoint：

```text
outputs/train/smpl_hsi_refine_20q/checkpoint_epoch_0121.pt
```

旧 HSI `0121` 在 200 样本 GT box prior 下有效：

```text
3D joints MPJPE: 0.07453m -> 0.06160m
Vertices PVE:    0.07888m -> 0.06732m
Translation L2:  0.06544m -> 0.04886m
Projected:       6.74px   -> 4.93px
Depth median L1: 5.826m   -> 0.259m
```

### 风险

`checkpoint_epoch_0122.pt` 之后出现 HSI scene scale 饱和：

```text
hsi_scene_scale ~= exp(3) = 20.0855
```

因此不能盲用 `checkpoint_latest.pt`，尤其是旧 HSI 分支。大训练前必须引入 validation/best checkpoint 选择，并监控 `hsi_scene_scale`。

## 1. 人物检测与跨帧 ID 跟踪支线

### 目标

补齐真实视频输入的 observation frontend：

```text
video / frame dir
  -> YOLO TorchScript person detector
  -> BoostTrack++ online tracking
  -> optional SAM2 masks
  -> sidecar
  -> smpl_query_boxes / smpl_track_ids
```

它不替换 VGGT baseline，也不改 `.reference/`。

### 已落地文件

```text
vggt_omega/tracking/schema.py
vggt_omega/tracking/detectors.py
vggt_omega/tracking/boosttrack_adapter.py
vggt_omega/tracking/postprocess.py
vggt_omega/tracking/clip_builder.py
vggt_omega/tracking/diagnostics.py
vggt_omega/tracking/track_memory.py
vggt_omega/tracking/sam2_masks.py

scripts/preprocess/prepare_video_person_tracks.py
scripts/preprocess/prepare_video_person_tracks.sh
scripts/preprocess/test_bedlam_person_tracks.sh
scripts/vis/visualize_video_person_tracks.py
scripts/eval/evaluate_bedlam_person_tracking.py
scripts/eval/eval_bedlam_person_tracking.sh
```

### sidecar 到模型输入

`build_clip_tensors_from_sidecar(...)` 输出：

```text
smpl_query_boxes      [1, S, Q, 4]
smpl_query_boxes_mask [1, S, Q]
smpl_track_ids        [1, S, Q]
smpl_track_mask       [1, S, Q]
```

模型入口已经支持：

```python
model(
    images,
    smpl_query_boxes=boxes,
    smpl_query_boxes_mask=box_mask,
    smpl_track_ids=track_ids,
    smpl_track_mask=track_mask,
)
```

### 当前本地 sidecar 统计

基于 `outputs/preprocess/video_tracks/Training/*/summary.json`：

```text
序列数: 101
总帧数: 3077
YOLO 检测框: 9358
最终 track observation: 8733
平均检测人数/帧: 3.04
平均跟踪人数/帧: 2.84
平均 track confidence: 0.988

总 track 数: 380
平均每序列 track 数: 3.76
中位数: 4
最少: 3
最多: 6

短轨 <= 5 帧: 49 / 380 = 12.9%
有 gap 的序列: 68 / 101
总 gap 数: 116
发生 stitching 的序列: 20
总 stitching 次数: 25
被改写 observation 数: 774
```

### 当前结论

主体 tracking 可用，stitching 确实修复了一部分遮挡断 ID。但不能把当前 sidecar 直接视作“身份全可靠”。本地没有看到：

```text
outputs/eval/bedlam_person_tracking/bedlam_person_tracking_eval.json
```

所以 GT-ID 指标还需要服务器侧确认。

### 分支方案

| 方案 | 状态 | 用途 | 风险 |
|---|---|---|---|
| YOLO + BoostTrack + stitching | 已实现 | 当前默认视频 observation frontend | 短轨、gap、ID switch 仍可能污染 HSI memory |
| SAM2 mask | 已接入但默认不用 | 后续可做遮挡/人体区域辅助 | 增加依赖和算力，当前不是主阻塞 |
| HSITrackMemory feedback | 留接口 | 用 HSI refined transl/betas 辅助后续 ID 合并 | 必须只在高置信 track 上用，避免错误身份越滚越大 |
| DWPose/keypoint confidence | 未接入 | 可补充人体可见性/局部质量 | 需要新依赖和验证，不应抢主线优先级 |

### 给模型侧的硬约束

```text
1. HSI 不负责发现 ID。
2. HSI temporal memory 的 key 应优先用 person_id / smpl_track_ids。
3. 不要假设 query slot index == person identity。
4. track_mask=False 的帧不要更新 memory。
5. 短轨、低置信、长 gap 恢复后的 track 应 mask 或降权。
```

## 2. Base SMPL Translation 支线

### 问题来源

早期坏帧诊断显示，有些人最早的 single-frame SMPL root/translation 已经错了。HSI 从错误 anchor 周围取场景上下文，很难完全救回来。

失败过的路线：

```text
outputs/train/smpl_geo_translation_depth_residual
```

典型症状：

```text
smpl_geo_gate_mean ~= 1e-13
smpl_geo_residual_l1 ~= 1e-6
anchor_depth_l1 ~= 3m
```

结论：不要重复未对齐 raw depth anchor + gate residual 的设计。raw VGGT depth 不能直接当 base SMPL root-depth teacher。

### 已成功分支：CameraRayTranslationRefiner

实现位置：

```text
vggt_omega/models/heads/smpl_head.py
```

核心思想：

```text
bbox center + VGGT camera intrinsics -> camera ray + tangent basis
base pred_transl_cam -> ray depth + tangent offsets
network predicts bounded ray/tangent/log-depth residuals
refined translation writes back to pred_transl_cam
pre-refiner value is preserved as base_pred_transl_cam
```

关键边界：

```text
监督目标是 dataset SMPL camera translation: gt_transl_cam
不使用 raw VGGT depth Z 作为 base translation 硬监督
```

训练链路：

```text
T0: eval original 0121
T1: train only camera-ray refiner
T2: train transl_cam_heads + refiner
T3: merge translation keys back into full HSI checkpoint
```

主脚本：

```text
scripts/train/train_smpl_translation_ray_refine_full_from0121.sh
scripts/diagnostics/merge_translation_refiner_into_hsi.py
scripts/train/train_smpl_hsi_after_translation_ray_refine.sh
```

### 已验证指标

base translation repair, 200 samples, GT box prior：

| Stage | transl L2 | Z L1 | XY L2 | MPJPE | PVE |
|---|---:|---:|---:|---:|---:|
| T0 original 0121 base | 0.06544 | 0.04406 | 0.04094 | 0.07453 | 0.07888 |
| T1 ray refiner only | 0.05205 | 0.03298 | 0.03452 | 0.06287 | 0.06765 |
| T2 transl heads + refiner | 0.04956 | 0.03117 | 0.03333 | 0.06184 | 0.06698 |

T2 相对 T0：

```text
transl L2: 65.44mm -> 49.56mm, +24.26%
Z L1:      44.06mm -> 31.17mm, +29.27%
XY L2:     40.94mm -> 33.33mm, +18.59%
MPJPE:     74.53mm -> 61.84mm, +17.03%
PVE:       78.88mm -> 66.98mm, +15.08%
```

### 主线决策

这个分支已经 closeout 成功。除非后续非 GT box 或 temporal eval 暴露新失败，不要继续把“平均 base translation 架构”当当前主阻塞。

推荐主线 checkpoint：

```text
outputs/train/smpl_hsi_after_translation_ray_refine/checkpoint_latest.pt
```

加载该 checkpoint 必须启用：

```yaml
model:
  smpl_enable_translation_refine: true
  smpl_translation_refine_max_ray_delta_m: 1.20
  smpl_translation_refine_max_tangent_delta_m: 0.60
  smpl_translation_refine_max_log_depth_delta: 0.85
  smpl_translation_refine_max_box_prior_weight: 1.00
```

注意：T2-derived checkpoint 下最终 translation 是 `pred_transl_cam`，不要把 `base_pred_transl_cam` 当最终结果。

## 3. HSI 单帧主线

### 新主线 HSI 结果

`outputs/eval/smpl_hsi_after_translation_ray_refine/hsi_refine_metrics.json`：

```text
checkpoint: outputs/train/smpl_hsi_after_translation_ray_refine/checkpoint_latest.pt
num_samples: 200
use_gt_box_prior: true

base_joints_mpjpe_m: 0.06184
hsi_joints_mpjpe_m:  0.04879

base_vertices_pve_m: 0.06698
hsi_vertices_pve_m:  0.05373

base_transl_l2_m: 0.04956
hsi_transl_l2_m:  0.03645

base_projected_joints_l2_px: 5.94315
hsi_projected_joints_l2_px:  4.40295

raw_depth_l1_median_m: 5.82600
hsi_depth_l1_median_m: 0.22121

hsi_scene_scale median: 7.743
hsi_scene_scale min/max: 6.698 / 11.156
hsi_worse_than_base_ratio_2cm: 0.0143
```

相对 repaired base，HSI 继续提升：

```text
MPJPE:       +21.10%
PVE:         +19.79%
Translation: +26.45%
Projected:   +25.92%
Depth median:+96.20%
```

相对旧 0121 HSI，文档中记录的新 HSI 提升：

```text
MPJPE:       0.06160m -> 0.04879m
PVE:         0.06732m -> 0.05373m
Translation: 0.04886m -> 0.03645m
Projected:   4.92777px -> 4.40295px
Depth median L1: 0.25903m -> 0.22121m
```

### 当前结论

HSI 仍然有效，且在 translation repair 后更强。当前 HSI 单帧主线不是“坏掉”，但它主要在 GT box prior 条件下验证。下一步必须验证：

```text
1. non-GT box / noisy box prior robustness
2. temporal/video stability
3. long-tail bad frames
```

## 4. HSI temporal / scene scale 主线

### 已落地能力

代码入口：

```text
vggt_omega/models/heads/hsi_refinement_head.py
vggt_omega/utils/hsi_affine.py
vggt_omega/training/hungarian_losses.py
```

已有设计：

```text
hsi_enable_temporal_momentum
hsi_temporal_momentum_decay
hsi_temporal_momentum_use_track_ids
hsi_scene_affine_mode: per_frame / clip_median / EMA-like modes
temporal velocity / acceleration / no-worse losses
```

主推 after-translation-ray-refine temporal 脚本：

```text
scripts/train/train_smpl_hsi_scene_then_temporal_noworse_after_translation_ray_refine.sh
scripts/eval/eval_smpl_hsi_scene_then_temporal_noworse_after_translation_ray_refine.sh
scripts/vis/vis_smpl_hsi_scene_then_temporal_noworse_after_translation_ray_refine_clip.sh
```

### 当前 temporal eval

`outputs/eval/hsi_temporal_after_translation_ray_refine_noworse/hsi_temporal_metrics.json`：

```text
checkpoint: outputs/train/smpl_hsi_temporal_after_translation_ray_refine/stage2_human_momentum_no_worse/checkpoint_latest.pt
num_sequences: 64
use_gt_box_prior: true

base_transl_velocity_l1_m: 0.03251
hsi_transl_velocity_l1_m:  0.02489

base_joints_velocity_l1_m: 0.03850
hsi_joints_velocity_l1_m:  0.03200

base_joints_acceleration_l1_m: 0.05402
hsi_joints_acceleration_l1_m:  0.04660

base_joints_jerk_l1_m: 0.09532
hsi_joints_jerk_l1_m:  0.08137

hsi_scene_scale_range: 0
hsi_scene_scale_saturation_rate: 0
hsi_scene_bias_range_m: 0
```

temporal 指标改善：

```text
translation velocity: +23.44%
joints velocity:      +16.89%
joints acceleration:  +13.74%
joints jerk:          +14.63%
```

### 当前结论

clip/sequence scene affine 稳定路线有效，`clip_median` 可以压住 scene scale/bias 抖动。temporal no-worse 分支能改善速度/加速度/jerk，但仍需要真实 tracking IDs 下的验证；当前 eval 里：

```text
track_explicit_per_frame = 0
track_person_index_per_frame = 2.99
```

说明这次主要还是用 GT/person-index 式轨迹对齐，不等价于真实视频 tracker ID。

### 分支方案

| 方案 | 状态 | 解决问题 | 风险 |
|---|---|---|---|
| loss-only temporal velocity/acceleration | 已落地 | 人物闪动 | 可能过平滑，需 no-worse guard |
| clip_median scene affine | 已验证 | 环境尺度跳变 | 只解决 scale/bias 输出稳定，不解决人根位置长尾 |
| HSI temporal momentum | 已实现 | 跨帧人体状态连续 | 依赖可靠 person_id；错 ID 会污染 memory |
| UniSH-style global scale token | 仅设计参考 | 更根本的 clip/global scale | 改动大，暂不优先 |
| DuoMo/HTD-style high-order dynamics | 概念已吸收 | velocity/acc/jerk | 不应引入完整 diffusion pipeline |
| contact/foot sliding | 有探索 | 接触和脚滑 | 不能替代 root translation 修复 |

## 5. 当前长尾 translation 问题

### 问题定义

camera-ray refiner 解决了平均质量，但仍存在少量 post-refiner 单帧 root/translation 大错。HSI 如果围绕错误 SMPL anchor 取场景上下文，会跟着错或只能部分纠正。

注意：在 after-translation-ray-refine 可视化中：

```text
pred_mesh_top03 使用 predictions["pred_transl_cam"]
如果 smpl_enable_translation_refine=true，这已经是 post-refiner translation
pre-refiner 值是 predictions["base_pred_transl_cam"]，当前 PLY 默认不单独导出
```

### 数据集级 scan

`outputs/eval/translation_good_bad_after_translation_ray_refine/translation_good_bad_summary.json`：

```text
deduped_frame_person_rows: 21852
num_frames: 7619
num_sequences: 250
bad threshold: 0.50m
severe threshold: 0.80m
focus frames: seq_000000_0085, seq_000000_0100
```

post-refiner base：

```text
mean: 0.0730m
p50:  0.0664m
p90:  0.1222m
p95:  0.1435m
p99:  0.2014m
max:  1.0267m
person >0.50m: 5
person >0.80m: 3
bad frames >0.50m: 5 / 7619
```

HSI refined in this scan：

```text
mean: 0.0914m
p50:  0.0817m
p90:  0.1586m
p95:  0.1846m
p99:  0.2431m
max:  1.0797m
person >0.50m: 5
person >0.80m: 2
```

paired comparison：

```text
hsi_better_person_ratio: 38.37%
hsi_worse_by_more_than_5cm_person_ratio: 23.68%
person_rescued_bad_to_good_count: 1
person_newly_bad_count: 1
person_both_bad_count: 4
```

这个 scan 和 200-sample HSI eval 口径不同，但它给出一个重要信号：HSI 不是一个可靠的“长尾 translation 后处理器”。长尾分支应优先让 post-refiner base 本身更稳。

### 当前最坏例

top bad base rows 中包含：

```text
seq_000032_0165: base 1.0267m, hsi 1.0797m
seq_000209_0140: base 0.8766m, hsi 0.9189m
seq_000018_0080: base 0.8566m, hsi 0.7691m
seq_000206_0080: base 0.6014m, hsi 0.6643m
seq_000209_0135: base 0.5171m, hsi 0.3370m
```

用户人工看过的 `seq_000000_0085` 在 scan 里不是极端坏例，但视觉上暴露了“环境/scene aligned 好，person root 不贴 GT”的诊断需求。后续需要用同一帧同时导出：

```text
pre-refiner base_pred_transl_cam mesh
post-refiner pred_transl_cam mesh
hsi_refined_pred_transl_cam mesh
GT mesh
HSI aligned scene
per-person translation JSON
```

### 长尾分支候选方案

| 方案 | 优先级 | 说明 |
|---|---:|---|
| 导出 pre/post/hsi/gt 同帧 PLY + per-person JSON | P0 | 先确认 refiner 在坏例上是没动、动错、还是动不够 |
| hard-frame mining / top-k tail loss | P0 | 目标是降 p95/p99 和 >0.5m 计数，不只降 mean |
| quantile / p90-p95 penalty | P1 | 防止平均指标掩盖极端错位 |
| 加强 ray/bbox geometry consistency | P1 | 继续沿当前成功方向，不重新押 raw depth |
| multi-hypothesis translation candidates | P2 | 对歧义帧预测多个 ray-depth/tangent 候选，再监督/选择 |
| aligned scene/depth as cue, not teacher | P2 | 可用 HSI-aligned human ROI depth 作为弱 cue，不用 raw depth 硬锚 |
| contact/foot constraints | P3 | 只适合作微调，不解决全局 root 错位 |

验收指标必须包含：

```text
mean/median transl_l2 不退化
p90/p95/p99 transl_l2 下降
>0.50m 和 >0.80m frame/person count 下降
known frames 可视化改善
HSI downstream 不明显退化
```

## 6. Translation head 设计调研结论

参考过的项目/思想：

```text
MetricHMSR / MetricHMR
GVHMR
PromptHMR
NLF / MNLF
Human3D
SAT-HMR-smpl
```

横向结论：几乎都不推荐“裸回归无几何约束的 xyz trans”。

可选设计范式：

| 设计 | 代表 | 适合场景 | 当前状态 |
|---|---|---|---|
| direct metric translation + ray/crop/intrinsics cue | MetricHMSR | 有可靠 metric GT 和 camera/crop 几何 | 参考思想 |
| pred_cam -> camera transl decode | GVHMR | 单帧 crop/HMR 风格 | 可作为后续轻量对照分支 |
| 2D offset + depth -> transl decode | PromptHMR | 显式拆横向和深度 | 可作为后续轻量对照分支 |
| velocity + rollout | GVHMR | 视频/global trajectory | 已部分吸收到 temporal branch |
| init_trans + delta_trans residual | PromptHMR/NLF scripts | 有粗初值再修正 | 当前 camera-ray refiner 属于此类成功实践 |

当前主线实际采用的是：

```text
camera ray basis + bounded residual refinement
```

未来若继续开 trans 结构分支，应保留 baseline switch，并优先做小对照，不要整体搬参考项目训练框架。

## 7. 分支之间的边界

### 主线 agent

负责：

```text
1. 以 outputs/train/smpl_hsi_after_translation_ray_refine/checkpoint_latest.pt 为新 baseline
2. 跑 noisy/non-GT box prior eval
3. 跑 temporal eval + video visualization
4. 维护 HSI scale/bias 稳定和 best checkpoint 选择
```

不应继续深挖：

```text
平均 base translation 架构
```

除非长尾/非 GT box 验证明确要求。

### 人物识别/ID 跟踪 agent

负责：

```text
1. 完成 BEDLAM GT-ID tracking eval
2. 给 sidecar 增加 track_quality / visible_ratio / gap statistics
3. 为 clip_builder 或模型输入生成质量 mask
4. 验证真实 video sidecar 下 HSI temporal memory 是否被错 ID 污染
```

不应：

```text
让 HSI 自己发现 ID
把 query slot 当身份
把短轨/低置信 track 写入长期 memory
```

### trans 长尾 agent

负责：

```text
1. 导出 pre-refiner / post-refiner / HSI / GT 对齐诊断
2. 从 scan 中定位 >0.5m / >0.8m 坏例
3. 设计 hard-mining 或 tail-aware training
4. 保持 camera-ray refiner 成功平均指标不退化
```

不应：

```text
重复 raw-depth residual gate route
只优化 mean transl_l2
用 contact 代替 root translation 修复
```

## 8. 当前最高优先级 To-Do

1. 补 tracking GT eval：

```text
RUN_TRACKER=0 bash scripts/eval/eval_bedlam_person_tracking.sh
```

检查：

```text
match_recall
match_precision
id_dominant_accuracy
id_switches_per_100_matched
fragmentations_per_100_matched
```

2. 做 noisy/non-GT box prior eval：

```text
目标：确认 translation-repaired HSI 不是只在 GT boxes 下有效。
```

3. 给 PLY 诊断补 pre-refiner mesh：

```text
predictions["base_pred_transl_cam"]
predictions["pred_transl_cam"]
predictions["hsi_refined_pred_transl_cam"]
batch["gt_transl_cam"]
```

4. 把长尾 scan 纳入 trans 分支验收：

```text
p95/p99
count_gt_0.50m
count_gt_0.80m
top bad frame list
```

5. 大训练前工程保护：

```text
validation split
checkpoint_best.pt
hsi_scene_scale saturation guard
checkpoint save cleanup / disk full protection
```

## 9. 判断一个新结果能否进入主线

必须同时回答：

```text
1. 改了哪个 VGGT/SMPL/HSI 组件？
2. 是否保留 baseline switch？
3. 是否使用 .reference/.paper 直接 import？如果是，应该拒绝。
4. 输入输出 shape、dtype、device 是否稳定？
5. GT box 和 noisy/non-GT box 是否都测过？
6. 单帧指标和 temporal 指标是否都不退化？
7. mean 之外，p95/p99 和坏例计数是否改善？
8. hsi_scene_scale 是否远离 exp(3)=20.0855？
9. 如果使用 track IDs，是否测过 ID switch/gap/短轨影响？
10. 是否能用脚本复现训练、评估、可视化？
```

## 10. 关联文档

```text
docs/smpl_hsi_full_training_chain.md
docs/base_smpl_translation_bad_frame_handoff.md
docs/smpl_translation_ray_refine_design.md
docs/base_smpl_translation_ray_refine_closeout_handoff.md
docs/hsi_mainline_after_translation_ray_refine.md
docs/base_smpl_translation_single_frame_long_tail_handoff.md
docs/video_hsi_temporal_scale_handoff.md
docs/video_person_tracking_frontend.md
docs/person_detection_tracking_design_handoff.md
docs/smpl_trans_design_survey.md
docs/crowd4d_cross_frame_id_tracking.md
```
