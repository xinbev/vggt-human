# Video HSI Temporal Stability And Scene Scale Handoff

本文档整理当前关于“从单帧训练扩展到视频输入”的 idea 沟通结论，供后续有完整代码上下文的对话继续设计和实现。

## 背景

当前项目主要围绕 VGGT-Omega 做人和环境重建改造。现有训练和 HSI refinement 虽然已经使用 `sequence_length` 形式的数据窗口，但主要监督和 refinement 仍偏单帧：每一帧的人体 pose、shape、translation 和 HSI scene affine 独立预测/约束。

实际项目输入应是视频，因此至少有两个需要控制的问题：

1. 人物帧间闪动：同一人物在相邻帧的姿态、位置、关节或接触状态发生不自然抖动。
2. 环境尺度突变：环境会被人物/HSI 校正一次，如果每帧 scene scale 自由变化，可能导致视频里环境突然变大或变小。

本轮只做论文和参考代码思路梳理，不要求立刻实现。

## 已确认的参考材料

本地论文和参考代码位于：

- `.paper/UniSH Unifying Scene and Human Reconstruction in a Feed-Forward Pass.pdf`
- `.paper/UniSH/`
- `.paper/DuoMo Dual Motion Diffusion for World-Space Human Reconstruction.pdf`
- `.paper/DuoMo/`
- `.paper/UniCon3R Unified Contact-aware 4D Human-Scene Reconstruction from Monocular Video.pdf`
- `.paper/Natural Human Motion Recovery by Aligning High-Order Temporal Dynamics from Monocular Videos.pdf`
- `.paper/MetricHMSR.pdf`
- `.paper/GRAFT/GRAFT.pdf`
- `.paper/Joint Optimization for 4D Human-Scene Reconstruction in the Wild.pdf`

当前主项目代码目录实际是 `vggt_omega/`。AGENTS 中提到 `vggt_human/`，但当前仓库可见代码在 `vggt_omega/`。

## 对两个问题的重新归类

### 问题 1：人物帧间闪动

本轮讨论里，以下思路都主要解决问题 1：

1. DuoMo / HTD-Refine 的 temporal loss 思路。
2. UniCon3R 的 HSI temporal/contact state 思路。

它们可以拆成两个阶段。

### 问题 2：环境尺度突变

UniSH 的 global/clip scale 思路可以解决问题 2，但和当前代码出入较大，暂时不建议第一步就改成完整 UniSH-style global scale head。

更保守的第一版处理是：

1. 先保留当前 HSI head 输出每帧 `hsi_scene_scale`。
2. 对一个序列内的所有 scale 做 robust aggregation。
3. 用聚合后的稳定 scale 代表整段视频或局部 clip。
4. 或者先只加 `log(scale)` 的 temporal smooth/prior loss。

## 论文思路对应关系

### DuoMo：适合借鉴 temporal motion loss/metric，不建议直接整体搬 diffusion

相关代码：

- `.paper/DuoMo/src/models/duomo_diffusion.py`
- `.paper/DuoMo/src/trainers/diffusion_trainer.py`
- `.paper/DuoMo/src/utils/eval_utils.py`
- `.paper/DuoMo/src/data/camera_motion.py`

可借鉴点：

- 把 world/root motion 表达为 velocity，而不是直接学习长序列 absolute root translation。
- 使用 root velocity、joint/mesh velocity、acceleration、jitter、foot skating/contact velocity 评价或约束人体视频稳定性。
- DuoMo 的 world-space model 使用 `lo6d_vel_1`，训练里包含 `vel_w_loss` 和 `foot_w_loss`。
- `compute_jitter` 使用有限差分的 jerk 评估 motion smoothness。

建议在当前项目中的轻量落地：

- 增加 `loss_hsi_transl_velocity`。
- 增加 `loss_hsi_joints_velocity`。
- 增加 `loss_hsi_joints_acceleration`。
- 可选增加 `loss_hsi_jitter` 或仅作为评估 metric。
- 可选增加 contact frame 下 foot sliding/foot velocity loss。

不建议第一步做的事情：

- 不建议直接引入完整 DuoMo diffusion pipeline。
- 不建议引入 DuoMo 的训练框架、依赖、数据预处理或 checkpoint 体系。

### HTD-Refine：适合转成高阶时序约束

可借鉴点：

- 论文强调 velocity、acceleration、jerk 是恢复自然人体运动的重要高阶时序信息。
- 它把这些高阶动态作为 soft constraints，用于优化 world-space trajectory。

建议在当前项目中的轻量落地：

- 第一阶段只把 HTD 思路转为 supervised temporal loss，不做额外 PVA-Net。
- 如果 batch 中已有 GT joints/translation，可以直接约束预测和 GT 的一阶/二阶差分。
- 如果 GT 不稳定或缺失，可以先做 prediction smoothness loss，但权重要小，避免过度平滑。

### UniCon3R：适合改 HSI head，引入 temporal/contact memory

可借鉴点：

- UniCon3R 使用 Temporal Momentum：上一帧的 contact token/status 被带入当前帧。
- 使用 scene-aware contact prompt、explicit metric geometry 和 contact-guided latent refinement。
- 目标是让人体和场景接触状态在时间上连续，降低 foot sliding 和 jitter。

与当前项目的关系：

- 当前 `HSIRefinementHead` 已经有 GRAFT-style HSI tokens 和 `hsi_contact_logits`。
- 但当前 HSI refinement 基本是按帧 flatten 后做，没有显式 previous-frame HSI/contact state。

建议落地方式：

- 第二阶段再改 HSI head。
- 给 HSI refinement 增加可选 temporal memory。
- 当前帧 token 可以 cross-attend 或 fuse 上一帧同一 person/query 的 HSI/contact token。
- 需要依赖 person identity 或匹配关系，当前数据里已有 `person_ids` / `person_id_mask`，但 query matching 仍要谨慎。

### UniSH：适合控制 scene scale，但第一阶段暂缓大改

相关代码：

- `.paper/UniSH/unish/heads/align_net.py`
- `.paper/UniSH/unish/pipeline.py`

可借鉴点：

- UniSH 的 `AlignNet` 使用一个 `scale_token` 预测 clip/global scale。
- 同时逐帧预测 `trans_cam`。
- 它不是每帧独立自由预测 scene scale，因此天然能避免环境尺度随帧跳变。
- `scale_predictions` 会把同一个 scale 应用到 `world_points`、`local_points`、camera translation 和 SMPL translation。

当前判断：

- 这个思路解决问题 2，但和当前代码结构差别较大。
- 当前项目里 `hsi_scene_scale` 是 HSI refinement head 从 per-query/per-frame 输出聚合而来。
- 第一阶段不建议直接重写成 UniSH-style global scale head。

保守替代方案：

- 对序列内已有 `hsi_scene_scale[t]` 做 robust aggregation。
- 推荐在 log 空间聚合：

```text
log_scale_seq = median(log(scale[t]))
scale_seq = exp(log_scale_seq)
```

- 如果需要考虑置信度，可以使用 confidence-weighted trimmed mean / median。
- 对异常 scale 需要过滤，尤其是当前已知饱和值 `20.0855`。

### MetricHMSR / GRAFT / JOSH

MetricHMSR：

- 可借鉴 spatially varying affine depth field 的思想。
- 它强调 affine depth correction 应该 smooth/coherent，不应让尺度任意跳动。
- 对当前项目的启发是：scale/bias 需要 prior、smooth 和范围约束。

GRAFT：

- 当前 HSI head 已经有 GRAFT-style 24 HSI tokens 和 iterative refinement 的味道。
- GRAFT 思路主要用于人-场景几何一致性，不代表 scene scale 可以逐帧自由变。

JOSH：

- 是 offline joint optimization 思路，包含 human motion smoothness、contact、scene/camera/human joint optimization。
- 适合作为 teacher 或诊断上限，不适合作为第一阶段工程集成目标。

## 当前项目中的关键代码锚点

### HSI head

文件：

- `vggt_omega/models/heads/hsi_refinement_head.py`

当前关键逻辑：

```text
per_query_log_scale = scale_delta(...)
log_scale = confidence-weighted aggregation over queries
hsi_scene_scale = exp(log_scale.clamp(-3, 3))
```

风险：

- `exp(3) = 20.0855`，文档中已经记录 `checkpoint_epoch_0122.pt` 出现这个饱和值。
- 当前 scale 是按帧输出，容易在视频中引入帧间 scene scale 跳变。

### HSI loss

文件：

- `vggt_omega/training/hungarian_losses.py`

已有相关能力：

- HSI refined pose/betas/transl/joints/vertices loss。
- HSI depth teacher。
- HSI anchor depth。
- HSI anchor scene xyz。
- HSI foot/contact/support-plane losses。
- teacher scene affine loss。

缺失或待补充：

- 显式 temporal velocity/acceleration/jitter loss。
- 显式 `hsi_scene_scale` temporal/prior/robust aggregation 机制。

### 数据和序列

文件：

- `vggt_omega/data/bedlam.py`

已有能力：

- `sequence_length`
- `stride`
- `person_ids`
- `person_id_mask`

说明：

- 当前数据管线已经有序列窗口，不是完全单帧。
- 可以利用 `person_ids` 做同一人物的相邻帧 temporal loss。

### Aggregator

文件：

- `vggt_omega/models/aggregator.py`

已有能力：

- 输入是 `[B, S, C, H, W]`。
- 有 inter-frame attention。

说明：

- 所以第一阶段不必先重写 video encoder。
- 更建议先在 loss 和 HSI head 的局部路径上加约束。

## 推荐实现路线

### 阶段 A：先解决人物闪动，做 loss-only 小改

目标：

- 尽量不动模型结构。
- 保留 baseline。
- 通过配置开关启用。

建议新增 loss：

```text
loss_hsi_transl_velocity
loss_hsi_joints_velocity
loss_hsi_joints_acceleration
loss_hsi_pose_delta_smooth, optional
loss_hsi_foot_sliding, optional
```

实现注意：

- 优先使用 `person_ids` 对齐同一人物。
- 若没有有效 `person_ids`，不能随意按 query index 对齐，除非确认 query 在时序上稳定。
- velocity loss 可以约束预测差分与 GT 差分：

```text
pred_vel[t] = pred[t] - pred[t-1]
gt_vel[t] = gt[t] - gt[t-1]
loss = smooth_l1(pred_vel, gt_vel)
```

- acceleration loss：

```text
pred_acc[t] = pred[t+1] - 2 * pred[t] + pred[t-1]
gt_acc[t] = gt[t+1] - 2 * gt[t] + gt[t-1]
loss = smooth_l1(pred_acc, gt_acc)
```

### 阶段 B：保守控制环境 scale，不大改 HSI head

目标：

- 暂不做 UniSH-style global scale head。
- 先防止明显 scale 跳变或饱和。

方案 1：序列内 robust scale 聚合

```text
scale_per_frame = hsi_scene_scale[:, t]
scale_seq = exp(median(log(scale_per_frame)))
```

然后用 `scale_seq` 替代每帧 scale 做视频输出或可选训练路径。

方案 2：scale temporal/prior loss

```text
loss_scale_temporal = smooth_l1(log(scale[t]) - log(scale[t-1]), 0)
loss_scale_prior = smooth_l1(log(scale), log(scale_ref))
```

其中 `scale_ref` 可以先取：

- 当前稳定 checkpoint 统计范围附近；
- 或 batch/sequence median；
- 或 teacher scale；
- 或配置给定值。

建议第一版：

- scale 用 robust sequence value。
- bias 暂时保留 per-frame，或只加 temporal smooth。
- 因为 scale 控制整体大小，bias 更像深度平移，风险相对不同。

### 阶段 C：再考虑结构改造

候选：

1. UniCon3R-style HSI temporal/contact memory。
2. UniSH-style clip/global scale token。
3. DuoMo-style post-process/teacher，而不是直接并入主训练。

## 当前优先级结论

```text
P0: 人物 temporal loss
P1: scene scale robust aggregation / log-scale temporal prior
P2: HSI temporal/contact memory
P3: UniSH-style global scale head
P4: DuoMo full diffusion teacher/post-process
```

## 需要避免的误区

1. 不要把 DuoMo 整个 diffusion 框架直接搬进主项目。
2. 不要让主项目直接 import `.paper/` 或 `.reference/` 下的代码。
3. 不要假设 query index 天然代表同一人物，除非有明确 matching 或 identity 约束。
4. 不要把 scene scale per-frame 自由预测继续放大训练，当前已有 `20.0855` 饱和风险。
5. 不要在第一版同时大改 loss、HSI head、data loader 和 inference，应该拆成可验证的小步骤。

## 可交给下一轮代码上下文对话的实现请求

建议 prompt：

```text
请根据 docs/video_hsi_temporal_scale_handoff.md 中的阶段 A 和阶段 B，先做最小代码改造：

1. 在 HungarianSMPLLoss 中新增配置开关控制的 HSI temporal loss：
   - hsi_transl_velocity_weight
   - hsi_joints_velocity_weight
   - hsi_joints_acceleration_weight
   优先用 batch 的 person_ids/person_id_mask 对齐同一人物相邻帧。

2. 为 hsi_scene_scale 增加轻量稳定控制：
   - log-scale temporal smooth loss，或
   - sequence-level robust scale aggregation 的可选 inference/eval 路径。

3. 保留 baseline，所有新能力默认关闭。

4. 不直接 import .paper 或 .reference 中的代码，只做概念改写。

5. 做 import check 和最小 dummy shape 测试；如果本地缺少训练环境或 ckpt，明确说明未验证完整训练。
```

