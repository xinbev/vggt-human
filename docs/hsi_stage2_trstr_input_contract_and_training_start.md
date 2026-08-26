# TRSTR 独立模块输入契约与训练起步

## 1. 模块职责

TRSTR（Track-Aware Regional Surface Translation Refiner，轨迹感知的区域表面平移修正器）是 Stage1 Scale 之后的独立模块。

它只做一件事：

```text
固定 pose / global orientation / betas
根据每个人的 SMPL 表面与 metric depth 的关系
预测该人的 delta_transl_cam
```

它不预测 Scale，不恢复 SMPL 动作，不修改体型，不输出自由顶点位移。

## 2. 统一张量符号

```text
B: batch 中的视频片段数量
S: 每个片段的帧数
Q: 每帧最大人体数量
A: 每个人的区域 query 数，当前默认 96；支持 48/72/96 消融
R: 每个区域的代表顶点数量
H,W: depth 分辨率
C: region token 特征维度
```

一帧多人时，每个人独立拥有 `A` 个 query。当前默认 `A=96`，同时支持 `A=48/72/96` 做区域数量消融：

```text
region_tokens [B,S,Q,A,C]
```

不同 `q` 的区域证据不能混合。

## 3. 最小 GT 训练输入

第一阶段建议做纯几何、单帧或短双帧训练，不运行 NLF，不依赖 Stage1，不要求 RGB/VGGT feature。

### 3.1 必需输入

| 字段 | Shape | dtype | 坐标/语义 |
| --- | --- | --- | --- |
| `base_pose_6d` | `[B,S,Q,144]` | float32 | GT SMPL 24 关节 6D rotation，只读 |
| `base_betas` | `[B,S,Q,10]` | float32 | GT SMPL shape，只读 |
| `base_transl_cam` | `[B,S,Q,3]` | float32 | 人工扰动后的 camera-space translation |
| `metric_depth` | `[B,S,1,H,W]` 或 `[B,S,H,W]` | float32 | 干净 GT metric depth，单位米 |
| `intrinsics` | `[B,S,3,3]` | float32 | 与 resize/pad 后 depth/image 平面对齐的相机内参 K |
| `person_valid` | `[B,S,Q]` | bool | 当前人体 slot 是否有效且满足在线可见性 |
| `image_size_hw` | `[B,S,2]` 或明确 tuple | int64 | K 所对应的图像平面尺寸 |

BEDLAM 当前字段映射：

```text
base_pose_6d <- batch["gt_pose_6d"]
base_betas <- batch["gt_betas"]
clean_transl_cam <- batch["gt_transl_cam"]
metric_depth <- batch["gt_depth"]
intrinsics <- batch["K_scal3r"]
person_valid <- online visibility filtered batch["smpl_mask"]
```

纯 GT 训练中的 `metric_depth` 已经是真实尺度，不需要再通过 Stage1。Stage1 只在真实推理桥接中使用。

### 3.2 训练 target

训练时保留干净 translation：

```text
clean_transl_cam [B,S,Q,3]
```

在线采样噪声：

```text
translation_noise = ray_noise + tangent_noise
base_transl_cam = clean_transl_cam + translation_noise
target_delta_transl = clean_transl_cam - base_transl_cam
```

每个有效区域的连续 vote target 相同：

```text
target_region_vote[b,s,q,a] = target_delta_transl[b,s,q]
```

Region query 的任务不是产生不同身体动作，而是从不同表面位置独立测量同一个人体平移。

### 3.3 可选输入

| 字段 | Shape | 用途 |
| --- | --- | --- |
| `depth_confidence` | `[B,S,1,H,W]` | 过滤低质量 depth；GT 阶段可为空 |
| `scene_features` | `[B,S,P,Cv]` 或多层 token list | 冻结 VGGT 局部视觉特征；第一版建议关闭 |
| `base_confidence` | `[B,S,Q,1]` | 模拟 NLF confidence；GT 阶段可全 1 |
| `camera_extrinsics` | `[B,S,3,4]` | world-space 时序；单帧训练不需要 |

第一版推荐 geometry-only（仅几何）训练。等纯几何路径通过后，再消融 VGGT scene feature 是否真正有增益。

`3x3` 和 `7x7` 是同时使用的两个 patch 尺度，不是二选一：`3x3` 保留投影中心的细粒度信息，`7x7` 提供更大的局部上下文并容忍少量投影误差。区域数量 `48/72/96` 才是独立的消融变量。

## 4. 建议 Forward 接口

```python
outputs = trstr(
    pose_6d=base_pose_6d,
    betas=base_betas,
    transl_cam=base_transl_cam,
    metric_depth=metric_depth,
    intrinsics=intrinsics,
    person_valid=person_valid,
    image_size_hw=image_size_hw,
    depth_confidence=None,
    scene_features=None,
    track_ids=None,
    track_quality=None,
    track_gap=None,
    track_memory=None,
)
```

核心输出：

```text
refined_transl_cam       [B,S,Q,3]
delta_transl_cam         [B,S,Q,3]
person_update_gate       [B,S,Q,1]
person_uncertainty       [B,S,Q,1]
region_displacement_vote [B,S,Q,A,3]
region_reliability       [B,S,Q,A,1]
region_uncertainty       [B,S,Q,A,1]
region_valid             [B,S,Q,A]
iteration_transl         [R_iter,B,S,Q,3]
```

Wrapper 同时原样返回输入 pose/betas，部署 smoke 必须检查逐元素完全一致。

## 5. 模块内部数据流

对每个有效人物 `q` 独立执行：

```text
pose/betas（固定）+ 当前 transl_cam
 -> 解码 6890 顶点
 -> 按固定 Region Bank 聚合为 96 个区域
 -> 区域代表顶点投影到 metric depth
 -> 采样 3x3 / 7x7 / adaptive / annulus patch
 -> 编码人体表面证据与环境上下文证据
 -> 每区域输出 translation vote + reliability + uncertainty
 -> 仅在人物 q 内做稳健聚合
 -> delta_transl_cam[q]
 -> 更新 transl_cam[q]
 -> 固定 pose/betas 重新生成网格并 re-probe
```

同帧多人在 GPU 上并行，但逻辑上相互隔离。

## 6. 时序训练新增输入

单帧 translation 路径稳定后，再增加：

| 字段 | Shape | 说明 |
| --- | --- | --- |
| `track_ids` | `[B,S,Q]` | 训练阶段使用 GT track ID |
| `track_mask` | `[B,S,Q]` | ID 是否有效 |
| `track_quality` | `[B,S,Q]` | 模拟推理 tracker 质量 |
| `track_gap` | `[B,S,Q]` | 距上次观测的帧间隔 |
| `camera_extrinsics` | `[B,S,3,4]` | 有可靠外参时使用 world translation |
| `track_memory` | ID-keyed state | 前一帧 refined translation、速度、置信度和区域 token |

时序 memory 只修正 translation trajectory（平移轨迹），不平滑 pose/betas。

建议噪声由三部分组成：

```text
track-level bias：整段序列共同平移偏差
frame jitter：逐帧随机抖动
slow drift：随时间缓慢漂移
```

这样比每帧完全独立噪声更接近真实 HMR translation 误差。

## 7. 真实推理桥接输入

部署时 TRSTR 严格输入应统一为：

```text
NLF pose/betas/transl_cam/confidence
frozen Stage1 输出的 metric_depth
VGGT K/camera
NLF detector boxes（仅供前置 tracker）
前置 tracker 输出的 track_id/quality/gap
上一帧 track memory
可选 frozen VGGT scene features
```

其中：

- NLF box 不参与旧 HMR box pooling；
- TRSTR 构造区域投影不需要 sidecar box；
- box 仅供 tracker 进行跨帧人员匹配；
- Stage1 scale/bias 在进入 TRSTR 前应用完成；
- TRSTR 不应同时接收 raw depth 和 metric depth 后自行猜使用哪个。

建议接口只接收命名明确的 `metric_depth`，由上层 pipeline 对 GT/Stage1 来源负责。

## 8. Loss 最小集合

第一阶段先使用：

```text
L_vote: 每个有效区域的 translation vote 监督
L_trans: 最终 refined translation 与 GT translation
L_vertex: 固定 pose/betas 下平移后的 vertex consistency
L_clean: 无扰动样本必须 no-op
L_gate: reliability validity/coverage，防止所有 gate 塌缩为 0
L_bound: translation 更新幅度与饱和约束
L_mono: 第二轮不能比第一轮更差
L_unc: uncertainty 与实际误差校准
```

暂时不加入：

```text
pose loss
rotation loss
joint-delta loss
betas loss
contact classification
temporal loss
```

时序阶段再加入 translation velocity、acceleration 和 no-worse loss。

## 9. 推荐训练顺序

### T0 输入与区域 Smoke

- 一个人、多人和空人物 batch；
- 96 区域覆盖 6890 顶点；
- pose/betas passthrough 完全一致；
- 每个人独立输出 `[3]` translation；
- 无有效区域时严格 no-op。

### T1 单步小样本过拟合

- `S=1`；
- 先只训练一次 translation update；
- 固定 32/64 个样本；
- 验证 ray/tangent 三个方向都可恢复；
- clean displacement 接近 0。

### T2 两轮 Re-probe

- 共享权重 2 次更新；
- 第三次 probe 只做最终检查；
- 第二轮 residual 应小于第一轮；
- 变差时 rollback。

### T3 单帧完整分布

- 扩大 translation 高斯扰动范围；
- 按人体深度、区域覆盖、遮挡和人数分组；
- 检查少量错误区域不会带偏整个人。

### T4 时序

- `S=4 -> 8 -> 12`；
- 加入 track bias/jitter/drift；
- 模拟漏检、gap、ID reset/switch；
- 只稳定 translation。

### T5 真实 Bridge

- 冻结 VGGT/NLF/Stage1；
- 使用 NLF detector；
- GT 只用于 loss 和人员匹配；
- 对比 GT 几何训练与真实推理输入的 domain gap。

## 10. 数据准备结论

开始 T1 前只需要现有 BEDLAM：

```text
RGB（可选）
GT SMPL pose/betas/transl_cam
GT K
GT metric depth
person mask / online visibility
```

Translation 扰动在线生成，不需要复制或预生成新 depth。人体框 sidecar 不需要。

开始时序 T4 前额外需要可靠 GT track ID 和 camera extrinsics 审计；开始 T5 前需要缓存或在线运行 frozen VGGT/NLF/Stage1 输出。

训练配置中的 `enable_hsi_refine=true` 只用于加载并冻结 Stage1 的 scale/bias 承载 head；`enable_hsi_human_scene_align`、`enable_hsi_translation_refine_v4`、`enable_hsi_contact_refine` 和 `enable_hsi_grounding` 均为 false，TRSTR 不依赖这些旧修正模块。

## 11. 当前第一版实现

已实现的第一版代码：

```text
vggt_omega/models/geometry/smpl_region_bank.py
vggt_omega/models/heads/hsi_regional_translation_refiner.py
vggt_omega/training/hungarian_losses.py
vggt_omega/models/vggt_omega.py
```

当前实现覆盖：

- 默认 96 个基于 SMPL LBS 主关节分组的确定性区域；同一实现支持 48/72/96 区域预算消融；
- 当前实现使用 LBS 主关节分组和模板空间最远点采样；拓扑连通性/测地距离审计仍是后续 G0 工作；
- 每个人独立的区域池化；
- 3x3/7x7 depth patch 几何统计；
- 每区域 translation vote、reliability 和 log variance；
- 每个人内部的 reliability/uncertainty 加权聚合；
- 默认两轮 translation re-probe；
- pose、global orientation、betas 原样透传；
- TRSTR 专用 loss、训练配置、训练 shell 和 standalone smoke。

运行服务器 smoke：

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
bash scripts/smoke/check_hsi_stage2_trstr.sh
```

短训练：

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

MAX_STEPS_PER_EPOCH=100 \
BATCH_SIZE=2 \
NUM_VIEWS=1 \
CUDA_VISIBLE_DEVICES_VALUE=7 \
bash scripts/train/train_smpl_hsi_stage2_trstr.sh
```

默认输出：

```text
outputs/train/smpl_hsi_stage2_trstr_translation_only
```

当前未实现：持久化 inference tracker、track memory、真实 NLF bridge cache、最终 Viser 区域可视化。这些必须在单帧 TRSTR smoke 和短训练通过后再接入。
