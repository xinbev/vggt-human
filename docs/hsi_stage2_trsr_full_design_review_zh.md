# HSI 第二阶段 TRSTR 完整设计评审

## 1. 评审结论

本文重新审阅 Scale 模型之后的人体场景修正方案。结论是：整体方向成立，且比项目中旧的全身均匀采样后直接回归平移更精细，但必须拆成可验证阶段，不能一次性实现“全身区域 query + translation refinement + 多人时序 memory”后直接长训。

新模块命名为：

```text
TRSTR = Track-Aware Regional Surface Translation Refiner
      = 轨迹感知的区域表面平移修正器
```

它的核心不是把人体顶点吸向最近的 depth，也不是枚举若干位移候选后做分类，而是：

```text
人体表面非均匀区域 query
 -> 查询局部多尺度 depth/camera 信息
 -> 每个区域预测同一人体平移的连续三维修正证据
 -> 根据区域可靠性做稳健集合聚合
 -> 输出唯一的 SMPL camera-space translation 更新
 -> 重新生成网格并重新查询场景
 -> 使用提前确定的 track ID 融合历史状态
```

完整方案需要满足四条硬约束：

1. Stage1 Scale、VGGT 和 NLF 在 Stage2 默认冻结。
2. Stage2 只更新 `pred_transl_cam`；NLF/HMR 的 pose、global orientation 和 betas 全部只读。
3. 无可靠场景证据时必须 no-op（不修改），不能为了降低平均 loss 强行移动人体。
4. 时序状态必须按稳定 track ID 隔离，不能按 query 下标继承。

## 2. 全流程定位

### 2.1 离线视频推理

```text
输入 RGB 序列 [B,S,3,H,W]
 -> VGGT
    -> camera pose encoding
    -> raw depth
    -> 多层 scene patch features
 -> 从 VGGT camera encoding 解码 K 和外参
 -> NLF internal detector
    -> 人体框、confidence、SMPL pose/betas/transl
 -> Stage1 Scale（冻结）
    -> 稳定 scene scale/bias
    -> metric depth
 -> persistent base-SMPL tracker（前置持久化人体跟踪器）
    -> track_id/quality/gap/source
 -> TRSTR 单帧区域平移修正
 -> TRSTR causal temporal memory（因果时序记忆）
 -> refined SMPL
 -> Viser / export
```

VGGT 可以一次处理完整 clip，TRSTR 的时序状态仍按帧从前向后更新，避免读取未来的人体修正结果。

### 2.2 在线或分块推理

```text
chunk_0 -> tracker/memory state -> chunk_1 -> tracker/memory state -> ...
```

跨 chunk 必须持久化：

- tracker 轨迹表；
- scene scale state；
- 每个 track 的人体运动和区域交互 state；
- 最后处理帧编号和 camera continuity（相机连续性）。

当前 `BaseSMPLTrackAssigner` 在一次调用内建立轨迹，调用结束后状态消失。因此它可复用匹配分数，但需要改造成 persistent tracker（持久化跟踪器）或增加显式 state 输入输出。

### 2.3 训练主线

```text
Stage2-A  区域库与局部观测审计，不训练 TRSTR
Stage2-B  单帧连续 translation 修正
Stage2-C  使用 GT track ID 的因果 translation 时序训练
Stage2-D  VGGT + NLF detector + frozen Stage1 的真实推理桥接
```

只有前一阶段通过 gate（验收门槛）后才进入下一阶段。

## 3. Stage1 到 Stage2 的输入契约

### 3.1 Stage1 输出

Stage1 输出：

```text
hsi_scene_scale       [B,S,1]
hsi_scene_depth_bias  [B,S,1]
```

修正深度：

```text
D_metric(u,v) = s * D_raw(u,v) + b
```

Stage2 只读取该结果，不更新 `scale_delta` 和 `bias_delta`。

### 3.2 Scale 稳定策略

Stage2 不应直接使用剧烈跳动的逐帧 scale：

- 离线 clip：优先在 log-scale（对数尺度）中做置信度加权 robust aggregation（稳健聚合），例如 trimmed median/median；
- 在线视频：使用跨 chunk 持久化的置信度 EMA（指数移动平均）；
- scale confidence 过低或相邻帧突变超阈值时，Stage2 降低世界坐标时序权重并倾向 no-op，不用 translation 去掩盖 scale 错误。

### 3.3 坐标一致性

单帧区域查询使用 camera coordinates（相机坐标）：

```text
SMPL vertices_cam + K + D_metric
```

跨帧时序使用 world coordinates（世界坐标）时，对于纯乘法 scale：

```text
depth_metric = s * depth_raw
camera_translation_metric = s * camera_translation_raw
camera_rotation_metric = camera_rotation_raw
K_metric = K_raw
```

相机旋转和内参不能缩放。

`bias` 是沿每个像素射线的深度偏移，不是刚体平移，不能直接加到 camera translation。若 bias 不可忽略，跨帧融合必须从每帧修正后的 depth 重新反投影并检查一致性；检查失败时只使用 camera-space 局部查询，不声称使用可靠 world-space surface memory。

当前 BEDLAM loader 只读取 `K_scal3r`，没有读 camera extrinsics（相机外参）。Stage2-C 前必须先审计真实 `cam/*.npz` 字段并增加明确的 camera-from-world tensor；不能把 `gt_transl_cam` 当成世界坐标。

## 4. 人体检测与 ID 跟踪

### 4.1 NLF 的职责

真实推理使用：

```text
RGB + VGGT predicted K
 -> NLF detect_smpl_batched
 -> boxes + confidence + SMPL
```

NLF 不使用 VGGT depth。Depth 在 Stage1 和 TRSTR 使用。

NLF detector 输出的人体框只用于检测结果表达和 tracker 匹配，不恢复旧 HMR 的 box pooling/query prior。

### 4.2 前置 tracker

顺序必须是：

```text
NLF -> base-SMPL tracker -> 固定本帧 track ID -> TRSTR -> 写回 memory
```

tracker 第一版可以不是神经网络。复用当前几何匹配因素：

- NLF box center/size；
- NLF `pred_transl_cam`；
- betas 相似度；
- NLF confidence；
- 可选 ID embedding（身份外观特征）；
- 上一帧速度预测。

输出：

```text
assigned_track_ids      [B,S,Q]
assigned_track_mask     [B,S,Q]
assigned_track_quality  [B,S,Q]
assigned_track_gap      [B,S,Q]
assigned_track_source   [B,S,Q]
```

“ID 已确定”表示 TRSTR 在当前 forward 中不会再重排 ID，但 ID 仍有置信度。低质量、新轨迹、长 gap 或疑似 ID switch（身份切换）时：

- 允许单帧 TRSTR；
- 禁止读取旧 memory 或显著降低 memory 权重；
- 清空有污染风险的区域状态；
- 默认不跨 ID 平滑。

Tracker 只给当前 NLF slot 绑定 `track_id`，不要求把所有帧的 slot 重排成固定顺序。TRSTR 和 memory 必须通过 `track_id -> state` 查询，因此同一个人即使从 query 2 变成 query 7，历史仍属于同一 ID。

漏检帧默认不渲染“幽灵人体”。Memory 可以短期保留以便重新连接，但不能凭 memory 无条件生成当前人体。

## 5. SMPL 区域库

### 5.1 术语

Region Bank（区域库）：将 SMPL 6890 个顶点划分为固定数量的表面测量区域，并记录每个区域对应的顶点和解剖语义。

LBS（Linear Blend Skinning，线性混合蒙皮）：SMPL 使用多个关节权重共同控制每个顶点的方法。

Geodesic Distance（测地距离）：沿网格表面计算的距离，不是穿过人体内部的直线距离。

### 5.2 构建方法

从 SMPL neutral template（中性模板）、faces（网格三角面）和 LBS weights 构建：

1. 根据 LBS 权重获得每个顶点的主控制关节和次级关节。
2. 使用 faces 建立网格邻接图，保证区域在表面上连通。
3. 按深度边界风险、遮挡概率和接触信息量分配 query budget（查询数量预算）。
4. 在每个解剖组内部使用 geodesic FPS（测地最远点采样）或邻接约束聚类。
5. 6890 个顶点必须且只能属于一个 pooling region（聚合区域）。
6. 记录区域的解剖分组，但不建立关节更新链；LBS 只用于稳定地划分和解释表面区域。

项目的 `SMPLLayer` 目前只显式暴露 faces，但底层 `smplx` layer 通常还包含 `lbs_weights`、`parents` 等属性。G0 smoke 必须确认服务器版本真实存在这些字段；若字段名或 shape 不符合预期，应报错，不允许悄悄退回均匀 FPS。

### 5.3 初始 96 区域预算

```text
头、颈、胸、背、腹、骨盆：20 个粗区域
上臂、前臂：               16 个中等区域
左右手：                   24 个细区域
大腿、小腿：               16 个中等区域
左右脚、脚跟、脚尖、脚底： 20 个细区域
```

手脚区域除了区域数量更多，每个 query 内的 representative vertices（代表顶点）数量也应更多。建议进行 `48/72/96` 区域消融，而不是把 96 当作固定真理。

### 5.4 区域元数据

每个区域记录：

```text
vertex_indices / pooling_weights
representative_vertex_indices
canonical centroid / normal / covariance / radius
mean LBS weights [24]（仅用于区域归属/语义）
risk group / left-right / semantic group
region-specific vote bound（区域投票上限）
```

审计标准：

- 6890 顶点覆盖率等于 100%；
- 没有重复和未分配顶点；
- 每个区域连通；
- 左右对称；
- 手、脚底、脚尖、脚跟、膝、肘、臀覆盖足够密；
- 每个区域有明确解剖归属和代表顶点。

## 6. 多尺度局部查询

### 6.1 Patch 的定义

Patch（局部窗口）：区域投影到 depth map 后，围绕投影位置采样的局部网格窗口。

建议基础尺度：

```text
3x3：中心精细表面
7x7：容忍投影和 NLF pose 小误差
adaptive window：根据区域投影半径动态调整
annulus：区域轮廓外环，用于环境上下文
```

固定像素窗口对远近人体代表的物理范围不同，因此 3x3/7x7 是最低配置。实际半径应使用区域投影 radius 自适应，并限制最大采样成本。

### 6.2 双通道 depth 语义

不能把中心 patch 内所有“人体 depth”删除，也不能把所有 depth 都当成环境。需要两个通道：

#### Human Surface Evidence（人体表面证据）

中心 patch 中与当前 SMPL 投影、深度顺序和人体轮廓一致的表面。它用于估计 NLF SMPL 的整体 translation 误差；若观察更像 pose 误差，只作为异常诊断，不修改 pose。

#### Environment Context Evidence（环境上下文证据）

人体投影轮廓外环、与人体表面深度不一致但几何可靠的局部点、以及可选跨帧融合后显露的支撑/碰撞面。它用于判断墙、地面、桌椅等环境关系。

两者必须分别编码 validity（有效性）和 confidence（置信度），不能先混在一起求平均。

BEDLAM GT depth 已有文档证据表明通常包含人体；真实 VGGT depth 也可能包含人体。因此训练时应审计 depth 语义，而不是假设 scene-only depth（仅环境深度）。

对投影点先计算：

```text
depth_order_residual = z_depth - z_mesh
```

- `abs(residual) <= tau_self`：可能是人体自身表面；
- `residual < -tau_front`：depth 在人体前方，可能是前景遮挡或潜在穿模；
- `residual > tau_back`：depth 在人体后方，不能自动吸附人体。

这些只是 depth-order class（深度顺序类别），不是接触标签。尤其“depth 在人体前方”必须结合 annulus 环境面、法线、边缘和多尺度一致性后才能形成环境修正证据。

### 6.3 Patch 内处理

每个采样像素使用 K 反投影为 camera-space 3D point，并计算：

```text
body-to-point offset
ray/normal/tangent signed residual
local surface normal
depth edge score
roughness / plane residual
valid ratio
VGGT depth confidence
self-depth probability
foreground/background depth-order class
```

Patch encoder（局部窗口编码器）不直接简单平均。建议：

```text
共享小型 MLP 编码每个3D采样点
 -> patch内 masked attention / robust pooling
 -> 3x3、7x7、adaptive、annulus 多尺度融合
 -> region scene token
```

Masked Attention（掩码注意力）：只允许有效采样点参与注意力计算。

Robust Pooling（稳健聚合）：对异常值不敏感的中位数、截断均值或可学习加权聚合。

第一版以几何为主，不依赖 RGB feature 才能工作。VGGT patch feature 可作为后续 ablation（消融实验），避免模型从纹理走捷径。

### 6.4 不可观测情况

以下情况 region query 应标记 invalid 或高 uncertainty：

- 投影在图外或 z<=0；
- patch 有效点太少；
- 位于强 depth discontinuity（深度断层）；
- 人体表面和环境通道无法分离；
- VGGT depth confidence 过低；
- 多尺度窗口给出冲突法线/距离；
- 当前区域被前景遮挡，单层 depth 无法观察后方接触。

不可观测不代表“区域正确”，而代表“当前帧没有足够证据修正”。

## 7. 区域 Query 表达

Query（查询向量）：代表一个人体表面区域、当前 SMPL 状态和局部场景观测的特征向量。

每个 query 输入：

```text
region canonical embedding
当前 centroid/normal/covariance/radius
代表顶点和 body-relative position
anatomical/LBS region embedding（只作为区域语义）
camera ray basis / projected location
human-surface patch token
environment-context patch token
多尺度 residual statistics: P10/P50/P90/MAD
visibility / edge / depth / NLF / Stage1 confidence
track motion prior / previous region memory
```

其中：

- P10/P50/P90：残差分布的 10%、50%、90% 分位数；
- MAD（Median Absolute Deviation，中位绝对偏差）：稳健衡量残差离散程度；
- covariance（协方差）：描述区域点云的局部形状和主方向。

推荐 shape：

```text
region_vertices_cam       [B,S,Q,A,R,3]
region_tokens             [B,S,Q,A,C]
region_valid              [B,S,Q,A]
region_uncertainty        [B,S,Q,A,1]
region_anatomical_group   [A]
```

`A` 是区域数，初始 96；`R` 是每区域代表顶点数；`C` 是 token channel（特征通道数）。

`Q` 是当前帧最多的人数/人体 slot 数。区域库在每个人上独立实例化，因此总区域 token 为 `[B,S,Q,A,C]`，不是整帧共享 96 个 query：

```text
person q=0 -> 96 region queries -> delta_transl_cam[q=0]
person q=1 -> 96 region queries -> delta_transl_cam[q=1]
...
person q=Q-1 -> 96 region queries -> delta_transl_cam[q=Q-1]
```

每个人只聚合自己的区域 vote。`person_valid/nlf_valid_mask` 屏蔽不存在的人体，`track_id` 只负责把该人的历史状态取回来。V1 不在不同人之间做 region attention，也不允许一个人的区域证据影响另一个人的 translation。

## 8. 连续区域平移证据场

### 8.1 区域连续位移

每个 query 直接预测：

```text
region_displacement_vote [B,S,Q,A,3]
region_reliability_gate  [B,S,Q,A,1]
region_log_variance      [B,S,Q,A,1]
```

Displacement Vote（位移投票）：该区域根据当前观测，对“整个人应该平移多少”提出的连续三维估计。它不是区域独立位移，更不是顶点 offset。

输出在稳定局部基底中参数化：

```text
camera ray + region normal + tangent
```

再转换为 XYZ。使用 bounded tanh（有界双曲正切）限制最大步长，但不设置离散的 2/5/8 cm 候选。

`region_reliability_gate` 不能完全无监督自由学习，否则模型可能把所有 gate 关掉来逃避 region loss。它至少需要：

- deterministic validity（由投影范围、有效 depth 数量、边缘和 confidence 得到的确定性有效标签）；
- synthetic observability（已知扰动下该区域是否有足够观测恢复目标）；
- gate coverage regularization（有效区域覆盖率正则）；
- 与预测 uncertainty 一致的校准 loss。

Gate 表示“当前观测是否可靠”，不是接触分类器。

### 8.2 连续监督目标

对于输入的扰动 SMPL 和干净 GT SMPL：

```text
delta_vertex[a,r] = V_gt[a,r] - V_input[a,r]
target_region_vote[a] = weighted robust mean(delta_vertex[a,*])
```

因为训练只扰动 translation，理论上所有区域的目标相同：

```text
target_region_vote[a] = gt_transl_cam - input_transl_cam
```

顶点差公式保留为一致性检查。区域 vote loss 只在该区域观测有效时计算。隐藏区域不能被要求从不存在的 depth 证据中猜平移。

clean sample（干净样本）所有有效区域 target vote 为零，用于训练 no-op。

### 8.3 区域投票的稳健聚合

所有有效区域共同估计唯一的 translation：

```text
delta_transl[b,s,q] = robust_set_aggregate(
    region_vote[b,s,q,a],
    reliability[b,s,q,a],
    uncertainty[b,s,q,a]
)
```

初版推荐使用可解释的“置信度加权截断均值或几何中位数 + 小型集合网络残差”，而不是一个无约束 MLP 直接吞掉全部 query。

区域之间的 disagreement（分歧）有三种用途：

1. 少量区域异常：视作遮挡、depth 边缘或局部场景噪声并降权；
2. 大量区域给出一致位移：形成高置信度 translation；
3. 区域残差呈旋转样、尺度样或局部非刚体模式：判为 translation-only 假设不成立，降低 person gate 并 no-op 或仅做保守修正。

可以额外做只读诊断拟合：

```text
vote_a ≈ t + omega_diag × (x_a-root) + alpha_diag*(x_a-camera_center) + residual_a
```

只有 `t` 可以应用。`omega_diag`、`alpha_diag` 和局部 `residual_a` 只用于识别 pose、scale 或 depth 不一致，绝不能更新 global orientation 或关节。

### 8.4 术语说明

Set Decoder（集合解码器）：对无固定顺序的多个区域 translation vote 做聚合，结果不依赖 query 排列。

Ridge Regression（岭回归）：在线性求解中增加正则项，使解在噪声下更稳定。

Geometric Median（几何中位数）：使到所有投票的加权距离和最小的稳健中心，用于减少离群区域影响。

## 9. Translation-only 参数解码

### 9.1 输出参数

```text
delta_transl_cam   [B,S,Q,3]
person_update_gate [B,S,Q,1]
person_uncertainty [B,S,Q,1]
region_inlier_mask [B,S,Q,A]
```

最终更新只有：

```text
refined_transl_cam = base_transl_cam + person_update_gate * delta_transl_cam
```

`pred_pose_6d`、`pred_poses`、global orientation 和 `pred_betas` 原样透传。它们可以参与解码网格、构造 query、tracking 和异常诊断，但不能接收梯度更新或被时序模块改写。

### 9.2 平移参数化

`delta_transl_cam` 可以在 camera ray basis（相机射线基底）中预测：

```text
delta = delta_ray * ray + delta_tx * tangent_x + delta_ty * tangent_y
```

这与项目旧 translation head 的几何含义一致，并允许分别限制：

- 沿相机射线的深度修正；
- 图像平面两个切向修正。

输出使用有界 `tanh`，每轮上限逐步减小。上限按人体深度和配置约束，但不能由单个异常 region 放大。

### 9.3 参数边界

- Stage2 只训练 translation 聚合器和可靠性/不确定度分支；
- pose、global orientation、betas、Stage1 scale 全部冻结；
- 不输出自由顶点 displacement；
- clean input 应接近严格 identity mapping（恒等映射）；
- 区域分歧过大、scale-like 或 rotation-like 诊断过强时，降低 person gate 或 no-op；
- hand/foot query 数量更多只是为了获得更细的几何测量，不代表手脚可以独立移动。

## 10. 真实迭代重查

Re-probing（重新探测）：应用一次 translation 更新后，用固定 pose/betas 和新 translation 重新生成网格并查询 depth，而不是在同一批旧 token 上重复 Transformer。

建议默认执行 2 次学习到的 translation 更新，最大允许 3 次。所有有效人体可以张量化并行处理，但每个人的更新彼此独立：

```text
for iteration r:
    for every valid person q in parallel:
        decode current SMPL with fixed pose/betas and current transl_cam[q]
        rebuild that person's 96 region geometries
        sample that person's multi-scale depth patches
        predict that person's continuous region votes
        robustly aggregate one delta_transl_cam[q]
        diagnose disagreement without applying pose/scale changes
        update that person's transl_cam only
```

“更新次数”和“几何查询次数”需要分开计数。默认 `R=2` 时：

```text
state 0: base SMPL       -> probe 0 -> translation update 1
state 1: updated SMPL    -> probe 1 -> translation update 2
state 2: final SMPL      -> probe 2 -> final energy check / rollback decision
```

因此默认是：

```text
每个人 2 次可学习平移更新
每个人 3 个网格状态的 depth 几何检查
VGGT depth 本身只预测一次，后续只是反复从同一张已缩放 depth 中采样
```

若最大 `R=3`，则是 3 次平移更新和 4 个几何状态检查。若第一轮后误差已足够小、有效区域不足或不确定度过高，则 early stopping（提前停止），不会强制执行后续更新。

这里的“优化”是 inference-time iterative refinement（推理时迭代修正），不是推理时运行 AdamW、反向传播或重新训练模型。

从完整推理流水线看，默认计算顺序是：

```text
VGGT forward：              1 次，生成 camera/depth/features
NLF detector + HMR forward：1 次，生成每个人的 base SMPL
Stage1 Scale forward/apply：1 次，生成统一 metric depth
前置 tracker assignment：   每帧 1 次，绑定 track ID
TRSTR translation update：  每人最多 2 次（默认）
TRSTR final geometry check：每人 1 次
temporal memory read/write：每帧每人 1 组状态操作，不额外生成修正轮次
```

这里“每人”是逻辑上的独立修正，GPU 实现应将同一 batch/帧中的多人并行计算，而不是按人物逐个执行完整 VGGT/NLF。

训练时每轮都有监督。推理时采用：

- step bound 随轮次减小；
- valid region 太少时停止；
- uncertainty 太高时停止；
- reliable geometric energy 明显变差时 rollback（回滚）到上一轮；
- 最大迭代数硬限制。

Reliable Geometric Energy（可靠几何能量）：仅对可观测且高置信度的人体表面/环境关系计算的误差，不能使用无效区域强行决定回滚。

## 11. 时序与多人状态

### 11.1 Scene State（场景状态）

每个视频流一份：

```text
stable log_scale / bias
scale confidence
camera continuity
last frame/chunk
```

### 11.2 Track State（人物轨迹状态）

每个 track ID 一份：

```text
last refined root in world/camera fallback
root velocity / acceleration
read-only pose/betas signature（只用于跟踪一致性和异常检测）
pooled region interaction token
per-region contact/collision confidence
last translation correction and uncertainty
track quality / gap / last_seen
```

Memory 默认 detach（与梯度图分离），防止跨无限视频反向传播。

### 11.3 时序融合

时序模块只修正 translation，不平均或改写 pose。它融合：

```text
previous refined state + velocity
 -> motion prediction
current NLF + current region observations
 -> current evidence
confidence gate
 -> correction prior
```

运动快时允许真实变化，观测弱时使用历史稳定，观测强且与历史冲突时相信当前帧并降低 memory confidence。

Hysteresis（滞回）：建立接触需要连续可靠证据，但已经建立的接触可以容忍一帧 depth 缺失。

### 11.4 轨迹异常

- 新 ID：无历史，单帧运行；
- gap 小且质量高：读取历史但按 gap 衰减；
- gap 过大：重置 memory；
- ID switch 风险：禁止继承旧 translation/region token；
- 当前无人：跳过 TRSTR，不更新 optimizer/global step；
- 某些 region 无效：只屏蔽这些 region，不一定丢弃整个人；
- 所有 region 无效：整个人 no-op。

### 11.5 多人边界

V1 每个人独立估计 translation。人与人重叠只作为异常/降权信号，不尝试通过独立平移自动解决 human-human collision（人与人碰撞），因为在 pose 固定时移动其中一人可能破坏真实交互。任何多人联合修正都属于后续独立课题。

## 12. 训练数据与扰动

### 12.1 Stage2-A/B 的 GT 路径

输入：

```text
GT SMPL + GT K + GT depth + RGB/VGGT frozen features(optional)
```

扰动采用连续分布，不使用离散位移候选：

- root ray/tangent translation：零均值高斯或 log-depth Gaussian；
- 可选 world XYZ translation：在可靠外参存在时使用；
- pose/global orientation/betas 不加扰动，直接使用干净 GT；
- clean/no-op 样本保持明确比例；
- 扰动幅度按 curriculum 从小到大。

### 12.2 两类训练信号

GT depth 通常包含人体，因此中心 patch 可监督 NLF/SMPL 与人体观测对齐。环境碰撞监督必须单独要求可靠环境上下文。

```text
人体重建修正：clean human depth surface + GT SMPL target
环境关系修正：可靠 environment channel + 几何有效 mask
```

不能把 BEDLAM 中所有 GT 人体都当成物理接触正确。浮空、坐姿、躺姿、遮挡和不可靠场景区域需要 mask 或 no-op/uncertainty 监督。

### 12.3 Region Vote Target

```text
target_vote[a] = robust_pool(V_gt[a] - V_perturbed[a])
```

在 translation-only 扰动下，它等于同一个 `gt_transl_cam-input_transl_cam`。只有观测有效区域计算直接 vote loss。最终 translation 和整体 vertex consistency loss 约束聚合结果；pose/betas 不计算修正 loss，因为它们不应变化。

### 12.4 Stage2-C 时序数据

序列长度逐级：

```text
4 -> 8 -> 12
```

训练用 GT track ID，同时模拟推理错误：

- missed detection（漏检）；
- track gap（断轨间隔）；
- quality drop；
- region observation dropout；
- base translation jitter；
- ID reset；
- ID switch negative（身份切换负样本）。

可以额外加入少量 pose/scale/depth-mismatch hard negatives（困难负样本），但它们的目标不是让 TRSTR 恢复 pose，而是训练 disagreement detector、低 person gate 和 no-op。训练集必须明确区分：

```text
translation-correctable sample -> 监督连续 translation
non-translation sample         -> 监督低置信度/no-op/上报诊断
```

### 12.5 Stage2-D 真实桥接

必须使用与部署一致的：

```text
VGGT + NLF detector + frozen Stage1 + persistent tracker
```

NLF 输出与 BEDLAM GT 的匹配只用于训练监督。可以在线投影 GT SMPL 生成训练匹配框，不需要恢复 sidecar 依赖，更不能让推理依赖 GT 框。

建议缓存 frozen outputs 到 versioned `outputs/preprocess/`，记录：

- 代码 commit；
- VGGT/NLF/Stage1 checkpoint hash；
- resize/camera convention；
- detector thresholds；
- person-to-GT match quality。

## 13. 损失函数

```text
L = lambda_vote   * 区域连续位移监督
  + lambda_trans  * camera/world translation loss
  + lambda_vertex * translation-induced vertex consistency
  + lambda_human  * 人体表面对齐 loss
  + lambda_env    * 可靠的单侧穿模/支撑 loss
  + lambda_clean  * clean identity loss
  + lambda_bound  * translation magnitude/bound regularization
  + lambda_mono   * 每轮不劣化/单调改善 loss
  + lambda_temp   * track translation velocity/acceleration residual
  + lambda_slide  * contact-conditioned foot sliding
  + lambda_gate   * reliability gate validity/coverage supervision
  + lambda_unc    * uncertainty calibration
```

No-worse Loss（不劣化损失）：相对于输入 base SMPL，修正后不能让可靠指标更差。

Monotonic Rollout Loss（迭代单调损失）：后一次 re-probe 的可靠误差不应高于前一次。

Uncertainty Calibration（不确定度校准）：预测高不确定度的样本确实更容易出错，低不确定度结果应更可靠。

关键 mask：

- invisible region 不计算 vote/env loss；
- rotation-like/scale-like/局部非刚体分歧只参与诊断和 gate，不产生 pose 更新；
- ambiguous contact 不计算强制吸附 loss；
- clean 样本要求 translation/vertex displacement 接近零；
- pose、global orientation、betas 必须与输入逐元素保持一致；
- Stage1、VGGT、NLF 通过 hash 检查保持冻结。

## 14. 鲁棒性与失败保护

### 输入级

- 无 NLF 人体：跳过；
- NLF confidence 低：no-op 或仅保守 root；
- depth 无效：不运行依赖 depth 的修正；
- Stage1 scale 低可信：降低 translation gate 或 no-op；
- camera discontinuity：清空 world temporal state。

### Region 级

- patch 无有效点：region invalid；
- human/environment channel 冲突：提高 uncertainty；
- 多尺度结果冲突：降低 gate；
- 运动链不允许：attention mask 阻止更新。

### Person 级

- 所有 region invalid：严格 no-op；
- scale-like/rotation-like/local residual 大：报告上游问题，不通过 translation 强行消化；
- predicted update 超限：clamp 并记录 saturation rate；
- re-probe energy 变差：rollback；
- clean displacement 超阈值：训练 gate 失败。

### Track 级

- 低 track quality：禁用 memory；
- gap 超限：重置；
- ID switch：隔离并重建 state；
- 一个人的 memory 永不写入另一个 ID。

### Epoch/训练级

- 无有效监督 batch 不更新 optimizer；
- 整轮无有效 step 直接失败；
- non-finite loss/gradient 直接失败；
- frozen prefix hash 改变直接失败；
- 不保存每个 epoch 大 checkpoint，只保留 latest 和有限 best。

## 15. 验证阶梯

### G0 区域库审计

- 6890 顶点完全覆盖；
- 拓扑连通和左右对称；
- LBS/parents/faces shape 正确；
- 96 区域 Viser 着色；
- 高风险区域密度符合设计。

### G1 可观测性与可控性

Observability（可观测性）：输入发生已知连续扰动时，相关 query 的 patch 特征是否稳定变化。

Controllability（可控性）：单一三维 translation 是否足以解释各区域观测变化，以及 query 特征能否恢复已知 translation 扰动。

不训练完整 refiner，报告：

- 3x3/7x7/adaptive/annulus 有效率；
- affected-region response；
- unaffected leakage；
- self-depth/environment 分离准确率；
- finite-difference sensitivity（有限差分敏感度）；
- 不同 body group 的有效覆盖。

G1 不生成候选位移，也不使用 oracle 选择器。

### G2 小样本过拟合与梯度

- 连续 root correction 能拟合固定小集合；
- 多区域聚合能恢复已知 ray/tangent/XYZ translation 扰动；
- clean sample 基本不动；
- 每轮 re-probe observation 真实改变；
- translation/region/gate head 有非零有限梯度；
- Stage1/upstream hash 不变。

### G3 合成分布

按 hand/foot/limb/torso 区域证据覆盖、translation 扰动方向/大小、可见性和 confidence 分组；要求相对输入改善，而不是只看总体 loss。

### G4 时序与 ID 污染

- translation acceleration、jerk（加加速度/抖动指标）；
- vertex jitter；
- foot sliding；
- missed detection/gap 恢复；
- ID switch 后 memory contamination rate；
- 真实快速运动不被过度平滑。

### G5 真实推理桥接

完整 held-out sequences 比较：

```text
NLF + frozen Stage1
旧 root-only align
旧 contact/grounding baseline
TRSTR single-frame
TRSTR temporal
```

检查 GT 路径成功但 NLF/VGGT 路径失败的 teacher-forcing gap（教师强制域差异）。

### G6 Viser 与真实视频

显示：

- base/refined SMPL；
- 区域颜色和代表顶点；
- human/environment 多尺度 patch；
- region vote/gate/uncertainty；
- aggregated translation、scale/rotation/local disagreement diagnostics；
- translation update；
- track ID/quality/gap；
- memory reset/rollback。

## 16. 指标

人体精度：

```text
MPJPE / PVE / translation error / pose-betas passthrough equality
```

人体场景关系：

```text
penetration depth/rate
collision ratio by body region
support/float distance
clean-person displacement
```

时序：

```text
translation/vertex velocity error
acceleration / jerk / jitter
foot sliding
```

可靠性：

```text
valid region coverage
uncertainty-error calibration
abstention/no-op rate
rollback rate
update saturation rate
scale-like residual rate
```

跟踪：

```text
ID switches
track fragmentation
memory contamination
gap recovery
```

工程：

```text
runtime per person/frame
peak GPU memory
number of decoded meshes/re-probes
```

## 17. 现有模块复用与弃用

可复用：

- `HSIHumanSceneAlignHead` 的局部反投影、robust residual、camera basis；
- `contact_geometry.py` 的 local plane 和 confidence filter；
- `HSIContactRefineHead` 的下肢边界与 contact metrics；
- `HSIGroundingHead` 的 no-op、deadzone、uncertainty 指标作为 baseline；
- 已有 temporal losses；
- `BaseSMPLTrackAssigner` 的匹配分数；
- `HSITrackMemory` 的按 ID 所有权思想；
- `apply_hsi_scene_affine_mode` 的 clip/EMA 思路。

不作为新主线：

- 均匀 FPS 后全身平均 residual；
- 人工 translation candidate bank；
- candidate selection classification；
- 只修 root 的旧 Stage2；
- 只处理脚的 grounding 作为全身方案；
- HSI 后的 display-only tracking；
- 任何 pose/global orientation/betas 更新或时序平滑；
- 不重新生成网格的伪迭代。

## 18. 代码边界

建议新增：

```text
vggt_omega/models/heads/hsi_regional_translation_refiner.py
vggt_omega/models/geometry/smpl_region_bank.py
vggt_omega/models/geometry/regional_scene_probe.py
vggt_omega/tracking/persistent_smpl_tracker.py
vggt_omega/tracking/hsi_regional_track_memory.py
configs/train_smpl_hsi_stage2_regional_*.yaml
scripts/smoke/check_hsi_stage2_regional_*.sh
scripts/train/train_smpl_hsi_stage2_regional_*.sh
scripts/vis/serve_hsi_stage2_regional_refiner.sh
```

启动契约必须拒绝：

```text
TRSTR 与 legacy align/contact/grounding 同时 overwrite hsi_refined_pred_transl_cam
Stage1 scale 仍可训练
没有前置 track ID 却启用 temporal memory
没有 metric camera contract 却声称使用 world temporal state
TRSTR 输出 pose/global orientation/betas delta
```

Checkpoint 组合：

```text
VGGT baseline
+ NLF external checkpoint
+ accepted Stage1 scale checkpoint (frozen)
+ TRSTR checkpoint
```

Stage2 保存部署所需的 Stage1 scale 和 TRSTR prefixes，或通过明确 overlay 顺序加载；不保存 VGGT/NLF 大权重，不生成逐 epoch 重复 checkpoint。

### 18.1 初始计算预算

`A=96`、多人 `Q<=20`、多尺度 patch 会快速增加显存。第一版必须向量化 gather（批量索引采样），同时提供 person/region chunking（按人物或区域分块）：

```text
region count A:              96
representative vertices R:   torso 2-4, hand/foot 6-8
base patch scales:           3x3 + 7x7 + sparse annulus
re-probe iterations:         2
training sequence length:    B1/B2=1-2, C starts at 4
```

不要在每个 query 上运行独立 Python 循环或完整局部 point cloud KNN。G0/G1 必须报告：

- 每人每帧采样点数；
- SMPL decode 时间；
- patch encoder 时间；
- 每轮 re-probe 时间；
- peak GPU memory；
- 有效 region 比例。

若 `B x S x Q x A x patch_points` 超出预算，优先减少代表点/annulus 稀疏度或分块，不先减少手脚区域覆盖。

## 19. 推荐实施顺序

第一批只实现：

1. `smpl_region_bank.py`：96 区域和 G0 Viser。
2. `regional_scene_probe.py`：双通道、多尺度 patch 和 G1 审计。
3. 连续 region vote target 与 synthetic perturb 数据。
4. 只输出 `delta_transl_cam` 的稳健区域集合聚合器。
5. 两轮真实 re-probe 和 rollback 指标。

通过后再实现：

6. persistent tracker 前置化。
7. causal translation track memory。
8. NLF/VGGT/Stage1 bridge。

## 20. 当前未闭合问题

开始第一阶段代码前需要确认：

1. 服务器 `smplx` layer 的 `lbs_weights/faces` 字段与 shape；`parents` 不再是 translation-only 解码的必要条件。
2. BEDLAM camera NPZ 是否包含可用外参及其 convention。
3. Stage1 最终选择 per-frame、clip median 还是 persistent EMA scale。
4. Stage1 bias 是否足够小，能否安全建立统一 world scene。
5. VGGT depth confidence 的数值方向和可用阈值。
6. NLF detector 在多人、遮挡和长视频上的 track front-end 指标。
7. GT depth 中人体/环境表面的具体语义和深度边缘统计。
8. NLF/HMR pose 被视为可信这一假设在真实数据上的误差统计；超出假设的样本只能 no-op/上报，TRSTR 不负责修 pose。

其中 1、2、3、4 是结构性前置条件；5、6、7、8 可以通过 smoke/audit 获得。

## 21. 术语表

| English | 中文 | 本项目含义 |
| --- | --- | --- |
| Query | 查询向量/查询单元 | 一个身体区域携带人体、场景和历史信息的 token |
| Token | 特征向量 | Transformer/attention 处理的固定维度表示 |
| Region Bank | 人体表面区域库 | 6890 顶点到约 96 区域的固定映射 |
| Patch | 局部窗口 | 区域投影周围的多尺度 depth 采样网格 |
| Attention | 注意力机制 | 根据内容动态加权多个特征 |
| Masked Attention | 掩码注意力 | 屏蔽 patch 中无效点和无效区域 |
| LBS | 线性混合蒙皮 | 多关节加权控制 SMPL 顶点 |
| FPS | 最远点采样 | 选择空间分布均匀的代表点 |
| Geodesic FPS | 测地最远点采样 | 沿网格表面距离选择代表点 |
| Observability | 可观测性 | 输入观测是否包含推断修正所需的信息 |
| Controllability | 可控性 | 单一 translation 是否足以解释区域变化并恢复已知平移扰动 |
| Robust Aggregation | 稳健聚合 | 对异常 depth/残差不敏感的聚合 |
| Displacement Vote | 平移投票 | 区域对整个人三维 translation 修正的连续估计 |
| Re-probing | 重新探测 | 更新 SMPL 后重新查询场景 |
| Rollout | 迭代展开 | 多轮共享权重修正过程 |
| No-op | 不执行修正 | 证据不足或输入已正确时保持原结果 |
| Abstention | 拒绝修正 | 模型明确表示当前证据不足 |
| Rollback | 回滚 | 新一轮更差时恢复上一轮结果 |
| Track ID | 轨迹身份编号 | 同一人在跨帧中的稳定标识 |
| Causal Memory | 因果记忆 | 只使用当前和过去帧的状态 |
| EMA | 指数移动平均 | 对历史状态递减加权的平滑 |
| Hysteresis | 滞回 | 状态建立和取消使用不同证据强度 |
| Teacher Forcing Gap | 教师强制域差异 | GT 训练输入与真实预测输入之间的分布差异 |
| Calibration | 校准 | 置信度/不确定度与真实错误概率一致 |
| Ablation | 消融实验 | 移除某模块验证其真实贡献 |
