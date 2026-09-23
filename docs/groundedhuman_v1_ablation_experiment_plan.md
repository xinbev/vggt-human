# GroundedHuman v1 消融实验设计

日期：2026-09-23

## 1. 目标与当前缺口

本文方法包含两个连续阶段：

1. **Body-Anchored Metric Calibration**：解析人体表面尺度初始化、Body-Anchored Scene Attention、残差 scale/bias readout、跨帧共识。
2. **Dynamic Interaction Grounding**：四类空间 support、self-surface/environment 双通道、区域 vote、region gate、log-variance、person gate、两次共享参数的动态重查询。

当前 v1 主文只有 `w/o Dynamic Interaction Grounding` 与完整模型两行结果。这能说明整体 placement refinement 有效，但还不能分别支持以下论文主张：

- learned body--scene query 是否优于解析尺度初始化；
- clip-level shared calibration 是否优于逐帧尺度；
- hierarchical context 是否确实需要多尺度 support；
- self-surface 与 environment 分流是否有贡献；
- 第二次更新的收益来自“多走一步”，还是来自更新位置后的关系重编码；
- gate 和 uncertainty weighting 是否能抑制错误更新；
- clean/no-worse/monotonic 等训练约束是否真正降低退化率。

因此建议把消融分成两张主表和三组附录实验。不要把所有指标塞进一张超宽表。

## 2. 统一实验纪律

- 所有变体固定 VGGT-Ω、NLF、数据划分、输入帧、person matching、RICH 物理评测分母与聚合规则。
- 不覆盖 baseline 配置或 checkpoint。每个正式变体使用独立配置、输出目录和 checkpoint。
- 能通过推理时关闭实现的实验称为 **inference decomposition**；改变特征结构、训练分布或损失的实验必须重新训练，不能只做 checkpoint surgery 后当作正式因果结论。
- 结构性训练变体至少采用相同初始化、训练步数、batch sampling 和 checkpoint selection。优先对最重要的 3--4 个变体运行 3 个随机种子，报告 mean ± std。
- 统计不把视频帧当作独立样本。建议按 sequence 做 paired bootstrap，报告相对 Full 的 95% CI。
- Bonn 深度不得做 GT scale/shift fitting。TUM 的尺度结论使用 SE(3)-aligned ATE 和 residual log-scale error；Sim(3)-ATE 只能说明轨迹形状，不能证明米制尺度恢复。
- Dynamic Interaction Grounding 不改变 pose/shape，因此 3DPW、EMDB-1 的 PA-MPJPE/PVE 不适合作为其主要消融指标。优先使用 RICH 物理指标、EMDB-2/RICH world-motion 指标和退化率。

## 3. 主表 A：Body-Anchored Metric Calibration

### 3.1 推荐变体

| ID | 解析初始化 | learned residual scale | depth bias | clip 共识 | 用途 | 是否需重训 |
|---|---:|---:|---:|---:|---|---:|
| M0 |  |  |  |  | 原始 VGGT-Ω，作为任意尺度诊断基线 | 否 |
| M1 | ✓ |  |  |  | 验证 human-surface analytic initialization | 否 |
| M2 | ✓ | ✓ |  |  | 隔离 Body-Anchored Scene Attention 的 residual scale 增益 | 否，bias 置零 |
| M3 | ✓ | ✓ | ✓ |  | 检验 affine depth readout 中 bias 的额外作用 | 否 |
| M4 | ✓ | ✓ | ✓ | ✓ | 完整 Body-Anchored Metric Calibration | 否 |

M0 不能与人体米制结果直接混为一个公平系统，只用于显示原始 gauge ambiguity。真正关键的比较是 M1 → M2 和 M3 → M4。

### 3.2 指标与数据集

主表建议列：

- **Bonn**：metric Abs Rel ↓、δ<1.25 ↑，禁止 GT scale/shift fitting。
- **TUM-Dynamic**：SE(3)-ATE (m) ↓、`|log a*|` ↓。`a*` 仅用于评测诊断，不回写预测。
- 可选 **EMDB-2**：W-MPJPE ↓、RTE ↓，用于说明共享 metric gauge 对 world motion 的影响；WA-MPJPE 经过全段 similarity alignment，对统一尺度变化不够敏感，应作为补充而非尺度主证据。

建议主表布局：

| Variant | Bonn Abs Rel ↓ | Bonn δ1 ↑ | TUM ATE-SE3 ↓ | TUM log-scale err. ↓ | EMDB-2 W-MPJPE ↓ | RTE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| M1 Analytic initialization |  |  |  |  |  |  |
| M2 + Body-Anchored residual scale |  |  |  |  |  |  |
| M3 + depth bias |  |  |  |  |  |  |
| M4 + clip consensus (Full calibration) |  |  |  |  |  |  |

### 3.3 附加尺度消融

1. **Temporal aggregation**：`per_frame`、EMA、`clip_median`。代码已有三种模式，属于低成本推理消融。
2. **聚合统计量**：mean、median、confidence-weighted median/trimmed mean。用于验证 median 的稳健性；新增实现后无需重训。
3. **Body anchor 组成**：24 anchors；去掉 hand anchors；只用 joint anchors；只用 global anchor。正式结果需重训。
4. **Anchor scene neighborhood**：1×1、3×3、5×5。若 cross-attention 接受可变 token 数，可先做 inference sensitivity；正式主张仍建议重训。
5. **关系特征**：去掉 normal、去掉 3D offset、去掉 depth residual、只保留 RGB/scene feature。建议采用 full-minus-one，需重训。
6. **人级聚合**：uniform 与 NLF-confidence weighting。可在推理端实现，适合多人/遮挡子集。

## 4. 主表 B：Dynamic Interaction Grounding

### 4.1 最有说服力的因果链

建议把当前两行表扩成下列六行。若版面紧张，D2、D3、D4 放主文，其他放附录。

| ID | Hierarchical supports | self/env 分流 | reliability weighting | 更新次数 | 更新后重编码 | 说明 |
|---|---:|---:|---:|---:|---:|---|
| D0 |  |  |  | 0 |  | w/o Dynamic Interaction Grounding |
| D1 | ✓ | ✓ | ✓ | 1 |  | 单步 placement readout |
| D2 | ✓ | ✓ | ✓ | 2 |  | 两步但第二步复用第一次 context，隔离“多一步” |
| D3 | ✓ | ✓ | ✓ | 2 | ✓ | 完整 dynamic recomputation |
| D4 | single 7×7 | ✓ | ✓ | 2 | ✓ | w/o hierarchical multi-scale context |
| D5 | ✓ | merged | ✓ | 2 | ✓ | w/o self/environment separation |
| D6 | ✓ | ✓ | uniform | 2 | ✓ | w/o learned reliability weighting |

其中 D2 对论文的“dynamic”主张最关键。若只有 D0 与 D3，审稿人无法判断提升来自区域关系建模、第二次更新，还是位置变化后的重新采样。D1、D2、D3 形成严格控制链：

`one step → two steps with frozen context → two steps with recomputed context`。

### 4.2 Reliability 的进一步拆分

完整权重为 `valid × region gate × exp(-logvar)`，之后还有 person gate。附录可做：

| Variant | region gate | uncertainty | person gate | 解释 |
|---|---:|---:|---:|---|
| Uniform valid-region mean |  |  |  | 无学习可靠性 |
| + region gate | ✓ |  |  | 局部有效性筛选 |
| + uncertainty | ✓ | ✓ |  | 区域不确定度加权 |
| + person gate | ✓ | ✓ | ✓ | 完整模型 |

这组正式消融最好保持特征维度不变并分别训练。仅在推理时把 gate 设为 1 或 logvar 设为 0，可以先作为诊断，但应在文中明确为 checkpoint decomposition。

### 4.3 指标

主指标：

- **RICH physical grounding**：Collision ratio ↓、Penetration ↓、Floating distance ↓、P.Max ↓；沿用当前 UniCon3R 协议。
- **RICH/EMDB-2 global motion**：W-MPJPE ↓、RTE ↓，避免只优化“看起来接触更好”却破坏全局轨迹。
- **Safety metrics**：worsening rate ↓、clean displacement ↓、strong-error improvement rate ↑、monotonic violation rate ↓。

建议增加一个 paired 指标：`Improved / unchanged / worsened (%)`。只报平均 Coll./Float 可能掩盖少量严重退化，尤其当前 P.Max 改善较小。

## 5. 训练策略与损失消融

该组放附录，主要解释为什么模型能在纠错同时尽量不破坏正确输入。

### 5.1 推荐训练变体

| ID | 训练状态/损失 | 核心问题 | 需重训 |
|---|---|---|---:|
| T0 | translation-perturbed only + vote/trans loss | 基础纠错能力 | ✓ |
| T1 | T0 + clean states + identity loss | 是否减少正确输入漂移 | ✓ |
| T2 | T1 + no-worse loss | 是否降低退化率 | ✓ |
| T3 | T2 + monotonic loss | 两步更新是否更稳定 | ✓ |
| T4 | T3 + scale-perturbed/mixed states | 是否处理 scale/placement 耦合 | ✓ |
| T5 | T4 + strong-error loss | 是否提升大误差恢复 | ✓，Full |

### 5.2 Vote target

比较：

- full remaining displacement；
- `remaining displacement / remaining iterations`（当前 staged target）；
- 只监督 final translation，不监督 regional vote。

报告每一步 translation error、第二步增益、monotonic violation 和最终 RICH 指标。该实验能验证 staged update supervision 是否与动态两步推理匹配。

## 6. 超参数与容量敏感性

这些实验不宜全部占主文，但适合附录回答“结果是否依赖特定设置”。

| 因素 | 建议取值 | 当前支持情况 | 主要指标 |
|---|---|---|---|
| 区域数 | 48 / 72 / 96 | 代码明确支持 | RICH 四指标、FPS、显存 |
| representative vertices | 4 / 8 / 16 | 配置可调，需重训 | RICH、有效 support ratio |
| 迭代次数 | 1 / 2 / 3 | 配置可调；共享参数 | RICH、worsening、FPS |
| fixed supports | [3] / [7] / [3,7] | 配置可调但会改变输入维度 | RICH、参数量；需重训 |
| adaptive radius max | 4 / 8 / 12 | 配置可调 | 遮挡与尺度分层结果 |
| min valid ratio | 0.10 / 0.25 / 0.40 | 配置可调 | coverage、worsening |
| human-depth tolerance | 0.10 / 0.18 / 0.25 m | 配置可调 | self/env 分类质量、RICH |
| translation cap | 0.10 / 0.22 / 0.50 m | 推理可调 | strong recovery、P.Max、worsening |
| clip length | 50 / 100 / 200 / 500 | 已有 Bonn 长度实验基础 | metric depth、scale variance |

不建议一次性做完整笛卡尔积。采用 one-factor-at-a-time，并始终保留当前 Full 设置为参照。

## 7. Robustness 与 failure-conditioned evaluation

这些不全是“消融”，但对 v1 的 limitation 最有补强价值。

1. **人体可见性/截断**：按有效 anchor pixel 数或可见 SMPL 表面比例分桶，报告尺度误差和 RICH 指标。
2. **人体尺度先验偏差**：对 body height/shape 施加 ±5%、±10%、±20% 的受控缩放，测 metric scale error。这直接对应儿童、非常规体型的 limitation。
3. **初始 translation error**：按 0--10、10--30、30--60、>60 cm 分桶，报告恢复率和退化率。
4. **scene depth 扰动**：乘性 scale noise、加性 bias、局部缺失/遮挡分别测试，避免把不同错误类型混在一起。
5. **多人干扰**：按 `other_human_ratio` 分桶；另比较“排除 other-human samples”与“当 environment 使用”。后者需新增开关并建议重训。
6. **支持类型**：standing/sitting/lying/hand support 分层。至少报告 sitting/lying，因为当前定性图重点展示这些场景。
7. **空 support 与低置信度**：报告 identity fallback、错误更新率和 coverage，而不只统计成功样本。
8. **输入长度稳定性**：除均值外，报告 per-frame scale 的 MAD/variance，解释 200 帧后 Bonn 曲线变差的原因。

## 8. 效率消融

报告同一 GPU、相同分辨率、相同人数和帧数下的增量成本：

| Variant | Params | FLOPs/frame | FPS | Peak memory | 说明 |
|---|---:|---:|---:|---:|---|
| VGGT-Ω + NLF |  |  |  |  | frozen baseline |
| + Metric Calibration |  |  |  |  | anchor query 开销 |
| + DIG, 1 iteration |  |  |  |  | regional context 开销 |
| + DIG, 2 iterations |  |  |  |  | Full |

同时画 `WA-MPJPE vs FPS` 或 `RICH Coll. vs FPS`，但不要将不同硬件、batch size 或预处理口径的公开 FPS 直接混成严格比较。

## 9. 推荐执行优先级

### P0：投稿前必须完成

1. M1/M2/M3/M4 的尺度校准主表。
2. TUM 改用保留尺度的 SE(3)-ATE；Sim(3)-ATE 只作辅助。
3. D0/D1/D2/D3，尤其固定上下文两步与动态重编码两步的对照。
4. D4、D5、D6 中至少完成两项，优先 D4（hierarchy）和 D6（reliability）。
5. 每个 placement 变体同时报告 RICH physical metrics 与 worsening/clean-drift 指标。

### P1：强烈建议

1. 48/72/96 regions 与 1/2/3 iterations。
2. T0→T5 中 identity/no-worse/monotonic 的训练消融。
3. 可见性、初始 translation error、other-human interference 分桶。
4. 模块级 FPS/显存/参数量。

### P2：有预算再做

1. Anchor 数量/组成与关系特征 full-minus-one。
2. support 半径、valid ratio、depth tolerance 的完整敏感性。
3. body-height bias、depth corruption 和接触类型分层。

## 10. 建议的文件与服务器组织

后续实施时建议新增而不是覆盖现有文件：

```text
configs/ablation/groundedhuman_v1/
  metric_calibration_*.yaml
  dynamic_grounding_*.yaml
  training_objective_*.yaml

scripts/eval/
  run_groundedhuman_v1_metric_ablation.sh
  run_groundedhuman_v1_grounding_ablation.sh
  run_groundedhuman_v1_robustness.sh

outputs/eval/groundedhuman_v1_ablation/
  metric_calibration/
  dynamic_grounding/
  robustness/
  efficiency/
```

本地仓库根目录为 `C:\Users\ROG\PycharmProjects\vggt-omega`，服务器对应根目录为 `/home/zhw/lab_users/xyb/home/projects/vggt-human`。服务器运行入口应统一为：

```bash
bash scripts/eval/run_groundedhuman_v1_metric_ablation.sh
bash scripts/eval/run_groundedhuman_v1_grounding_ablation.sh
bash scripts/eval/run_groundedhuman_v1_robustness.sh
```

当前仅完成实验设计，尚未创建上述配置/脚本，也未运行新实验。

## 11. 与当前实现的对应关系

- `configs/infer_smpl_hsi_v3_trstr_spatial.yaml`：当前 Full 推理设置，96 regions、8 representatives、2 iterations、[3,7] fixed supports、adaptive radius 8、annulus width 2、spatial-only。
- `configs/train_smpl_hsi_stage2_trstr_v3_refine.yaml`：当前主要 TRSTR 训练状态、损失与 selection metrics。
- `vggt_omega/models/heads/hsi_regional_translation_refiner.py`：区域 vote、region gate、logvar、person gate，以及每次迭代重新 decode/project/probe 的实现。
- `vggt_omega/models/geometry/regional_scene_probe.py`：fixed/adaptive/annulus supports，self-surface/environment 分流和 other-human interference。
- `vggt_omega/utils/hsi_affine.py`：`per_frame`、`clip_median`、EMA 三种 scene affine 模式。
- `scripts/eval/evaluate_rich_physical_grounding.py`：现有 RICH Coll./Pen./Float/P.Max 评测入口。

## 12. 论文表述边界

- 只有 M1→M2 改善，才能将尺度收益明确归因于 learned Body-Anchored Scene Attention，而非人体解析尺度本身。
- 只有 D2→D3 改善，才能有力支持“更新人体位置后重新编码 interaction relation”这一 dynamic 主张。
- 只有 D4/D5 改善，才能分别支持 hierarchical support 和 self/environment separation，而不是仅说明整个 TRSTR 有效。
- P.Max 若仍改善有限，应保持当前诚实解释：translation-only 无法修复 pose、shape 或局部 scene geometry 错误。
- 不得用 placeholder、未核对 checkpoint 的日志或 GT-aligned depth/camera 结果填表。
