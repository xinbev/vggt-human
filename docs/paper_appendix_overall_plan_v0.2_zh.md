# GroundedHuman 正文与附录联合规划（v0.2 中文主版本）

## 1. 规划目标与边界

本文档面向 ICLR 2027 初投稿，目标是在不改变 GroundedHuman 核心研究主张的前提下，将当前约 16 页主文压缩为 9 页左右的审稿主线，并建立一套能够支撑方法复现、贡献归因和失败分析的附录结构。

当前唯一主版本为：

- `.paper/my_paper/versions/v0.2/paper_v0.2/main_zh.tex`
- 最新核验 PDF：`outputs/debug/paper_v0.2_zh_intro_update/main_zh.pdf`

英文稿不自动同步。只有用户明确要求时，才从当时最新的中文版同步英文稿。

本规划参考：

- `.paper/iclr2027/iclr2027_conference.tex`
- `.paper/base_pdf/Human3R.pdf`
- `.paper/base_pdf/UniSH.pdf`
- `.paper/base_pdf/UniCon3R.pdf`
- `.paper/base_pdf/HAMSt3R Human-Aware Multi-view Stereo 3D Reconstruction.pdf`
- `.paper/base_pdf/MetricHMSR.pdf`
- `.paper/base_pdf/GRAFT.pdf`
- `.paper/base_pdf/JOSH.pdf`
- `.paper/base_pdf/DuoMo Dual Motion Diffusion for World-Space Human Reconstruction.pdf`
- `.paper/base_pdf/Crowd4D.pdf`
- `.paper/base_pdf/VGGT-omega.pdf`

## 2. ICLR 2027 版式约束

根据仓库中的 ICLR 2027 模板：

1. 初投稿主文严格上限为 9 页；rebuttal/camera-ready 为 10 页。
2. 参考文献页数不限。
3. 模板在参考文献之后提供 `\appendix` 入口，但本地模板没有明确写明附录是否完全豁免页数；最终投稿前必须再核对 ICLR 2027 官网和 OpenReview submission instructions，不能仅依据模板假定附录页数无限。
4. 摘要必须为单段。
5. AI use statement 为必需内容，不计入主文页数，且不超过 1 页。
6. Ethics statement 和 Reproducibility statement 为推荐内容，不计入主文页数，应放在主文结尾、参考文献之前。
7. PDF 使用 US Letter，不能调整官方样式文件、字号或版心来压页。

当前中文 PDF 共 18 页，其中主文延伸到第 16 页，参考文献位于第 17--18 页。主文需要压缩约 7 页，不能只靠缩图或缩字号解决，必须重新分配正文和附录的证据职责。

## 3. 论文主线与必须闭合的证据链

GroundedHuman 的正文应围绕一个中心判断展开：具有物理尺寸的参数化人体可以成为相对场景、相机平移和人体世界轨迹之间的共享 metric anchor。

正文必须依次回答四个问题：

1. **为什么需要人体度量锚点？** 相对场景与米制人体存在 gauge mismatch，深度与相机平移可以共同缩放而保持投影不变。
2. **如何恢复共享尺度？** 可见人体前表面给出 analytic coarse scale，person-conditioned residual affine 修正残余尺度和 depth bias，clip-level consensus 将尺度传播到场景深度与相机平移。
3. **如何改善人体在场景中的局部落位？** TRSTR 只更新 root translation，以 canonical regions、ray/tangent votes、gates 和 uncertainty 聚合局部几何证据，不改变 pose 与 shape。
4. **证据是否覆盖全部主张？** 3DPW 验证局部人体不退化，EMDB-2 验证世界运动，Bonn 验证无 GT scale/shift 的米制深度，核心消融验证各模块因果贡献，定性/物理指标验证人体--场景几何关系。

TUM-Dynamic 的 ATE 主要验证冻结的 VGGT-Ω camera backbone 是否为长序列提供稳定基础，不能写成 GroundedHuman 新模块的直接性能增益。

## 4. 推荐的 9 页正文结构

| 目标页 | 内容 | 必须保留 | 应压缩或迁移 |
|---|---|---|---|
| 1 | 标题、摘要、缩小后的 teaser、引言开头 | 任务、metric gauge 问题、核心洞察 | teaser 说明性文字压缩 |
| 2 | 引言完成、紧凑 Related Work | 单段贡献、最接近工作差异 | 四个相关工作小节压为约 0.7 页，扩展比较移附录 |
| 3 | 方法总览、问题定义、坐标与共享尺度 | 主方法图、输入输出、metricization 总式 | VGGT-Ω/NLF 输出字段细节移附录 |
| 4 | analytic coarse scale、residual affine、clip consensus | 前表面鲁棒比例、残差校准、camera translation propagation | 阈值、异常处理、anchor 网络细节移附录 |
| 5 | TRSTR 与训练概述 | translation-only 约束、区域投票聚合、总训练目标 | 96 区域构造、probe 细节、完整 loss 移附录 |
| 6 | 实验设置、局部人体结果、主模块消融 | 数据集/指标一句话、NLF/Human3R/Ours、compact causal ablation | 完整 baseline 表和协议移附录 |
| 7 | EMDB-2 世界运动结果 | 主要对比、W-MPJPE/RTE 解释 | RICH、完整输入/输出属性表移附录 |
| 8 | Bonn 米制深度、核心人体--场景定性图 | 无 GT scale/shift、图 5、必要物理解释 | 图 4、图 6、图 7及完整深度 baseline 表移附录 |
| 9 | 统一分析、局限、结论 | 证据闭环、适用边界、结论 | 重复实验数字和大段未来工作压缩 |

正文之后按模板顺序放置：AI use statement、Ethics statement、Reproducibility statement、References、Appendix。

### 4.1 正文建议保留的图表

- 图 1：teaser，缩短图注并适度压缩高度。
- 图 2：完整方法总览，必须保留。
- 图 5：人体--场景几何交互定性比较，作为 TRSTR 的可视证据保留。
- 表 1（压缩版）：只保留最相关局部 HMR 对比，必须加入 NLF 行，因为正文声称 Ours 保持 NLF 输出。
- 表 2（压缩版）：正文只放 EMDB-2，并保留 JOSH、Human3R、MetricHMSR、UniCon3R、UniSH、Ours 等最接近方法；RICH 与完整 capability columns 移附录。
- 表 3（压缩版）：正文保留 VGGT、$\pi^3$、Human3R、UniSH、Ours 及少量关键基线；完整方法列表移附录。
- 新增一张 compact causal ablation：这是正文当前最重要的缺失证据。

### 4.2 正文建议迁出的内容

- 当前图 3：TRSTR 详细结构图，迁入附录 B。
- 当前图 4：三组时序 4D 重建，迁入附录 I。
- 当前图 6：TUM-Dynamic 与 Bonn 多视图曲线，迁入附录 H；正文仅保留一段 backbone diagnostic。
- 当前图 7：真实长度测量，迁入附录 I。
- 18 个编号公式压缩为约 4--6 个决定性公式；完整推导和训练损失迁入附录 A--C。
- BEDLAM 扰动课程、teacher 过滤、各 loss 解释迁入附录 C。

## 5. 推荐附录目录与逐节内容

建议附录总长约 9--12 页。若投稿规则限制补充页数，优先保留 A--G，减少 I 中的普通定性图。

### Appendix A. Notation, Coordinate Systems, and Metric Gauge（约 1--1.5 页）

**目的：** 消除尺度、坐标方向和相机变换歧义，使后续算法可复现。

应写内容：

1. 符号表：batch、time、person、vertex、anchor、region、feature level 维度。
2. camera/world 坐标方向，camera-to-world 与 world-to-camera 约定。
3. Z-depth、Euclidean depth、ray depth 的区别。
4. relative depth、metric depth、global scale、depth bias 的定义。
5. SMPL 以米为单位的具体接口约定。
6. 共享尺度如何作用于 scene depth、camera translation 和 human root。
7. gauge ambiguity 与人体尺度可辨识性的简短推导。
8. scale/bias clamp、$\varepsilon$、有效范围和单位。

建议资产：一张符号表、一张坐标链示意图、完整 metric propagation 公式。

### Appendix B. Complete GroundedHuman Architecture and Inference（约 1.5--2 页）

**目的：** 给出正文方法总览图无法容纳的接口、网络和完整推理流程。

#### B.1 VGGT-Ω 与 NLF 接口及 baseline 路径

- 输入分辨率、clip shape、person slots。
- VGGT-Ω 提供的 depth、camera、multi-level features。
- NLF 提供的 SMPL pose、shape、root translation、confidence。
- 冻结模块、可训练模块、梯度边界。
- 保留的 baseline 路径与实验开关。

#### B.2 Visible-front-surface analytic scale

- SMPL 投影和 z-buffer-like front-surface 筛选。
- depth correspondence 构造。
- median/MAD 或其他 robust estimator。
- 有效像素最小数、置信阈值、异常 ratio 过滤。
- 单人和多人 scale proposals 的聚合。

#### B.3 HSI residual affine

- 24 anchors 的来源、特征组成和 shape。
- 局部窗口、窗口回退和 scene-token 查询。
- Transformer/MLP 层数、hidden dimension、attention heads。
- residual scale 与 depth bias 的参数化、初始化和范围。

#### B.4 Clip-level scale propagation and fallback

- frame/person/clip 三层聚合。
- 低置信帧与无人帧的 fallback。
- camera translation propagation。
- 长序列或滑动 clip 的状态衔接。

#### B.5 TRSTR

- 96 canonical regions 的 dominant-LBS 构造。
- 每区采样点数和多尺度 probe 半径。
- self-surface、environment、other-human 三类证据分流。
- ray/tangent basis 和 bounded regional vote。
- region gate、log-variance、person gate。
- 两轮迭代与重新投影/重新采样。
- translation-only invariance：pose 与 shape 的变化应为 0 或浮点误差范围。

建议资产：当前图 3、完整推理伪代码（Algorithm 1）、shape/dtype/device 表。

### Appendix C. Training Objectives and Implementation Details（约 1.5 页）

**目的：** 完整回答“模型如何训练、哪些部分可学习、如何复现”。

#### C.1 Training data and sampling

- BEDLAM 训练/验证切分、序列筛选、人物可见性规则。
- clip 长度、帧采样、人数分布、crop/resize。
- 是否混用 3DPW/EMDB/RICH 或伪标签；若没有则明确写没有。

#### C.2 Scale-stage curriculum

- coarse-residual 扰动类型、幅度、采样概率和阶段变化。
- teacher/target 的生成和过滤。
- scale、bias、identity、no-worse 等损失的完整公式。

#### C.3 TRSTR curriculum

- clean、scale-error、NLF-like、mixed-error 状态的构造。
- staged remaining vote target。
- vote、translation、gate、regularization、identity、safe/no-worse、monotonic losses。

#### C.4 Optimization and hardware

- optimizer、learning rate、scheduler、warm-up、weight decay。
- batch size、gradient accumulation、gradient clipping、mixed precision。
- epochs/steps、随机种子、checkpoint selection。
- GPU 型号/数量、训练时长、软件版本。
- 所有 loss 权重及其选择依据；建议说明是否通过 held-out validation 的量级匹配或敏感性实验确定。

建议资产：完整超参数表、loss-weight 表、训练流程 Algorithm 2。

### Appendix D. Datasets and Evaluation Protocols（约 1--1.5 页）

**目的：** 统一解释正文四类任务的输入、对齐和 GT 使用边界。

建议先给一张总表，列：Dataset、Task、Split、Input frames/views、Alignment、Metrics、Test-time GT scale/shift、Output object。

逐数据集说明：

- 3DPW/EMDB-1：PA-MPJPE、MPJPE、PVE 的关节集合、对齐和单位。
- EMDB-2/RICH：100-frame clip 划分、WA-MPJPE、W-MPJPE、RTE 对齐方式和平均规则。
- TUM-Dynamic：view sampling、Sim(3) 对齐、ATE 计算及为何该实验属于 backbone diagnostic。
- Bonn：valid mask、crop/resize、请求视图数与实际帧数、Abs Rel、$\delta<1.25$，明确测试时不使用 GT scale/shift。
- 定性比较：相机视角、mesh/scene rendering、颜色、是否使用相同输入和后处理。
- baseline 数值来源；Human3R 曲线若由图中数字化，应说明工具、采样误差和“approximate”属性。

### Appendix E. Component Ablations and Mechanism Diagnostics（约 2--3 页）

**目的：** 证明收益来自本文模块，而不是 VGGT-Ω 与 NLF 的简单拼接。

#### E.1 主模块累加消融

必须包含：

1. VGGT-Ω + NLF naive composition。
2. + analytic coarse scale。
3. + residual scale。
4. + depth bias。
5. + clip-level consensus / camera propagation。
6. + TRSTR（full model）。

正文保留压缩版，至少报告 EMDB-2 的 W-MPJPE/RTE 和 Bonn 的 Abs Rel/$\delta<1.25$；附录给完整 WA-MPJPE、W-MPJPE、RTE、Abs Rel、$\delta$ 及 per-sequence 结果。

#### E.2 Analytic scale design

- visible front surface vs root/joint/单点深度。
- median vs mean/trimmed mean。
- z-buffer 筛选 on/off。
- 单帧 vs clip-level scale。
- 单人 vs multi-person confidence aggregation。

#### E.3 HSI residual affine

- no residual affine。
- scale-only vs bias-only vs scale+bias。
- 去掉 local scene features。
- anchor 数量与特征层数。
- shuffle person conditioning。
- zero-residual counterfactual。

#### E.4 TRSTR

- no TRSTR。
- joint tokens vs 96 regions。
- single-scale vs multi-scale probes。
- other-human filtering on/off。
- XYZ vote vs ray/tangent vote。
- region gate、uncertainty weighting、person gate 分别移除。
- 1/2/3 iterations 与最大步长。
- identity/no-worse/monotonic losses 的移除实验。
- 数值验证 pose 与 shape 不变。

#### E.5 机制可视化

- 区域 vote、gate 和 uncertainty 的人体表面热图。
- 同一案例中 coarse scale、residual scale、bias 和 root update 的逐阶段变化。
- matched control：与等尺寸普通 scene-token slice 比较，避免只用 attention map 作为因果解释。

所有尚未运行的数值必须标记为“待实验”，不得根据趋势或正文结果补写。

### Appendix F. Robustness and Sensitivity（约 1.5--2 页）

**目的：** 回答人体度量锚点在现实退化条件下是否可靠。

优先实验：

1. 遮挡比例、截断比例、人体可见面积分桶。
2. 连续无人/低置信帧比例与 fallback。
3. 单人、双人和多人；多人尺度不一致或互遮挡。
4. SMPL height/shape 受控扰动：建议 $\pm5\%$、$\pm10\%$、$\pm15\%$。
5. relative depth 噪声、局部结构错误和内参误差。
6. clip length、anchor 数、region 数、probe radius、confidence threshold。
7. TRSTR 对已经正确初始化的样本是否 no-worse。

其中“人体真实尺寸或 SMPL shape 偏差对绝对尺度的影响”是本文最关键的稳健性实验，应优先于增加普通定性图。

### Appendix G. Efficiency, Parameters, and Scalability（约 0.5--1 页）

**目的：** 量化模块代价并说明视频长度/人数扩展性。

- 总参数、可训练参数、FLOPs（若可可靠计算）。
- VGGT-Ω、NLF、front-surface extraction、HSI、TRSTR、fusion 的分项时间。
- 固定 GPU、分辨率、clip length、人数和是否含数据加载。
- 不同 clip length 与人数下的峰值显存、latency、FPS。
- TRSTR 迭代次数的 quality--runtime trade-off。

### Appendix H. Additional Quantitative Results（约 1--1.5 页）

**目的：** 保存完整比较矩阵，但不稀释正文主结论。

- 完整表 1：3DPW 与 EMDB-1 全部方法。
- 完整表 2：Offline/Online 或 World-HMR/Unified 分组、RICH 列、capability columns。
- 完整表 3：全部视频深度方法。
- TUM-Dynamic 全 view-count 曲线及数值表。
- Bonn oracle-scale diagnostic，明确它只测 VGGT-Ω 相对几何上界，不是方法预测结果。
- per-sequence/per-category 结果和多随机种子方差（若可运行）。
- 若已有可靠 evaluator，可加入 Float、Penetration、Maximum Penetration、Collision Ratio、support distance 等局部几何指标；若没有，不应把表 2/表 3 描述为对局部接触质量的直接定量证明。

### Appendix I. Additional Qualitative Results and Supplementary Video（约 1.5--2 页）

**目的：** 展示静态指标难以表达的时序与局部几何行为。

- 当前图 4：时序人体--场景重建与轨迹。
- 当前图 6：相机/深度曲线。
- 当前图 7：真实世界长度测量。
- 图 5 的高分辨率放大版本。
- 更多遮挡、躺卧、倚靠、坐姿、多人、户外和长序列案例。
- 同一案例的 `relative baseline -> coarse scale -> residual affine -> TRSTR` 阶段结果。
- 补充视频 roadmap：输入、世界场景、人体/相机轨迹、逐阶段 scale、TRSTR before/after、失败案例。

### Appendix J. Failure Cases, Limitations, and Responsible Use（约 1 页）

**目的：** 将局限连接到明确的失败因果链，而不是重复结论段。

建议按原因组织：

1. 人体长期不可见/严重截断 -> anchor support 不足 -> coarse scale 不稳定。
2. SMPL shape/height 错误 -> metric anchor 本身有偏 -> 全局尺度偏差。
3. VGGT-Ω 相对结构错误 -> local probes 错误 -> residual/TRSTR 无法恢复。
4. 动态物体污染或多人互遮挡 -> environment/other-human 分流失败。
5. TRSTR 为 translation-only，无法修正错误 pose、shape 或细粒度 contact。
6. 规范人体尺寸对不同体型、儿童或特殊人群可能产生系统偏差。
7. 人体视频数据的隐私、授权与潜在监控误用风险。

每类至少配一张失败案例或一个量化分桶结果，并写清检测信号、当前 fallback 和仍未解决的边界。

## 6. 主文必须新增的一张消融表

建议表结构：

| Variant | Analytic scale | Residual scale | Depth bias | Clip consensus / camera propagation | TRSTR | EMDB-2 W-MPJPE ↓ | EMDB-2 RTE ↓ | Bonn Abs Rel ↓ | Bonn $\delta<1.25$ ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VGGT-Ω + NLF |  |  |  |  |  | 待实验 | 待实验 | 待实验 | 待实验 |
| + coarse scale | ✓ |  |  |  |  | 待实验 | 待实验 | 待实验 | 待实验 |
| + residual scale | ✓ | ✓ |  |  |  | 待实验 | 待实验 | 待实验 | 待实验 |
| + depth bias | ✓ | ✓ | ✓ |  |  | 待实验 | 待实验 | 待实验 | 待实验 |
| + clip consensus | ✓ | ✓ | ✓ | ✓ |  | 待实验 | 待实验 | 待实验 | 待实验 |
| Full GroundedHuman | ✓ | ✓ | ✓ | ✓ | ✓ | 待实验 | 待实验 | 待实验 | 待实验 |

若一张表过宽，正文只保留 W-MPJPE、RTE、Abs Rel 三个指标，完整四指标表放附录。

## 7. 复现声明与责任声明规划

### 7.1 Reproducibility statement（主文末、参考文献前）

只写一段导航文字，不在声明里重复实现细节。应明确指向：

- Appendix A：坐标和尺度规范。
- Appendix B：完整推理算法与网络结构。
- Appendix C：训练配置、loss 和硬件。
- Appendix D：数据集与评测协议。
- Appendix E--F：消融与鲁棒性。
- 匿名代码、配置、checkpoint 和评测脚本的补充材料位置（若投稿时可提供）。

### 7.2 Ethics statement（主文末、参考文献前）

说明使用公开人体视频数据集、隐私与许可遵循、体型偏差、监控误用风险，以及不进行身份识别的研究边界。没有核实的数据许可证或 IRB 信息不得推测，应逐数据集核对后填写。

### 7.3 AI use statement（主文末、参考文献前）

沿用 ICLR 2027 模板要求，明确 AI 仅参与哪些写作/图示任务、哪些技术内容由作者独立完成和验证，以及作者对最终内容负责。最终英文同步时再根据官方措辞制作英文版。

## 8. 实验优先级

### P0：投稿前必须完成

1. 主模块累加消融，并在正文保留 compact 版本。
2. 完整训练配置、loss 权重、冻结边界和推理伪代码。
3. 所有数据集、对齐、mask、GT scale/shift 使用规则。
4. 人体尺度/shape 偏差敏感性。
5. TRSTR pose/shape 不变性数值核验。
6. runtime、参数量与显存。
7. 失败案例及其成因。

### P1：强烈建议完成

1. 遮挡、截断、人体像素面积分桶。
2. 多人体 scale consensus 与 other-human filtering。
3. 无人体/低置信度帧 fallback。
4. relative depth 与内参噪声鲁棒性。
5. TRSTR 的 float/penetration/support distance 定量指标。
6. per-sequence 结果与随机种子方差。

### P2：有余力再做

1. 更多 object metric measurements。
2. 额外 backbone 或模型规模迁移。
3. 交互网页和大规模补充视频。
4. 更多普通成功案例。

## 9. 正文与附录交叉引用规则

1. 正文每个被压缩的细节都应有明确附录入口，例如“完整坐标约定见 Appendix A”。
2. 附录不重复正文主结果，只补全公式、设置、诊断和完整表。
3. 正文的每项 contribution 至少对应一张正文证据表/图，并在附录有进一步验证。
4. oracle、GT-assisted diagnostic 和正式方法预测必须在标题、图注和正文中明确区分。
5. baseline 数值来源、公平比较组和输入依赖必须一致；跨设置方法只作背景参照。
6. 图表编号全部使用 `\label`/`\ref`，不要手写固定编号。
7. 尚未运行的实验统一写“待实验”，不得用预期趋势或合成数字替代。

## 10. 推荐实施顺序

### 阶段 1：先补证据，不先压页

- 建立消融配置与服务器 `.sh` 脚本。
- 补 P0 实验并保存到 `outputs/eval/`。
- 核对所有指标与 baseline 来源。

### 阶段 2：搭建附录骨架

- 在中文版 `main_zh.tex` 的参考文献后加入 `\appendix`。
- 先完成 A--D 的复现性内容，再写 E--J。
- 图表和导出资产放在 `outputs/`，确认后再纳入论文版本目录。

### 阶段 3：正文重构到 9 页

- 按第 4 节页级规划压缩 Related Work、Method 和 Experiments。
- 新增正文主消融表。
- 将完整公式、完整 baseline、扩展定性移入附录。
- 不修改 ICLR style 文件、不缩字号、不压边距。

### 阶段 4：一致性与投稿检查

- 检查 contribution -> method -> experiment -> appendix 的一一映射。
- 检查正文 9 页边界、参考文献起始页和附录起始页。
- 检查匿名性、US Letter、公式/表图可读性、引用完整性。
- 核对 ICLR 2027 官网对 appendix/supplementary 的最新页数规则。
- 只有用户明确要求时，才同步英文版。

## 11. 当前主要风险

1. **主文超页：** 当前主文约 16 页，必须结构性压缩约 7 页。
2. **贡献归因不足：** 当前没有核心模块消融，无法区分 analytic scale、residual affine、clip consensus 和 TRSTR 的独立作用。
3. **人体尺度先验风险：** 真实身高/shape 偏差可能系统性影响绝对尺度，当前尚无敏感性证据。
4. **局部交互证据不足：** 图 5 只有定性结果；若继续强调 contact-aware 或支撑关系，应补充 float/penetration/support 指标或弱化表述。
5. **比较口径风险：** TUM ATE 是 backbone diagnostic；Human3R 曲线为近似数字化；完整 baseline 的预处理与输入依赖需要分组说明。
6. **复现信息缺失：** 正文已经声称完整训练配置见附录，但当前实际没有附录。
7. **投稿规则不确定：** 本地模板没有明确保证附录不计页数，必须在提交前核对官网规则。

## 12. 完成判据

附录设计完成并可进入英文同步前，应同时满足：

- 正文不超过 ICLR 初投稿主文页数上限。
- 正文保留一张核心消融表。
- 所有主要模块有可复现的结构、超参数和 loss 描述。
- 所有 benchmark 有明确 split、alignment、mask 和 GT 使用说明。
- 人体尺度偏差、遮挡/截断和无人帧至少有一组稳健性结果。
- TRSTR 的 translation-only 不变量得到数值验证。
- runtime/显存/参数量可复核。
- 至少一组失败案例按因果链解释。
- AI、Ethics、Reproducibility 三项声明位置符合模板。
- 中英文若存在差异，以中文版为准；英文仅在用户明确要求后同步。

