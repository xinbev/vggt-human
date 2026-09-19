# 实验章节重构：以现有结果为中心

日期：2026-09-16

## 1. 本轮目标

把 v0.3 已有结果组织成清晰、有重点的论文实验章节，不重新设计一项必须新增大量实验才能完成的研究。参考 Human3R 的任务覆盖和结果解释方式，但不复制其章节目录或能力主张。

不新增训练、不重新评估、不编造数值、不修改 `.paper/` 原稿。本轮只重构文稿组织，所有数据和图均来自初稿材料；沿用不等于本轮已经复现或独立核验。使用正文样例的前提是文章仍描述产生这些结果的方法版本。

参考材料：

- 本地 `.paper/base_pdf/Human3R.pdf`，第 6–10 页，实验总述及第 4 节。
- `.paper/my_paper/versions/v0.3/paper_v0.3/main_en.tex`，现有方法和实验。
- 同目录 `appendix_en.tex`，Bonn 主表及补充材料。

本次未调用现有 skill。Human3R PDF 通过文本提取阅读，未对其中数值进行复算。

## 2. 从 Human3R 借鉴什么

Human3R 将局部人体、全局运动、相机/深度、拥挤场景展示及分析分开组织。它将多任务结果用于支撑一个系统级定位，详细比较则服务于该定位。本轮参考这种组织方式。

但它的 Analysis 有实际组件对照，不是只改写其他结果。我们没有对应结果时，不照搬相同验证口吻。其在线、实时、检测依赖及拥挤泛化论述也不自动适用于本系统。

**我们的定位建议：一个聚焦尺度与全局放置、同时保留局部人体质量的度量人景重建系统。**

写作优势应来自现有世界运动结果、可评价的度量深度，以及完整的人景输出。无需把论文写成全新的 HMR、相机骨干或专门的拥挤场景模型。

## 3. 最终推荐目录

```text
4. Experiments
   [One opening paragraph on evaluation scope]

   4.1 World-Space Human Motion
       Protocol and comparison
       Quantitative results
       Temporal reconstruction

   4.2 Metric Scene Reconstruction
       Metric video depth
       Camera geometry

   4.3 Local Human Reconstruction
       Local reconstruction quality
       Relation to the fixed human prior

   4.4 Human–Scene Placement
       Geometric placement
       Metric interpretation
       Short synthesis
```

缩进项为段落功能或加粗段首，不建议全部设为编号小节。

推荐篇幅：4.1 约 35%，4.2 约 30%，4.3 约 10%，4.4 约 25%。这是编辑优先级，不是会议规定。

**这一顺序让读者先看到核心结果，再看到场景输出，随后理解局部人体质量的来源，最后通过可视化理解完整重建。**

不新增独立的 Experimental Setup 大节；短协议说明就近放置。也不设一个没有新证据、仅重复前文数值的 Analysis 大节。

## 4. 三项创新怎样融入现有证据

| 选定创新 | 当前写作落点 | 对应现有结果 | 结果的解释层级 |
| --- | --- | --- | --- |
| 共享尺度与局部位移的结构化校准 | 把场景尺度和人物位置置于同一系统中组织处理 | EMDB-2、Bonn、世界重建图 | 支持系统能够输出世界运动与度量场景，不隔离各模块因果贡献 |
| 局部结构保持的受限校准 | 固定人体姿态、形状，更新尺度及根平移 | 原方法定义、3DPW 局部人体记录 | 说明更新变量和保留的人体先验，不宣称重新提高了 pose/shape |
| 证据驱动的选择性校准 | 人体区域探测、方向参数化、门控及重投影 | 原 TRSTR 描述、人景放置图 | 说明实现及展示效果，不虚构方向可观测性消融 |

实验标题用任务名，段落解释用贡献逻辑。读者不必按 C1/C2/C3 阅读三套独立实验，也能理解系统的设计分工。

关于贡献用语：当前材料描述级联校准和标量不确定性，推荐在现版本正文用“结构化尺度与位移校准”“局部结构保持”“区域证据驱动的选择性修正”。更强的“联合消解歧义”“方向可观测性”可作为研究动机或后续目标，但不能仅靠重命名就当成已实现、已实证的机制。

## 5. 实验开场

### 写作目标

三至四句交代整体评价目标与四类输出。无需在开场给出全部数值，也无需逐条宣告“验证贡献一、二、三”。

### 英文草稿

> We evaluate GroundedHuman from the perspective of metric human–scene reconstruction. We first assess human motion in world coordinates, followed by scene depth and camera geometry. We then report local human reconstruction quality and examine human placement in the reconstructed environment. Together, these evaluations characterize the system's global geometry, retained human detail, and joint human–scene output.

这段给出阅读路线，不声称已经完成模块因果验证。

## 6. 4.1 World-Space Human Motion

### 材料安排

- 使用原稿 EMDB-2 主表，现有指标为 58.468 mm WA-MPJPE、149.959 mm W-MPJPE、1.079% RTE。
- 将 `fig3_temporal_4d.png` 对应的时序重建图移到世界运动结果附近。
- 只评测 EMDB-2 时，正文表聚焦 EMDB-2；无需为模仿参考论文保留自己没有结果的 RICH 列。明确本节覆盖范围，不暗示在 RICH 上也已验证。

### 段落 1：协议

一句交代数据集和 clip 长度，随后用简短语句区分 WA-MPJPE、W-MPJPE、RTE 的对齐方式和单位。具体协议仍按现有 evaluator 的实际定义填写。

### 段落 2：最强结果

集中报告三个指标，挑一组最能解释优势的比较。不要一口气对三篇论文各列三个百分比。

初稿中已有：

- 相对 Human3R，W-MPJPE 由 267.9 降至 149.959 mm，约降低 44.0%。
- 相对 JOSH，W-MPJPE 由 174.7 降至 149.959 mm，约降低 14.2%。

上述比较可择一作为主句，其他结果留表中。它们是原稿表格的比较，不是本轮重新运行方法获得的数据。

### 段落 3：解释与展示

强调系统的世界坐标运动质量与整体人景输出，配合时序图说明输出形式。图中的曲线用于展示，不额外称为“准确度证明”。

### 英文草稿

> On EMDB-2, GroundedHuman achieves 58.468 mm WA-MPJPE, 149.959 mm W-MPJPE, and 1.079% RTE. Relative to the Human3R result reported in the comparison table, W-MPJPE is reduced by 44.0%. The results establish the system's world-space motion accuracy while it also produces scene geometry and camera trajectories. Figure X complements the quantitative comparison by visualizing human motion within the reconstructed environment.

### 有力但不过界的结论

“The system achieves strong world-space motion accuracy” 可以由主表承载；“TRSTR alone causes the improvement” 或 “the ambiguity is provably resolved” 则不是当前比较能够支持的结论。

不要在没有同骨干对照时说“把两个模型拼起来绝对做不到”；无需主动贬低自己的模块组合，直接解释具体几何组织及结果。

## 7. 4.2 Metric Scene Reconstruction

### 本节主次

Bonn 度量深度为核心，TUM 相机结果为几何基础的补充。把“我们不仅重建人体，还得到可解释的场景尺度”写清楚。

### 第一部分：Metric Video Depth

使用原附录 `tab:bonn-depth`，将自身实际度量结果移到正文。主指标为 0.08599 Abs Rel 与 0.96377 threshold accuracy，明确评估时不使用 GT scale/shift。

写三层内容：任务协议 → 两个结果 → 与论文目标的关系。

### 英文草稿

> We evaluate metric video depth on Bonn without ground-truth scale or shift alignment at test time. GroundedHuman obtains an Abs Rel of 0.08599 and a threshold accuracy of 96.377% at δ < 1.25. This evaluation complements the world-motion results by assessing the calibrated scene depth in physical units, rather than only the placement of the reconstructed humans.

若引用原稿 Human3R 对照，写 Abs Rel 相近、threshold accuracy 较高即可，不写整个 depth benchmark 全面最优。原稿表中存在更优方法，不删掉这些方法来制造最优排名。

VGGT 等相对深度方法若需要 GT 对齐才能得到表中结果，必须注明并与无 GT 对齐结果区分；协议未明确的行不能无说明地当成同口径排名。

### 第二部分：Camera Geometry

沿用原稿 TUM 曲线，主要事实是现有 50–500 views 结果与 500 views 时 0.00692 m 的 Sim(3)-aligned ATE。保留“VGGT-Ω backbone”的结果归属。

### 英文草稿

> We additionally report the camera geometry supplied by the VGGT-Ω backbone on TUM-Dynamic. Under Sim(3) alignment, the reported ATE remains below 7 mm over the evaluated 50–500-view range. These results characterize the relative camera trajectory used for world-space reconstruction.

不把 Sim(3)-aligned ATE 解释成度量尺度恢复效果。无需在正文展开一整段防御性讨论，准确写出归属与协议即可。

### 原组合曲线怎样处理

原 `fig_generic_3d_reconstruction_curves.png` 同时包含相机和 GT-scale oracle depth 曲线。当前可将完整图放入附录，正文用相机数字和实际 Bonn 表；这样无需生成新结果。若以后单独重排图版，保持相机曲线来源不变。

oracle 只标作使用 GT 的诊断参照，不叫本方法成绩，也不称已证明的性能上界。不同样本范围、聚合方式下的 oracle 与实际结果不直接作百分比排名。

## 8. 4.3 Local Human Reconstruction

### 写作任务

解释系统保留了怎样的人体先验。篇幅要紧凑，不让这一节变成与论文核心不同的新 HMR 论文。

原稿局部值为 37.3 mm PA-MPJPE、60.3 mm MPJPE、71.4 mm PVE，并明确与 NLF 一致。正文保留这个归属，避免把观察器性能算成新增校准模块的精度提升。

如果这些值仅来自 NLF 原报告，而没有本系统独立重跑，应在表中标成观察器参考结果，不能改写为独立系统实测。这样的标注不需要新增实验。

### 英文草稿

> GroundedHuman retains the local pose and shape provided by its frozen NLF observer. The 3DPW values reported in our draft are 37.3 mm PA-MPJPE, 60.3 mm MPJPE, and 71.4 mm PVE, matching the observer's local reconstruction results. Our calibration operates on scene scale and human root translation; it does not introduce a new pose or shape estimator.

投稿正文将 “reported in our draft” 换成准确来源，如 “reported for the frozen observer” 或实际系统评测描述；来源未厘清前，不默认选更强表述。

可用的收束句：

> This separation allows the calibration stage to address global placement while leaving the local human representation unchanged.

无需新增“允许姿态更新”的模型对照。结构不变是变量定义层面的说明，不扩展成“我们证明冻结优于所有联合优化”。

若保留方法排名，保留同范围内的重要竞争方法和 NLF 行；不能将 NLF 隐去再把相同数字作为独立 HMR 突破。

## 9. 4.4 Human–Scene Placement

### 本节作用

将原稿分散的交互图和实物尺度图集中到一个完整定性小节，使读者看到世界运动与场景尺度如何体现在实际重建中。

这是 Human3R 展示广泛场景能力的写作思路在本项目上的适配，不沿用其拥挤泛化标题，也不将双人样例扩大成 crowd benchmark。

### Geometric Placement

使用原 `fig_human_scene_interaction.png`。按照实际展示的坐、躺、倚靠、双人样例讨论人体与邻近表面的相对位置。

正文观察应指向具体局部，例如躯干与沙发、骨盆与座面或人体前后深度关系；不要每行都泛泛说“更真实、更自然、更一致”。

可直接使用的中性开头：

> Figure X compares human–scene placement in sitting, lying, leaning, and two-person examples. These visualizations complement the trajectory metrics by showing the relation between the reconstructed body and nearby surfaces.

逐图结论需要以真实图像内容为准。本轮只读取了原稿的图注与论述，未独立检查图片，不自动将原稿的每个视觉优劣判断升级为已确认观察。

方法连接句：

> Our refinement acts on root translation while retaining the predicted local pose and shape, concentrating the geometric correction on human placement.

这个句子说明变量作用，不宣称定性比较隔离了某一个模块。

### Metric Interpretation

使用原 `fig_metric_measurement.png`，说明人体与环境在同一度量重建中可作长度读取。篇幅不够时，图移附录，主文保留一句引用。

图中标注若来自真实重建和对应测量，可写 “illustrates metric interpretation”；不要称“达到测量级精度”，也不要把单个物体的标注写成新数据集平均误差。

### 章末收束

> Taken together, the evaluations characterize GroundedHuman as a metric human–scene reconstruction system: it estimates world-space human motion, produces metric scene depth, retains the local human prior, and supports joint visualization of people and their environment.

无需再重复四组数值，也不写“上述结果证明每一模块均不可或缺”。

## 10. 图表与现有资产映射

| 现有材料 | 当前建议位置 | 具体动作 |
| --- | --- | --- |
| EMDB-2 主表 | 4.1 首表 | 保留现有结果，聚焦已经评测的数据集 |
| `fig3_temporal_4d.png` | 4.1 表后 | 从实验开头的独立展示移到对应任务 |
| Bonn `tab:bonn-depth` | 4.2 主表 | 从附录移到正文，标清对齐条件 |
| TUM 相机数值 | 4.2 Camera Geometry | 保留骨干归属，简短呈现 |
| `fig_generic_3d_reconstruction_curves.png` | 附录 | oracle 单独说明，不当实际输出 |
| 3DPW 主表 | 4.3 紧凑表 | 标明 NLF 关系，无需保留空 EMDB-1 列 |
| `fig_human_scene_interaction.png` | 4.4 主图 | 聚焦已有样例中的放置关系 |
| `fig_metric_measurement.png` | 4.4 或附录 | 作为度量解释的辅助展示 |
| 原 Scale-consistency Analysis | 融入小节末尾 | 删除重复数字和过度因果解释 |
| 未完成的 ablation/robustness/efficiency 表 | 内部工作材料 | 不作为结果提交，不声称附录已有实验 |

删除空结果列是收紧评价范围，不是删掉不利结果。重要对照、较差成绩、GT 使用和外部依赖不能因影响叙事而隐去。

全章建议只保留三类主要表格：世界运动、度量深度、局部人体。定性图用来补足输出的可读性，不强行替代缺失的组件对照。

## 11. 语言策略

### 应重点强化

- 研究对象明确：从局部人体恢复走向共同世界坐标中的度量人景重建。
- 输出完整：世界运动、场景深度、相机与局部人体同时进入评价视野。
- 变量分工清楚：共享尺度与人体根平移承担不同角色。
- 核心任务结果靠前：最强的世界运动结果不被背景和模块细节淹没。
- 设计与结果连接具体：每节只解释该结果实际说明的一个层面。

### 推荐表述替换

| 过弱或散乱的表述 | 更集中、适合当前材料的表述 |
| --- | --- |
| 我们用了 VGGT 和 NLF，又加了两个模块 | 我们围绕共享场景尺度与人体全局位置组织几何校准，同时保持局部人体表示 |
| 我们在很多数据集上效果很好 | 我们从世界人体运动、度量场景与局部人体三个层面评价系统，并展示联合重建 |
| 冻结人体是因为不训练人体网络 | 校准的更新空间保留已有局部人体结构，将调整集中在全局几何变量 |
| 本方法所有任务都是最优 | 本方法在世界运动上取得较强结果，同时提供度量场景输出并保留局部人体质量 |
| 所有收益都是 TRSTR 带来的 | 完整系统表现出这些结果；TRSTR 的具体作用通过方法定义和放置示例说明 |

强表达可以来自准确的动词和明确的对象，不需要加入“首次”“彻底消除”“任意遮挡”“严格保证”等目前没有支撑的限定。

## 12. 对上一篇大纲的修正

本轮撤下作为必做项的：误差注入新实验、联合求解新模型、pose/shape 更新对照、方向不确定性消融、多人/多帧扩展基准。

它们仍可作为后续研究选项，但本次论文重写不等待这些工作。

保留并强化：世界运动主结果、Bonn 实际度量结果、局部人体归属、现有放置与时序图、精确但有力度的章节语言。

如果之后更改实际方法，旧结果应继续标明旧版本，不能沿用作新算法成绩。当前方案的目标是把现有版本讲清楚并突出已有优势，而不是用文字声称新版本已经完成。

## 13. 交付与验证

本轮仅新增本说明并同步修改总大纲的实验目录、图表分配和写作步骤。未修改原稿、模型、配置、权重或参考 PDF；不需要服务器执行，也不产生实验结果文件。

本地检查包括文件结构、标题与章节对应关系，以及所列数值/图文件名在原稿中的出处。没有运行新训练或评估，没有独立认证原稿中的指标与视觉比较。
