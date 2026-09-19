# GroundedHuman 论文写作大纲

## Title

**GroundedHuman: Metric Reconstruction of Humans and Scenes from Monocular Video**

**GroundedHuman：从单目视频进行人体与场景的度量重建**

## 论文目录

```text
Abstract

1. Introduction

2. Related Work

3. Method
   3.1 Scale Calibration
   3.2 Translation Refinement

4. Experiments
   4.1 Global Human Motion Estimation
   4.2 Scene Reconstruction
   4.3 Human Mesh Recovery
   4.4 Qualitative Results

5. Conclusion

References

Appendix
   A. Implementation Details
   B. Evaluation Protocols
   C. Additional Results
   D. Failure Cases
```

方法部分按两个实际模块展开：尺度校准与人体平移修正。局部姿态和形状保持不变是平移模块的参数约束；区域可靠性与重投影是该模块的内部机制。三项贡献体现在模块设计中，不分别扩展为三个抽象原则小节。

正文使用两级编号。模块内部以简短段首组织内容，训练说明放在方法末尾，局限放在结论之后的一个短段落中。

## Abstract

写成一个自然段，按以下顺序展开：

1. **任务与问题。**从单目视频重建人体、场景与相机，需要将人体的度量先验和场景的相对几何连接起来。人体位置和场景尺度中的误差会影响这一过程。
2. **整体方案。**提出 GroundedHuman，通过尺度校准和人体平移修正，将已有场景与人体预测表达在共同的世界坐标系中。
3. **尺度模块。**从可见人体表面估计初始尺度，再结合人体特征与局部场景特征预测残差，将共享尺度应用于场景深度和相机平移。
4. **平移模块。**从人体周围采集区域几何证据，聚合有界平移更新，并在重投影后继续修正；全过程保持人体姿态与形状不变。
5. **主要结果。**报告 EMDB-2 世界运动与 Bonn 度量深度的关键结果，概括局部人体质量及联合重建输出。

不在摘要中展开 HSI、TRSTR、anchor 数量或网络层数。最多保留两组关键数值，不将摘要写成指标清单。

## 1. Introduction

### 第 1 段：研究任务

从单目视频中的完整几何理解切入：不仅要恢复人体形态与周围环境，还需要确定人在世界中的位置、运动以及与场景的空间关系。

指出分别合理的人体和场景预测不一定具有一致的尺度与坐标。

**段末落点：**共同的度量坐标是将局部重建转化为完整人景重建的基础。

### 第 2 段：问题来源

场景模型能提供相对深度与相机运动，人体模型则提供具有物理单位的几何先验。但人体的预测位置也存在误差，直接将其作为精确尺度参照可能把定位误差传递到场景中。

用直观几何关系解释尺度与人体位置的相互影响，不在引言里建立完整优化问题。

**段末落点：**需要处理共享场景尺度与人物位置，同时保留可靠的局部人体表示。

### 第 3 段：尺度校准

介绍尺度模块的思路：从可见人体表面得到几何初始化，利用人体与局部场景特征估计残差，再跨有效观测形成片段共享尺度。

解释共享尺度同时作用于场景深度与相机平移，使后续人体位置有共同的几何参考。

**段末落点：**尺度模块连接人体的度量先验与场景的相对几何。

### 第 4 段：人体平移修正

说明尺度统一后，人体仍可能与周围表面存在位置偏差。

介绍人体区域探测、视线和切向位移、可靠性加权与重投影。明确只更新根平移，不改写姿态和形状。

**段末落点：**平移模块利用场景信息修正人体放置，同时保留局部人体结构。

### 第 5 段：结果与贡献

用一至两句概括世界运动、度量场景、局部人体与定性重建结果，随后列出三项贡献：

- **尺度与位置校准。**通过可见人体表面、残差估计与共享尺度传播，建立人体、场景和相机之间的度量关系，并进一步修正人物位置。
- **人体结构保持。**将几何修正限制在场景尺度与人体根平移，使全局校准不通过改变人体姿态和形状来吸收残差。
- **区域几何修正。**利用不同人体区域的几何支持、预测可靠性和重投影迭代，聚合人体平移更新。

贡献表述必须对应实际模块。“联合”如用于描述系统输出，不应被解释为已经实现了同时求解尺度与位移的优化器。方向参数化与标量置信度也不等同于完整的方向可观测性估计。

**Fig. 1：**输入视频与共同世界坐标中的场景、人体及相机轨迹。

## 2. Related Work

不设编号小节，用三个段首组织。

### Scene Reconstruction.

简述学习式场景重建提供的深度、相机与场景特征，以及相对尺度和动态人体带来的问题。引用原稿中与模型基础直接相关的文献，不铺陈完整发展史。

结尾明确：本方法沿用场景模型的几何能力，主要处理度量尺度与人体放置。

### Human Reconstruction.

区分局部人体恢复和世界人体运动，讨论人体先验、相机运动与全局位置之间的关系。

说明本文保留已有局部人体预测，将方法重点放在场景尺度与人体根平移。

### Joint Reconstruction.

讨论尺度校准和人体修正。围绕原稿中的 JOSH、Human3R、UniSH、GRAFT 等最相关方法展开。

比较尺度来源、可更新变量和几何交互方式。用一两句定位本文，不以模型名列表代替比较，也不根据新术语声称已有工作从未研究相近问题。   

## 3. Method

### 开场：两段引导，不单独编号

**第 1 段：输入与输出。**

定义输入视频 \(I_{1:T}\)。VGGT-Ω 提供相对深度 \(D_t^r\)、相机内参 \(K_t\)、外参 \([R_{cw,t}\mid t_{cw,t}^r]\) 和场景特征。NLF 提供姿态 \(\theta_{tq}\)、形状 \(\beta_{tq}\)、根平移 \(\tau_{tq}\) 及置信度。

两路预测使用相同图像坐标与相机内参。输出为度量场景和相机，以及世界坐标中的人体序列。

**第 2 段：模块关系。**

结合 Fig. 2 介绍尺度校准与人体平移修正。前者得到场景与相机的共享尺度，后者利用标定后的几何更新人体位置。两个模块均保留上游人体姿态和形状。

建议英文开头：

> Given a monocular video, GroundedHuman reconstructs the scene, camera trajectory, and human meshes in a common world coordinate system. The method consists of two modules: scale calibration and translation refinement.

紧接着说明观察器与输入输出，不再另设问题定义、变量空间或方法哲学章节。详细符号表及张量约定放入附录。

**Fig. 2：**总结构图只标出实际组件，例如 VGGT-Ω、NLF、Scale Calibration 和 Translation Refinement。图中的重投影反馈连接到人体区域采样，不绘制不存在的尺度反馈分支。

### 3.1 Scale Calibration

**本节讲清一个模块：怎样从人体和相对场景得到共享尺度。**

#### Scale Initialization.

说明人体表面的米制几何及其与相对深度的对应关系。将有效人体表面投影到深度图，按前表面可见性处理同像素冲突，过滤越界和无效深度。

保留一个组合公式：

\[
u_{tqi}=\pi(K_t,V_{tqi}^{0}+\tau_{tq}),\qquad
s_t^c=
\operatorname{median}_{(q,i)\in\mathcal A_t}
\frac{[V_{tqi}^{0}+\tau_{tq}]_z}{D_t^r(u_{tqi})}.
\]

解释表面共识相比单点尺度的作用，以及多个人体观测如何进入同一估计。采样间隔、有效像素数和异常值阈值放入附录。

#### Residual Prediction.

介绍现有 24 个人体 anchor，编码人体状态、投影位置与局部深度残差，并查询场景特征。

按“输入表示 → 局部特征交互 → 参数输出”写一至两段：人体特征与附近场景特征通过 HSI 模块交互，输出残差尺度 \(\rho_t\) 与深度偏置 \(b_t\)。

\[
D_t^m=\rho_t s_t^cD_t^r+b_t.
\]

解释几何初始化与残差预测的分工：前者提供主要尺度，后者处理剩余误差。正文保留核心交互方式，不逐层描述 Transformer 配置。

#### Scale Aggregation.

说明人物间的置信度聚合，以及有效帧尺度如何形成片段共享尺度：

\[
s^{\mathrm{clip}}=
\exp\!\left[
\operatorname{median}_{t\in\mathcal V}
\log(s_t^c\rho_t)
\right].
\]

随后给出实际应用：

\[
D_t^m=s^{\mathrm{clip}}D_t^r+b^{\mathrm{clip}},\qquad
t_{cw,t}^{m}=s^{\mathrm{clip}}t_{cw,t}^{r}.
\]

说明片段内的相对几何共享同一尺度，旋转不变；深度偏置不加入相机平移。有效帧选择和无人体时的回退规则在附录明确。

**本节收束：**尺度校准为场景与相机提供度量参考；下一模块在这一参考下修正人体位置。

### 3.2 Translation Refinement

**本节讲清一个模块：怎样根据人体周围的几何修正根平移。**

#### Human Representation.

先用短段落说明只更新根平移，姿态和形状固定。将人体表面组织为 96 个具有固定解剖对应关系的区域，用区域中心和代表顶点定位采样位置。

\[
V_{tq}^{c,(k)}=
\operatorname{SMPL}(\theta_{tq},\beta_{tq})
+\tau_{tq}^{(k)}.
\]

在解释这个表达式时交代局部结构不变，不另设“更新空间”小节。区域构造、代表点数量与分组规则放入附录。

#### Geometry Sampling.

描述各人体区域周围的多尺度几何探测。区分自身表面、环境与其他人物，说明有效支持和遮挡信息如何进入区域表示。

强调对应关系：采样区域由当前人体位置决定，因此位置更新后需要重新采样。

#### Translation Prediction.

区域输出沿相机视线和切向的位移：

\[
v_r=\delta_{\parallel,r}e_{\parallel,r}
+\delta_{x,r}e_{x,r}
+\delta_{y,r}e_{y,r}.
\]

说明不同方向的更新范围，再介绍区域门控和标量不确定性：

\[
\alpha_r=m_rg_r\exp(-\ell_r),\qquad
\widehat v=
\frac{\sum_r\alpha_rv_r}
{\max(\sum_r\alpha_r,\varepsilon)}.
\]

用文字解释有效性掩码 \(m_r\)、门控 \(g_r\) 与 log variance \(\ell_r\) 的作用。再说明人物级门控和更新限幅怎样得到最终 \(\Delta\tau\)。

重点是具体预测和聚合过程。不要用未实现的协方差矩阵、可辨识性求解或几何信息矩阵替代现有机制。

#### Iterative Refinement.

给出更新：

\[
\tau_{tq}^{(k+1)}
=\tau_{tq}^{(k)}+\Delta\tau_{tq}^{(k)}.
\]

解释两次迭代共享参数，更新后重新投影并采集区域证据，第二步使用变化后的残差。

最后写世界坐标转换：

\[
V_{tq}^{w}=
R_{cw,t}^{\mathsf T}
\left(
V_{tq}^{c,*}-s^{\mathrm{clip}}t_{cw,t}^{r}
\right).
\]

**本节收束：**人体位置在同一场景尺度下得到修正，局部姿态和形状保持固定。

### Training.：方法末尾的段首，不单独编号

用一至两段交代分阶段训练，随后指向附录：

- 尺度阶段：解释尺度扰动、粗尺度误差与残差监督，明确参与训练的模块。
- 平移阶段：解释 clean、scale error、NLF error 和 mixed 四类训练状态，以及区域 vote、最终位置和更新约束的监督作用。

正文可给出两项实际总损失的紧凑表达；全部损失定义、权重、训练配置与冻结策略放附录。

推理顺序已经由 3.1 和 3.2 说明，不再设置独立的 Inference 小节。完整伪代码放附录，不重复占用正文。

## 4. Experiments

### 开场：评价范围与必要设置

一段概括评价对象：世界人体运动、场景深度与相机、局部人体、完整重建效果。

数据与对齐协议在对应小节说明。训练超参数与实现细节进入附录，不增加一个主要介绍工程配置的正文小节。

以下数值为原稿已有记录，保持对应方法版本、结果来源与评价协议。

### 4.1 Global Human Motion Estimation

**第 1 段：数据与指标。**介绍 EMDB-2、片段划分与关节集合，简述 WA-MPJPE、W-MPJPE 和 RTE 的对齐方式及单位。

**第 2 段：主结果。**用 Table 1 比较现有方法，集中报告：

| 指标 | 现有结果 |
| --- | ---: |
| WA-MPJPE | 58.468 mm |
| W-MPJPE | 149.959 mm |
| RTE | 1.079% |

正文选择一组最关键的比较。例如，原稿中 W-MPJPE 相对 Human3R 的 267.9 mm 降低约 44.0%。其他比较留在表中，不连续罗列多组百分比。

**第 3 段：时序结果。**结合 Fig. 3 展示人体序列、场景与相机轨迹。解释系统在世界坐标中的输出，不将轨迹可视化作为额外定量指标。

**小节结论：**突出完整系统的世界运动结果及联合场景输出，不声称跨方法差异已经隔离了某一个模块的贡献。

### 4.2 Scene Reconstruction

用两个段首组织，不再增加编号层级。

#### Video Depth Estimation.

介绍 Bonn 及 Abs Rel、\(\delta<1.25\)，明确测试时不使用 GT scale 或 shift 对齐。

用 Table 2 报告已有实际结果：

| 指标 | 现有结果 |
| --- | ---: |
| Abs Rel | 0.08599 |
| \(\delta<1.25\) | 0.96377 |

解释这项评价测量的是校准后场景的度量深度，与世界人体运动结果互补。

保留重要对照方法及评价条件。相对深度模型经 GT 对齐得到的结果应标注协议，不能与无 GT 对齐结果无说明地并列。

#### Camera Pose Estimation.

介绍 TUM-Dynamic 和 Sim(3) 对齐下的 ATE。报告 VGGT-Ω 骨干在现有 50–500 views 范围内低于 7 mm 的结果，其中 500 views 为 0.00692 m。

明确结果归属与协议：它说明使用的相对相机几何质量，不是尺度模块改善绝对尺度的独立证据。

原稿包含 GT-scale oracle 的完整曲线进入附录，正文优先展示实际深度结果与必要相机指标。

### 4.3 Human Mesh Recovery

用一个紧凑小节交代局部人体质量。

**第 1 段：协议。**介绍 3DPW 及局部指标的对齐方式，明确 NLF 为固定人体观察器。

**第 2 段：结果与解释。**

| 指标 | 原稿记录 |
| --- | ---: |
| PA-MPJPE | 37.3 mm |
| MPJPE | 60.3 mm |
| PVE | 71.4 mm |

Table 3 保留这些数值与 NLF 的关系。来自观察器原报告的数值标为参考结果；来自系统评测的数值采用相应记录。

解释根平移修正保持局部姿态与形状，不将继承的人体质量写成新的姿态估计精度提升。

### 4.4 Qualitative Results

将现有几何展示集中到此节，以两个段首组织。

#### Human Placement.

使用 Fig. 4 展示坐、躺、倚靠和双人样例，逐个讨论人体与邻近表面的具体位置关系，如躯干与沙发、骨盆与座面、前后深度偏差。

说明方法通过根平移修正放置，局部人体参数保持不变。视觉比较限定于展示样例，不扩展成未进行的接触或拥挤场景定量评价。

#### Metric Measurements.

引用已有实物尺度图，展示人体与环境在共同重建中的长度关系。图优先置于附录，篇幅允许时并入主定性图。

用一至两句话收束全章：系统恢复世界人体运动，提供度量场景，保留局部人体表示，并支持完整人景可视化。

不添加只有数值复述的 Analysis 章节，也不设置没有实际结果的 Ablation Study。

## 5. Conclusion

用一个短段落完成：

1. 总结从人体先验与相对场景几何出发的重建问题。
2. 概括尺度校准与人体平移修正两个模块。
3. 指出世界运动、场景深度与局部人体结果共同呈现的系统能力。
4. 简要说明后续方向。

不再重复全部数据集和指标。

### Limitations.：结论后的短段落，不单独编号

讨论人体初始化、人体可见性和场景深度误差带来的限制。固定姿态和形状无法修复局部关节错误或所有接触问题；缺少可靠人体观测时尺度约束会减弱。

明确片段聚合与未来帧使用范围，避免将非因果处理称为在线推理。详细失败样例放附录。

## 图表安排

| 编号 | 内容 | 位置 | 现有材料 |
| --- | --- | --- | --- |
| Fig. 1 | 视频与完整重建 | 首页或引言 | `fig1_teaser_real.png` |
| Fig. 2 | 两个校准模块及观察器 | 第 3 节开头 | `main_method.png`，同步模块名称 |
| Fig. 3 | 时序人体、场景与相机 | 4.1 | `fig3_temporal_4d.png` |
| Fig. 4 | 人体与环境的放置比较 | 4.4 | `fig_human_scene_interaction.png` |
| Table 1 | EMDB-2 世界运动 | 4.1 | 原 `tab:global-motion` |
| Table 2 | Bonn 度量深度 | 4.2 | 原 `tab:bonn-depth` |
| Table 3 | 3DPW 局部人体 | 4.3 | 原 `tab:local-hmr` |
| Fig. A1 | 模块结构细节 | 附录 A | `fig3_scale_trstr_detail_final.png` |
| Fig. A2 | 相机与 oracle 深度诊断 | 附录 C | `fig_generic_3d_reconstruction_curves.png` |
| Fig. A3 | 实物尺度展示 | 附录 C | `fig_metric_measurement.png` |

图注介绍展示对象、比较条件和读图方式；正文解释结果。两者不复制同一段结论。

## Appendix

### A. Implementation Details

给出坐标、Z-depth 和米制人体约定，列出张量维度、dtype/device、图像预处理与相机内参处理。

补充观察器配置、24-anchor 表示、96-region 构造、采样窗口、门控和迭代细节，以及完整推理伪代码。

训练部分列出实际冻结与训练参数、初始化、数据划分、扰动、损失权重、优化器和 checkpoint 选择。

### B. Evaluation Protocols

分别说明 EMDB-2、Bonn、3DPW 和 TUM-Dynamic 的样本范围、指标实现、对齐、有效性过滤及统计方式。

明确引用结果、观察器结果、骨干结果和完整系统结果的来源。

### C. Additional Results

放置已存在的完整曲线、逐序列结果与更多定性展示。GT-scale oracle 明确标注为使用 GT 的诊断，不作为方法实际成绩。

不放入未完成的消融占位表，也不引用不存在的补充实验。

### D. Failure Cases

展示已有失败样例，围绕人体定位、遮挡、场景误差及固定人体参数的限制解释结果。

## 语言与结构约束

### 标题

- 标题直接命名模块、任务或内容，例如 Scale Calibration、Translation Refinement、Human Mesh Recovery。
- 禁止用多个连字符把一串修饰语拼成自造术语；英文标题中不使用此类复合修饰结构。
- 不为每条贡献新造模块名或缩写。保留 GroundedHuman、HSI、TRSTR 等已有名称时，首次出现解释一次。
- “保持人体结构”“共享变量与局部变量”“选择性更新”等设计原则写在相关模块的正文中，不单独凑成章节。
- 技术名词、数据集名、指标名和代码标识保持标准拼写；标题命名约束不改变它们。

### 句子

- 优先写具体动作：estimate、predict、sample、aggregate、refine、keep、apply。
- 先交代操作对象，再说明操作和结果，避免连续堆叠抽象名词。
- 用短从句表达条件和约束，不把整句内容压成名词前的长修饰语。
- 一句话表达一个主要技术动作。条件、动机和作用复杂时拆成两句。
- 不反复使用 novel、powerful、elegant、seamless、holistic、robust 等评价词代替技术解释。
- “we propose”用于引出方法或主要模块，不作为每段固定开头。

### 推荐英文表达

| 要表达的内容 | 推荐写法 |
| --- | --- |
| 保持姿态与形状 | We keep pose and shape fixed and update only root translation. |
| 人体提供尺度 | We estimate scene scale from the visible human surface. |
| 残差预测 | The module predicts a residual scale and a depth offset. |
| 共享尺度传播 | We apply the same scale to scene depth and camera translation. |
| 区域探测 | We sample scene geometry around each body region. |
| 可靠性聚合 | We weight regional updates by their predicted confidence. |
| 重投影 | After each update, we reproject the body and sample the scene again. |
| 世界坐标输出 | We transform the refined meshes into world coordinates using the calibrated cameras. |

### 段落

- 方法段落按具体模块写清输入、计算、输出及必要理由，不强制每段复述贡献。
- 实验段落先给协议和结果，再解释结果与任务的关系。
- 不在相邻段落反复用同义表达重述“统一”“一致”“共享”。
- 数值比较只突出一两个关键结果，其余交给表格。
- 结论动词与证据对应：任务比较报告性能，定性图展示样例，消融才能隔离组件贡献。

### LaTeX 层级

使用本地 `iclr2027_conference.tex` 与 `iclr2027_conference.sty` 的层级和排版命令，不修改样式文件。模板控制版式，本大纲的章节名称与组织是文章自身的写作选择。

```latex
\section{Method}
% Two opening paragraphs and the architecture figure.

\subsection{Scale Calibration}
\paragraph{Scale Initialization.}
\paragraph{Residual Prediction.}
\paragraph{Scale Aggregation.}

\subsection{Translation Refinement}
\paragraph{Human Representation.}
\paragraph{Geometry Sampling.}
\paragraph{Translation Prediction.}
\paragraph{Iterative Refinement.}
\paragraph{Training.}
```

摘要使用单个自然段。引用采用模板中的 `\citet{}` 与 `\citep{}`，图表编号和标题格式交由模板处理。不通过新增大量编号小节、手工修改字体或挤压行距制造层次。
