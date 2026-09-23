# GroundedHuman v1 正文命名与图表同步核查

核查日期：2026-09-23。状态：**待办清单已按用户确认更新；已完成条目移除，未决条目保留原编号。**

## 范围与判断基准

只检查 `.paper/my_paper/versions/v1/main.tex` 实际引用的正文：摘要、引言、相关工作、方法、实验、结论及正文末尾声明；逐张查看正文实际使用的 5 个图片文件。方法图当前为 LaTeX 占位框，因此正文共 6 个图位。**未检查附录**，未把 `trash/`、未引用图片、此前生成但尚未采用的 NBAI 图片计入正文问题。

图表编号已与修订后编译记录 `outputs/paper/groundedhuman_v1_ablation_revision_20260923/main.aux` 核对。下面的旧源文件行号为首次扫描时行号，后续修改可造成小幅移动。这里核查的是命名、表达及其与当前方法的对应关系，不独立核实实验数值真实性或外部方法的性能。

统一基准取自 [当前方法第 13 行](C:/Users/ROG/PycharmProjects/vggt-omega/.paper/my_paper/versions/v1/tex/03_method_v7.tex:13)：

| 层级 | 当前正式名称 / 含义 |
|---|---|
| 论文标题 | GroundedHuman: Implicit Interaction Grounding for Dynamic Human–Scene Reconstruction |
| 模块一 | **Body-Anchored Metric Calibration** |
| 模块一核心机制 | **Body-Anchored Scene Query** |
| 模块二 | **Dynamic Interaction Grounding** |
| 模块二核心机制 | **Hierarchical Context Attention** |
| 共同思想 | 人体与场景关联 → 隐式交互表示 → 几何读出；分别支持 metric calibration 和 human placement |
| Dynamic | 人体位置更新后重新采样、重建交互上下文；两步整体平移精修，不代表新增时序注意力 |
| 输出边界 | 保留人体姿态与形状；没有显式 contact-label prediction |

**当前剩余重点是 Fig. 1 首页 teaser、方法图接入和 B5 尺度验证。§4.4 命名与协议说明、Contact Control 定义已修订；Online 分类按用户要求保持。** 下文将确定的同步缺口、需要确认的实验口径、可选整理分别列出，避免直接替换后误标实验。

## A. 明确存在的同步缺口

### A1｜Fig. 1(a)：第一模块仍使用旧标题和旧层级

- 位置：[fig1_teaser_v2.jpg](C:/Users/ROG/PycharmProjects/vggt-omega/.paper/my_paper/versions/v1/figures/fig1_teaser_v2.jpg) 左上；引用入口 [main.tex:49](C:/Users/ROG/PycharmProjects/vggt-omega/.paper/my_paper/versions/v1/main.tex:49)。
- 图内原文：`(a) Human-anchored scale alignment`；下一行 `Anchor–Scene Cross Attention`。
- 问题：没有采用模块一正式名称，也没有呈现“模块 / 核心机制”的新层级。`scale alignment` 还弱化了深度偏置校准的含义。
- 建议核对后改为：主标题 **Body-Anchored Metric Calibration**，机制名 **Body-Anchored Scene Query**。Cross-attention 可继续作为机制内部的算子标注，**不是要删除或否定现有交叉注意力**。

### A2｜Fig. 1(b)：第二模块仍被命名为 Query

- 位置：同图左下。
- 图内原文：`(b) Body Region Context Query`、`Region-conditioned geometry for body placement`，以及 `Query q_r`。
- 问题：主标题仍沿用原区域 query 叙事，读者无法与正文的 **Dynamic Interaction Grounding / Hierarchical Context Attention** 对应；也削弱了“一阶段 Query 对齐、二阶段 Attention 精修”的结构。
- 建议：主标题用 **Dynamic Interaction Grounding**，机制名用 **Hierarchical Context Attention**。内部 `Query q_r` 是否改成 regional interaction representation，要与具体图中节点含义一起确认，不能只做字符串替换。

### A3｜Fig. 1 与 caption 尚未表达新的共同思想

- 位置：[main.tex:50](C:/Users/ROG/PycharmProjects/vggt-omega/.paper/my_paper/versions/v1/main.tex:50) 及 teaser 左侧两个机制示意。
- Caption 原文：`body-region context queries aggregate multi-scale human--scene evidence ...`。
- 问题：caption 仍用旧的第二阶段名称；图内没有明确可辨识的“两种 interaction representation → 各自 geometric readout”对应关系。现有图主要讲尺度修正和区域几何汇总，还没有同步当前方法的高层组织方式。
- 建议：caption 使用两组正式名称；图内在两个阶段分别突出 interaction representation 和 metric / placement readout。可保留现有几何示意，未必需要把整个 teaser 重画。

### A7｜Fig. 2 方法总图仍是占位框，尚未接入新设计

- 位置：[03_method_v7.tex:22](C:/Users/ROG/PycharmProjects/vggt-omega/.paper/my_paper/versions/v1/tex/03_method_v7.tex:22)。
- 当前显示：`Method overview to be redrawn.`，加两个新模块名称。
- 判断：这是**此前已知的待完成项**，不是旧图被误用。Caption 已同步新思想。旧 `main_method.png` 未被当前正文引用，近期 NBAI 图也尚未插入。
- 后续动作：作者确认最终图后再替换占位框；本轮不自动选图、不接入任何生成版本。

## B. 涉及实验含义，不能直接改名的待确认项

### B1｜Fig. 1 速度–精度图的 Ours / Ours + HSI 对应关系不明【优先确认】

- 位置：teaser 右侧散点图及图例。
- 原文：蓝星 `Ours`，橙星 `Ours + HSI`。蓝星大致位于 58.5 mm、10 FPS；橙星大致位于 70 mm、5 FPS，具体值以绘图数据为准。
- 问题：当前 GroundedHuman 已将隐式交互作为完整方法组成，图中的 `+ HSI` 却像额外扩展。两点的含义没有在 caption 或正文解释，不能猜测哪一个是完整方法。
- 请确认：两点分别对应什么 checkpoint / 开关？HSI 到底是什么？为何增加 HSI 的点在该指标上更慢且误差更大？蓝星是否对应 Table 1 的完整 Ours？
- 确认后再统一为 **GroundedHuman (full)** / **GroundedHuman w/o ...** 等可追溯名字；不得只交换图例来追求视觉一致。

### B5｜Fig. 4 的 Sim(3)-aligned ATE 不能单独证明米制尺度校准有效

- 位置：[04_exp.tex:68](C:/Users/ROG/PycharmProjects/vggt-omega/.paper/my_paper/versions/v1/tex/04_exp.tex:68)、[04_exp.tex:73](C:/Users/ROG/PycharmProjects/vggt-omega/.paper/my_paper/versions/v1/tex/04_exp.tex:73)。
- 原文在同一句中把应用 video-level scale 与 `producing stable trajectory accuracy` 相连；评估明确先做 Sim(3) 对齐。
- 问题：Sim(3) 包含尺度拟合，统一缩放相机轨迹通常会被对齐吸收。该指标更适合证明相机轨迹形状的质量/保持情况，不能单独归因于恢复了绝对米制尺度。
- 最新处理：用户希望将其设计为尺度有效性实验。推荐以相同 backbone 输出比较解析初始化与完整校准，采用 SE(3)-ATE 和残余尺度误差；具体协议、代码依据和英文表述见 [尺度评测设计](C:/Users/ROG/PycharmProjects/vggt-omega/docs/groundedhuman_v1_metric_scale_evaluation.md)。新结果尚未运行，原图及原评测名称未改，保留本条待办。

### B6｜Table 2 / Human Mesh Recovery 应明确是冻结 NLF 的局部质量

- 位置：[04_exp.tex:79](C:/Users/ROG/PycharmProjects/vggt-omega/.paper/my_paper/versions/v1/tex/04_exp.tex:79)、[Ours 行第 100 行](C:/Users/ROG/PycharmProjects/vggt-omega/.paper/my_paper/versions/v1/tex/04_exp.tex:100)、[解释段第 108 行](C:/Users/ROG/PycharmProjects/vggt-omega/.paper/my_paper/versions/v1/tex/04_exp.tex:108)。
- 已有正确说明：第 108 行明确冻结 NLF、局部姿态与形状不变，因此**不是仍宣称局部姿态精修**。
- 待确认：表中 Ours 的 PA-MPJPE / MPJPE / PVE 是否采用与 NLF 一致的局部对齐、输入和协议？是否需要用表注或 NLF 行让贡献归属更直接？
- 小标题 `Human Mesh Recovery` 本身可以保留，不应机械替换成 Dynamic Interaction Grounding。

### B7｜Fig. 1 的速度与精度比较缺少正文可见的评测身份说明

- 位置：teaser 右图、[main.tex:50](C:/Users/ROG/PycharmProjects/vggt-omega/.paper/my_paper/versions/v1/main.tex:50)、[04_exp.tex:14](C:/Users/ROG/PycharmProjects/vggt-omega/.paper/my_paper/versions/v1/tex/04_exp.tex:14)。
- 右图横轴是 `WA-MPJPE_100 (mm)`；主表/实验正文用 `WA-MPJPE`，正文没有定义下标 100 的窗口含义，也没有在 teaser caption 指明数据集。
- FPS 未在本次检查的正文中解释硬件、序列长度和计时范围；尤其不清楚 Ours / Ours + HSI 是否都计入 VGGT-Ω、NLF 和第二模块。
- 建议：在图注或正文给出最小协议说明，并与 B1 一起核对。不据此判断图中数值有误；附录是否有说明不在本轮范围内。

## C. 可选统一与一般图表表达问题

| 编号 | 位置 | 观察与建议 | 性质 |
|---|---|---|---|
| C1 | Fig. 3/4/5/6、Table 1/2 | 图中与表行普遍用 `Ours`，部分 caption 用 `GroundedHuman`。两者都合理；如统一视觉品牌可改为 GroundedHuman。真正必须澄清的是 Fig. 1 同时存在的 `Ours + HSI`。 | 可选，不是旧方法名错误 |
| C3 | §4.2 `Scene Reconstruction`；Fig. 4 caption `generic 3D reconstruction` | 覆盖 camera 与 metric depth，现有名称不算错，但较泛。可考虑 `Scene and Camera Reconstruction`，使实验职责更清楚。图中指标名称与 caption 基本一致。 | 可选 |
| C4 | Fig. 3 `Temporal human–scene reconstruction` / `Metric 4D Reconstruction` | 是视频重建结果的名称，可以保留；不表示引入了 Temporal Stabilization。若解释 Dynamic，仍应以方法中的位置依赖重采样为准。底部轨迹没有标出坐标轴含义/单位，可在 caption 补充。 | 名称无需强制更换；图注可完善 |
| C5 | Fig. 5；[04_exp.tex:137](C:/Users/ROG/PycharmProjects/vggt-omega/.paper/my_paper/versions/v1/tex/04_exp.tex:137) | `Human–Scene Spatial Alignment` 与当前 placement 目标相容；可以加一句对应模块二，但不能把跨方法定性比较当作模块二单独消融证据。 | 可选解释 |
| C6 | Fig. 6；[04_exp.tex:153](C:/Users/ROG/PycharmProjects/vggt-omega/.paper/my_paper/versions/v1/tex/04_exp.tex:153) | `Comparison of the metric measurement on PROX.` 表达含糊、单复数不自然，未点明测量人体身高/场景尺寸以及参照行。可用 `Metric measurements of human and scene dimensions on PROX` 等更具体标题。图内 Ground truth / MapAnything / Ours 无旧模块名。 | 图注改进 |
| C7 | Table 1 Ours 行 | RICH 的 WA-MPJPE 87.7 高于 GVHMR 78.8、UniCon3R 81.5；RTE 2.5 高于多项对比值，但 Ours 全行数字均加粗。若粗体表示最优，则不一致；若仅强调本方法，应注明或只加粗方法名。 | 一般表格语义，不是命名残留 |

## D. 已检查且不应误判为待替换的内容

- **摘要、引言、贡献点、结论的有效正文**已经采用新模块名和 interaction representation / geometric readout 思想；无需整段重写来“同步命名”。相关工作未发现以旧正式名称介绍本方法。
- `Human-Anchored Metric Attention`、`Multi-Scale Regional Interaction Attention`、`hierarchical spatial grounding` 的历史稿文字仍存在于 `00_abs.tex`、`01_intro.tex` 等文件的 `%` 注释中；`main.tex` 也保留注释掉的旧标题。它们**不出现在当前排版正文**，可留作历史记录。
- `sec:contact-ablation`、`tab:body-region-ablation` 等 LaTeX label 仍是旧内部标识。正常编译时不显示其字符串，本轮不把它们算作读者可见的命名问题；之后如改 label 必须同步引用。
- `main_method.png`、`fig3_scale_trstr_detail_final.png` 等旧资产虽在目录内，但未被当前正文引用。不能据文件名判断正文还存在旧模块。
- `Temporal Aggregation` 是模块一仍在使用的跨帧尺度汇总，应保留。它与已弃用的时序稳定模块不是一回事。
- 相关工作介绍 UniCon3R / WHAM 的 `contact-aware`，以及一般的 contact / penetration / floating 描述，不应一律删除；本方法消融中的旧 contact-aware variant 表述已修订。
- Fig. 4 caption 和视频深度段已经写入 **Body-Anchored Metric Calibration**，无旧模块名残留；其需要注意的是 B5 的指标解释。

## 正文图片与表格检查台账

| 图表 | 实际素材 / 入口 | 检查结果 |
|---|---|---|
| Fig. 1 | `fig1_teaser_v2.jpg`；main.tex:49–50 | 主要遗留集中处：A1–A3、B1、B7 |
| Fig. 2 | 方法文件中的 LaTeX 占位框 | 新 caption 正确，最终图未接入：A7 |
| Fig. 3 | `fig3_temporal_4d_round2.png`；04_exp.tex:57–58 | 无旧模块名；时间/4D 结果描述可保留：C1、C4 |
| Fig. 4 | `fig4_v2.jpg`；04_exp.tex:67–76 | 图内名称基本对应；关注 Sim(3) 解释：B5、C1、C3 |
| Fig. 5 | `fig_human_scene_interaction.png`；04_exp.tex:141–142 | 无旧模块名：C1、C5 |
| Fig. 6 | `fig_metric_measurement.png`；04_exp.tex:152–157 | 无旧模块名；caption 可具体化：C1、C6 |
| Table 1 | 04_exp.tex，主比较表 | Contact Control 定义已修订；Online 保持；剩余 C7 |
| Table 2 | 04_exp.tex:83–108 | Ours 的 NLF 质量归属：B6；无旧模块名 |
| Table 3 | 04_exp.tex，§4.4 | 模块名称、协议、正文和 caption 已按作者确认修订 |

## 建议作者核对顺序

1. 确认 Fig. 1 的 Ours / Ours + HSI 实验身份，再同步 A1–A3 的图内文字和 caption。
2. 决定 B5 的尺度验证方案，见 [尺度评测设计与推荐表述](C:/Users/ROG/PycharmProjects/vggt-omega/docs/groundedhuman_v1_metric_scale_evaluation.md)。新数值尚未评估，因此 B5 保留待办。
3. 确认最终方法图后处理 A7。
4. 核对 B6/B7 及剩余 C 类条目。

已处理项与用户选择单独记录于 [本轮修订记录](C:/Users/ROG/PycharmProjects/vggt-omega/docs/groundedhuman_v1_ablation_revision_20260923.md)。本清单不再保留其原问题段落。附录仍未检查或修改；实验数值和图像资产未改变。旧行号为首次核查时定位，修改后可按段落标题或表格标签定位。
