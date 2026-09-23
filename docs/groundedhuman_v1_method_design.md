# GroundedHuman v1：交互表示主线与本轮修订说明

本次严格基于 .paper/my_paper/versions/v1，继续禁用所有 skill。正式方法文件仍为 tex/03_method_v7.tex。两大模块按老师指定的结构保留；内部小标题尽可能恢复原稿。没有修改模型、训练代码、配置或实验结果。

## 现在要传达的统一思想

**人体与场景的关联形成隐式交互表示；度量校准和人体放置是从这些关系表示中推断出的几何修正。**

\[
(\text{Human},\text{Scene})
\xrightarrow{\text{Query / Attention}}
\text{Implicit Interaction Representation}
\xrightarrow{\text{Geometric Readout}}
(\text{Metric},\text{Placement}).
\]

这对应老师的两行直觉公式：Human × Scene 强调关系的来源；Q/K/V 强调如何建立、读取关联；Metric 与 Placement 强调关系表示的几何用途。“×”表达关联，不规定具体矩阵乘法。Metric 指米制尺度与几何校准，不是评估指标。

人体决定在哪里、以什么几何条件查询；场景提供被关联的观测；注意力组织证据；预测头读出相应的几何修正。交互无需先经过接触标签预测。

## 这次怎样把思想落实到正文

1. **方法开头**：明确以隐式交互表示建模 grounding，并增加一个不编号的概念式，不改变原公式 2 的编号。
2. **第一模块**：最终锚点响应经过池化，显式定义度量交互表示 z_met；尺度/偏置头是 metric readout。
3. **第二模块**：分通道、多空间支持的证据与区域几何融合后，显式定义区域交互表示 z_place；平移、门控与不确定性预测属于 placement readout。
4. **迭代循环**：位置变化使人—场景关系改变，因此重新编码交互，再进行下一次几何读出。Dynamic 的含义落实于这个循环。
5. **摘要、引言、贡献与结论**：同步围绕“交互表示 → 几何读出”展开，避免统一思想只出现于方法开头。
6. **图注**：新图中明确需要出现关系表示及读出，而不能只画两个直接输出的模块盒子。

两个模块分别构建表示、使用各自的读出，顺序执行。统一的是关系推理原则，没有声称使用一个共享权重的网络同时输出所有量，也没有新增第三个 interaction 模块。

## 保留的结构和名称

| 一级模块 | 小标题 |
|---|---|
| Body-Anchored Metric Calibration | Analytic Scale Initialization |
| Body-Anchored Metric Calibration | Body-Anchored Scene Query |
| Body-Anchored Metric Calibration | Temporal Aggregation |
| Dynamic Interaction Grounding | Hierarchical Context Attention |
| Dynamic Interaction Grounding | Interaction-Aware Placement |
| Dynamic Interaction Grounding | Translation Refinement |

保留 Training Objectives 小节。上一轮临时引入的 Metric Alignment Query 与 Placement-Conditioned Interaction Attention 不再作为方法名使用，恢复老师已看过的内部标题。第二大模块保留老师指定的 Dynamic Interaction Grounding。

名称恢复不妨碍思想表达：交互表示在每个模块内部明确定义，两个 readout 的用途也直接解释，不需要靠新命名承载全部理念。

## 公式的取舍

正文保留 7 个编号公式：

| 编号 | 内容 | 用途 |
|---|---|---|
| 1 | 人体表面深度比的尺度初始化 | 交代米制参照的来源 |
| 2 | 原来的 SelfAttn / CrossAttn 两式 | 表达人体内部关联与人—场景关联 |
| 3 | 视频级尺度/偏置汇总及几何传播 | 交代公共参照怎样作用于深度与相机 |
| 4 | 区域建议的可靠性加权融合 | 交代区域交互如何形成整体平移建议 |
| 5 | 根平移迭代更新 | 连接当前关系、新位置和重新编码 |
| 6 | 校准训练目标 | 校准读出的监督 |
| 7 | 精修训练目标 | 位置读出的监督 |

公式 2 恢复原来的抽象层次，不展开 Q/K/V 投影、LayerNorm 或 softmax。正文用一句话说明 query 来自人体、key/value 来自场景，并说明略去的标准块。

注意力池化的 softmax 展开与世界坐标变换移至附录；裁剪及人物级门控的完整公式在附录原本就有，正文保留更新主式。只有与主线直接相关的操作在正文单独编号。

## 新方法图绘制依据

旧图继续停用；只保留待重绘图位及图注，图片文件不删除。

建议图中把两个模块共有的结构画清楚：

- 第一模块：人体/场景观测 → 初始几何对应 → body query 与 scene key/value → **metric interaction representation** → **metric readout** → 统一尺度的场景与相机。
- 第二模块：当前人体/米制场景 → 多空间支持和两类证据 → hierarchical attention → **regional interaction representation** → **placement readout** → 整体平移。
- 反馈箭头：更新人体位置 → 改变人体—场景关系 → 重采样并重新编码 → 下一次位置读出。
- 模块间传递的是校准后的几何参照；第二模块回环不返回重估尺度。
- 标注两步更新、参数共享、姿态与形状固定。多人/跨帧尺度汇总与空间迭代回环分别表达。

图中不增加时序稳定器、接触标签头、人体姿态更新头或第三个独立阶段。

## D4RT 的启发与边界

参考 .paper/base_pdf/D4RT_paper.pdf 的第 2 节、2.1–2.2 节和图 2。D4RT 将场景表示和查询接口连接起来，使同一个点查询接口组织多种重建任务。本项目借鉴的是“用具体的表示与读取操作承载统一思想”的组织方式，没有移植其算法。

GroundedHuman 保留当前两阶段设计：解析初始化与残余尺度校准、区域注意力与两步平移精修。第一阶段是锚点—场景交叉注意力；第二阶段是以人体相对几何为条件的 masked attention pooling。两者各自形成面向不同修正的交互表示。

## 文件、备份与验证

- 英文方法：.paper/my_paper/versions/v1/tex/03_method_v7.tex。
- 中文对照：docs/groundedhuman_v1_method_zh.md。
- 本轮英文方法预览：outputs/paper/groundedhuman_v1_method_rewrite/method_interaction_revision.pdf。上一轮 method_excerpt.pdf 保留不覆盖。
- 整稿预览：outputs/paper/groundedhuman_v1_method_rewrite/main.pdf。
- 本轮修改前的源文件、两份中文文档和方法 PDF：outputs/debug/groundedhuman_v1_method_rewrite/before_relation_revision/。
- 初始 v1 备份：outputs/debug/groundedhuman_v1_method_rewrite/before/。
- 差异与验证记录：outputs/debug/groundedhuman_v1_method_rewrite/。

本轮只进行论文静态检查、LaTeX 编译和方法页视觉检查；不涉及训练或推理，不需要服务器运行脚本。训练权重、checkpoint 对应关系等既有待核验内容保留原状态。参考文献数据库未改，原有会议缩写缺失问题仍待处理。编译后的具体页码与结果记录在 verification.json。

