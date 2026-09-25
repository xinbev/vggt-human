# v2 附录结构重构方案与实现记录

## 目标

将附录从当前的“评测开头、技术细节居中、评测协议重复出现”调整为一条清晰的审稿阅读路径：

1. 先定义坐标、尺度和张量约定；
2. 再说明模型模块、接口和完整推理流程；
3. 然后说明训练目标与冻结策略；
4. 最后集中给出评测协议、补充结果、失败边界和责任使用说明。

标题结构已完成，并已在 `.paper/my_paper/versions/v2/appendix_en.tex` 中展开正文。原正文 4.3 节 Human Mesh Recovery 已移至 D.2，使用 `v1_online` 的实验文字和表格。

## 建议的最终标题结构

```text
Appendix

A. Notation and Coordinate Conventions                         [app:coordinates]
   A.1 Coordinate Frames and Extrinsics
   A.2 Depth, Metric Scale, and Camera Translation
   A.3 Tensor Shapes, Precision, and Device Conventions

B. Model Architecture and Inference Details                    [app:architecture]
   B.1 Frozen Baselines and Module Interfaces
   B.2 Body-Anchored Metric Calibration
       B.2.1 Surface Sampling and Analytic Scale Initialization
       B.2.2 Body-Anchored Scene Query and Residual Calibration
       B.2.3 Clip-Level Consensus and Camera Propagation
   B.3 Dynamic Interaction Grounding
       B.3.1 Region Partition and Context Construction
       B.3.2 Translation Proposal Aggregation
       B.3.3 Iterative Refinement and World-Space Output
   B.4 Complete Inference Algorithm

C. Training Details                                             [app:training]
   C.1 Training Data and Perturbation States
   C.2 Metric Calibration Objective
   C.3 Dynamic Interaction Grounding Objective
   C.4 Training Schedule and Freezing Strategy
   C.5 Implementation Configuration

D. Evaluation Protocols and Supplementary Results                [app:evaluation]
   D.1 Datasets, Splits, and Metrics
   D.2 Global Motion and Local Mesh Evaluation
   D.3 Scene Depth and Camera Evaluation
   D.4 Human--Scene Consistency Evaluation
   D.5 Sequence-Length and Oracle Diagnostics                    [app:bonn-results]

E. Additional Qualitative Results and Limitations                 [app:limitations]
   E.1 Additional Qualitative Results
   E.2 Failure Cases and Reliability Boundaries
   E.3 Responsible Use and Dataset Scope
```

## 当前内容到新标题的映射

| 当前内容 | 新位置 |
|---|---|
| `Coordinate and Depth Conventions` | A.1--A.2 |
| `Tensor, Precision, and Device Conventions` | A.3 |
| `VGGT-Ω, NLF, and Baseline Boundaries` | B.1 |
| `Surface Sampling and Scale Initialization` | B.2.1 |
| `Person-conditioned Residual Affine` | B.2.2 |
| `Clip-level Consensus and Camera Propagation` | B.2.3 |
| `Complete Dynamic Interaction Grounding Stage` | B.3.1--B.3.3 |
| `Inference Algorithm` | B.4 |
| `Staged Training` | C.1 / C.4 |
| `Complete Losses and Weights` | C.2--C.3 |
| `Training Stages and Freezing Strategy` | C.4 |
| 训练配置 placeholder 表 | C.5 |
| `Bonn Depth Protocols` | D.3 / D.5 |
| `Camera Geometry and Oracle Diagnostics` | D.3 / D.5 |
| 当前 `Complete Evaluation Protocols` | D.1--D.4，拆分后不再单独保留重复标题 |
| 当前 `Additional Qualitative Results, Failure Cases, and Responsible Use` | E.1--E.3 |

## 标签与术语处理

- 保留 `app:coordinates`、`app:architecture`、`app:training`、`app:evaluation`、`app:limitations`，避免正文引用失效。
- 将 `app:bonn-results` 放到 D.5，以兼容已有的 Bonn/TUM 补充评测引用。
- 模型术语按当前 v2 主体统一为 `Body-Anchored Scene Query` 和 `Hierarchical Context Attention`。
- “Metric Gauge”并入 A 节的尺度约定标题，避免与模型模块标题并列造成概念层级混乱。
- `Responsible Use` 单独成为 E.3，避免与技术失败模式混在同一小节。

## 重写顺序

1. 先移动标题和标签，不改正文句子；
2. 将当前段落逐一放入新标题下；
3. 合并重复的评测协议说明；
4. 再补写缺失的 D.1--D.4 和 C.5 配置说明；
5. 最后统一正文中的 Appendix 引用和术语。

## 已实现内容

- A--E 五个附录主节及其子节已写入 v2 附录。
- D.2 保留局部人体网格评测的文字、指标和 3DPW/EMDB-1 对比表。
- D.2 加入 NBAI 生成的概念性实验流程图 `figures/appendix_local_hmr_protocol_concept.png`。
- 图注明确说明指标条仅为示意，服务器结果确认后可替换为正式实验图。
- v2 已完成本地 XeLaTeX 编译和附录页渲染检查。
