# v1：新增 Human–Scene Consistency 实验

## 2026-09-24 最新数值更新

按作者最新数据，Table 3 本方法行改为 **Ours**：3DPW 为 **0.0155 / 0.0181 / 4.99 / 4.86**，EMDB-1 为 **0.0102 / 0.0181 / 3.75 / 3.67**，依次对应 HS-V5、HS-V10、HS-CF5、HS-CF10。此前留白及旧 EMDB-1 数值已被替换，下文为新增表格时的历史记录。

已重新计算八列最优/次优标记，并同步 §4 实验概述及 §4.4 的双数据集分析：Ours 在两数据集全部指标上优于 UniSH；3DPW 全部指标优于 SHOW；EMDB-1 的 HS-V10 略高于 SHOW（0.0181 vs 0.017），其余三项更低。Human3R 仍在两数据集 HS-CF 上最优。消融 Table 4 不变。

当前 PDF：`outputs/paper/groundedhuman_v1_table3_values_20260924/main.pdf`。修改前备份与数值/排名验证：`outputs/debug/groundedhuman_v1_table3_values_20260924/`。本轮未重新运行实验。

来源：用户提供两张结果截图，以及本地 `.paper/base_pdf/SHOW.pdf` 第 6 页 Table 3 和 §4.2 指标定义。用户所写路径在本地未找到，已通过文件检索定位到该 PDF。

本轮仅使用参考论文的指标定义、公开比较数值及引用信息，属于论文内容适配，不涉及参考代码移植或模型、训练框架变更。VGGT baseline 不变，不新增模型配置。继续以 v1 为唯一论文工作版本，附录未修改。

## 表格内容与排版

新增 §4.4 Human–Scene Consistency，位于 Human Mesh Recovery 之后、消融实验之前。新表为 Table 3；原 Dynamic Interaction Grounding 消融顺延为 §4.5 / Table 4，原 LaTeX 标签保持，交叉引用由编译自动更新。

用户后续要求补入 3DPW 分组：已按 SHOW 第 6 页录入 Human3R、UniSH、SHOW 的四项 3DPW 数据，GroundedHuman 的四个 3DPW 单元格保持真正空白，不填零或推测值。EMDB-1 数据不变。SHOW 原表中的 Ours 已转换为 SHOW，GroundedHuman 为用户结果。正文性能分析明确限定在 EMDB-1。

3DPW 数值按 HS-V5、HS-V10、HS-CF5、HS-CF10 排列：Human3R 为 0.652、0.640、1.30、1.32；UniSH 为 0.028、0.028、7.65、7.52；SHOW 为 0.030、0.026、5.69、5.44。按已报告结果重新标注最优和次优，包含 Human3R 在两项 HS-CF 上的最优值。

双数据集版本 PDF 位于 `outputs/paper/groundedhuman_v1_consistency_3dpw_20260923/main.pdf`；Table 3 仍在第 10 页。对应修改前备份位于 `outputs/debug/groundedhuman_v1_consistency_3dpw_20260923/`。

| Method | HS-V5 | HS-V10 | HS-CF5 | HS-CF10 |
|---|---:|---:|---:|---:|
| Human3R | 3.66 | 2.91 | 1.13 | 1.31 |
| UniSH | 0.033 | 0.037 | 7.10 | 6.99 |
| SHOW | 0.013 | 0.017 | 5.88 | 5.79 |
| GroundedHuman | 0.0087 | 0.0117 | 4.99 | 4.86 |

所有指标越低越好；逐列加粗最优、下划线次优。保留用户提供的数字精度。原表下方两条训练数据备注及其上标均不添加；caption 保留统一来源引用，未把基线称为本项目复现实验。

## 文字说明

- HS-V：可见人体表面与人体 mask 内场景点云在相机水平/垂直方向的空间方差差异。
- HS-CF：源论文定义的归一化最近邻表面距离；不误写成双向距离或碰撞率。
- 下标 5 / 10 分别为 5–95 和 10–90 百分位过滤，并非厘米阈值。
- 结果说明准确区分：GroundedHuman 两项 HS-V 最好、两项 HS-CF 次优；全部指标优于 SHOW 和 UniSH，Human3R 仍有最低 HS-CF。
- 这些指标描述重建内部的人体–场景表面一致性，不单独等价于真实米制精度或物理接触控制，不能据此关闭 B5 的尺度验证待办。

新增参考文献键 `shi2026show`：Boao Shi、Qiao Feng、Yiming Huang、Lingjie Liu，2026，arXiv:2606.27720；标题为 Scene and Human in One World: Reconstruction in a Feedforward Pass。依据 PDF 首页面元信息使用 arXiv 引用，不照搬该 PDF 中占位的 ACM 卷期/DOI。

## 文件与验证

- 修改 `v1/tex/04_exp.tex` 和 `v1/references.bib`。
- 修订前备份、来源文本和原文第 6 页渲染：`outputs/debug/groundedhuman_v1_consistency_20260923/`。
- 新 PDF 与日志：`outputs/paper/groundedhuman_v1_consistency_20260923/`。
- 本轮只录入作者提供结果和参考论文数值，没有重新运行实验或独立确认作者评测实现；数字核对与编译/版面验证另存 verification.json。
