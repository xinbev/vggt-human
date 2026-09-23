# v1 消融实验与 Contact 定义修订记录

日期：2026-09-23。依据：用户确认两行消融配置对应是否启用 Dynamic Interaction Grounding，并确认使用 RICH / UniCon3R 物理评测协议（稳健地面估计、5 mm 容差、相同划分、分母与聚合方式、共享输入及权重和样本）。这些属于作者提供的实验信息，本轮未重新运行或独立复核实验。

## 已完成

- §4.4 更名为 `Ablation of Dynamic Interaction Grounding`。
- Table 3 第一行 `GroundedHuman w/o Dynamic Interaction Grounding`，第二行 `GroundedHuman`。第一行在单元格内换行，避免过长名称挤压数据列。
- 同步重写 §4.4 正文与 caption：相同观测、预训练估计器、尺度校准和共享权重；完整模型增加区域交互编码与两步整体平移更新；保留姿态和形状。说明 RICH 的共同协议，并解释四项现有结果。移除已由用户确认的协议 TODO。
- Table 1 的 `Output` 分组更名为 `Capabilities`，`Contact` 更名为 `Contact Control`；caption 定义为通过显式约束或学习的几何更新主动调整人体放置、减少穿模，不要求输出 contact labels。
- 保留 Ours 的 Contact 勾选及原表其他勾选和数值；没有将接触控制表述成保证完全无穿模。
- B3 按用户指示关闭，不改 Online / Offline 分类；这不等于本轮新增因果在线验证。
- 从待办清单删除 A4、A5、A6、B2、B4；C2 因 §4.4 标题和 caption 已明确模块范围一并关闭；B3 作为用户明确跳过项移出待办。

## 保留待确认

B5 未对论文指标或曲线做文字替换，另见 [尺度评测设计](C:/Users/ROG/PycharmProjects/vggt-omega/docs/groundedhuman_v1_metric_scale_evaluation.md)。需要用保留尺度的 SE(3)-ATE 重新评测后，才能将该图用于证明米制校准贡献。

## 文件与验证

- 论文修改仅涉及 `.paper/my_paper/versions/v1/tex/04_exp.tex`；附录、模型代码、实验配置、图像资产和实验数值未修改。
- 修订前快照位于 `outputs/debug/groundedhuman_v1_ablation_revision_20260923/before/`。
- 重新编译的 PDF 与编译日志位于 `outputs/paper/groundedhuman_v1_ablation_revision_20260923/`。最终编译及版面检查结果记录于对应输出目录的验证文件。
