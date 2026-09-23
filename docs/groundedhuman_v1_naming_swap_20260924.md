# v1 机制标题 Query / Attention 对换

按用户 2026-09-24 指示同步整个当前 v1 论文可编辑文字，图片不动。

| 旧机制标题 | 当前机制标题 |
|---|---|
| Body-Anchored Scene Query | Body-Anchored Scene Attention |
| Hierarchical Context Attention | Hierarchical Context Query |

同步范围：摘要、引言及贡献、方法小标题与概述、主文和附录图注、实验描述、结论、附录架构解释，以及当前中文对照稿、设计说明、图稿规格和相关待办/实验设计文档。

Body-Anchored Metric Calibration 和 Dynamic Interaction Grounding 两个上层模块名不变。真正的 query/key/value、自注意力、交叉注意力、masked attention pooling 仍按算法原意描述；方法公式、数据、训练与模型实现未更改。旧版本 trash、历史生成提示词、历史 PDF 和源码注释不追溯重写。

所有图片文件保持原样，图内旧称呼仍是已知待办；当前生成的 NBAI 图也未接入论文。

修改前文本备份与校验位于 `outputs/debug/groundedhuman_v1_naming_swap_20260924/`；修订后的 PDF 与编译日志位于 `outputs/paper/groundedhuman_v1_naming_swap_20260924/`。
