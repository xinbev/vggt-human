# Body Region Context Query — B 部分设计

## 交付

- `body_region_context_query.png` / `.svg`：B 部分独立图，可编辑文字与形状。
- `teaser_body_region_query.png` / `.svg`：置回左侧下半部分的完整 teaser。
- `compose.py`：独立可复现的本地排版脚本。
- `build_variant.py`：从上一版排版构造本次版本。

保留用户粉色人体和真实采样截图；本次未运行推理、未更改模型。参考图只借鉴其彩色信息来源汇入 query 字段的视觉语言，没有使用蝴蝶、时间坐标、Fourier embedding 或 encoder/decoder 结构。

## 核心叙事

以人体区域为条件，读取多尺度的自身表面与环境几何证据，构造区域上下文表示，预测该区域对整体人体位置的修正建议；最终可靠性加权汇聚为一个整体平移。

上方：区域采样素材 + 四个明确的方形支持域。3×3 和 7×7 图标分别显示对应数量的位置；Adaptive 和 Annulus 为支持域形状示意，不表示当前素材测得的实际半径。图标中的粉色中心是区域查询中心，尤其方环中心不属于该支持域的采样点。

中间：每个尺度下的 Self 与 Env. 两个槽，共 8 个 pooled context tokens。淡色色带表示信息来源及组合关系；不表示 attention 数值、置信度或实测权重。

下方：以 query 组成条承接四类信息，构成 q_r。没有在 query 与结果之间展开完整网络，以保持 teaser 的信息密度。

## 与现有实现的对应

这里的 **Body Region Context Query** / **q_r** 是对区域条件表示的概念性命名，不是当前代码中已经存在的 Transformer Q tensor。当前实现没有本图参考来源中的 Cross-Attention Decoder。

| 图中字段 | 当前实现 |
|---|---|
| Identity | `region_group_ids` 对应的 anatomical body-part group embedding；并非每个细分 region 独立的唯一 embedding |
| Geometry | region centroid、representative vertices 的坐标标准差、归一化投影坐标、自适应半径 |
| Context | 4 个支持域 × self/environment 两通道，point encoder 后经 masked attention pooling，得到 8 个 context tokens |
| Validity | 各支持域/通道的 valid ratios 与 other-human ratio；不将预测后的 gate/logvar 误画作 query 输入 |

主要依据：

- `vggt_omega/models/heads/hsi_regional_translation_refiner.py` 第 203–225 行附近：组 embedding、features 拼接、feature MLP、vote/gate/logvar prediction。
- `vggt_omega/models/geometry/regional_scene_probe.py` 第 147–174 行附近：point features、point encoding、per-scale/per-channel attention pooling。
- `.paper/my_paper/GroundedHuman.pdf` 的 Multi-Scale Region Sampling、Regional Translation Prediction。

实际流程：采样与池化 → 拼接区域特征 → MLP → regional proposals → 有效性/门控/不确定性加权 → clipping/person gate → 整体 translation update。图中省略 clipping 等细节，保留区域建议到整体平移的关系。建议正文首次使用 Query 时说明这是 region-conditioned context representation，避免读者将其理解为新增 Transformer decoder。

## 图注候选

**Body Region Context Query.** Each body region gathers self-surface and environmental evidence over four image-space supports. The pooled multi-scale context is combined with body-part identity, regional geometry, and support validity to form a region-conditioned descriptor. Reliability-weighted regional proposals then refine whole-body translation while preserving pose and shape.

## 验证与限制

已检查 B 独立图与整体排版，完成 SVG XML 结构检查。区域图仍为用户截图裁切；正式投稿应替换高清原始素材。新增支持域图标、色带和 token 槽均为概念示意，没有伪造测量或实验结果。无需服务器执行，也没有 baseline 改动。
