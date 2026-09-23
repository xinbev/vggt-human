# Body Region Context Query 精修版

沿用已确认的布局和叙事，仅调整采样符号、色带、字体、信息层级与 query 组成条；前一版保留在相邻 `groundedhuman_body_region_query_20260922` 目录。

## 文件

- `body_region_context_query_polished.png`：B 部分独立预览。
- `body_region_context_query_polished.svg`：B 部分可编辑矢量排版，内部嵌入截图素材。
- `teaser_body_region_query_polished.png` / `.svg`：整体 teaser 排版。
- `compose.py`：独立本地生成脚本；`build_polished.py` 记录相对上一版的构建。

## 精修细节

1. 四种支持域使用相同像素点距、统一边框与中心标记。固定窗口为 3×3 与 7×7；自适应窗口以 9×9 为示意，外围方环内边界与其一致，环宽示意为两像素。自适应和方环图标不是该截图测得的实际半径。
2. 截图下补充 self / environment 图例，颜色与右侧双通道保持一致。
3. 8 个相同 token 槽象征 4 个尺度 × 2 类证据。槽内等距细分线仅表示向量，不表示实际向量维数、权重或置信度。
4. 用统一色带把证据来源关联到 Identity / Geometry / Context / Validity，强调区域条件表示的组合。
5. Query 字段从三行说明缩为两行；外侧括号表示特征拼接，不是对不同维度输入求和。
6. 增加 query 至预测结果的短连接，保留 reliability-weighted aggregation 与 whole-body translation。

## 科学含义

与上一版一致：Identity 指 body-part group embedding；Geometry 概括区域中心、代表点分布、投影坐标与自适应半径；Context 为四尺度双通道 attention-pooled tokens；Validity 概括支持率及 other-human overlap 信息。Query 为视觉层面的 region-conditioned descriptor 名称，未添加 Transformer cross-attention decoder。

模型依据和完整图注见上一版 README。没有运行推理或修改模型/配置。色带、中心标记和窗口图标是概念图形，原采样素材来自用户截图。正式论文仍应替换高清采样原图。

完成 B 独立图、缩放后整体图的视觉检查以及两份 SVG 的 XML 解析检查。
