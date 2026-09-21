# 左列双层方法细节 + 右侧大场景

`layout_existing_assets.png`：从用户提供的素材总览中裁切 scene、anchor20、hands r54 并排版。采样图样原样保留，截图本身限制了清晰度。引线只是排版来源提示，不是 measured attention 或平移向量。

NBAI 生成尝试：两次请求均在 90 秒超时后停止，没有收到生成图片，不存在已成功生成的 teaser_stacked_concept.png。当前交付为 layout_existing_assets.png。其尺度标记与方法标题说明设计语义，不是额外的测量结果。

设计逻辑：

- 左上是人体尺度先验与局部 scene token 的 cross attention，并以深度仿射关系说明作用；s 为包含初始化的有效尺度，b 为深度偏置。
- 左下是区域四尺度几何采样，区分 self-surface/environment，语义是 regional proposals 共同支持全身平移。
- 右侧以 curled sitting 姿态的人体和沙发作为主视觉，保持当前照片的姿势而不是改成双脚落地。
- 仅用尺度标记、稀疏 anchor、区域建议与整体位移说明科学叙事，不添加网络方块、张量路径或性能数字。
- 本文件夹全部属于左侧 teaser 主视觉的设计，用户计划的右侧定量图尚未加入。

现有截图没有 per-iteration vector 或可靠性原始文件，因此不声称示意箭头的位置、长度或方向来自推理。终稿以 `translation_iterations.npz`、RGBA 和 PLY 实际导出替换。
