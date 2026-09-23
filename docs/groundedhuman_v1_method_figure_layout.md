# GroundedHuman v1：NBAI 方法图

## 最新要求

用户最新要求改为纯文字生成，不向 NBAI 上传参考图。保留此前确定的单排布局和 v1 方法内容，整图仍必须由 NBAI 生成，不采用纯手绘替代。

当前生成目录：outputs/vis/groundedhuman_method_nbai_textonly_20260923/。generation_prompt.txt 为完整纯文字提示词。本次调用不传 --image，尺寸为 3072×1024，quality=high，单次等待上限 600 秒，总尝试上限 4 次。原参考图请求的本地客户端已停止，以避免继续自动重试；已提交的远端任务是否停止无法从本地确认。

前一轮纯结构参考图请求保留在原目录，仅作过程记录。其前两次尝试均约 301 秒收到 HTTP 504，第三次等待中按用户新要求停止。当前纯文字请求的最终状态以实际返回为准。

## 结构

单目视频 → 上下排列的 VGGT-Ω / NLF → 两个并列虚线模块框 → 米制重建输出。

第一模块：Body-Anchored Metric Calibration，内部机制名 Body-Anchored Scene Query。人体锚点与场景特征形成 Interaction representation，然后通过 Metric readout 预测尺度和偏置，并进行时间汇总。

第二模块：Dynamic Interaction Grounding，内部机制名 Hierarchical Context Attention。自身表面/环境证据形成区域 Interaction representation，然后通过 Placement readout 预测 Root translation。框内回环标明 Update → resample ×2。

两模块重复呈现“人体与场景证据 → 交互表示 → 几何修正”。没有添加接触标签预测、时序稳定器或姿态精修。

## 输出与来源

原参考图版本目录：outputs/vis/groundedhuman_method_nbai_structure_20260923/。当前纯文字版本目录：outputs/vis/groundedhuman_method_nbai_textonly_20260923/。

- layout_reference.png：用户新上传的纯结构布局图副本。
- generation_prompt.txt：NBAI 完整生成说明。
- groundedhuman_method：NBAI 返回的原始图像，实际扩展名以生成结果为准。

整图由 NBAI 的 gpt-image-2.5-sunburst 生成。仅允许在输出阶段进行白边裁切和文件格式导出，不用本地手绘图冒充模型生成。

人体、RGB 小图与重建结果均是方法示意，不是本项目的实验结果。论文采用前可换成真实样例。当前不修改 v1 LaTeX 图位，先交付独立设计图供作者审阅。

## 本轮扩展等待与重试

按用户 2026-09-23 的明确要求，本轮单次生成超时设为 600 秒，总尝试次数上限为 4，顺序执行。原 skill 的两次限制仅在 outputs/debug/nbai_extended_generation_20260923/ 下的任务副本中调整，原 skill、API key 和节点配置均未修改。临时副本仍读取原配置路径，并保留仅对瞬时故障重试的规则。

延长时间与增加次数不改变图的结构、提示词或生成模型。最终产物仍须来自 NBAI；本地只裁白边和导出格式。
