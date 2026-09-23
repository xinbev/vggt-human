# 基于用户粉色 teaser 的精修设计

## 文件

- `teaser_refined.png`：3720 × 2160 排版预览。
- `teaser_refined.svg`：嵌入截图素材，标题、说明、图例、引线均为可编辑矢量对象。
- `comparison.png`：用户原稿与精修稿的并排比较。
- `compose.py`：本机 Pillow 排版源文件，可重新生成以上文件。
- 四个素材 PNG：从用户上传的设计截图裁切，未重新生成或运行模型。

输入：`C:/Users/ROG/AppData/Local/Temp/codex-clipboard-d90a79ac-3330-48fe-a550-51247d07f03f.png`。

## 保留与补充

保留粉色人体、透视倾斜场景、人体与图像的投影示意、左上 anchor / 左下 region 的纵向布局。移除旧标签边缘，并统一字体、编号和说明。

上方叙事：人体米制先验通过 anatomical anchors 与局部 scene tokens/几何 probes 关联，用于尺度与深度偏差校正。明确区分 3×3 **特征 token** 邻域与几何 probe 的 9×9 **深度像素**搜索。9×9 为当前实现参数，并非概念上必需的固定值。24 是方法使用的 anchor 总数，截图只示意其中若干个。

下方叙事：以 region 为中心的四类**图像空间方形支持域**（3×3、7×7、随投影区域自适应的窗口、外围方环）提供不同尺度的几何证据。点颜色区分 self-surface 与 environment；支持窗口的轮廓颜色另有含义。每个区域对整个人体平移提出建议，可靠性加权后形成整体平移更新；姿态与形状保持不变。

右侧叙事：a 对应场景校准，b 对应整体人体位置细化，两种作用在同一主场景中汇合。顶部横括线是文字说明的范围标记，**不是带单位的尺度测量尺**。髋部圆圈和引线是目视定位的说明标记，**不是精确投影坐标、接触检测或预测平移向量**。

依据：只读参考 `.paper/my_paper/GroundedHuman.pdf` 的 Anchor–Scene Cross Attention、Multi-Scale Region Sampling 和 Regional Translation Prediction，以及现有 `scripts/vis/teaser_asset_utils.py` 的方形支持域与颜色约定。本次没有修改模型、推理或 baseline。

## 可用于论文的图注草稿

**GroundedHuman connects human metric priors with local scene evidence for human–scene alignment.** (a) Body anchors query local scene tokens and gather geometric probes to calibrate scene scale and depth bias. (b) Multi-scale regional sampling separates self-surface and environmental evidence; reliability-weighted regional proposals refine whole-body translation while preserving body pose and shape.

该文字描述方法意图；仅凭这张构图稿不能宣称定量提升、精确接触或无穿模。

## 投稿前素材替换

目前使用的是用户设计截图，所以 PNG 导出尺寸提高不代表素材本身细节增加。正式稿应在相同位置替换为原始高清 attention / sampling 图片、透明人体渲染和场景背景。大人体外沿白边及原人物头发残留来自输入图；此次没有把它们伪修成新的重建结果。SVG 中可独立替换四个 image 节点，保留全部矢量标签。

已完成 PNG 视觉检查与 SVG XML 结构检查。本设计不需要服务器或 checkpoint。
