# Person Detection And Cross-Frame Tracking Handoff

本文档给后续修改模型部件逻辑的 agent 使用，汇总当前项目里“人物框获取 + 跨帧人物 ID 跟踪”的设计、实现位置、数据格式和模型侧接入点。

## 目标

当前要补的是视频输入里的 observation frontend：

```text
原始视频 / BEDLAM rgb 序列
  -> 每帧人物检测 bbox
  -> 跨帧人物 ID 跟踪
  -> 可选遮挡断轨后处理
  -> 输出 sidecar
  -> 模型侧使用 bbox 初始化 SMPL query，使用 track_id 驱动 HSI temporal memory
```

这个模块不替换 VGGT baseline，也不改 `.reference/`。它只给现有 `VGGTOmega.forward(...)` 提供更真实的视频人物观测输入。

## 已实现文件

核心 tracking 包：

```text
vggt_omega/tracking/
  schema.py                 # Detection / TrackObservation / FrameObservations 数据结构
  detectors.py              # YOLO TorchScript person detector
  boosttrack_adapter.py     # BoostTrack++ wrapper
  postprocess.py            # tracklet stitching / ID 压紧
  clip_builder.py           # sidecar -> smpl_query_boxes / smpl_track_ids
  diagnostics.py            # tracking summary
  track_memory.py           # HSI-aware model feedback memory
  sam2_masks.py             # 可选 SAM2 mask，当前测试暂不使用
```

预处理和测试入口：

```text
scripts/preprocess/prepare_video_person_tracks.py
scripts/preprocess/prepare_video_person_tracks.sh
scripts/preprocess/test_bedlam_person_tracks.sh
```

可视化和评估：

```text
scripts/vis/visualize_video_person_tracks.py
scripts/eval/evaluate_bedlam_person_tracking.py
scripts/eval/eval_bedlam_person_tracking.sh
```

路径配置：

```text
configs/path.yaml
```

相关说明：

```text
docs/video_person_tracking_frontend.md
docs/crowd4d_cross_frame_id_tracking.md
```

## 第三方依赖与权重

当前默认使用：

```text
YOLO TorchScript detector:
  checkpoints.yolo8x

BoostTrack++:
  third_party/BoostTrack

BoostTrack/ReID weights:
  third_party/weights/BoostTrack

SAM2:
  third_party/sam2
  third_party/weights/sam/sam2.1_hiera_large.pt
```

`configs/path.yaml` 中第三方路径使用项目相对路径：

```yaml
third_party:
  boosttrack_root: "third_party/BoostTrack"
  boosttrack_weights_root: "third_party/weights/BoostTrack"
  sam2_root: "third_party/sam2"
  sam2_checkpoint: "third_party/weights/sam/sam2.1_hiera_large.pt"
```

原因：服务器 checkout 可能叫 `vggt-human`，本地/另一个环境可能叫 `vggt-omega`。脚本里也做了 fallback，如果旧绝对路径不存在，会尝试当前仓库根目录下的相对路径。

## 主流程

### 1. Person detector

实现位置：

```text
vggt_omega/tracking/detectors.py
```

类：

```python
TorchScriptYOLOPersonDetector
```

输入：

```text
frame_bgr: np.ndarray [H,W,3]
```

输出：

```python
list[Detection]
```

每个 detection 包括：

```python
{
  "bbox_xyxy_pixels": [x1, y1, x2, y2],
  "bbox_cxcywh_norm": [cx, cy, w, h],
  "det_score": float,
  "class_id": 0,
  "class_name": "person",
}
```

YOLO parser 做了兼容：

```text
1. post-NMS 输出: [N, 6] / [B, N, 6]
2. YOLOv8 raw 输出: [N, 84] / [B, N, 84] / [B, 84, N]
3. bbox 可能是输入尺寸像素坐标，也可能是 0-1 归一化坐标
```

### 2. Online tracker

实现位置：

```text
vggt_omega/tracking/boosttrack_adapter.py
```

类：

```python
BoostTrackPersonTracker
```

它直接调用 `third_party/BoostTrack/tracker/boost_track.py` 中的 `BoostTrack.update(...)`，不走 BoostTrack 自带 YOLOX detector。

输入：

```text
frame_bgr
detections: list[Detection]
frame_id
frame_index
video_name
```

输出：

```python
list[TrackObservation]
```

每个 observation 包括：

```python
{
  "frame_id": "000012",
  "frame_index": 12,
  "person_id": 3,
  "person_id_valid": true,
  "bbox_xyxy_pixels": [120.0, 44.0, 210.0, 310.0],
  "bbox_cxcywh_norm": [0.32, 0.41, 0.12, 0.35],
  "bbox_valid": true,
  "bbox_source": "yolo_torchscript+boosttrack",
  "det_score": 0.93,
  "track_confidence": 0.91,
  "missing_count": 0,
  "valid": true
}
```

注意：

```text
BoostTrack 原始 ID 不保证连续。
可能出现外部可见 ID 为 1,2,3,5，没有 4。
这是因为内部创建过短轨迹，但没有通过输出过滤。
```

### 3. Tracklet stitching 后处理

实现位置：

```text
vggt_omega/tracking/postprocess.py
```

函数：

```python
postprocess_sidecar_tracks(...)
```

目的：

```text
解决遮挡后 ID 没接回来的问题。
例如:
  ID 1: frame 0-7
  遮挡: frame 8-9
  ID 5: frame 10-26
后处理将:
  raw ID 5 -> final ID 1
```

判断依据：

```text
1. 两段 tracklet 时间上前后相接
2. gap <= stitch_max_gap
3. 前一段 bbox 按速度外推后，接近后一段首帧 bbox
4. bbox 尺度接近
5. 预测框与新框 IoU 合理
```

默认参数：

```text
stitch_max_gap = 30
stitch_center_thresh = 1.25
stitch_size_log_thresh = 0.70
stitch_min_score = 0.25
compact_ids = true
```

后处理会改写：

```text
smpl_boxes/*.pkl
observations.jsonl
summary.json
```

并新增：

```text
track_id_postprocess.json
```

如果 ID 被改写，会保留原始 ID：

```python
{
  "person_id": 1,
  "track_id_before_postprocess": 5
}
```

可视化时会显示：

```text
ID 1 raw5
```

意思是最终 ID 是 1，原始 BoostTrack ID 是 5。

## Sidecar 输出

默认输出目录：

```text
outputs/preprocess/video_tracks/<source_name>/
```

BEDLAM 输出目录：

```text
outputs/preprocess/video_tracks/Training/<sequence_name>/
```

文件结构：

```text
observations.jsonl
summary.json
track_id_postprocess.json
smpl_boxes/<frame_id>.pkl
masks/<frame_id>.npz        # 仅 enable_sam2_masks 时
```

`smpl_boxes/<frame_id>.pkl` 中的 frame dict：

```python
{
  "frame_id": str,
  "frame_index": int,
  "image_path": str,
  "image_hw": [height, width],
  "detections": [...],
  "persons": [...]
}
```

`persons` 里的每个元素是模型真正需要的 observation：

```python
{
  "person_id": int,
  "person_id_valid": bool,
  "bbox_xyxy_pixels": [x1, y1, x2, y2],
  "bbox_cxcywh_norm": [cx, cy, w, h],
  "bbox_valid": bool,
  "track_confidence": float,
  "valid": bool,
}
```

## 模型侧输入张量

已有模型接口：

```python
VGGTOmega.forward(
    images,
    smpl_query_boxes=None,
    smpl_query_boxes_mask=None,
    smpl_track_ids=None,
    smpl_track_mask=None,
)
```

sidecar 转 tensor 的工具：

```text
vggt_omega/tracking/clip_builder.py
```

函数：

```python
build_clip_tensors_from_sidecar(sidecar_root, frame_ids=None, max_humans=None, device=None)
```

返回：

```python
{
  "smpl_query_boxes":      Tensor[1, S, Q, 4],
  "smpl_query_boxes_mask": Tensor[1, S, Q],
  "smpl_track_ids":        Tensor[1, S, Q],
  "smpl_track_mask":       Tensor[1, S, Q],
  "frame_ids":             list[str],
  "frame_indices":         list[int],
  "slot_track_ids":        list[int],
}
```

约定：

```text
B = 1
S = clip 帧数
Q = clip 内 track slot 数
4 = 归一化 cx, cy, w, h
```

slot 映射：

```text
slot 0 -> person_id 1
slot 1 -> person_id 2
slot 2 -> person_id 3
```

某个人某帧不可见：

```python
smpl_query_boxes_mask = False
smpl_track_mask = False
```

## 当前 VGGTOmega / SMPL / HSI 关系

现有代码里，顺序是：

```text
Aggregator
  -> CameraHead / DenseHead
  -> SMPLHead
  -> HSIRefinementHead
```

位置：

```text
vggt_omega/models/vggt_omega.py
```

SMPLHead 先输出 base 人体：

```text
pred_pose_6d
pred_poses
pred_betas
pred_transl_cam
pred_confs
```

HSIRefinementHead 再输入 base SMPL + scene/depth/camera：

```text
pred_pose_6d / pred_betas / pred_transl_cam / pred_confs
depth
pose_enc
aggregated scene tokens
track_ids
track_mask
```

HSI 输出 refined 人体：

```text
hsi_refined_pred_pose_6d
hsi_refined_pred_poses
hsi_refined_pred_betas
hsi_refined_pred_transl_cam
hsi_contact_logits
hsi_scene_scale
hsi_scene_depth_bias
```

关键点：

```text
HSI 不负责发现 ID。
HSI 使用 tracker 给出的 track_id 作为 temporal memory 的 key。
如果 track_id 错了，HSI memory 会把不同人的历史状态混在一起。
```

HSI temporal memory 位置：

```text
vggt_omega/models/heads/hsi_refinement_head.py
```

相关函数：

```python
_inject_temporal_memory(...)
_update_temporal_memory(...)
_temporal_memory_key(...)
```

`_temporal_memory_key(...)` 当前逻辑：

```text
优先使用 smpl_track_ids；
如果没有有效 track_id，则 fallback 到负 query index。
```

因此模型侧修改时要注意：

```text
1. 不能假设 query slot index 等于 person identity。
2. 同一个 person_id 在不同帧可能占同一个 slot，clip_builder 当前会保持这一点。
3. HSI temporal memory 应只在 track_mask=True 且 track_id>=0 时强使用历史。
4. 对低质量/短轨/低置信 track，应降低 memory 权重或跳过。
```

## HSI-aware Track Memory

实现位置：

```text
vggt_omega/tracking/track_memory.py
```

类：

```python
HSITrackMemory
```

用途：

```text
保存模型输出的人体状态，供后续更强的 HSI-aware tracking / stitching 使用。
```

可写入：

```text
base pose
base betas
base transl_cam
hsi_refined_pose
hsi_refined_betas
hsi_refined_transl_cam
confidence
```

设计意图：

```text
第一阶段：tracking 主要靠 bbox + BoostTrack + stitching。
第二阶段：可在高置信情况下使用 HSI refined transl / betas 辅助 ID 合并。
```

建议优先使用：

```text
transl_cam / hsi_refined_transl_cam: 3D 位置连续性
betas / hsi_refined_betas: 体型一致性
pose: 只做弱约束，不要作为身份主依据
```

## 运行命令

BEDLAM 单序列测试：

```bash
bash scripts/preprocess/test_bedlam_person_tracks.sh
```

指定序列和帧数：

```bash
SEQ_INDEX=3 MAX_FRAMES=240 SPLIT=Training bash scripts/preprocess/test_bedlam_person_tracks.sh
```

直接调用：

```bash
python scripts/preprocess/prepare_video_person_tracks.py \
  --bedlam-sequence-index 0 \
  --bedlam-split Training \
  --path-config configs/path.yaml \
  --output-root outputs/preprocess/video_tracks \
  --overwrite \
  --max-frames 120
```

可视化：

```bash
python scripts/vis/visualize_video_person_tracks.py \
  --sidecar-root outputs/preprocess/video_tracks/Training/<sequence_name> \
  --output-dir outputs/vis/video_person_tracks/Training/<sequence_name> \
  --write-video
```

BEDLAM tracking 评估：

```bash
bash scripts/eval/eval_bedlam_person_tracking.sh
```

小规模评估：

```bash
MAX_SEQUENCES=2 MAX_FRAMES_PER_SEQUENCE=120 bash scripts/eval/eval_bedlam_person_tracking.sh
```

只评估已有 sidecar：

```bash
RUN_TRACKER=0 bash scripts/eval/eval_bedlam_person_tracking.sh
```

## 当前 101 个序列测试观察

基于本地已同步的 `outputs/preprocess/video_tracks/Training/*/summary.json`，当前 tracking sidecar 统计如下：

```text
序列数: 101
总帧数: 3077
YOLO 检测框: 9358
最终 track observation: 8733
平均检测人数/帧: 3.04
平均跟踪人数/帧: 2.84
平均 track confidence: 0.988
```

轨迹数量：

```text
总 track 数: 380
平均每序列 track 数: 3.76
中位数: 4
最少: 3
最多: 6
```

分布：

```text
3 tracks: 48 个序列
4 tracks: 36 个序列
5 tracks: 10 个序列
6 tracks: 7 个序列
```

短轨：

```text
总 track: 380
长度 <= 2 帧: 24 条
长度 <= 5 帧: 49 条
track 长度中位数: 25 帧
track 长度均值: 22.98 帧
最长 track: 48 帧
```

gap / 断观测：

```text
有 gap 的序列: 68 / 101
总 gap 数: 116
单序列最大 gap 数: 5
```

stitching：

```text
发生 stitching 的序列: 20
总 stitching 次数: 25
被改写的 observation 数: 774
```

这说明：

```text
1. 主体 track 基本可用。
2. stitching 确实修复了一部分遮挡断 ID。
3. 仍存在短轨和漏检 gap。
4. 如果直接喂 HSI temporal memory，建议对低质量 track 做 mask 或降权。
```

## 目前风险

### 1. 短轨风险

有一些长度很短的 track：

```text
<= 5 帧短轨占比约 12.9%
```

建议模型侧使用前过滤：

```text
min_track_length >= 5
或 visible_ratio >= 0.15
```

也可以在 `clip_builder` 或模型 forward 前生成：

```text
track_quality_mask
```

### 2. gap 风险

有 gap 不一定是 ID switch，也可能只是检测漏人。

模型侧建议：

```text
1. track_mask=False 的帧不要更新 temporal memory。
2. 如果同一 track gap 太长，恢复后 memory gate 应降低。
3. HSI temporal memory 可以引入 gap-aware decay。
```

### 3. ID switch 风险

当前 summary 只能看 tracking 自身情况，不等于 GT ID accuracy。

真实 ID 准确度需要看：

```text
outputs/eval/bedlam_person_tracking/bedlam_person_tracking_eval.json
outputs/eval/bedlam_person_tracking/bedlam_person_tracking_eval.csv
```

评估脚本指标：

```text
match_recall
match_precision
id_dominant_accuracy
id_switches
id_switches_per_100_matched
fragmentations
fragmentations_per_100_matched
```

如果其他 agent 要改模型侧逻辑，应优先以这些 GT eval 指标为准。

## 给模型侧 agent 的建议

### 最小接入

模型侧只需要从 sidecar 构造：

```python
smpl_query_boxes
smpl_query_boxes_mask
smpl_track_ids
smpl_track_mask
```

然后传入：

```python
predictions = model(
    images,
    smpl_query_boxes=boxes,
    smpl_query_boxes_mask=box_mask,
    smpl_track_ids=track_ids,
    smpl_track_mask=track_mask,
)
```

### HSI memory 推荐改动

建议模型侧支持：

```text
1. track_quality / track_confidence 输入
2. gap-aware memory decay
3. 短轨不更新 memory
4. 低置信 track 只使用当前帧，不注入历史
5. memory key 必须使用 person_id，而不是 query index
```

### 不建议

```text
1. 不要让 HSI 自己发现 ID。
2. 不要默认 query slot index 就是身份。
3. 不要对 track_mask=False 的帧更新 memory。
4. 不要把短轨、低置信 track 的状态写入长期 memory。
```

## 与 Crowd4D 思路的关系

当前方案借鉴的是 Crowd4D 的 observation frontend 思路：

```text
YOLO/YOLOX 检人
BoostTrack++ 做跨帧 ID
可选 SAM2 mask
可选 DWPose keypoint confidence
按 person_id 组织后续人体重建
```

我们当前额外做了：

```text
1. project-local sidecar 格式
2. 与 VGGTOmega.forward 的 smpl_query_boxes / smpl_track_ids 对齐
3. tracklet stitching 后处理
4. HSITrackMemory，为后续人体状态反馈 tracking 留接口
```

当前 SAM2 已接入但默认不用；DWPose 尚未接入。

