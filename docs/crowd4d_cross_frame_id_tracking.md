# Crowd4D Cross-Frame Person ID Tracking Notes

## 背景

Crowd4D 的前端 observation pipeline 负责把原始单目 RGB 视频整理成后续 4D 人群重建可用的结构化输入。这里的 YOLOX、SAM2、DWPose 和 BoostTrack++ 不是论文的核心创新模块，但它们决定了后续 SMPL motion sequence、temporal loss、CSCR 人群结构约束能否稳定工作。

整体流程可以概括为：

```text
RGB video frame I_t
-> YOLOX 检测每帧的人
-> SAM2 给每个检测到的人生成 instance mask
-> DWPose 估计 2D joints 和 per-joint confidence
-> BoostTrack++ 将不同帧中的 detection 关联为 persistent person ID
-> 得到每个 valid person-frame pair 的观测
   bbox b_{n,t}, mask m_{n,t}, keypoint confidence Jconf_{n,t}
-> 按 person ID 初始化和优化 SMPL motion sequence Q_n
```

## 各模块职责

### YOLOX：逐帧行人检测

YOLOX 只负责在每一帧里检测人物，输出每个检测到的人的 bounding box。

论文中的记号为：

```text
b_{n,t} in R^4
```

它提供的是“这一帧哪里有人”，但不提供跨帧 ID。也就是说，YOLOX 本身不知道第 `t` 帧的某个 box 和第 `t+1` 帧的某个 box 是否属于同一个人。

YOLOX 输出后续主要用于：

- 给 BoostTrack++ 做跨帧关联。
- 给 SAM2 提供 person region / prompt。
- 给人体 motion reconstruction 方法提供人物定位。
- 在 reprojection loss 中用 bbox 尺度做归一化，缓解远近人物尺度差异带来的梯度不平衡。

### SAM2：实例 mask

SAM2 根据检测框生成每个人的 instance mask：

```text
m_{n,t} in {0, 1}^{H x W}
```

YOLOX 只给框，框内可能有背景、地面、其他人。SAM2 mask 更精细地告诉系统“哪些像素属于这个人”。

在 Crowd4D 中，mask 的明确用途之一是在 close-range HSIP 处理里辅助估计局部支撑位置。近景人体可能遮挡地面支撑区域，此时作者会从可靠 body joints 或 person mask 附近的 local scene-depth observations 估计支撑位置。

### DWPose：2D 关键点与置信度

DWPose 提供每个人的 2D joints 和 per-joint confidence。论文中记录的是：

```text
Jconf_{n,t} in [0, 1]^17
```

它的核心作用是告诉后续优化哪些图像关键点可信。大场景远处人很小，遮挡严重，如果所有 2D joints 都等权使用，容易污染 reprojection constraint。

### BoostTrack++：跨帧 ID 关联

BoostTrack++ 是这里真正负责跨帧人物 ID 的模块。

YOLOX 的逐帧输出类似：

```text
frame 1: box A, box B, box C
frame 2: box D, box E, box F
```

BoostTrack++ 会根据检测框位置、运动连续性、tracklet 信息以及可能的外观线索，将不同帧中的检测结果关联起来，输出 persistent identities：

```text
person id 1:
  frame 1 -> bbox b_{1,1}, mask m_{1,1}, conf Jconf_{1,1}
  frame 2 -> bbox b_{1,2}, mask m_{1,2}, conf Jconf_{1,2}
  frame 3 -> missing or valid

person id 2:
  frame 1 -> bbox b_{2,1}
  frame 2 -> bbox b_{2,2}
  ...
```

论文中写作：

```text
for each valid person-frame pair (n, t):
  b_{n,t} in R^4
  m_{n,t} in {0,1}^{H x W}
  Jconf_{n,t} in [0,1]^17
```

这里的 `n` 就是 BoostTrack++ 维护出来的跨帧人物 ID。

## 跨帧 ID 在后续重建中的作用

一旦有了 person ID，Crowd4D 就可以把同一个人的多帧观测组织成一条 SMPL motion sequence：

```text
Q_n = { Q_{n,t} }_{t=1}^T
Q_{n,t} = (theta_{n,t}, beta_{n,t}, gamma_{n,t}, tau_{n,t})
```

后续几个关键环节都依赖 ID：

- 每个 identity 单独初始化 temporally consistent SMPL motion。
- Stage-1 按 ID 优化 root translation `tau_{n,t}` 和 global orientation `gamma_{n,t}`。
- temporal smoothness 约束同一个人的相邻帧 motion。
- CSCR 在人群邻接图中约束局部邻居之间的相对位移和方向随时间保持结构一致。

如果没有稳定 ID，后续 temporal loss 和 crowd structural loss 会把不同人的轨迹误当作同一个人，导致 motion sequence 被污染。

## 遮挡丢帧时 ID 如何处理

如果一个人在中间因为遮挡丢失 `n` 帧，到 `t+n+1` 帧重新出现，通常有三种情况。

### 情况 1：短遮挡，恢复原 ID

如果遮挡时间短、重新出现的位置合理、运动轨迹可预测，tracker 通常会维护一个 lost track 状态：

```text
t       : ID 7 visible
t+1...t+n: ID 7 missing, but track remains alive
t+n+1   : new detection appears near predicted location
-> matched to lost track
-> still ID 7
```

这类恢复一般依赖：

- 运动预测，例如 Kalman filter。
- bbox 位置、大小、速度连续性。
- detection score。
- tracklet 信息。
- 有些 tracker 会额外使用 ReID appearance feature。

### 情况 2：遮挡太久，生成新 ID

如果遮挡太久、场景过密、重新出现位置不确定，tracker 可能会终止旧轨迹并创建新 ID：

```text
t       : ID 7 visible
t+1...t+n: missing
t+n+1   : detection appears, but cannot be reliably matched
-> create ID 23
```

这会造成 track fragmentation。Crowd4D 论文没有声称完全解决这个问题，因此实验里区分了：

```text
Unified Tracking-by-Detection:
  使用检测和跟踪结果，包含真实 tracking error。

Ground-truth Object Tracking:
  使用 GT identity association，用来隔离重建方法本身的质量。
```

### 情况 3：ID switch

最麻烦的是接错 ID：

```text
ID 7 被遮挡
ID 12 经过附近
tracker 把后续 detection 错接给 ID 7
```

这会污染整条 SMPL motion sequence。CSCR 可以缓解局部结构突变，但无法从根上保证 ID 一定正确。

## Crowd4D 对 missing frames 的后端处理

论文没有展开完整 tracking recovery 细节，但从公式和描述可以看出：

1. 只对 valid person-frame pair 记录观测：

```text
(n, t) in Omega
```

2. temporal loss 和 CSCR 只在有效追踪帧之间计算。

CSCR 中明确提到：loss 只在 pair 在 `t` 和 `t-Delta` 都 validly tracked 时计算。

因此：

- 如果 tracker 能保持同一个 ID，后端知道这是同一个人，中间缺失帧没有观测，可以靠 motion prior、temporal smoothness 和 group structure 补稳定。
- 如果 tracker 把轨迹断成新 ID，Crowd4D 后端通常会把它当成两个人或两段 track。

## 对 VGGT-Omega / 当前 HSI 的启发

当前项目中的视频 HSI 不应只依赖 query 顺序，而应该显式传递：

```text
track_id
track_mask
missing_count
track_confidence
```

你的 `HSIRefinementHead` 已经有相关入口：

```text
enable_temporal_momentum
temporal_momentum_use_track_ids
track_ids
track_mask
```

这和 Crowd4D 的 person ID 角色相似。推荐策略是：

```text
ID 连续可见:
  沿用 previous HSI/contact tokens。

ID 短暂丢失:
  memory 挂起若干帧，不立即删除。

ID 重新出现且 tracker/ReID 置信高:
  恢复旧 ID 的 HSI temporal memory。

ID 不可靠或遮挡太久:
  降低 memory 权重或重置为新 ID。
```

重要原则：

```text
遮挡 n 帧后是否还是同一人，不应主要由 HSI 判断；
应由 tracker / ReID / tracklet stitching 判断。
HSI 只应该在 ID 可信时使用历史状态。
```

## 可选工程增强：Tracklet Stitching

为了提升视频 HSI 稳定性，可以在 HSI 前增加 tracklet stitching 后处理：

```text
短 tracklet A 结束
短 tracklet B 出现
如果满足：
  时间间隔小
  空间预测合理
  bbox 尺度连续
  外观相似
  运动方向连续
则合并为同一个 person_id
否则保持新 ID
```

这一步对视频人群重建很关键。Crowd4D 的重建质量很大程度依赖 tracking front-end 的质量；如果 ID 断裂或 switch，后面的 SMPL motion、temporal loss、CSCR 都会受到影响。

## 最小建议

如果要在当前项目中借鉴 Crowd4D 的 ID 传递思路，建议优先做三件事：

1. 在视频数据/推理输出中统一保存 `(frame_id, person_id, bbox, mask, keypoint_conf, valid)`。
2. 让 `HSIRefinementHead` 的 temporal memory 严格按 `track_id` 读取和更新，而不是依赖 query index。
3. 增加 ID 质量诊断指标，例如 ID switch 数、track fragmentation 数、missing gap 长度分布、memory reuse 次数。
