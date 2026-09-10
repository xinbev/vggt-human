# Stage2 可视化尺度策略与人数显示控制

## 当前 scale 应用策略

`serve_stage2_walking_coarse_scale_hsi_cascade.sh` 当前固定使用：

- `SCENE_SCALE_PREALIGN=smpl_median`
- `HSI_SCENE_AFFINE_MODE=per_frame`
- `CASCADE_EFFECTIVE_AFFINE_MODE=clip_median`（当前 Stage2 cascade 脚本默认启用）
- coarse scale 范围 `[0.10, 10.0]`
- anchor vertex stride 为 8，最少有效 anchor pixels 为 32
- 无有效粗尺度的帧使用有效帧 coarse scale 的 log-space sequence median
- `HSI_VISUAL_SCALE=1.0`

处理链路：

1. 首次 VGGT/NLF forward 得到 raw VGGT depth、相机和 base SMPL。启用 TRSTR 的配置会在这一步临时跳过 TRSTR，避免其在 metric depth 构建前运行。
2. 每帧用置信度通过阈值的 base SMPL 与 raw depth 投影锚点估计解析式 `coarse_scale`。
3. 使用 `coarse_depth = raw_depth * coarse_scale` 进行第二次 model forward，并从 HSI head 得到每帧 residual scale 与 residual bias。
4. 先逐帧合成候选 affine：

   `effective_scale[f] = coarse_scale[f] * hsi_residual_scale[f]`

   `effective_bias[f] = hsi_residual_bias[f]`

5. 当前 `clip_median` 对所有选中帧的候选值做稳健聚合：

   `shared_scale = exp(median(log(frame_effective_scale[f])))`

   `shared_bias = median(frame_effective_bias[f])`

   `final_depth[f] = raw_depth[f] * shared_scale + shared_bias`

6. HSI world 构建时，所有帧 w2c extrinsic 的 translation 同步乘同一个 `shared_scale`；最终 depth 使用上述共享 scale+bias。Stage2/TRSTR 随后修正 SMPL translation，它不再次修改场景 scale。
7. UI 中的 `Visual Scale Multiplier` 是额外的 viewer-only 校准：只缩放 HSI 环境点和 HSI 相机位置，不回写模型结果，默认值 1.0。

因此当前虽然逐帧估计 coarse 与 residual 候选值，但最终整个输入序列应用同一个 scale 和同一个 bias。需要恢复旧行为时，可显式设置 `CASCADE_EFFECTIVE_AFFINE_MODE=per_frame`。

## MAX_FRAMES 三态语义

- `MAX_FRAMES=0`：全部候选帧进入模型。
- `MAX_FRAMES=N` 且 `N>0`：保持原有行为，按照 `FRAME_SAMPLING` 限制到最多 N 帧。
- `MAX_FRAMES=-1`：启用长序列模式；候选帧不超过 500 时全部进入，超过 500 时从完整范围均匀抽取恰好 500 帧并覆盖首尾。
- 小于 `-1` 的值视为配置错误。

## 开始与结束帧范围

- `START_INDEX=0`：排序后源图片列表的起始下标，包含该帧。
- `END_INDEX=-1`：默认读取到序列末尾。
- `END_INDEX=N`：结束下标，包含该帧。例如 `START_INDEX=100 END_INDEX=299` 的候选范围是 200 帧。
- 处理顺序固定为：闭区间范围 `[START_INDEX, END_INDEX]` → `FRAME_STRIDE` → `MAX_FRAMES` 三态限制。
- `END_INDEX < START_INDEX` 或 `END_INDEX < -1` 会直接报错；超过序列末尾的结束下标会自然截到最后一帧。

## 推荐的自动目标帧数接口

日常使用推荐只设置：

```bash
START_INDEX=250
END_INDEX=1749
INFERENCE_FRAMES=300
```

程序会在闭区间 `[250,1749]` 内自动均匀选择 300 帧，包含范围首尾。不再需要手算 `FRAME_STRIDE`。

- `INFERENCE_FRAMES=0`：范围内所有帧进入。
- `INFERENCE_FRAMES=1`：选择范围中间帧。
- 请求数量大于范围长度：范围内全部帧进入，不重复。
- 只要设置了 `INFERENCE_FRAMES`，旧的 `FRAME_STRIDE`、`FRAME_SAMPLING` 和 `MAX_FRAMES` 就会被忽略；未设置时旧采样接口完全保留。
- 实际选中的源序列下标和文件名会写入 `run_summary.json` 的 `frame_selection`。

## NLF 检测置信度

- `NLF_DETECTOR_THRESHOLD` 控制 NLF `detect_smpl_batched()` 是否生成候选，默认 `0.3`。
- `CONF_THRESHOLD` 是检测后的使用阈值，控制人物是否进入 coarse scale、TRSTR、跟踪、人体 mask 和 Viewer scene，Stage2 cascade 默认 `0.05`。
- 对漏检序列建议依次比较 `NLF_DETECTOR_THRESHOLD=0.30/0.15/0.10`，不要直接降到极低值，以免假阳性污染共享尺度和人体点云过滤。
- `run_summary.json` 的 `nlf_detection` 会记录两个阈值、逐帧原始检测人数、通过下游阈值的人数，以及两种空人物帧列表和比例。

## 人数显示控制

原 `SequenceViewer` 新增 `Max People Per Frame` 滑条：

- 默认值 0 对应启动时显示所有已解码人物，保持 baseline。
- UI 滑条范围为 1 到该序列单帧实际最大人数。
- 设置为 N 时，每帧最多显示置信度最高的 N 个 SMPL 和对应 track label。
- 只改变 Viser handle visibility；NLF 检测、最多 query 数、track assignment、人体点云过滤 mask、scale 估计、模型输出和缓存都不变。
- 该限制与 `Accumulated SMPL Frames` 正交：前者控制每个采样时刻的人数，后者控制累计序列展示多少个 SMPL 时刻。
- 完整缓存 viewer 因为复用原 `SequenceViewer`，自动具有同一控件；轻量缓存 viewer 也同步提供该控件。

如需指定初始值，可在原推理/Viewer 启动链路设置：

```bash
DISPLAY_PEOPLE=1 bash scripts/vis/serve_stage2_walking_coarse_scale_hsi_cascade.sh
```

若不设置或设置为 0，启动行为与修改前一致。

## 推荐的两阶段处理命令

第一阶段只推理并写完整 Viewer 缓存：

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

FRAMES_DIR=/path/to/sequence/images \
STAGE2_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full \
CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt \
SCALE_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt \
OUTPUT_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/my_sequence \
FULL_VIEWER_CACHE_OUTPUT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/my_sequence/full_viewer_cache \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}" \
START_INDEX=250 \
END_INDEX=-1 \
INFERENCE_FRAMES=300 \
NLF_DETECTOR_THRESHOLD=0.15 \
CONF_THRESHOLD=0.05 \
CASCADE_EFFECTIVE_AFFINE_MODE=clip_median \
SMOKE_ONLY=true \
bash scripts/vis/serve_stage2_walking_coarse_scale_hsi_cascade.sh
```

第二阶段只读取缓存启动原 `SequenceViewer`：

```bash
CACHE_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/my_sequence/full_viewer_cache \
PORT=8080 \
bash scripts/vis/serve_full_sequence_viewer_cache.sh
```
