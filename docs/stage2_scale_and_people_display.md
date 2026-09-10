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
