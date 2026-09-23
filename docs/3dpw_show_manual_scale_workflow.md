# 3DPW SHOW 人工 scale 评测流程

这套流程把推理与人工校准解耦。每条 3DPW 序列只取前 `min(200, N)` 帧，其中 `N` 是原始序列长度；不做均匀采样，也不继续评测第 201 帧之后的内容。缓存阶段复用当前 HMR4D 3DPW 适配器和可视化脚本使用的 analytic-coarse + residual-scale + Stage-2 cascade；缓存完成后，Viser 不再加载模型或 checkpoint。

汇总时先计算每条序列采样帧的均值，再按原始长度 `N` 加权。例如一条原始 1000 帧的序列只推理前 200 帧，但其采样均值以 1000 帧权重参与最终结果。换句话说，每条有效采样帧的展开权重约为 `N / 有效采样帧数`。

人工 scale 的语义与项目现有 RICH manual-scale 流程一致：

```text
adjusted_scene_depth = cached_scene_depth * manual_scale
predicted SMPL vertices 不缩放
```

因此 SHOW 的 GT-SMPL 投影区域、预测人体网格和场景深度仍在同一相机坐标系中，只把场景 metric depth 做序列级校准。

## 1. 生成缓存

服务器项目路径：`/home/zhw/lab_users/xyb/home/projects/vggt-human`

```bash
CHECKPOINT=/path/to/stage2-checkpoint.pt \
SCALE_CHECKPOINT=/path/to/coarse-scale-checkpoint.pt \
SUPPORT_ROOT=/path/to/hmr4d_support \
FRAMES_ROOT=/path/to/3DPW/imageFiles \
DEVICE=cuda:0 \
WINDOW_SIZE=200 \
bash scripts/eval/prepare_3dpw_show_manual_scale_cache.sh
```

默认输出：`outputs/eval/show_3dpw_manual_scale/cache/`。

每条序列生成一个 pickle，保存前 200 帧的 scene depth、confidence mask、预测 SMPL、GT SMPL、内参和源帧编号；`manifest.json` 同时记录 `sampled_frame_count`、`original_frame_count`、checkpoint 和 metric 配置。

## 2. Viser 人工调 scale

```bash
CACHE_DIR=outputs/eval/show_3dpw_manual_scale/cache \
PORT=8080 \
bash scripts/vis/serve_3dpw_show_manual_scale_viewer.sh
```

在页面中逐序列选择采样帧观察，调节 `Manual Scale (log10)`，点击 `Save Sequence Scale`。scale 会保存到 `cache/manual_scales.json`，每条序列只有一个 scale。

## 3. 重评测

```bash
CACHE_DIR=outputs/eval/show_3dpw_manual_scale/cache \
SCALE_MODE=manual \
bash scripts/eval/evaluate_3dpw_show_manual_scale.sh
```

结果在 `outputs/eval/show_3dpw_manual_scale/metrics/`：

- `summary.json`：HS-V5/10、HS-CF5/10 和协议元数据；
- `per_sequence.csv`：每条序列的采样均值、原始长度、展开权重及其 scale；
- `per_frame.csv`：逐帧指标；
- `applied_scales.json`：实际采用的 scale 映射。

正式 manual 评测要求所有序列都有保存的 scale。若只想检查未校准基线，可使用 `SCALE_MODE=base`；若只做中途诊断，可显式设置 `ALLOW_MISSING_MANUAL_SCALES=true`，缺失序列按 1.0 处理。

