# 3DPW / EMDB-1 SHOW 人工 scale 评测流程

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

3DPW support 的一条记录代表原视频中的一个 GT 人物轨迹，例如 `flat_guitar_01_0` 和 `flat_guitar_01_1` 是同一视频中的两个不同目标。缓存推理保留默认 8 个检测候选，每帧使用该记录的预处理 GT bbox 与候选 bbox 做最大 IoU 匹配；候选 query 编号可以变化，但目标人物不能变化。默认要求 IoU 至少为 0.30，低于阈值的帧标记为关联无效，不参与指标汇总。Viser 的 Frame Status 会显示 `GT-box IoU` 和 `match_valid`。

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

## EMDB-1：同一套前 200 帧协议

EMDB-1 使用完全相同的缓存字段、GT bbox 人物匹配、人工 scale 语义和按原始长度 `N` 加权方式。区别仅为数据集与输出目录。

生成缓存：

```bash
CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt \
SCALE_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt \
SUPPORT_ROOT=/home/zhw/xyb_space/emdb/hmr4d_support \
FRAMES_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/preprocess/hmr4d_eval_frames \
WINDOW_SIZE=200 \
MAX_HUMANS=8 \
TARGET_MIN_IOU=0.30 \
DEVICE=cuda:0 \
bash scripts/eval/prepare_emdb1_show_manual_scale_cache.sh
```

启动 Viser：

```bash
CACHE_DIR=outputs/eval/show_emdb1_manual_scale/cache \
PORT=8080 \
bash scripts/vis/serve_emdb1_show_manual_scale_viewer.sh
```

保存完 17 条序列的 scale 后评测：

```bash
CACHE_DIR=outputs/eval/show_emdb1_manual_scale/cache \
SCALE_MODE=manual \
DEVICE=cuda:0 \
bash scripts/eval/evaluate_emdb1_show_manual_scale.sh
```

结果写入 `outputs/eval/show_emdb1_manual_scale/metrics/`。

