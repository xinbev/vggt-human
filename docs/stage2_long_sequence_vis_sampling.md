# Stage2 长序列可视化抽样说明

## 目标

为 EMDB `P4/36_outdoor_long_walk` 这类长序列提供两层互不混淆的抽样：

1. 推理帧抽样：从约 2200 帧的完整候选范围内均匀选取最多 500 帧，避免只处理序列开头。
2. 展示帧抽样：保留全部 500 帧点云，在 Viser 的 `3D accumulate` 模式中均匀展示用户指定数量的 SMPL，例如 50 帧。

## Baseline 与改动边界

- `FRAME_SAMPLING=head` 仍是默认值，保持原来“按 `START_INDEX`/`FRAME_STRIDE` 后取前 `MAX_FRAMES` 帧”的行为。
- `SMPL_DISPLAY_FRAMES=0` 保留原 `SMPL Downsample` 步长控件，且默认步长为 1、展示全部已选帧的 SMPL。
- 当 `SMPL_DISPLAY_FRAMES>0` 时，UI 默认选择新增的 `target frame count` 模式；用户仍可切回 baseline 步长模式。
- 新的 SMPL 数量控件只改变 Viser mesh/track-label 的可见性，不改变推理、SMPL 解码、点云、相机或坐标。
- 未使用或移植外部论文/第三方项目代码。

## 数据与坐标约定

- 模型输入仍为单次 forward 的 `[1, S, 3, H, W]`，本任务默认 `S=500`。
- dtype/device 沿用现有 viewer：图像在 forward 前搬到所选 CUDA device，模型和 SMPL layer 均使用现有配置。
- 500 个已选帧仍在一次 VGGT forward 中处理，以维持同一个 VGGT camera/world frame。
- UI 的 `Accumulated SMPL Frames=N` 使用包含首尾帧的确定性均匀索引；当 `N=50, S=500` 时最终累计视图恰好显示 50 个采样时刻的 SMPL。
- `4D current frame` 与 `Hybrid` 继续显示当前帧人物；数量限制只在 `3D accumulate` 生效。
- 控件限制的是采样时刻数量；如果某个采样帧没有有效人体检测，该时刻不会产生 SMPL mesh，实际 mesh 数可能少于目标值。

## 启动与输出

服务器项目路径：`/home/zhw/lab_users/xyb/home/projects/vggt-human`

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
bash scripts/vis/serve_stage2_emdb_p4_36_long_walk_uniform.sh
```

默认输入：`/home/zhw/xyb_space/emdb/P4/36_outdoor_long_walk/images`

默认输出：`outputs/vis/stage2_emdb_p4_36_long_walk_coarse_residual_v3/`

轻量抽样检查：

```bash
bash scripts/smoke/check_sequence_viewer_sampling.sh
```

## 尚需服务器验证的风险

- 500 帧仍需一次性进入 VGGT forward；显存峰值还受图像分辨率、检测人数和模型配置影响。
- 均匀抽样后相邻输入大约跨越 4--5 个原始视频帧，时序模块和跟踪器会把这些采样帧当作相邻时刻；论文结果中应明确该时间采样策略。
- Windows 本地没有 checkpoint/GPU，只能进行语法、抽样逻辑和 shell 静态检查；完整点云/SMPL 对齐需在服务器确认。
