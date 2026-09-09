# EMDB P3/28 Stage2 推理与完整 Viser 拆分

## 目标与 baseline

本拆分对应原始输入 `P3/28_outdoor_walk_lunges/images/`、最多前 200 帧、端口 8088 的 Stage2 coarse-scale + HSI cascade 可视化命令。

模型、权重、输入选帧、置信度、点云采样、人体过滤、坐标系以及 `SequenceViewer` UI 均不改变。唯一变化是把原来同一进程中的两个阶段拆开：

1. GPU 推理、完整场景构建、缓存写盘，然后退出。
2. 从缓存恢复同一份 scene，再交给原来的 `SequenceViewer`。

没有使用论文或第三方项目实现，也没有修改 VGGT forward、SMPL 解码或 HSI 几何。

## 第一步：处理并保存

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
CUDA_VISIBLE_DEVICES_VALUE=0 \
bash scripts/vis/export_stage2_emdb_p3_28_walk_lunges_full_cache.sh
```

输出保持原命令不变：

`outputs/vis/stage2_walking_coarse_residual_v3/`

完整缓存位于：

`outputs/vis/stage2_walking_coarse_residual_v3/full_viewer_cache/`

缓存包含原 viewer 使用的每帧 raw/HSI 点云、full/filtered 点云、depth map、RGB、人体 exclusion mask、相机、SMPL、track、alignment 和场景尺度信息。SMPL faces 单独保存一次并在加载时恢复。

## 第二步：只加载 Viser

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
PORT=8088 \
bash scripts/vis/serve_stage2_emdb_p3_28_walk_lunges_full_cache.sh
```

此阶段不加载 checkpoint、不调用 NLF、不执行 VGGT/HSI/SMPL forward。它使用缓存中记录的原始参数初始化相同的 `SequenceViewer`，所以 UI 和原命令一致。

## 验证与风险

轻量缓存往返检查：

```bash
bash scripts/smoke/check_full_sequence_viewer_cache.sh
```

完整缓存为了保持所有 UI 功能，会明显大于仅保存最终 HSI 点云/SMPL 的轻量缓存。缓存使用项目本地 pickle 文件，只应读取本项目自己生成、可信来源的缓存。
