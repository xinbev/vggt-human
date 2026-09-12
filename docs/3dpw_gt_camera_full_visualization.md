# 3DPW 全序列 GT 相机辅助可视化

`downtown_walkBridge_01` 有 1371 帧，而单次 VGGT/NLF 推理显存上限约为 550 帧。新增工作流把推理与可视化拆开：

1. `run_3dpw_gt_camera_full_sequence.sh` 按 500 帧窗口生成完整 viewer cache，默认不抽帧，所有 1371 张图都会进入推理。
2. `merge_3dpw_gt_camera_full_caches.py` 读取每段 cache，用原生 3DPW `cam_poses`（`T_w2c`）把预测的 camera-space 点云和 SMPL 放到统一 GT 世界坐标。GT 只用于展示坐标拼接，不替换 RGB 模型预测的人体、深度或尺度。
3. `serve_3dpw_gt_camera_full_sequence.sh` 只加载合并 cache，不再运行模型。

默认窗口之间无重叠；如果需要检查接缝，可设置 `CHUNK_OVERLAP=50`，合并时重复帧保留第一次出现的版本。

服务器执行：

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
FRAMES_DIR=/home/zhw/xyb_space/3DPW/imageFiles/downtown_walkBridge_01 \
GT_PKL=/home/zhw/xyb_space/3DPW/sequenceFiles/test/downtown_walkBridge_01.pkl \
OUTPUT_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/3dpw_downtown_walkBridge_01_gt_camera_full \
CUDA_VISIBLE_DEVICES_VALUE=0 \
TOTAL_FRAMES=1371 \
CHUNK_SIZE=500 \
CHUNK_OVERLAP=0 \
SERVE_VIEWER=false \
bash scripts/vis/run_3dpw_gt_camera_full_sequence.sh
```

该命令会依次处理 `[0,499]`、`[500,999]`、`[1000,1370]` 三段。每段都使用完整的连续帧，不做抽帧；`SMOKE_ONLY=true` 在这里表示“推理并写缓存后退出”，不是只跑少量帧。

推理完成后单独启动：

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
CACHE_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/3dpw_downtown_walkBridge_01_gt_camera_full/full_viewer_cache_gt_camera \
PORT=8090 \
VIEWER_MODE=Hybrid \
SMPL_DISPLAY_FRAMES=50 \
bash scripts/vis/serve_3dpw_gt_camera_full_sequence.sh
```

缓存服务阶段不会重新加载模型。Viewer 中的点云、SMPL 和相机都来自合并缓存；SMPL、深度和尺度仍是模型预测，GT 仅用于把各段放进同一个世界坐标系，因此该结果应标注为 `GT-camera oracle visualization`。

如果分段推理已经完成，也可以只执行合并阶段：

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
INPUT_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/3dpw_downtown_walkBridge_01_gt_camera_full \
GT_PKL=/home/zhw/xyb_space/3DPW/sequenceFiles/test/downtown_walkBridge_01.pkl \
OUTPUT_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/3dpw_downtown_walkBridge_01_gt_camera_full/full_viewer_cache_gt_camera \
TOTAL_FRAMES=1371 \
bash scripts/vis/merge_3dpw_gt_camera_full_caches.sh
```

输出目录：

```text
outputs/vis/3dpw_downtown_walkBridge_01_gt_camera_full/
  chunk_0000/full_viewer_cache/
  chunk_0001/full_viewer_cache/
  chunk_0002/full_viewer_cache/
  full_viewer_cache_gt_camera/
```

注意：这是 GT-camera oracle visualization。若要报告 RGB-only 结果，必须使用原始预测相机，不能使用本工作流的 GT 外参。
