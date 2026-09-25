# Cached SAM2 mask visualization

这个工具读取已有的 `full_viewer_cache` 或轻量 `viewer_cache`，不重新执行 VGGT、NLF 或 SMPL 推理。每一帧先将 cache 中的 SMPL 顶点投影回原始 RGB 图像，并把每个人的投影包围盒作为 SAM2 的 box prompt；最终的像素 mask 来自 SAM2。

## 服务器入口

本地脚本：

`C:\Users\ROG\PycharmProjects\vggt-omega\scripts\vis\visualize_cached_sam2_masks.sh`

同步到服务器后：

`/home/zhw/lab_users/xyb/home/projects/vggt-human/scripts/vis/visualize_cached_sam2_masks.sh`

单帧运行：

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
CACHE_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/my_sequence/full_viewer_cache \
IMAGE_ROOT=/home/zhw/xyb_space/3DPW/imageFiles/downtown_walkBridge_01 \
FRAME_INDEX=0 \
bash scripts/vis/visualize_cached_sam2_masks.sh
```

处理全部 cache 帧：

```bash
RENDER_ALL=true \
OUTPUT_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/my_sequence/cached_sam2_masks \
bash scripts/vis/visualize_cached_sam2_masks.sh
```

如果 SAM2 权重或目录不在 `configs/path.yaml` 约定的默认位置，可以设置 `SAM2_ROOT`、`SAM2_CHECKPOINT` 和 `SAM2_MODEL_CFG`。`BOX_EXPAND_RATIO` 默认是 `0.05`，表示在投影 SMPL 框四周扩展 5%；SAM2 的多个候选 mask 默认选择得分最高者。

## 输出

默认写入 `outputs/vis/cached_sam2_masks/`：

- `*_sam2_overlay.png`：SAM2 mask 叠加到原始 RGB 的可视化；
- `*_sam2_union_mask.png`：所有人物 mask 的联合二值 mask；
- `*_sam2_masks.npz`：按 `person_000000` 等 key 保存的单人 mask；
- `*_sam2.json`：投影内参映射、SAM2 分数、面积、track/query id 和 prompt 框；
- `sam2_mask_summary.json`：本次运行的汇总。

脚本不会改写输入 cache，也不会修改现有 SMPL 投影或 Viser viewer。`prompt_source` 和 `mask_source` 会写入元数据，便于后续对比 SMPL 投影 mask 与 SAM2 mask。

对完整 `full_viewer_cache`，可以在处理全部帧时额外导出一个可直接被现有 Viser viewer 加载的新 cache：

```bash
RENDER_ALL=true \
EXPORT_CACHE_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/my_sequence/full_viewer_cache_sam2 \
bash scripts/vis/visualize_cached_sam2_masks.sh

CACHE_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/my_sequence/full_viewer_cache_sam2 \
bash scripts/vis/serve_full_sequence_viewer_cache.sh
```

导出的 cache 保留原始完整点云、SMPL、相机和 manifest，只用 SAM2 联合 mask 重新生成 `raw_points`、`raw_colors`、`hsi_points`、`hsi_colors` 及对应 exclusion mask。轻量 `viewer_cache` 没有深度图和完整点云，因此只支持 mask/overlay 导出，不支持 cache 重写。

## 已验证与限制

本地完成 Python 语法检查、cache 格式兼容性检查和脚本参数/路径静态检查。Windows 本地没有 SAM2 checkpoint 和完整 CUDA 环境，因此需要在 Linux 服务器执行上述 `.sh`；首次运行的剩余风险是服务器上的 SAM2 配置路径、权重版本和源图像目录是否与 cache 对应。
