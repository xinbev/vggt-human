# Cached SMPL RGB 投影

这个工具不重新推理，直接读取已有 `viewer_cache` 或 `full_viewer_cache`，把缓存中的 SMPL 网格投影到对应原图。它使用缓存里的 `intrinsic` 和 `raw_extrinsic`/`hsi_extrinsic`；`*_vertices_cam` 与同名相机匹配时直接投影，其他组合则先用 `R_w2c X_world + t_w2c` 转回相机坐标。

## 本地与服务器入口

本地脚本：

`C:\Users\ROG\PycharmProjects\vggt-omega\scripts\vis\project_cached_smpl_on_rgb.sh`

同步到服务器后：

`/home/zhw/lab_users/xyb/home/projects/vggt-human/scripts/vis/project_cached_smpl_on_rgb.sh`

服务器上运行示例：

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
CACHE_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/stage2_walking_coarse_residual_v3/full_viewer_cache \
OUTPUT_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/cached_smpl_projection \
FRAME_INDEX=0 \
MESH_SOURCE=hsi \
CAMERA_SOURCE=hsi \
bash scripts/vis/project_cached_smpl_on_rgb.sh
```

如果 cache 中保存的是另一台机器的绝对图片路径，使用 `IMAGE_ROOT` 将图片文件名映射到服务器目录：

```bash
IMAGE_ROOT=/home/zhw/xyb_space/3DPW/imageFiles/downtown_walkBridge_01 \
RENDER_ALL=true \
bash scripts/vis/project_cached_smpl_on_rgb.sh
```

也可以直接按 `frame_id` 选择一帧：

```bash
FRAME_ID=image_00042 bash scripts/vis/project_cached_smpl_on_rgb.sh
```

输出写入 `outputs/vis/cached_smpl_projection/`（或 `OUTPUT_DIR` 指定的目录）：

- `*_smpl_overlay.png`：原图上的平滑光照 SMPL 表面；默认使用 2 倍超采样抗锯齿，不画线框；
- `*_smpl_overlay.json`：原图尺寸、处理后尺寸、映射后的内参、外参、坐标来源和绘制统计；
- `projection_summary.json`：本次选择与输出汇总。

## 尺寸与坐标注意事项

推理输入可能经过长宽比 crop、resize 和 batch padding。脚本默认按项目的 `balanced/512/patch=16` 复原这段几何，并将缓存内参映射到原图像素坐标。如果生成 cache 时使用了其他设置，需同步设置 `RESIZE_MODE`、`IMAGE_RESOLUTION` 和 `PATCH_SIZE`，否则网格会出现整体缩放或偏移。

`MESH_SOURCE=hsi`、`CAMERA_SOURCE=hsi` 是当前 HSI viewer cache 的默认组合；若要查看 baseline 几何，可改为 `MESH_SOURCE=base CAMERA_SOURCE=raw`。`RENDER_SCALE=2` 使用超采样平滑轮廓；CPU 时间紧张时可改成 `RENDER_SCALE=1`。如果需要检查三角拓扑，可设置 `WIREFRAME=true`，但论文/展示图建议保持默认关闭。`FACE_STRIDE=1` 保留完整网格，CPU 时间紧张时可增大为 2 或 4。

由于 Windows 本地没有服务器 ckpt、依赖和完整图片树，本次只做了静态语法检查与已有缓存格式检查；正式投影请在 Linux 服务器执行上述 `.sh` 脚本。 
