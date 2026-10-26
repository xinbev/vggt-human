# 全序列 viewer 场景导出（逐帧独立文件）

## 用法

用 `scripts/vis/serve_full_sequence_viewer_cache.sh` 启动原有缓存 viewer，调好场景后点击 **Sequence Export → Export All Frames**。按钮会暂停播放，在后台遍历缓存中的全部帧。默认创建 `outputs/vis/sequence_scene_exports/sequence_时间戳/`，其下每帧一个 `frame_XXXX/` 目录和一份总 `manifest.json`。每次点击创建新目录，不覆盖旧结果。可用 `SEQUENCE_EXPORT_DIR` 调整 `outputs/` 内的导出子目录。

应在完成调整的同一次 viewer 会话中点击导出，并在状态显示完成之前保持编辑设置不变。现有 `SMPL_EDIT_OUTPUT` / “Save Viewer Edits” 只写编辑记录，启动 viewer 时不会自动回放该 JSON。

启动示例：

```bash
CACHE_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/video/P0_07_outdoor_push_ups/outputs/full_viewer_cache \
SMPL_EDIT_OUTPUT=/home/zhw/lab_users/xyb/home/projects/vggt-human/video/P0_07_outdoor_push_ups/outputs/smpl_edit_offsets.json \
POINT_SIZE=0.006 FILTER_HUMAN_POINTS=true PORT=8087 \
bash scripts/vis/serve_full_sequence_viewer_cache.sh
```

## 文件与约定

- 每个 `frame_XXXX/` 都有独立的 `manifest.json`、几何文件和相机文件，可以单独复制到其他项目使用；序列根目录的 `manifest.json` 给出全部帧的索引。
- `environment_raw.ply` / `environment_hsi.ply`：当前 Depth Source 设置应用于每帧后的 RGB 点云。过滤、擦除和 HSI Visual Scale 已烘焙。
- `environment_*_mesh.ply`：当 Environment Display 为 `mesh` 或 `both` 时额外导出带顶点色的三角网格，与 viewer 的网格路径一致。viewer 的网格路径本身不应用点云擦除和人物点过滤；导出的点云与网格也保持这一差异。
- `person_*.obj` + 同名 `.mtl`：当前可见的人物分支，已烘焙 SMPL Visual Scale、位移、HSI 相机位置补偿和颜色。重新指定的 ID 反映在文件名和 manifest 中。若开启 GT SMPL，也会导出。
- `cameras.json`：每帧原始 raw 与调整后 HSI 相机的 3×3 内参、4×4 camera-to-world/world-to-camera 位姿、FOV。若 Viser 点击事件带客户端信息，点击时的浏览视角记录在序列根目录 `manifest.json` 的 `viewport_at_export`。
- `manifest.json`：帧索引、原图路径、viewer 设置和文件清单。`point_size` 只是 viewer 显示参数，PLY 本身没有点半径。

PLY、OBJ 与相机使用同一 viewer 世界坐标。相机局部 +Z 朝前、+X 向右、+Y 向下，4×4 矩阵按列向量使用。HSI Visual Scale 是 viewer 展示变换，导出的位姿与几何已一起变换；世界单位不保证为米。每个子目录只含对应的一帧，积累模式也不会把其他帧混入该帧。OBJ/PLY 可直接导入 Blender、Open3D、MeshLab 等；其他程序按 `cameras.json` 读取内参与位姿。

这仅扩展可视化导出，不修改 VGGT/SMPL 预测、缓存或 baseline。未使用 `.paper/` 参考代码。服务器上可运行 `bash scripts/smoke/check_sequence_scene_export.sh` 检查文件格式和全序列索引；仍需用真实缓存人工确认界面按钮及目标软件的坐标方向。
