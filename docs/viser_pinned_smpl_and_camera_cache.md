# Viser 固定 SMPL 帧与相机缓存

## 目标

在不改变模型推理、点云累积、SMPL 解码和已有编辑行为的前提下，为 `SequenceViewer` 增加 Hybrid 固定 SMPL 帧功能，并明确保存缓存中的相机参数。

未使用论文或第三方参考实现；这是对现有 Viser UI 和本项目缓存格式的适配扩展。

## Hybrid 固定 SMPL

Viser 新增默认折叠的 `Pinned SMPL Frames` 文件夹。长序列按每 50 帧一个子文件夹组织，每帧提供一个复选框，标签包括 Viewer 帧序号、原始 frame ID 和该帧人物数。

Hybrid 模式行为：

- 环境点云仍随 timestep 累积。
- 当前 timestep 的 SMPL 仍动态显示。
- 勾选帧的 SMPL 始终叠加显示，不受当前 timestep 影响。
- `Max People Per Frame`、Base/HSI SMPL 开关、TRSTR 开关、颜色、透明度和 Track ID 同样作用于固定帧。
- `Pin Current Frame` 固定当前帧；`Clear Pinned Frames` 清空选择。

固定帧使用原有 mesh handle。点击固定 SMPL 后仍进入 `Selected SMPL`，并同步到该 timestep；位移滑条、transform controls、`selected frame`/`same track all frames` scope 以及 `Save SMPL Offsets` 都沿用原实现。编辑后的 handle 保持固定显示。

固定帧选择属于当前 Viewer 运行状态，不写回模型结果，也不写入 cache。重新启动 Viewer 后复选框恢复为未选择。

## 相机缓存

完整 `full_viewer_cache` 原本已经保存：

- 每帧 `intrinsic`: `float32 [3,3]`。
- 每帧 `raw_extrinsic` 与 `hsi_extrinsic`: `float32 [3,4]`，world-to-camera。
- 每帧 `raw_camera`、`hsi_camera`、`camera`：包含 `rotation_c2w [3,3]`、`position [3]`、`fov`、`aspect`。
- scene metadata 中的 `camera_trajectory_raw`、`camera_trajectory_hsi` 和兼容别名 `camera_trajectory`: `float32 [S,3]`。

本次在完整 cache manifest 中显式声明以上字段，便于检查。

轻量 `viewer_cache` 原本没有相机数据。本次新增：

- 每帧 NPZ 中保存 intrinsic、raw/HSI extrinsic、raw/HSI camera rotation/position/fov/aspect。
- cache 根目录保存 `camera_trajectory_raw.npy` 和 `camera_trajectory_hsi.npy`。
- manifest 的 `camera_parameters` 记录字段与轨迹文件名。

旧的完整缓存本身已经包含相机数据，可直接使用新 UI，无需重新推理。旧的轻量缓存没有新增字段；如果需要读取轻量 cache 的相机参数，需要重新导出轻量缓存。

## 验证

```bash
bash scripts/smoke/check_pinned_smpl_visibility.sh
bash scripts/smoke/check_sequence_viewer_cache.sh
bash scripts/smoke/check_full_sequence_viewer_cache.sh
```

Windows 本地仅验证了 Hybrid 可见性规则、缓存字段往返、Python/Shell 语法。Viser 折叠列表、mesh 点击和 transform gizmo 的实际交互需要在 Linux 服务器 Viewer 中确认。
