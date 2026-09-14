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

固定帧使用原有 mesh handle。点击固定 SMPL 后仍进入 `Selected SMPL`，并同步到该 timestep；位移滑条、transform controls、`selected frame`/`same track all frames` scope 以及 `Save Viewer Edits` 都沿用原实现。编辑后的 handle 保持固定显示。

固定帧选择属于当前 Viewer 运行状态，不写回模型结果，也不写入 cache。重新启动 Viewer 后复选框恢复为未选择。

## 固定 SMPL 手动重上色

`Pinned SMPL Recolor` 面板提供独立于全局 `SMPL Color` 的手动上色：

1. 将 `Mode` 设为 `Hybrid`，并在 `Pinned SMPL Frames` 中固定需要检查的帧。
2. 在 `Recolor Target` 指定颜色，在 `Right-click Scope` 选择作用范围。
3. 打开 `Enable Right-click Recolor`，右键单击固定帧中的 SMPL mesh。
4. 完成后关闭该模式，mesh 恢复原有的左键选择和平移编辑。

作用范围包括单个 mesh、点击帧、固定帧中的同 track 同分支，以及全部固定帧。`Apply Color to All Pinned SMPL` 可一次批量应用颜色，`Restore Pinned SMPL Colors` 恢复固定 mesh 的基础颜色。手动颜色覆盖不会被后续全局 `SMPL Color` 改写；恢复后重新跟随全局颜色。

精确区分左/右键使用 Viser 1.1.0 的 click binding，因此服务器环境需要 `viser>=1.1.0`。旧版 Viser 会禁用右键模式并在状态栏提示，但批量应用与恢复按钮仍可使用。

## 点云橡皮擦

`Point Cloud Eraser` 是 viewer 内的三维球形笔刷：

1. 保持 `Environment Display` 为 `points` 或 `both`。
2. 打开 `Enable Eraser`，设置 `Eraser Radius (world units)`。
3. 左键单击可见点云，删除射线命中点周围球形范围内的点。

每次点击只编辑命中的一个 frame/source 点云。`Undo Last Erase` 撤销最后一次操作，`Restore All Erased Points` 恢复本次 Viewer 会话删除的全部点。橡皮擦与点云测距互斥，启用其中一个会关闭另一个。

删除掩码保存在 source 的未缩放世界坐标中，因此调整 HSI visual scale、人体过滤或点采样后仍会重新应用。它只影响 points 显示，不修改环境 mesh，也不写回 `full_viewer_cache`。

`Save Viewer Edits` 继续写入 `SMPL_EDIT_OUTPUT`。JSON 保留兼容字段 `offsets`，并新增 `recolors` 与 `point_erase_strokes`；这些记录用于审计和后续处理，当前 Viewer 启动时不会自动重放。

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

## Viewer 启动时的点大小与人体过滤

人体过滤在第一阶段构建 scene 时已经执行，完整缓存同时保存过滤前/后的点云、depth、RGB、相机参数、人体 mask 和 SMPL。第二阶段默认直接读取缓存结果。

完整缓存 Viewer 现在支持启动时覆盖：

```bash
CACHE_DIR=/path/to/full_viewer_cache \
POINT_SIZE=0.006 \
HUMAN_MASK_DILATION_PX=12 \
FILTER_HUMAN_POINTS=true \
PORT=8080 \
bash scripts/vis/serve_full_sequence_viewer_cache.sh
```

- `POINT_SIZE`：Viser 渲染点大小，范围 `[0.0005,0.08]`，只影响显示。
- `HUMAN_MASK_DILATION_PX`：SMPL 投影轮廓向外扩张的像素数，范围 `[0,32]`。
- `FILTER_HUMAN_POINTS=true/false`：启动时默认显示过滤后或完整点云。

如果启动命令覆盖 `HUMAN_MASK_DILATION_PX`，Viewer 会在启动阶段利用完整缓存内的 depth、RGB、intrinsic/extrinsic 和 SMPL 重新计算 mask 与过滤点云，不重新执行 VGGT/NLF/HSI。500 帧缓存可能需要一些 CPU 处理时间，终端会每 25 帧打印进度。

第一阶段推理/缓存导出脚本也已暴露同名参数。若不设置，继续使用 `POINT_SIZE=0.006`、`HUMAN_MASK_DILATION_PX=12` 和 `FILTER_HUMAN_POINTS=true`。

## 验证

```bash
bash scripts/smoke/check_pinned_smpl_visibility.sh
bash scripts/smoke/check_viewer_edit_tools.sh
bash scripts/smoke/check_sequence_viewer_cache.sh
bash scripts/smoke/check_full_sequence_viewer_cache.sh
```

Windows 本地仅验证了 Hybrid 可见性规则、重上色 scope、球形删除 mask、缓存字段往返、Python/Shell 语法。Viser 右键绑定、折叠列表、点选射线和 transform gizmo 的实际交互需要在 Linux 服务器 Viewer 中确认。
