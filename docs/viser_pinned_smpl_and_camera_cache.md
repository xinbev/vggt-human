# Viser 固定 SMPL 帧与相机缓存

## 目标

在不改变模型推理、点云累积、SMPL 解码和已有编辑行为的前提下，为 `SequenceViewer` 增加 Hybrid 固定 SMPL 帧功能，并明确保存缓存中的相机参数。

未使用论文或第三方参考实现；这是对现有 Viser UI 和本项目缓存格式的适配扩展。

## Hybrid 固定 SMPL

Viser 新增默认折叠的 `Pinned SMPL Frames` 文件夹。长序列按每 50 帧一个按钮组组织，帧按钮会根据 Viser 控制面板当前宽度自动换行成多列。按钮标签格式为 `0042/2`，分别表示 Viewer 帧序号和该帧人物数；已固定帧显示为 `*0042/2`，再次点击即可取消固定。

Hybrid 模式行为：

- 环境点云仍随 timestep 累积。
- 当前 timestep 的 SMPL 仍动态显示。
- 勾选帧的 SMPL 始终叠加显示，不受当前 timestep 影响。
- `Max People Per Frame`、Base/HSI SMPL 开关、TRSTR 开关、颜色、透明度和 Track ID 同样作用于固定帧。
- `Pin Current Frame` 固定当前帧；`Clear Pinned Frames` 清空选择。

固定帧使用原有 mesh handle。点击固定 SMPL 后仍进入 `Selected SMPL`，并同步到该 timestep；位移滑条、transform controls、`selected frame`/`same track all frames` scope 以及 `Save Viewer Edits` 都沿用原实现。编辑后的 handle 保持固定显示。

固定帧选择属于当前 Viewer 运行状态，不写回模型结果，也不写入 cache。重新启动 Viewer 后所有帧按钮恢复为未固定状态。

## SMPL Track ID 重分配

`SMPL ID Reassignment` 用于修正 Viewer 中的错配 Track ID：

1. 左键点击 SMPL，或在 `Selected SMPL` 下拉框中选择目标。
2. 在 `Assign to Existing ID` 中选择序列里已经出现过的目标 ID。
3. 选择作用范围并点击 `Apply ID Reassignment`。

作用范围包括当前帧中的这个人、从当前帧开始的同 ID，以及全序列同 ID。一次修改会同步该人物同一帧的 Base/HSI mesh、Track ID 标签、选择列表和按 ID 区分的基础颜色。若目标 ID 在受影响帧已属于另一个人，操作会被拒绝，防止同一帧出现重复 ID。`Undo Last ID Reassignment` 按操作撤销，`Restore Original IDs` 恢复 cache 中的全部原始 ID。

ID 修改仅影响当前 Viewer 状态，不修改模型预测和 `full_viewer_cache`。`Save Viewer Edits` 会将仍然有效的修改写入 JSON 的 `id_reassignments` 字段；当前 Viewer 启动时不会自动重放。

## Track ID 颜色列表

`Track ID Colors` 面板会按序列中出现过的 Track ID 显示颜色列表。每一行的 `ID N` RGB 色块既用于查看该 ID 当前对应的颜色，也可以直接指定新颜色；修改后，该 ID 在所有帧中的 Base/HSI SMPL 会立即更新。执行 ID 重分配后，被分配的人会自动采用目标 ID 的颜色。

`Restore Initial ID Colors` 恢复 Viewer 启动时从 cache 读取的 ID 配色。顶部原有的 `SMPL Color` 仍可一次把所有 ID 设成同一种颜色，并会同步更新列表中的色块。Pin 右键重上色属于局部覆盖，不会被 ID 基础色改写；用 `Restore Pinned SMPL Colors` 清除覆盖后，SMPL 会重新跟随当前 ID 色。

自定义 ID 基础色属于 Viewer 状态，不写回 cache。`Save Viewer Edits` 会把与初始值不同的 ID 配色写入 JSON 的 `track_id_colors` 字段，用于审计或后续处理；当前 Viewer 启动时不会自动重放。

## 固定 SMPL 手动重上色

`Pinned SMPL Recolor` 面板提供独立于全局 `SMPL Color` 的手动上色：

1. 将 `Mode` 设为 `Hybrid`，并在 `Pinned SMPL Frames` 中固定需要检查的帧。
2. 在 `Recolor Target` 指定颜色，在 `Right-click Scope` 选择作用范围。
3. 打开 `Enable Right-click Recolor`，右键单击固定帧中的 SMPL mesh。
4. 完成后关闭该模式，mesh 恢复原有的左键选择和平移编辑。

作用范围包括单个 mesh、点击帧、固定帧中的同 track 同分支，以及全部固定帧。`Apply Color to All Pinned SMPL` 可一次批量应用颜色，`Restore Pinned SMPL Colors` 恢复固定 mesh 的基础颜色。手动颜色覆盖不会被后续全局 `SMPL Color` 改写；恢复后重新跟随全局颜色。

精确区分左/右键使用 Viser 1.1.0 的 click binding，因此服务器环境需要 `viser>=1.1.0`。旧版 Viser 会禁用右键模式并在状态栏提示，但批量应用与恢复按钮仍可使用。

## 点云橡皮擦

`Point Cloud Eraser` 是 viewer 内可见、可移动的三维球形笔刷。红色线框球表示实际删除范围，中心的三轴/平面 gizmo 用来精确移动：

1. 保持 `Environment Display` 为 `points` 或 `both`。
2. 打开 `Enable Eraser`，球形笔刷会出现在当前帧点云中心。
3. 左键单击点云可将笔刷快速吸附到该处；这一步只放置笔刷，不会删除。
4. 拖动 gizmo 的轴或平面精确调整位置，通过 `Eraser Radius (world units)` 调整球体大小。
5. 点击 `Erase Points Inside Brush` 执行删除。

`Eraser Scope=picked cloud` 只编辑最后吸附时命中的 frame/source；`all visible clouds` 会擦除球体覆盖的全部可见点云。`Center Brush on Current Frame` 可随时把笔刷找回当前帧中心。`Undo Last Erase` 按一次完整笔刷操作撤销，`Restore All Erased Points` 恢复本次 Viewer 会话删除的全部点。橡皮擦与点云测距互斥，启用其中一个会关闭另一个。

删除掩码保存在 source 的未缩放世界坐标中，因此调整 HSI visual scale、人体过滤或点采样后仍会重新应用。它只影响 points 显示，不修改环境 mesh，也不写回 `full_viewer_cache`。

`Save Viewer Edits` 继续写入 `SMPL_EDIT_OUTPUT`。JSON 保留兼容字段 `offsets`，并新增 `recolors`、`track_id_colors`、`id_reassignments` 与 `point_erase_strokes`；这些记录用于审计和后续处理，当前 Viewer 启动时不会自动重放。

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

Windows 本地仅验证了 Hybrid 可见性规则、ID 重分配 scope、重上色 scope、球形删除 mask、缓存字段往返、Python/Shell 语法。Viser 右键绑定、多列按钮组、动态下拉列表、点选射线和 transform gizmo 的实际交互需要在 Linux 服务器 Viewer 中确认。
