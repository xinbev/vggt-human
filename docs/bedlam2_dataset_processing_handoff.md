# BEDLAM2 数据集处理交接说明

## 任务目标

将 BEDLAM2 场景 `20241213_1_250_rome_tracking` 处理为当前项目
`vggt_omega.data.BedlamDataset` 可直接读取的训练数据。保留现有 BEDLAM
baseline 和原始数据，不修改 `.paper/` 资料，不把原始 EXR 或大文件提交到 git。

最终要得到一个独立的 processed tree：

```text
<processed_root>/
  Training/
    20241213_1_250_rome_tracking_seq_000002/
      rgb/seq_000002_0000.png
      depth/seq_000002_0000.npy       # float32, HxW, 相机坐标 Z-depth，单位 m
      cam/seq_000002_0000.npz          # intrinsics (3,3), pose (4,4)
      smpl/seq_000002_0000.pkl         # list[person dict]
```

默认输出根目录：

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/preprocess/bedlam2_processed
```

## 已确认的输入数据

服务器项目目录：

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human
```

原始 RGB：

```text
/home/zhw/xyb_space/bedlam2/hf_raw/BEDLAM2/
  20241213_1_250_rome_tracking/png/seq_xxxxxx/*.png
```

原始 EXR：

```text
/home/zhw/xyb_space/bedlam2/hf_raw/BEDLAM2-depth/
  20241213_1_250_rome_tracking/exr_depth/seq_xxxxxx/*.exr
```

SMPL 6 FPS 标签：

```text
/home/zhw/xyb_space/bedlam2/bedlam_data/labels_smpl_6fps/
  20241213_1_250_rome_tracking.npz
```

标签包含 12,342 个 record，字段已确认包括：

```text
imgname, smpl_pose_cam, smpl_betas, smpl_trans_cam,
cam_int, cam_ext, gender
```

`imgname` 与 RGB/EXR 可按同一 stem 对应，例如：

```text
imgname:   seq_000002/seq_000002_0000.png
RGB:       .../BEDLAM2/<scene>/png/seq_000002/seq_000002_0000.png
EXR:       .../BEDLAM2-depth/<scene>/exr_depth/seq_000002/seq_000002_0000.exr
```

深度侧元数据仍保留于：

```text
.../BEDLAM2-depth/<scene>/ground_truth/meta_exr_depth_csv/
.../BEDLAM2-depth/<scene>/be_camera_animations_depth.json
```

## 当前 EXR 的实际检查结论

已执行输出报告（本地副本）：

```text
outputs/20241213_1_250_rome_tracking_exr_inspection.json
```

第一帧的 RGB 与 EXR 分辨率相同，均为 `720 x 1280`。EXR 有：

```text
RGBA                                  float16, HxWx4
ActorHitProxyMask00                   float32, HxWx4
ActorHitProxyMask01/02                float32, HxWx4
FinalImageMovieRenderQueue_WorldDepth float16, HxWx4
```

有效几何 payload 是 `FinalImageMovieRenderQueue_WorldDepth`，**但不能把
任意一个分量直接保存为 depth**。同一中心像素的值为：

```text
[191.125, 191.125, 104.375, 0.0]
```

三个非零分量的样本中位数约为：

```text
component 0: 485.5
component 1: 587.0
component 2:   3.67
```

它们不是重复 scalar-depth，极可能是 3D point/vector 的不同坐标分量。`RGBA`
是颜色，`ActorHitProxyMask*` 是 mask，均不能作为深度。

### 明确禁止

- 不要使用 `DEPTH_COMPONENT=0/1/2` 猜测性转换。
- 不要沿用旧 BEDLAM 脚本中的 `Depth` 通道选择和 `depth / 100`。
- 不要把 EXR 直接交给当前 loader；loader 只读 `.npy` 的 HxW float32。
- 不要因为 RGB 与 EXR 分辨率相同，就假定其相机坐标系已一致。
- 不要删除 raw RGB、EXR、标签 NPZ 或相机 metadata。

## 已存在的项目内实现

这些文件是为该任务新增/修改的项目本地适配，不依赖 `.paper/` import：

| 文件 | 作用 | 当前状态 |
| --- | --- | --- |
| `scripts/preprocess/prepare_bedlam2_scene.py` | 物化 RGB、depth NPY、cam NPZ、SMPL PKL | 仅适用于已经确定 scalar depth 的场景；目前不能正确处理 BEDLAM2 的 vector `WorldDepth` |
| `scripts/preprocess/prepare_bedlam2_scene.sh` | 上述物化脚本的服务器入口 | 同上 |
| `scripts/diagnostics/inspect_bedlam2_world_depth.py` | 用 `cam_int/cam_ext` 诊断 `WorldDepth[..., :3]` 的坐标含义 | 下一步必须执行 |
| `scripts/diagnostics/inspect_bedlam2_world_depth.sh` | 只读诊断入口 | 下一步必须执行 |
| `docs/bedlam2_depth_preprocessing.md` | 早期处理流程说明 | 以本交接文档的约束为准 |

本地已完成的验证仅包括 Python 语法检查和 synthetic reprojection 单元检查；
没有本地 ckpt/完整 EXR 运行环境，不能把它视为数据正确性验证。

## 下一步：先做坐标语义诊断（只读）

同步最新代码后，在服务器运行：

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
bash scripts/diagnostics/inspect_bedlam2_world_depth.sh
```

默认检查帧：

```text
seq_000002/seq_000002_0000.png
```

报告输出：

```text
outputs/debug/bedlam2_world_depth/
  20241213_1_250_rome_tracking_coordinate_check.json
```

诊断会将 EXR 前 3 个分量解释为以下候选：

1. 原始 vector 已是 camera XYZ；
2. UE XYZ 经 `UE -> OpenCV` 轴变换后已是 camera XYZ；
3. 以上两者为 world XYZ，再分别经 `cam_ext` 和 `inverse(cam_ext)` 转到 camera；
4. 对 world candidate 同时测试 raw 值为 m 与 cm (`1.0` / `0.01`)。

每个 candidate 用标签 `cam_int` 重投影到原像素坐标，并输出：

```text
positive_z_fraction
median_z
median_reprojection_error_px
p95_reprojection_error_px
```

### 诊断通过条件

不能只看 “误差最小”。可接受候选至少需要：

- `positive_z_fraction >= 0.99`
- `median_reprojection_error_px <= 2.0`
- 在至少 3 个不同 sequence/frame 上均满足上述条件
- 如果是 world coordinate，要确认其 camera transform 的方向和单位一致

如果没有候选通过，停止转换；检查 `meta_exr_depth_csv`、
`be_camera_animations_depth.json`，并根据其字段添加新的候选变换。不能绕过
此检查直接落盘。

## 得到正确几何解释后的实现要求

现有 `prepare_bedlam2_scene.py` 只能从一个 scalar channel 分量生成 NPY，
因此需进行最小改动，新增一个显式的 vector-depth 路径（推荐独立 helper）：

```python
world_or_camera_xyz = exr_world_depth[..., :3]
camera_xyz = convert_with_validated_convention(world_or_camera_xyz, cam_ext)
depth_m = camera_xyz[..., 2]
depth_m[~isfinite | (depth_m <= 0)] = 0
```

要求：

- 只保存 validated `camera_xyz[...,2]`，不能保存 X/Y 或向量范数。
- 记录实际采用的 channel、坐标变换、`cam_ext` 方向和 depth unit/scale 到
  summary JSON。
- 如 EXR raw vector 是厘米，必须在变换前/后一致地乘 `0.01`；若是米，乘 `1.0`。
- RGB 拷贝默认可 hardlink，跨文件系统失败时 copy fallback；raw 不可变。
- 使用 NPZ `cam_int` 作为 `cam/*.npz` 中 `intrinsics`，并将 `cam_ext` 作为 `pose`。
- 每个 `imgname` 的标签 record 要按 image grouping 生成一个 person-list PKL。
- 当前 loader 的历史字典键仍是 `smplx_*`；即使用 SMPL 标签也需写：

```text
smplx_root_pose = smpl_pose_cam[:3].reshape(1, 3)
smplx_body_pose = smpl_pose_cam[3:66].reshape(21, 3)
smplx_shape = smpl_betas[:10]
smplx_gender = gender
smplx_transl = smpl_trans_cam + cam_ext[:3, 3]
```

最后一项是项目现有 `gt_transl_cam` baseline 约定；不要在本任务中改成
`smpl_trans_cam` direct，除非单独完成坐标审计并记录原因。

## 分阶段执行方式

### 1. 坐标诊断

按前述命令执行，并人工审计 JSON。

### 2. 最小 smoke materialization

在 vector-depth 路径完成并被诊断证明后，先限制一段 sequence 的少量帧：

```bash
SCENE=20241213_1_250_rome_tracking \
SEQUENCE=seq_000002 \
MAX_FRAMES=8 \
DEPTH_SCALE=<validated scale> \
bash scripts/preprocess/prepare_bedlam2_scene.sh
```

这一步应同时检查：

- 每个 `depth/*.npy` 是 `(720, 1280)`、`float32`、有限、正值为主；
- 将 NPY 以对应 `cam/*.npz` 回投影后，可还原 EXR 已验证点的 pixel geometry；
- 深度的中位数有合理米制量级，和 `smplx_transl[:, 2]` / 场景关系不冲突；
- 读取 `BedlamDataset(require_depth=True)` 时不报错；
- SMPL 投影 box 与 RGB 人物大致对齐。

### 3. 全量 materialization

仅当 smoke 通过后执行：

```bash
SCENE=20241213_1_250_rome_tracking \
DEPTH_SCALE=<validated scale> \
bash scripts/preprocess/prepare_bedlam2_scene.sh
```

预期总帧/标签 person record 为 `12,342`。保存 summary：

```text
outputs/preprocess/bedlam2_processed/_preprocess_summaries/
  20241213_1_250_rome_tracking_bedlam2_summary.json
```

### 4. 下游 sidecar 与 loader 验证

```bash
BEDLAM_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/preprocess/bedlam2_processed \
  bash scripts/preprocess/prepare_bedlam_boxes.sh

BEDLAM_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/preprocess/bedlam2_processed \
  bash scripts/diagnostics/check_bedlam_full_system_data.sh
```

`check_bedlam_full_system_data.sh` 已支持 `BEDLAM_ROOT` 覆盖，不需要修改
`configs/path.yaml` 的 baseline 路径。

## 最终交付与必须汇报项

完成时应在 `docs/` 或 summary 中明确记录：

1. EXR channel 名称、采用的 vector/scalar 解释和每一步坐标变换；
2. EXR 原始单位、最终 `depth_scale_to_m`；
3. 至少 3 个 frame 的重投影误差和 positive-Z 比例；
4. 输出帧数、person record 数、缺失 RGB/EXR 数量；
5. 生成的 processed root、box sidecar root、summary 路径；
6. 已运行的服务器 `.sh` 脚本与 loader smoke result；
7. 未消除的风险，尤其 RGB/depth camera metadata 是否完全一致。

## 2026-08-25 当前服务器诊断结果

已在服务器执行只读诊断：

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
bash scripts/diagnostics/inspect_bedlam2_world_depth.sh
```

诊断帧：

```text
seq_000002/seq_000002_0000.png
```

实际 EXR channel：

```text
FinalImageMovieRenderQueue_WorldDepth
```

实际 payload：

```text
(720, 1280, 4), float16
```

其中前三个分量有数值，第四个分量全为 0。该 channel 不是可直接写入 loader 的标量 Z-depth。

已测试的解释包括：

```text
raw XYZ 作为 camera XYZ
Unreal XYZ -> OpenCV XYZ
world XYZ -> depth CSV camera world-to-camera
world XYZ -> inverse camera transform
标签 cam_ext 作为 world-to-camera 或 camera-to-world
raw scale = 1.0 和 0.01
```

没有候选满足交接要求：

```text
positive_z_fraction >= 0.99
median_reprojection_error_px <= 2.0
```

原始诊断中最优候选仍约为：

```text
positive_z_fraction: 1.0
median_z: 7.113 m (仅按 0.01 缩放后的候选)
median_reprojection_error_px: 424.5 px
p95_reprojection_error_px: 1230.8 px
```

使用项目已有的 `prepare_bedlam_raw_scene.py:get_frame_w2c`，基于 depth 侧
`meta_exr_depth_csv/seq_000002_camera.csv` 重建相机外参后，最优误差仍约为
380 px。使用 EXR header 中的真实相机位置和旋转进行常见 Unreal 轴变换也没有达到
可接受误差。

RGB 与 depth 的 camera CSV 数值几乎完全一致。例如首帧两侧都为：

```text
x=3544.677
y=1415.285
z=163.030
yaw=-14.5035
pitch=-17.8091
roll=0.3587
focal_length=15.487783
```

因此当前阻塞点不是 RGB/depth camera CSV 数值不一致，而是
`FinalImageMovieRenderQueue_WorldDepth` 的 vector 语义与当前相机变换/像素投影约定尚未确认。

EXR header 也确认包含同一帧相机信息：

```text
unreal/camera/curPos/{x,y,z}
unreal/camera/curRot/{yaw,pitch,roll}
unreal/camera/FinalImage/focalLength
unreal/camera/FinalImage/fov
```

当前不得执行以下转换：

```text
WorldDepth[..., 0] / 100
WorldDepth[..., 1] / 100
WorldDepth[..., 2] / 100
WorldDepth[..., :3] 直接当 camera XYZ
```

因此尚未生成：

```text
outputs/preprocess/bedlam2_processed/
```

也没有写入任何 `.npy` depth 或新的 processed tree。原始 RGB、EXR、SMPL 标签和 metadata
均保持不变。

当前已确认可用的 SMPL label 仍为：

```text
/home/zhw/xyb_space/bedlam2/bedlam_data/labels_smpl_6fps/20241213_1_250_rome_tracking.npz
```

后续官方信息已澄清：BEDLAM2 EXR depth sequences 为 1280x720、30fps，depth 是
`FinalImageMovieRenderQueue_WorldDepth.R` 中的 16-bit float 标量深度。此前把前三个分量
当作 vector/point 做重投影诊断是不适用的；正确读取方式是 channel
`FinalImageMovieRenderQueue_WorldDepth` 的 component 0。

已按官方说明执行只读 inspection：

```bash
EXR_CHANNEL=FinalImageMovieRenderQueue_WorldDepth DEPTH_COMPONENT=0 INSPECT_ONLY=true \
  bash scripts/preprocess/prepare_bedlam2_scene.sh
```

前三个抽样 label 帧均为 720x1280，component 0 统计如下：

```text
seq_000002_0000: min=172.0, median=485.5, max=2050.0
seq_000002_0005: min=171.875, median=488.5, max=2045.0
seq_000002_0010: min=171.5, median=489.75, max=2047.0
```

结合 Unreal/BEDLAM 常用厘米单位，当前 materialization 采用：

```text
EXR_CHANNEL=FinalImageMovieRenderQueue_WorldDepth
DEPTH_COMPONENT=0
DEPTH_SCALE=0.01
```

即保存 `WorldDepth.R * 0.01` 为 meter-scale `float32` HxW `.npy`。注意：camera ground truth
仍为 Unreal coordinates，类似 BEDLAM；processed `cam/*.npz` 当前仍沿用已有 adapter 的
NPZ `cam_int` 和 `cam_ext` 字段约定，后续如要严格使用 depth-side Unreal camera GT，需要单独做相机约定审计。
