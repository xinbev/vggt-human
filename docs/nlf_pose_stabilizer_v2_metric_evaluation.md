# RGB -> VGGT -> NLF -> Pose Temporal Stabilizer V2 指标评测

## 目标

该评测用于回答当前真实系统在 3DPW 和 EMDB-1 上的相机空间 SMPL 指标，而不是合成 GT 扰动上的 E0/E1 指标。

```text
RGB (9-frame window)
  -> VGGT runtime camera prediction
  -> NLF internal detector + metric SMPL
  -> model-side BaseSMPL track assignment
  -> PoseTemporalStabilizer V2 (optional pose-only refinement)
  -> 3DPW / EMDB-1 GT SMPL metrics
```

本评测**明确关闭** HSI、TRSTR、contact、grounding 和外部 sidecar。它对应用户当前指定的 `RGB -> VGGT -> NLF -> 时序稳定器` 链路，不能被解释为 HSI/人景对齐全系统指标。

## 主表口径

每个 9 帧窗口只评估中间第 5 帧，因此每个时序可用 frame 只会计一次；序列开头/结尾各 4 帧不属于时序中心帧，不能混入 temporal 结果。base 与 temporal 两行都在相同 centre frame 子集上统计。

主指标均以毫米输出：

| 指标 | 当前实现 |
| --- | --- |
| PA-MPJPE | 24 joints 的 Procrustes-aligned error |
| MPJPE | 24 joints 的 pelvis-aligned error |
| PVE | 6890 SMPL vertices 的 pelvis-aligned error |

JSON/CSV 额外报告 `cam_mpjpe_no_align_mm` 与 `cam_pve_no_align_mm`，用于诊断 NLF 全局 translation 误差。论文与其他项目做数值比较前，必须确认目标论文的 PVE 是否也使用 pelvis alignment；若对方使用 camera-space PVE，必须使用这里的 `*_cam_pve_no_align_mm` 或重新统一协议，不能混用。

## NLF 检测与时序 track

评测不将 GT pose、GT translation 或 GT track 输入 NLF 或稳定器。GT bbox 仅用于**评测匹配**：从当前中心帧的 NLF detections 中选择与目标人体 IoU 最大者；没有可用 GT bbox 时才退化到最高 NLF confidence。

V2 只接受所选中心检测的相同 `assigned_track_id`：

- 当邻帧中同一 ID 每帧恰好出现一次，应用 9 帧 pose stabilizer；
- track 不连续、ID 缺失/重复或边界不足时，严格输出 NLF base pose（temporal no-op）；
- summary 的 `coverage.temporal_applied_rate` 必须与主表一并汇报，避免把低覆盖的 temporal 数字误解为整段视频均被稳定。

## 运行

先做小样本接口运行：

```bash
CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/vggt_omega_1b_512.pt \
TEMPORAL_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/<v2_run>/checkpoint_best.pt \
DATASETS="3dpw emdb1" \
MAX_WINDOWS=20 \
OUT_ROOT=outputs/debug/nlf_pose_stabilizer_v2_smoke \
bash scripts/eval/evaluate_nlf_pose_stabilizer_v2.sh
```

确认每个 dataset JSON 的 `num_metric_rows > 0`、`coverage.temporal_applied_rate` 合理、无 unsupported label 后，去掉 `MAX_WINDOWS` 跑完整评测：

```bash
CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/vggt_omega_1b_512.pt \
TEMPORAL_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/<v2_run>/checkpoint_best.pt \
DATASETS="3dpw emdb1" \
OUT_ROOT=outputs/eval/nlf_pose_stabilizer_v2_<v2_run> \
bash scripts/eval/evaluate_nlf_pose_stabilizer_v2.sh
```

主要产物：

```text
outputs/eval/.../3dpw/3dpw_nlf_pose_stabilizer_v2_metrics.json
outputs/eval/.../emdb1/emdb1_nlf_pose_stabilizer_v2_metrics.json
outputs/eval/.../summary.md
outputs/eval/.../summary.json
```

`summary.md` 自动生成 NLF base 与 NLF+V2 temporal 两行、六个主表数值。它们只能代表实际计入的 centre frames，不应描述成“完整视频所有帧”的结果。

## 原始 3DPW `imageFiles` 直接输入

无需先抽帧时，可把 `FRAMES_ROOT` 指向**上层**原始图像目录：

```text
/home/zhw/xyb_space/3DPW/imageFiles
```

不要把它设为单独的 `.../imageFiles/courtyard_arguing_00`，因为评测器会从 HMR4D support label 的 `vname` 自动拼接 sequence 目录。先对指定序列做 smoke：

```bash
CUDA_VISIBLE_DEVICES_VALUE=7 \
DEVICE=cuda:0 \
FRAMES_ROOT=/home/zhw/xyb_space/3DPW/imageFiles \
SEQUENCE_FILTER=courtyard_arguing_00 \
DATASETS=3dpw \
MAX_WINDOWS=20 \
CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/vggt_omega_1b_512.pt \
TEMPORAL_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/<v2_run>/checkpoint_best.pt \
OUT_ROOT=outputs/debug/nlf_pose_stabilizer_v2_3dpw_courtyard_smoke \
bash scripts/eval/evaluate_nlf_pose_stabilizer_v2.sh
```

`CUDA_VISIBLE_DEVICES_VALUE=7` 选择物理 GPU 7；该环境变量会把它映射为进程内唯一可见设备，因此 `DEVICE` 必须为 `cuda:0`，不要设置为 `cuda:7`。EMDB-1 不能使用 3DPW 的 imageFiles root；它需要独立的 EMDB RGB frame root 或 HMR4D 抽帧目录。

## 当前风险

- V2 是合成 GT 扰动训练，尚未使用真实 NLF error cache 微调；真实指标可能无改善或退化。
- 本模块当前只修 pose，translation/betas 不变；MPJPE/PVE 的变化应主要来自局部姿态，未对齐诊断指标通常不会显著改善。
- 评测调用完整 VGGT+NLF，速度和显存开销高；必须先跑 `MAX_WINDOWS=20`。
- HMR4D support 的 EMDB-1 名单当前加载 17 个序列；它与某些论文表中标注的 24 个序列不自动等价，最终报告必须写入 JSON 实际覆盖数和 metric rows。
