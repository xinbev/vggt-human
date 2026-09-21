# RICH 数据集与评测交接说明

本文档用于将当前 RICH 数据准备、物理接地评测和人工 scale 实验交接给新的会话。实际运行环境是 Linux 服务器，不是 Windows 本地环境。

## 1. 项目路径

Windows 本地仓库：

```text
C:\Users\ROG\PycharmProjects\vggt-omega
```

Linux 服务器仓库：

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human
```

服务器 Conda 环境：`zhw_env`。服务器运行都应该通过仓库内的 `.sh` 脚本完成。

## 2. RICH 官方数据位置

RICH 官方数据根目录：

```text
/home/zhw/xyb_space/RICH/official
```

主要目录：

```text
/home/zhw/xyb_space/RICH/official/test
/home/zhw/xyb_space/RICH/official/scan_calibration
/home/zhw/xyb_space/RICH/official/multicam2world
```

压缩包目录：`/home/zhw/xyb_space/RICH/official_downloads`

已确认下载完成的有效压缩包：

```text
rich_test_jpg_images.tar.gz       # 约 226G
rich_scan_calibration.zip        # 约 863M
rich_multicam2world.zip
```

解压后的目录结构：

```text
/home/zhw/xyb_space/RICH/official/test/<recording>/cam_XX/*.jpg
/home/zhw/xyb_space/RICH/official/scan_calibration/<scene>/calibration/*.xml
/home/zhw/xyb_space/RICH/official/scan_calibration/<scene>/*.ply
/home/zhw/xyb_space/RICH/official/multicam2world/*.json
```

`.invalid.*` 文件是下载失败时保存的 HTML 登录页面备份，不是有效数据。

## 3. HMR4D 辅助文件

```text
/home/zhw/xyb_space/RICH/hmr4d_support/rich_test_labels.pt
/home/zhw/xyb_space/RICH/hmr4d_support/rich_test_preproc.pt
/home/zhw/xyb_space/RICH/hmr4d_support/cam2params.pt
```

前两个文件用于旧的 HMR4D 静态相机视角协议。当前移动相机 `cam_10` 不在 HMR4D 标签中，因此 test-9 移动相机协议不使用 `bbx_xys`。

## 4. RICH moving-camera 划分

RICH metadata 中 `moving_cam=V` 的数量为：train 22 条、val 9 条、test 9 条，总计 40 条。

当前项目只评测已经准备好的 **test 9 条**，不是完整 40 条，也不是 UniCon3R Table 3 的完整官方协议。

Manifest：

```text
configs/eval/rich_test_moving_camera_9.txt
```

当前 test-9 序列：

```text
test/ParkingLot2_017_burpeejump2/cam_10
test/ParkingLot2_017_burpeejump1/cam_10
test/ParkingLot2_017_overfence1/cam_10
test/ParkingLot2_017_overfence2/cam_10
test/ParkingLot2_017_eating1/cam_10
test/ParkingLot2_017_pushup2/cam_10
test/Gym_011_cooking1/cam_10
test/Gym_011_cooking2/cam_10
test/Gym_012_cooking2/cam_10
```

不要假设每条 recording 都有 `cam_00`。当前协议明确使用 `cam_10`。

## 5. 数据接口与协议

主要数据接口：`vggt_omega/data/rich_physical_grounding.py`

```text
rich_test_moving_camera_9   # 当前默认协议
hmr4d_test_camera_views     # 保留的旧静态相机协议
```

移动 test-9 协议读取每条 `cam_10` 的所有 JPG；不使用 HMR4D 帧索引或 `bbx_xys`；每帧选择 NLF 置信度最高的有效人体；按非重叠 100 帧窗口处理；最后不足 100 帧的窗口也保留；每个 100 帧窗口只使用一个共享 scale。当前 9 条序列合计约 78 个窗口，具体以缓存 manifest 为准。

## 6. 当前 checkpoint 与配置

```text
训练配置:
/home/zhw/lab_users/xyb/home/projects/vggt-human/configs/train_smpl_hsi_nlf_stage2_human_scene_align.yaml

Stage-2 checkpoint:
/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt

Coarse residual scale checkpoint:
/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt

VGGT baseline:
/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/vggt_omega_1b_512.pt

NLF TorchScript:
/home/zhw/lab_users/xyb/home/projects/vggt-human/third_party/weights/nlf/nlf_l_multi_0.3.2.torchscript

SMPL 模型目录:
/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/body_models/smpl
```

当前 accepted cascade：`stage2 align input_dim=25`、`legacy_scale_bias_v0`、`hsi_scene_affine_mode=per_frame`、`smpl_provider=nlf`、`nlf_use_detector=true`、`num_smpl_queries=8`。

## 7. 正式物理 grounding 评测

入口：`scripts/eval/evaluate_rich_physical_grounding.sh`

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
CUDA_VISIBLE_DEVICES_VALUE=0 \
MODE=full \
WINDOW_SIZE=100 \
MAX_FRAMES_PER_SEQUENCE=0 \
bash scripts/eval/evaluate_rich_physical_grounding.sh
```

输出目录：`outputs/eval/rich_physical_grounding_test_moving9_cam10/`

主要文件：`summary.json`、`summary.csv`、`per_sequence.csv`、`per_window.csv`、`per_frame.csv`、`sequences/*.json`。

## 8. 四个物理 grounding 指标

所有指标都越低越好：

```text
Collision Ratio (%)  有效帧中人体穿过地面的帧比例
Penetrate (cm)       穿地帧的平均穿透深度
Float (cm)           悬空帧中人体最低点到地面的平均距离
Penetration Max (cm) 最大穿透深度
```

接触容差为 `0.005m`。clearance 小于 `-0.005m` 计为 collision，大于等于 `0.005m` 计为 floating。

## 9. 已完成的正式 test-9 结果

序列平均：

```text
Collision Ratio : 6.48%
Penetrate       : 4.65 cm
Float           : 17.15 cm
Penetration Max : 7.31 cm
```

所有有效帧 pooled：

```text
valid_frames       : 7371
collision_frames   : 449
floating_frames    : 6806
neutral_frames     : 116
collision_ratio    : 6.09%
penetrate          : 6.77 cm
float              : 17.47 cm
penetration_max    : 20.98 cm
```

这些结果属于 `rich_test_moving_camera_9` reduced protocol，不能直接声称复现完整 Table 3。

## 10. 人工 scale 实验

相关文件：

```text
vggt_omega/evaluation/rich_scale_oracle.py
scripts/eval/prepare_rich_manual_scale_cache.sh
scripts/vis/serve_rich_manual_scale_viewer.sh
scripts/eval/evaluate_rich_manual_scale.sh
scripts/eval/reset_rich_manual_scale.sh
```

生成缓存：

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
CUDA_VISIBLE_DEVICES_VALUE=0 \
bash scripts/eval/prepare_rich_manual_scale_cache.sh
```

缓存目录：`outputs/eval/rich_manual_scale/cache/`

打开 Viser：

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
PORT=8080 bash scripts/vis/serve_rich_manual_scale_viewer.sh
```

每个 100 帧窗口只保存一个 scale。`Preview Window Metrics` 只在内存中计算；`Save Window Scale` 才会写入：`outputs/eval/rich_manual_scale/cache/manual_scales.json`。

人工评测：

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
SCALE_MODE=manual bash scripts/eval/evaluate_rich_manual_scale.sh
```

输出目录：`outputs/eval/rich_manual_scale/manual_metrics/`

当前人工 scale 结果：

```text
valid_frames       : 7158
invalid_frames     : 337
collision_ratio    : 11.76% pooled / 12.83% mean-of-sequences
penetrate          : 4.10 cm pooled / 2.89 cm mean-of-sequences
float              : 14.99 cm pooled / 13.47 cm mean-of-sequences
penetration_max    : 16.59 cm pooled / 7.99 cm mean-of-sequences
```

结论：人工 scale 减少了平均悬空和平均穿透深度，但增加了 collision ratio，不能同时优化四项指标。

## 11. 当前 oracle_grid 异常

运行命令：

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
SCALE_MODE=oracle_grid \
ORACLE_SCALE_MIN=0.25 \
ORACLE_SCALE_MAX=4.0 \
ORACLE_STEPS=41 \
bash scripts/eval/evaluate_rich_manual_scale.sh
```

当前输出显示 `selected_frames=7495`、`invalid_frames=7495`、`valid_frames=0`。这个全零结果不能解释为 scale 极限，也不能作为正式结果。

下一会话应优先检查：

1. `choose_oracle_scale()` 的候选选择和 invalid penalty。
2. 每个候选 scale 的 `valid_frames` 和 invalid reason。
3. `scale=1.0` 在同一缓存上的结果是否有效。
4. 自动搜索是否应始终把 `x1.0` 作为保底候选。
5. 找不到任何有效候选时应直接报错，而不是输出全零指标。

## 12. 重置人工 scale

只清除人工 scale 和评测结果，保留昂贵的推理缓存：

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
bash scripts/eval/reset_rich_manual_scale.sh
```

保留：`outputs/eval/rich_manual_scale/cache/windows/`、`outputs/eval/rich_manual_scale/cache/manifest.json`。

删除：`outputs/eval/rich_manual_scale/cache/manual_scales.json`、`manual_metrics/`、`oracle_grid_metrics/`。

## 13. 重要限制

1. 当前只评测 test-9 `cam_10`，不是完整 RICH moving-camera 40 条。
2. 当前结果不能直接声称复现 UniCon3R Table 3。
3. 当前指标使用模型重建的 metric depth 估计地面，不是直接使用 RICH reference scan 作为地面真值。
4. 人工 scale 是诊断性 upper-bound 实验，不是可部署模型结果。
5. 评测时人体区域从 scene support points 中排除；Viser 保留人体点云只是为了观察对齐效果。
6. 下一步评测其他指标时，必须记录 protocol、manifest、window size、checkpoint 和 aggregation 方式。

## 14. 下一会话建议

先运行 base 模式确认缓存评测仍有效：

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
SCALE_MODE=base bash scripts/eval/evaluate_rich_manual_scale.sh
```

确认 base 有有效帧后，再修改 oracle 搜索逻辑并重新运行 `oracle_grid`。不要引用当前全零 oracle 输出。
