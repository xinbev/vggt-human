# NLF-HSI Depth Flattening Debug

## 目的

当 viser 中看到 HSI 输出把 depth 拉成近似平面时，不直接判断是训练坏了，而是拆成以下链路逐项检查：

1. VGGT raw depth 是否本身有结构。
2. HSI affine depth 是否因为 `hsi_scene_scale` 太小被压平。
3. GT SMPL + dataset K 投影到 GT depth 是否闭合。
4. GT SMPL + VGGT K 投影到 GT depth 是否错位。
5. NLF base / HSI refined SMPL 投影到 raw / HSI / GT depth 后采样是否合理。
6. human ROI 内 robust affine raw->GT 的 scale/bias 和 HSI 预测的 scale/bias 是否差太多。

## 脚本

本地路径：

`scripts/diagnostics/debug_nlf_hsi_depth_flattening.py`

服务器路径：

`/home/zhw/lab_users/xyb/home/projects/vggt-human/scripts/diagnostics/debug_nlf_hsi_depth_flattening.py`

包装脚本：

`scripts/diagnostics/debug_nlf_hsi_depth_flattening.sh`

默认输出：

`outputs/debug/nlf_hsi_depth_flattening/`

## 关键输出

每个样本会生成：

- `processed_rgb.png`：模型实际吃到的 processed image。
- `depth_gt_raw_hsi_errors.png`：GT depth、VGGT raw depth、HSI depth、raw error、HSI error。
- `human_roi_mask.png`：depth teacher 当前关注的人体 ROI。
- `depth_centerline_profiles.png`：GT/raw/HSI 的中心行深度曲线，快速判断 HSI 是否被压平。
- `projection_overlay_rgb.png`：RGB 上的投影检查。
- `projection_overlay_gt_depth.png`：GT depth 上的投影检查。
- `projection_depth_samples.csv`：逐点投影和 depth 采样表。
- `frame_summary.json`：单帧诊断结果和 alerts。
- `summary.json`：所有样本汇总。

投影颜色：

- green：GT SMPL 使用 dataset K 投影。
- yellow：GT SMPL 使用 VGGT predicted K 投影。
- cyan：NLF/base SMPL 使用 VGGT K 投影。
- magenta：HSI refined SMPL 使用 VGGT K 投影。

## 重点告警含义

- `hsi_scene_scale_is_too_small_depth_can_collapse_to_plane`
  - HSI 预测的 scene scale 太接近 0，`hsi_depth = raw_depth * scale + bias` 会把 raw depth 的结构压没。

- `hsi_depth_std_much_smaller_than_raw_depth_flattening_likely`
  - HSI depth 的方差远小于 raw depth，说明视觉上的平面化很可能是真的。

- `hsi_depth_gradient_much_smaller_than_raw_depth_flattening_likely`
  - HSI depth 的局部梯度远小于 raw depth，说明几何细节被压平。

- `predicted_hsi_scale_far_from_roi_robust_affine_fit`
  - human ROI 内用 raw depth 拟合 GT depth 得到的合理 scale/bias，和 HSI 预测的 scale/bias 差很远。

- `gt_smpl_matches_depth_with_datasetK_but_not_vggtK_camera_mismatch_likely`
  - GT SMPL 用 dataset K 能贴住 GT depth，但用 VGGT K 贴不住，说明 HSI/NLF 使用的 VGGT 相机可能和 BEDLAM GT 相机存在明显偏差。

## 运行方式

默认检查 stage2 最新 checkpoint 的第一个训练窗口：

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

DATA_ROOT=/home/zhw/xyb_space \
BEDLAM_ROOT=/home/zhw/xyb_space/bedlam/processed_bedlam \
PREPROCESSED_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/preprocess/bedlam_boxes \
PIPELINE_OUTPUT_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_full_b12_20260710 \
CUDA_VISIBLE_DEVICES_VALUE=6 \
NUM_SAMPLES=1 \
bash scripts/diagnostics/debug_nlf_hsi_depth_flattening.sh
```

指定 viser 中出问题的图像：

```bash
IMAGE_PATH=/home/zhw/xyb_space/bedlam/processed_bedlam/Training/<sequence>/rgb/<frame>.png \
SMPL_CKPT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_full_b12_20260710/stage2_anchor_transl/checkpoint_latest.pt \
bash scripts/diagnostics/debug_nlf_hsi_depth_flattening.sh
```

检查更多样本：

```bash
START_INDEX=0 NUM_SAMPLES=8 bash scripts/diagnostics/debug_nlf_hsi_depth_flattening.sh
```
