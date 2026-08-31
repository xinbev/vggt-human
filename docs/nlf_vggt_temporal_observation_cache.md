# 可复用 RGB -> VGGT -> NLF 时序观测缓存

## 目标

本缓存只保存真实推理的原始观测：

```text
RGB 9-frame context
  -> VGGT runtime camera
  -> NLF internal detector（仅中心帧）
  -> BaseSMPL track assignment
  -> cache
```

缓存不含 V2 pose stabilizer、HSI、TRSTR、GT SMPL 或任何评测数字。因此，重训/更换 V2 checkpoint 后可直接复用缓存，避免重新跑慢速 VGGT/NLF 推理。

## 文件内容

每个唯一原始视频保存一个 `.pt`：

```text
pred_pose_6d       [T,Q,144]
pred_betas         [T,Q,10]
pred_transl_cam    [T,Q,3]
pred_confs         [T,Q,1]
pred_boxes         [T,Q,4]
assigned_track_*   BaseSMPL track assignment
center_valid       哪些 support frame 具有 9-frame camera context
frame_id           support 到源 RGB 的帧号映射
```

缓存以原始 RGB 视频 `vname` 去重。例如 `downtown_arguing_00_0` 与 `_1` 共用一个 `downtown_arguing_00.pt`，因为它们是同一视频的两个 GT 人物轨迹。

## 3DPW test 构建

原始 RGB root 应为 `imageFiles` 的上层目录：

```bash
CUDA_VISIBLE_DEVICES_VALUE=7 \
DEVICE=cuda:0 \
DATASET=3dpw \
FRAMES_ROOT=/home/zhw/xyb_space/3DPW/imageFiles \
CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/vggt_omega_1b_512.pt \
CACHE_ROOT=outputs/preprocess/nlf_vggt_temporal_cache_v1 \
bash scripts/preprocess/cache_nlf_vggt_temporal_inputs.sh
```

首次先限定一个正式 test 视频验证：

```bash
CUDA_VISIBLE_DEVICES_VALUE=7 \
DEVICE=cuda:0 \
DATASET=3dpw \
FRAMES_ROOT=/home/zhw/xyb_space/3DPW/imageFiles \
SEQUENCE_FILTER=downtown_arguing_00 \
CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/vggt_omega_1b_512.pt \
CACHE_ROOT=outputs/debug/nlf_vggt_temporal_cache_3dpw_smoke \
bash scripts/preprocess/cache_nlf_vggt_temporal_inputs.sh
```

`CUDA_VISIBLE_DEVICES_VALUE=7` 选择物理 GPU 7；该卡在进程内重新编号为 `cuda:0`，因此 `DEVICE` 必须保持 `cuda:0`。

## 可复用边界

可以复用的前提是以下内容不变：原始 RGB、VGGT camera checkpoint、NLF checkpoint、image resize/patch 配置、NLF detection 参数和 9-frame centre protocol。任意一项变化都应使用新的 `CACHE_ROOT`，不能覆盖旧缓存。

V2 稳定器 checkpoint 可以任意改变；它不会导致该缓存失效。
