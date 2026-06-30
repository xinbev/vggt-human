# Full-System Dataset Pipeline

## Goal

把训练和评测数据统一成当前系统需要的四类输入：

- RGB frames: 模型主输入。
- detection query sidecars: 每帧 YOLO person bbox，驱动单帧 human query。
- SAM2 detection masks: 可选的人体 patch mask，只用于 query pooling。
- external track priors: BoostTrack 输出的可选跨帧 ID prior，base SMPL 后再做项目内 ID assignment。
- GT supervision: BEDLAM 保留 SMPL、depth、camera；EMDB/3DPW/RICH 评测保留 hmr4d_support 标签。

## BEDLAM Training Data

原始 BEDLAM 不改动。预处理结果写到：

```text
outputs/preprocess/video_tracks/<split>/<sequence>/
  smpl_boxes/*.pkl
  masks/*.npz
  observations.jsonl
  summary.json
```

一键处理 Training split：

```bash
bash scripts/preprocess/prepare_bedlam_full_system_tracks.sh
```

常用小样本检查：

```bash
MAX_SEQUENCES=2 MAX_FRAMES=120 OVERWRITE_FLAG=1 bash scripts/preprocess/prepare_bedlam_full_system_tracks.sh
```

生成后检查训练接口：

```bash
bash scripts/diagnostics/check_bedlam_full_system_data.sh
```

训练配置使用：

```yaml
data:
  boxes_root_key: datasets.bedlam_tracks_root
  query_source: detections
  require_boxes: true
  mask_patch_threshold: 0.10
  min_mask_patches: 4
```

数据检查通过后启动训练：

```bash
bash scripts/train/train_smpl_hsi_full_system_restructure.sh
```

训练 batch 关键字段：

```text
images                       [B,S,3,H,W]
smpl_query_boxes             [B,S,Q,4]
smpl_query_boxes_mask        [B,S,Q]
smpl_query_patch_masks       [B,S,Q,P]
smpl_query_patch_masks_valid [B,S,Q]
external_track_ids           [B,S,Q]
external_track_mask          [B,S,Q]
external_track_confidence    [B,S,Q]
gt_pose_6d / gt_betas / gt_transl_cam / gt_depth / K_scal3r
```

## EMDB / RICH / 3DPW Evaluation Data

评测数据走 hmr4d_support 协议。先抽帧，再对抽帧跑同样的 YOLO/SAM2/BoostTrack sidecar。

一键准备默认四个数据集：

```bash
bash scripts/preprocess/prepare_hmr4d_eval_full_system.sh
```

只跑一个小样本：

```bash
DATASETS="emdb1" MAX_SEQUENCES=1 MAX_FRAMES=120 OVERWRITE_FLAG=1 \
  bash scripts/preprocess/prepare_hmr4d_eval_full_system.sh
```

输出目录：

```text
outputs/preprocess/hmr4d_eval_frames/<dataset>/<safe_vid>/rgb/*.png
outputs/preprocess/hmr4d_eval_tracks/<dataset>/<safe_vid>/
```

接口检查会由一键脚本自动执行，也可以单独跑：

```bash
DATASET=emdb1 bash scripts/diagnostics/check_hmr4d_eval_data_interface.sh
```

## Evaluation Run

EMDB/3DPW 可先跑项目内 SMPL-24 指标：

```bash
CHECKPOINT=outputs/train/smpl_hsi_full_system_restructure/checkpoint_latest.pt \
DATASETS="emdb1 emdb2 3dpw" \
  bash scripts/eval/evaluate_hmr4d_full_system.sh
```

输出：

```text
outputs/eval/hmr4d_smpl_metrics/<dataset>/<dataset>_smpl_metrics.json
outputs/eval/hmr4d_smpl_metrics/<dataset>/<dataset>_smpl_metrics_rows.csv
```

当前评测脚本的 metric protocol 是 `project_native_smpl24`，包含：

```text
MPJPE
PA-MPJPE
camera-space MPJPE
PVE
acceleration error
```

## Known Missing Information

RICH 的 hmr4d_support 通常提供 SMPL-X GT。要做严格 RICH 指标，还需要确认服务器上是否有：

```text
SMPL-X body model assets
smplx2smpl_sparse.pt
smpl_neutral_J_regressor.pt
RICH/GVHMR 评测所需 joint regressor
```

没有这些资产时，脚本不会生成假的 RICH 指标，只会给出 unsupported summary。
