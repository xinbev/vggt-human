# 3DPW SMPL Base Training

## Stage Goal

第一阶段只训练 SMPL base head。输入只需要：

```text
3DPW imageFiles
3DPW sequenceFiles
SMPL body model assets
```

不需要 YOLO、SAM2、BoostTrack，也不复制图片。

## Preprocess

服务器原始数据结构：

```text
/home/zhw/xyb_space/3DPW/
  imageFiles/
  sequenceFiles/
    train/
    validation/
    test/
```

生成 compact annotation cache：

```bash
bash scripts/preprocess/prepare_3dpw_smpl_base.sh
```

小样本调试：

```bash
MAX_SEQUENCES=2 DEVICE=cuda bash scripts/preprocess/prepare_3dpw_smpl_base.sh
```

输出：

```text
outputs/preprocess/3dpw_smpl_base/train.pkl
outputs/preprocess/3dpw_smpl_base/validation.pkl
outputs/preprocess/3dpw_smpl_base/test.pkl
outputs/preprocess/3dpw_smpl_base/summary.json
```

cache 内容是相机系 SMPL pose/beta/transl、intrinsics、2D bbox，不包含图片副本。

## Check Data

```bash
bash scripts/diagnostics/check_3dpw_smpl_base_data.sh
```

检查 validation/test：

```bash
SPLIT=validation bash scripts/diagnostics/check_3dpw_smpl_base_data.sh
SPLIT=test bash scripts/diagnostics/check_3dpw_smpl_base_data.sh
```

## Train

```bash
bash scripts/train/train_smpl_base_3dpw.sh
```

配置：

```text
configs/train_smpl_base_3dpw.yaml
```

输出：

```text
outputs/train/smpl_base_3dpw/checkpoint_latest.pt
outputs/train/smpl_base_3dpw/checkpoint_top01.pt
outputs/train/smpl_base_3dpw/checkpoint_top02.pt
outputs/train/smpl_base_3dpw/checkpoint_top03.pt
outputs/train/smpl_base_3dpw/resolved_config.json
```

top-3 根据 validation `loss_total` 越低越好保存。

## Evaluate

```bash
CHECKPOINT=outputs/train/smpl_base_3dpw/checkpoint_top01.pt \
  bash scripts/eval/evaluate_3dpw_smpl_base_metrics.sh
```

输出：

```text
outputs/eval/3dpw_smpl_base/3dpw_smpl_base_metrics.json
outputs/eval/3dpw_smpl_base/3dpw_smpl_base_metric_rows.csv
```

指标：

```text
PA-MPJPE
MPJPE
PVE
```

当前协议是 project-native SMPL24 camera-coordinate pelvis-aligned metric，参考 Human3R/GVHMR 的 3DPW 相机坐标评测思路，但不直接 import 外部项目。
