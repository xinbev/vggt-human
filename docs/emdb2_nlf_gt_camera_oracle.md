# EMDB-2 NLF + GT Camera Oracle

## Goal

Measure NLF human reconstruction when camera uncertainty is removed. This is
an oracle ablation, not a pure-RGB benchmark result.

## Data Flow

```text
EMDB RGB
  -> project-standard crop/resize/pad
  -> transform native EMDB GT K into the processed image plane
  -> NLF detector and SMPL reconstruction
  -> select the same single GT person with Human3R-style 2D matching
  -> NLF SMPL-24 joints in camera coordinates
  -> inverse native EMDB GT T_w2c
  -> predicted world joints
  -> W-MPJPE / WA-MPJPE / RTE
```

Only NLF predicts the human. GT SMPL pose, shape, translation, and world joints
are not used to construct predictions. GT SMPL is used only as the metric target
and for the existing 2D person-association protocol.

The experiment uses both parts of the GT camera:

- `K` (intrinsics): supplied to NLF after crop/resize/pad adjustment.
- `T_w2c` (extrinsics): inverted and used to map NLF camera joints to world.

## Protocol

- Dataset: native EMDB-2 25-sequence list.
- Frames: `good_frames_mask[::7]`.
- Maximum selected frames per sequence: 500.
- W/WA chunk length: `int(100 / 7) = 14`.
- Person matching: `human3r_gt_smpl2d_iou_v1`.
- Outputs: `outputs/eval/emdb2_s7_nlf_gt_camera/`.

Because GT camera parameters are privileged test-time data, these values show
the error remaining in NLF after camera error is removed. They must not be
presented as the official RGB-only EMDB-2 result.

## Server Command

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

CUDA_VISIBLE_DEVICES_VALUE=0 \
bash benchmarks/emdb2_global/run_gt_camera_nlf_stride7.sh
```

For a one-sequence smoke run:

```bash
MAX_SEQUENCES=1 \
CUDA_VISIBLE_DEVICES_VALUE=0 \
bash benchmarks/emdb2_global/run_gt_camera_nlf_stride7.sh
```

The final values are written to:

```text
outputs/eval/emdb2_s7_nlf_gt_camera/metrics/summary.json
outputs/eval/emdb2_s7_nlf_gt_camera/metrics/stage_metrics.csv
outputs/eval/emdb2_s7_nlf_gt_camera/metrics/sequence_metrics.csv
```
