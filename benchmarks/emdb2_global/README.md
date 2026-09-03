# EMDB-2 Global Human Benchmark

This benchmark is an adaptation rewrite of Human3R's
`eval/global_human` EMDB-2 protocol. It does not import or modify `.paper`.

## Metrics

| Paper metric | Human3R internal key | Protocol | Unit |
| --- | --- | --- | --- |
| W-MPJPE | `wa2_mpjpe` | Split into <=100-frame chunks; estimate one Sim(3) from the first two frames, apply to the chunk | mm |
| WA-MPJPE | `waa_mpjpe` | Split into <=100-frame chunks; estimate one Sim(3) from all joints in the chunk | mm |
| RTE | `rte` | Rigid-align the complete root trajectory without scale; divide frame root error by total GT root displacement | % |

Human3R's text summary labels the global metric section as `cm`, but its code
multiplies W/WA joint errors by `1000`. This implementation follows the actual
computation and reports W-MPJPE/WA-MPJPE in millimeters.

The paper summary is frame-weighted across all evaluated sequences, matching
Human3R's `n_human` weighted aggregation. Sequence-macro values are exported as
diagnostics only.

## EMDB-2 GT

The protocol contains the same 25 native EMDB-2 sequences listed by Human3R.
Only `good_frames_mask` frames are evaluated. GT joints are decoded from native
world-space gender-specific SMPL using:

```text
poses_root + poses_body + betas + world transl
```

The evaluated skeleton is this project's SMPL-24, with root joint index 0.

## Prediction Archive Contract

Create one archive per sequence, for example:

```text
<PREDICTIONS_ROOT>/P9_80_outdoor_walk_big_circle.npz
```

The current multi-stage evaluator requires all three stage arrays:

```python
np.savez(
    path,
    sequence_name="P9/80_outdoor_walk_big_circle",
    frame_indices=np.asarray([...], dtype=np.int64),  # original EMDB frame IDs
    stage_names=np.asarray([
        "vggt_nlf",
        "vggt_nlf_hsi_scale",
        "vggt_nlf_hsi_scale_trstr",
    ]),
    pred_joints_world__vggt_nlf=stage_a,                    # [F,24,3], m
    pred_joints_world__vggt_nlf_hsi_scale=stage_b,          # [F,24,3], m
    pred_joints_world__vggt_nlf_hsi_scale_trstr=stage_c,    # [F,24,3], m
    valid=np.asarray(..., dtype=bool),
    joint_format="smpl24",
    units="m",
)
```

Each world array may instead be represented by its stage-specific camera-space
joints and model-predicted continuous camera-to-world trajectory:

```python
np.savez(
    path,
    sequence_name="P9/80_outdoor_walk_big_circle",
    frame_indices=frame_indices,
    pred_joints_cam__vggt_nlf=stage_a_cam,
    pred_T_c2w__vggt_nlf=stage_a_predicted_T_c2w,
    # Repeat the same pair for vggt_nlf_hsi_scale and
    # vggt_nlf_hsi_scale_trstr.
    valid=valid,
    joint_format="smpl24",
    units="m",
)
```

Do not use EMDB GT camera extrinsics to transform predictions. That evaluates
human reconstruction with oracle camera motion and is not Human3R's predicted
global-human setting.

## Run

First test the metric implementation:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
bash benchmarks/emdb2_global/test.sh
```

Validate the configured native EMDB-2 tree:

```bash
bash benchmarks/emdb2_global/check_data.sh
```

Then evaluate prediction archives:

```bash
PREDICTIONS_ROOT=/path/to/emdb2_world_predictions \
EMDB_ROOT=/home/zhw/xyb_space/emdb \
OUTPUT_DIR=outputs/eval/emdb2_global/my_model \
CUDA_VISIBLE_DEVICES_VALUE=7 \
bash benchmarks/emdb2_global/run.sh
```

Outputs:

```text
summary.json
sequence_metrics.csv
frame_metrics.csv
```

## Official Full-Frame Integration Blocker

The current VGGT-Omega full pipeline can evaluate short windows, but EMDB-2
sequences contain roughly 700-3300 frames. Processing each sequence in
independent VGGT chunks creates independent Sim(3) camera worlds. Concatenating
those chunks would corrupt W-MPJPE, WA-MPJPE and RTE. A valid exporter needs a
verified long-sequence camera-state path or overlap-based Sim(3) stitching with
explicit camera-trajectory tests. This benchmark intentionally refuses to use
GT camera extrinsics as a shortcut.

The stride-7 baseline below avoids chunk gauges with one forward per sampled
sequence. The blocker remains only for the official all-good-frame protocol.

## Stride-7 Unchunked Two-Pass Workflow

The first practical pipeline uses a fixed stride of seven after EMDB's
`good_frames_mask`. The longest protocol sequence then contains fewer than 500
selected frames, so every selected sequence is processed without independent
chunks. The current coarse/HSI cascade makes two complete passes over the same
selected frames: pass one estimates analytic coarse scale; pass two applies
coarse + v3 residual + TRSTR. Final world joints use only pass-two predicted
camera poses.

Per sequence:

```text
native EMDB good frames
-> good_frame_indices[::7]
-> <=500 RGB frames in one unchunked batch per pass
-> analytic coarse + v3 residual with one shared effective affine
-> TRSTR spatial correction
-> decode selected SMPL-24 camera joints
-> invert scaled predicted VGGT T_w2c
-> predicted world joints
-> NPZ archive
```

The exporter never reads EMDB GT camera extrinsics. It uses GT annotations only
to select the protocol good-frame IDs; GT joints are decoded later by the metric
process.

Run one sequence first:

```bash
TRSTR_CHECKPOINT=/path/to/trstr_checkpoint.pt \
MAX_SEQUENCES=1 \
CUDA_VISIBLE_DEVICES_VALUE=7 \
bash benchmarks/emdb2_global/run_stride7_full.sh
```

Full 25-sequence run:

```bash
TRSTR_CHECKPOINT=/path/to/trstr_checkpoint.pt \
CUDA_VISIBLE_DEVICES_VALUE=7 \
bash benchmarks/emdb2_global/run_stride7_full.sh
```

Default outputs:

```text
outputs/eval/emdb2_global_stride7/predictions/*.npz
outputs/eval/emdb2_global_stride7/predictions/manifest.json
outputs/eval/emdb2_global_stride7/metrics/summary.json
outputs/eval/emdb2_global_stride7/metrics/stage_metrics.csv
outputs/eval/emdb2_global_stride7/metrics/sequence_metrics.csv
outputs/eval/emdb2_global_stride7/metrics/frame_metrics.csv
```

For stride seven the Human3R chunk length is `int(100/7)=14`. Results are
labelled `EMDB-2-S7` and must not be presented as official full-frame EMDB-2
numbers.

## Multi-Stage Contribution Report

One prediction archive stores three outputs built from the same detector
selection, pose/betas, and final-pass predicted camera rotation:

```text
A: RGB-VGGT-NLF + shared analytic coarse gauge
B: A + HSI Scale v3 residual
C: B + TRSTR translation
```

The analytic coarse gauge in A is required because NLF camera-space SMPL is in
meters while raw VGGT camera translation has an arbitrary scale. Directly
combining those units would make baseline RTE meaningless. A uses one
sequence-wide log-median coarse scale, B uses one sequence-wide final effective
scale, and C keeps B's camera while changing only SMPL translation.

`summary.json` contains metrics for all three stages and error reductions:

```text
contributions_error_reduction.hsi_scale = A - B
contributions_error_reduction.trstr     = B - C
contributions_error_reduction.total     = A - C
```

Positive contribution means the added module lowered error. Negative means it
degraded that metric. `sequence_metrics.csv` and `frame_metrics.csv` include a
`stage` column for direct plotting.

The shared detector query is selected with Human3R's matching principle:
gender-specific EMDB world GT SMPL is projected with the EMDB GT camera for 2D
association only, predicted neutral SMPL-24 joints are projected with the
predicted VGGT intrinsics, and the lowest-error prediction with bbox IoU at
least 0.05 is matched. EMDB GT camera is never used to construct predicted
world coordinates.
