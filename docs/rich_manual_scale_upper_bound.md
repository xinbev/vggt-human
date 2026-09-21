# RICH Manual Scale Upper-Bound Experiment

## Purpose

This is a diagnostic upper-bound experiment for the reduced RICH test-9
`cam_10` protocol. It asks how far the four physical-grounding metrics can be
improved if a human supplies one scale multiplier for every non-overlapping
100-frame window.

The multiplier uses the existing viewer calibration convention:

- scale reconstructed scene points and camera centers about the shared world
  origin;
- preserve the metric size and pose of the SMPL body;
- move the SMPL body with its camera center;
- do not rerun the network after scale selection.

This is an oracle/diagnostic result, not a deployable model result and not the
official UniCon3R Table 3 protocol.

## Step 1: Infer and cache

Local script:

`scripts/eval/prepare_rich_manual_scale_cache.sh`

Server command:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
CUDA_VISIBLE_DEVICES_VALUE=0 \
bash scripts/eval/prepare_rich_manual_scale_cache.sh
```

This evaluates all nine test recordings in 100-frame windows and writes the
reusable inference cache to:

`outputs/eval/rich_manual_scale/cache/`

The command is resumable. Cached windows contain sampled metric depth, RGB,
camera matrices, the selected SMPL mesh, target box, confidence, and model
scale. They do not contain checkpoints or model modules.

## Step 2: Tune and save in Viser

Local script:

`scripts/vis/serve_rich_manual_scale_viewer.sh`

Server command:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
PORT=8080 bash scripts/vis/serve_rich_manual_scale_viewer.sh
```

For each window:

1. Select the window.
2. Move `Scale Multiplier (log10)`.
3. Inspect multiple frames with `Frame in Window`.
4. Click `Preview Window Metrics`.
5. Click `Save Window Scale`.
6. Move to the next window.

Selections are written immediately to:

`outputs/eval/rich_manual_scale/cache/manual_scales.json`

The final evaluator requires every window to have an explicit saved value.
Saving `x1.0` is valid when no adjustment is wanted.

## Step 3: Evaluate manual scales

Local script:

`scripts/eval/evaluate_rich_manual_scale.sh`

Server command:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
SCALE_MODE=manual bash scripts/eval/evaluate_rich_manual_scale.sh
```

Outputs:

`outputs/eval/rich_manual_scale/manual_metrics/`

`summary.json` contains both `adjusted` and `baseline_from_same_cache`. This
same-cache comparison is the main result of the experiment.

## Optional automatic scale oracle

The automatic grid search selects one factor per window by minimizing mean
absolute clearance plus a penalty for invalid frames:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
SCALE_MODE=oracle_grid \
ORACLE_SCALE_MIN=0.25 \
ORACLE_SCALE_MAX=4.0 \
ORACLE_STEPS=41 \
bash scripts/eval/evaluate_rich_manual_scale.sh
```

Outputs are written to:

`outputs/eval/rich_manual_scale/oracle_grid_metrics/`

This grid search can be substantially slower than evaluating saved manual
scales, but it requires no new GPU inference.
