# Interactive SMPL And HSI Point Viewer

## Current Behavior

The Stage2 walking viewer uses the original RGB point-cloud environment again.
The generic viewer still retains the experimental mesh path for compatibility,
but `serve_stage2_human_scene_align_walking_id_overlay.sh` now starts with
`ENVIRONMENT_DISPLAY=points`.

The viewer supports:

- current-frame, accumulated, and hybrid sequence display;
- raw VGGT depth, HSI-corrected depth, or both;
- optional track ID labels;
- click/dropdown SMPL selection and viewer-only XYZ translation;
- display of the active HSI scale strategy and scale/bias values;
- a viewer-only HSI environment scale multiplier.

## HSI Scale Strategy

The Stage2 walking wrapper loads
`configs/train_smpl_hsi_nlf_stage2_human_scene_align.yaml`, whose model setting
is:

```yaml
hsi_scene_affine_mode: clip_median
```

The HSI refinement head first predicts one scale and depth bias per frame. It
confidence-weights all valid human-query predictions inside each frame. The
`clip_median` strategy then takes the sequence median in log-scale space and
the sequence median depth bias, and broadcasts that robust pair to every frame.
Therefore this viewer uses one shared model scale/bias pair for the selected
sequence, although the pre-aggregation per-frame values can differ.

The calibrated depth is:

```text
hsi_depth = raw_depth * hsi_scene_scale + hsi_scene_depth_bias
```

The GUI has a dedicated, expanded `HSI Scale Controls` folder with separate
read-only rows for:

- the configured strategy;
- the current model scale and bias actually used by geometry;
- the original per-frame prediction before sequence aggregation;
- model and raw sequence min/median/max scale statistics;
- the applied and pending viewer-only multipliers.

## Manual Visualization Scale

The `Visual Scale Multiplier` slider defaults to `1.0`, ranges from `0.5` to
`2.0`, and uses a `0.01` step. Move the slider and click `Apply Scale` to update
the complete HSI point sequence. `Reset Scale to 1.0` restores the model
visualization.

The multiplier scales HSI environment world points, HSI camera positions, and
the HSI camera trajectory about the shared world origin. It does not edit model
outputs, the checkpoint, raw VGGT points, HSI SMPL vertices, or saved alignment
diagnostics. Scaling both the corrected depth and HSI camera translation is
equivalent to multiplying the model scale and bias by the displayed viewer
multiplier. The GUI therefore also reports the effective visual scale/bias.

The initial multiplier can be set from the shell with
`HSI_VISUAL_SCALE=<value>`; the supported range is `0.5` to `2.0`.

## Server Usage

Local script:

`scripts/vis/serve_stage2_human_scene_align_walking_id_overlay.sh`

Server script:

`/home/zhw/lab_users/xyb/home/projects/vggt-human/scripts/vis/serve_stage2_human_scene_align_walking_id_overlay.sh`

Run:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
FRAMES_DIR=/home/zhw/lab_users/xyb/home/projects/Human3R-master/outputs/walking/color \
STAGE2_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full \
CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt \
OUTPUT_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/stage2_human_scene_align_walking_viewer_id_overlay \
CUDA_VISIBLE_DEVICES_VALUE=7 \
PORT=8080 \
MAX_FRAMES=64 \
HSI_VISUAL_SCALE=1.0 \
bash scripts/vis/serve_stage2_human_scene_align_walking_id_overlay.sh
```

The viewer summary remains under the configured `outputs/vis/...` directory.
`run_summary.json` now records the affine strategy, final model scales, raw
per-frame scales, and the initial viewer-only multiplier.

## Validation Boundary

Windows validation covers syntax, shell wiring, and static geometry flow. Full
Viser interaction still needs the Linux server environment, checkpoint, SMPL
assets, and CUDA runtime.
