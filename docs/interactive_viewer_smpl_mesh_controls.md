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
- a single-frame live scale calibration mode using the complete HSI point cloud.

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

Enable `Single-Frame Scale Calibration` to pause playback and temporarily force
the viewer to the current frame, HSI depth points, point rendering, and HSI
SMPL. The point cloud remains complete, including person pixels; no human mask
is subtracted. The scale slider then updates only the selected frame live, so
the point-cloud person shape can be compared with SMPL without rebuilding the
whole sequence on every slider event. `Timestep`, `Prev Frame`, and `Next Frame`
remain available for choosing a representative frame.

Turning calibration mode off applies the selected multiplier to the complete
sequence and restores the playback, display mode, depth source, SMPL, and camera
visibility settings that were active before calibration.

## Human Point Removal

Normal sequence display removes human depth points; calibration mode bypasses
this removal and uses the stored complete point cloud. The removal mask is built
per person and per frame as follows:

1. Project valid HSI SMPL vertices into the depth image.
2. Rasterize every valid SMPL triangle to obtain the actual projected body
   silhouette rather than a coarse bounding box or convex hull.
3. Dilate the silhouette by the applied pixel radius (`12` pixels by default)
   to cover projection and boundary error.
4. Remove every sampled depth point covered by the dilated silhouette. No depth
   distance condition is applied.

`HUMAN_MASK_DILATION_PX` sets the initial boundary expansion. The startup
`run_summary.json` records the method, initial dilation, full point counts, and
removed HSI point counts.

The Viser `Filter Human Points` checkbox switches both raw and HSI point clouds
between the cached filtered and complete versions without rerunning inference.
It defaults to enabled and can also be initialized with
`FILTER_HUMAN_POINTS=true/false`. Single-frame scale calibration temporarily
forces filtering off, then restores the previous checkbox state when closed.

`Human Filter Dilation (px)` adjusts the projected silhouette expansion from
`0` to `32` pixels in one-pixel steps. `0` uses the exact rasterized SMPL
silhouette; larger values remove a wider boundary around it. Moving the slider
only marks the value as pending. Click `Apply Human Filter Size` to rebuild the
filtered raw and HSI point caches for the sequence, or use
`Reset Human Filter Size to 12 px` to restore the default. The complete point
cloud caches are unchanged. Calibration mode disables these controls while it
is active.

The multiplier scales HSI environment world points, HSI camera positions, and
the HSI camera trajectory about the shared world origin. It does not edit model
outputs, the checkpoint, raw VGGT points, HSI SMPL vertices, or saved alignment
diagnostics. Scaling both the corrected depth and HSI camera translation is
equivalent to multiplying the model scale and bias by the displayed viewer
multiplier. The GUI therefore also reports the effective visual scale/bias.

The initial multiplier can be set from the shell with
`HSI_VISUAL_SCALE=<value>`; the supported range is `0.5` to `2.0`.

## Point Cloud Measurement

Open `Point Cloud Measurement` and enable `Enable Point Measurement`. Playback
pauses so that the displayed cloud stays stable while selecting points. Click
two visible raw or HSI environment points. The viewer marks both endpoints
without text labels, draws a line between them, and displays only an `x.xxm`
distance label at the line midpoint. The value is also shown in `Measurement
Result`, together with each source frame, point index, and world coordinate.

Point selection uses Viser's scene-level click ray because Viser `0.2.x` point
cloud handles do not expose node click callbacks. `Measurement Pick Radius`
sets the maximum world-space distance between that ray and the selected point.
Increase it when a sparse cloud is difficult to click; the status reports the
nearest distance when a click falls outside the current radius.

`Measurement Line Width` changes the line thickness, `Distance Font Size`
changes the colored distance label size, and `Measurement Color` controls both
the line and label background. These style controls update the current
measurement immediately. The image-based label follows the active viewer camera
so that its colored text remains readable while orbiting the scene.

Distances use the coordinates currently displayed by the viewer and are shown
in meters. This includes the current HSI visual scale multiplier. A third
point click clears the previous measurement and starts a new one. `Clear Current
Measurement` removes the endpoints, labels, line, and value. Changing point
filtering, point sampling, or HSI visual scale also clears the current result so
that measurements cannot remain attached to stale geometry.

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
