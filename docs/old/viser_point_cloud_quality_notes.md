# Viser Point Cloud Quality Notes

## Context

This note records the display-side differences between the VGGT-Omega NLF/HSI sequence viewer and the Human3R viewer. It is meant to avoid confusing visualization quality with geometry quality.

## VGGT-Omega Viewer

Main script:

`scripts/vis/serve_nlf_hsi_vggt_sequence_viewer.py`

The scene point cloud is generated from per-frame VGGT depth and RGB:

`VGGT depth -> camera points -> world points -> Viser point cloud`

Important display controls:

- `DEPTH_POINT_STRIDE`: samples every N pixels along both image axes. `6` means roughly `1/36` of pixels are rendered.
- `MAX_SCENE_DEPTH`: clips points farther than the threshold. `0` disables far-depth clipping.
- `POINT_SIZE`: Viser point size.

The viewer now exposes these as live GUI controls:

- `Point Density Preset`
- `Depth Point Stride`
- `Max Scene Depth`
- `Point Size`

Changing stride or max depth rebuilds raw and HSI point clouds from stored depth/RGB/camera data, so the user can move between fast sparse display and dense inspection without restarting the model forward pass.

## Human3R Viewer

Reference file:

`C:\Users\ROG\PycharmProjects\Human3R\viser_utils.py`

Human3R's viewer keeps the source point cloud data and regenerates Viser point clouds when display settings change. The important display logic is in `SceneHumanViewer.parse_pc_data`.

Human3R applies several visual-quality filters before rendering:

- confidence threshold filtering via `conf > vis_threshold`
- foreground/background selection via `msk`
- optional mask dilation/erosion via `Mask Morphology`
- dynamic downsample via `Downsample Factor`
- smaller default point size, around `0.005`

This makes the background look cleaner because low-confidence or masked-out points can be removed before display. It is not only a Viser rendering difference; the displayed point set is different.

## Practical Interpretation

Human3R looking more realistic does not automatically mean its geometry is better. It may be using:

- denser points
- smaller point size
- confidence filtering
- foreground/background mask filtering
- morphology smoothing
- possibly original or less aggressively resized RGB colors
- different temporal accumulation choices

For fair comparison, both systems should be visualized with the same:

- point density
- point size
- far-depth clipping
- camera scale
- RGB resolution/color source
- confidence or foreground mask filtering

## Recommended Dense Inspection Settings

For short sequence inspection:

```bash
DEPTH_POINT_STRIDE=1
POINT_SIZE=0.003
MAX_SCENE_DEPTH=80
MAX_FRAMES=16
```

For a balanced full-sequence view:

```bash
DEPTH_POINT_STRIDE=2
POINT_SIZE=0.006
MAX_SCENE_DEPTH=80
MAX_FRAMES=64
```

For interactive full walking playback:

```bash
DEPTH_POINT_STRIDE=6
POINT_SIZE=0.014
MAX_SCENE_DEPTH=30
MAX_FRAMES=0
```

