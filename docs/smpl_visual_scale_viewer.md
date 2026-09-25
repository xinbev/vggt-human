# SMPL Visual Scale Control

## Purpose

The cached full-sequence viewer now has an independent visual scale control for
SMPL meshes. It applies to predicted Base SMPL, HSI/TRSTR SMPL, and attached
RICH GT SMPL meshes. It does not modify inference outputs, cached point clouds,
camera poses, or saved model checkpoints.

## Behavior

- `SMPL Visual Scale Multiplier (log10)` uses the same `0.1` to `10.0`
  logarithmic range as the environment visual scale control.
- Click `Apply SMPL Scale` to rebuild the meshes with the requested multiplier.
- Click `Reset SMPL Scale to 1.0` to restore the original display size.
- Scaling is performed around each mesh's own geometric center. This changes
  the visible body size without moving the person toward the global origin.
- The HSI camera visual offset, SMPL opacity, colors, selection, and track
  visibility are retained after a mesh rebuild.

## Server usage

The local script is:

`scripts/vis/serve_full_sequence_viewer_cache.sh`

On the Linux server:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
SMPL_VISUAL_SCALE=1.0 \
bash scripts/vis/serve_full_sequence_viewer_cache.sh
```

`SMPL_VISUAL_SCALE` is only the initial value. The same setting can be changed
live in Viser with the slider and Apply button. For example, `1.05` makes each
SMPL mesh 5% larger while preserving its center.
