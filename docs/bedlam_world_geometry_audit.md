# BEDLAM processed data: world geometry audit

This viewer checks semantic consistency of an already processed sequence. It
overlays the actual RGB-coloured metric-depth scene and GT SMPL meshes in one
world frame, which makes it more useful than a directory or loader check.

## Coordinate contract under test

The audit assumes the current project convention:

```text
K                 = pixel intrinsics
depth[u, v]       = camera-coordinate Z depth in metres
cam pose          = camera-from-world / world-to-camera
                    x_cam = R_w2c x_world + t_w2c
smplx_transl      = camera-coordinate SMPL root translation
```

For each sampled RGB/depth pixel it computes:

```text
x_cam = ((u-cx) z/fx, (v-cy) z/fy, z)
x_world = R_w2c^T (x_cam - t_w2c)
```

For each GT person it decodes SMPL, adds `smplx_transl` in camera coordinates,
then applies the identical camera-to-world transform. Thus depth and SMPL
cannot appear aligned merely because one was left in a camera frame while the
other was moved to world.

The script also writes an RGB-side `*_smpl_projection_overlay.png`. It projects
the same camera-space SMPL vertices and joints with `K`; it catches wrong
intrinsics, pose/root orientation, and translation conventions without relying
on the world transform.

## Server entry point

Local script: `scripts/vis/serve_bedlam2_single_frame_viser.sh`.

After Git sync, run on the server:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
bash scripts/vis/serve_bedlam2_single_frame_viser.sh
```

For the new data path explicitly:

```bash
SEQUENCE_DIR=/home/zhw/xyb_space/bedlam2/bedlam2_processed/Training/20241213_1_250_rome_tracking_seq_000002 \
PORT=8088 \
bash scripts/vis/serve_bedlam2_single_frame_viser.sh --frame-id seq_000002_0000
```

The terminal prints the Viser address. `outputs/vis/bedlam_world_geometry_audit/`
receives `run_summary.json` and the RGB/SMPL projection overlay. Use the same
script with an old processed BEDLAM sequence to form a direct baseline
comparison.

## What to inspect

1. In Viser, the coloured point cloud should show a recognisable scene and the
   camera frustum should sit at a sensible scene location.
2. Each coloured SMPL mesh should occupy its visible human-shaped depth region;
   small occlusion gaps are expected, metre-scale displacement is not.
3. In the RGB overlay, the projected SMPL point cloud, joints and rectangle
   should lie on the photographed person. If this is wrong while the mesh is
   plausible in Viser, inspect `K` and the SMPL camera convention.
4. Compare `depth.median_m` in `run_summary.json` with person translation Z. A
   systematic factor near 100 usually means centimetres/metres were mixed.

The script does not claim that a visually acceptable single frame proves every
training condition. It does prove or refute the most important geometric
condition for the planned GT depth + GT SMPL Stage1/TRSTR training: RGB, depth,
camera and SMPL describe the same physical frame and compatible scale.

## Coordinate-hypothesis audit

When SMPL penetrates the depth scene, run the dedicated enumerator instead of
changing the processed data immediately:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

SEQUENCE_DIR=/home/zhw/xyb_space/bedlam2/bedlam2_processed/Training/20241213_1_250_rome_tracking_seq_000002 \
OUTPUT_DIR=outputs/vis/bedlam2_coordinate_hypothesis_audit \
PORT=8090 \
FRAME_COUNT=3 \
FRAME_STEP=5 \
bash scripts/vis/serve_bedlam_coordinate_hypothesis_viser.sh
```

The Viser dropdown enumerates these choices:

1. camera coordinates with the stored `smplx_transl`;
2. camera coordinates with `smplx_transl - pose.t`;
3. world coordinates treating `pose` as world-to-camera (W2C), with each
   translation option;
4. world coordinates treating `pose` as camera-to-world (C2W), with each
   translation option.

The two camera choices are decisive for whether depth and SMPL share the
current training contract. The world choices show several frames together. The
correct external-camera direction is the candidate where static walls, floor,
and furniture align across frames; do not use human motion alone to decide.
The output directory receives two RGB projection overlays (stored translation
and `stored - pose.t`) plus `coordinate_hypothesis_summary.json`.
