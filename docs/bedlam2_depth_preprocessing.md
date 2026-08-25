# BEDLAM2 depth materialization

## Why the current labels do not contain depth

`labels_smpl_6fps/20241213_1_250_rome_tracking.npz` is a label table, not a
packaged training dataset.  It provides one or more SMPL records per RGB
`imgname`, but deliberately leaves RGB PNG and depth EXR in their raw roots.
The project loader, `vggt_omega.data.BedlamDataset`, instead reads one
float32 depth map per RGB frame from `depth/<frame>.npy`.

Do not add EXR arrays to the label NPZ and do not point the loader at EXR
files.  Materialize a separate processed tree while retaining the raw data.

## BEDLAM2 adapter

The project-local adapter is:

`scripts/preprocess/prepare_bedlam2_scene.py`

It is an adaptation, not a direct import, of the old BEDLAM preprocessor.  It
uses BEDLAM2 `smpl_pose_cam`, `smpl_betas`, `smpl_trans_cam`, `cam_int`, and
`cam_ext`; it groups duplicate `imgname` rows into the people list expected by
the existing loader.  It keeps the baseline loader untouched.

The output layout is:

```text
outputs/preprocess/bedlam2_processed/
  Training/
    20241213_1_250_rome_tracking_seq_000000/
      rgb/seq_000000_0000.png
      depth/seq_000000_0000.npy       # float32 depth in metres
      cam/seq_000000_0000.npz          # intrinsics, pose
      smpl/seq_000000_0000.pkl         # list of per-person dictionaries
```

`smpl_trans_cam + cam_ext[:3, 3]` is stored as `smplx_transl`, matching the
project's established camera-space target convention.  The adapter exposes
`--translation-mode direct` only for a later, evidence-backed convention
change; do not use it for the initial baseline.

## Server workflow

Local script path: `C:\Users\ROG\PycharmProjects\vggt-omega\scripts\preprocess\prepare_bedlam2_scene.sh`.
After git sync, server path: `/home/zhw/lab_users/xyb/home/projects/vggt-human/scripts/preprocess/prepare_bedlam2_scene.sh`.

First inspect the actual EXR channel and raw values.  This does not write any
dataset files:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
INSPECT_ONLY=true bash scripts/preprocess/prepare_bedlam2_scene.sh
```

The report prints `raw_depth_positive_*`, the selected EXR channel, and the
RGB/depth raster dimensions.  Confirm the EXR unit from BEDLAM2 metadata or a
known scene distance before choosing `DEPTH_SCALE`.  The legacy BEDLAM
converter used `0.01` because its EXR values were centimetres, but that is an
initial candidate—not an assumption applied by this adapter.

After confirmation, materialize the scene (for centimetre EXR values):

```bash
DEPTH_SCALE=0.01 bash scripts/preprocess/prepare_bedlam2_scene.sh
```

The conversion keeps the native 6-FPS sampling encoded in the label NPZ.  Its
results are written to
`/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/preprocess/bedlam2_processed`.
`hardlink` is the default RGB strategy and falls back to a copy across file
systems; raw RGB and raw EXR are not modified.

Before a full conversion, a small no-write pairing check is useful:

```bash
DEPTH_SCALE=0.01 SEQUENCE=seq_000000 MAX_FRAMES=8 DRY_RUN=true \
  bash scripts/preprocess/prepare_bedlam2_scene.sh
```

The adapter fails rather than resizing when a same-stem RGB/EXR pair has
different raster dimensions.  In that case inspect
`ground_truth/meta_exr_depth_csv/` and the depth camera animation metadata;
the RGB intrinsics in `cam_int` must not be silently paired with an unaligned
depth render.

## Follow-up checks

Once the tree has been materialized, create the existing projected-box
sidecars and run the standard dataset smoke test.  The first command's outputs
are under `outputs/preprocess/bedlam_boxes`; the data-check only prints a
tensor summary.

```bash
BEDLAM_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/preprocess/bedlam2_processed \
  bash scripts/preprocess/prepare_bedlam_boxes.sh

BEDLAM_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/preprocess/bedlam2_processed \
  bash scripts/diagnostics/check_bedlam_full_system_data.sh
```

For the second command, pass the same `datasets.bedlam_root` override if the
selected training config does not already resolve `BEDLAM_ROOT`.  Confirm that
`gt_depth` is non-zero and that projected SMPL boxes agree with the RGB before
enabling any depth-supervised experiment.
