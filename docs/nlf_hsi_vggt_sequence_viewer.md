# NLF-HSI VGGT Sequence Viser Viewer

This viewer is the project-native interactive inspection path for the NLF SMPL
provider plus HSI refinement checkpoints.

## Purpose

The viewer runs one VGGT-Omega forward over a selected RGB frame sequence and
serves a Viser scene containing:

- world-space point clouds from raw VGGT depth and HSI affine depth,
- world-space base NLF SMPL meshes,
- world-space HSI-refined SMPL meshes,
- predicted VGGT camera frustums.

It is intended for inspecting whether HSI-refined humans are located at the
corresponding human-shaped regions in the reconstructed depth point cloud.

## Coordinate Contract

- The full selected frame sequence is forwarded together as `[1,S,3,H,W]`.
- `encoding_to_camera(pose_enc, image_size_hw=(H,W))` provides the same camera
  convention used by the existing PLY exporters.
- Depth is unprojected in the processed image plane, then transformed by:
  `world = (camera - t) @ R`.
- SMPL vertices are decoded in camera coordinates, translated by the predicted
  root translation, then transformed to the same world coordinates.
- NLF sees the same processed/padded image plane as VGGT camera intrinsics.

## Run

First run the smoke check. It uses 8 frames, validates tensors/point
clouds/SMPL meshes, writes a summary, and exits without starting a persistent
Viser server:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

DATA_ROOT=/home/zhw/xyb_space \
BEDLAM_ROOT=/home/zhw/xyb_space/bedlam/processed_bedlam \
PREPROCESSED_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/preprocess/bedlam_boxes \
STAGE2_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_full_b12_20260710/stage2_anchor_transl \
FRAMES_DIR=/home/zhw/xyb_space/bedlam/processed_bedlam/Training/20221013_3_250_batch01hand_orbit_bigOffice_seq_000000/rgb \
QUERY_SOURCE=bedlam_sidecar \
MAX_FRAMES=8 \
CUDA_VISIBLE_DEVICES_VALUE=6 \
bash scripts/smoke/check_nlf_hsi_vggt_sequence_viewer.sh
```

Then start the interactive viewer:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

DATA_ROOT=/home/zhw/xyb_space \
BEDLAM_ROOT=/home/zhw/xyb_space/bedlam/processed_bedlam \
PREPROCESSED_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/preprocess/bedlam_boxes \
STAGE2_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_full_b12_20260710/stage2_anchor_transl \
FRAMES_DIR=/home/zhw/xyb_space/bedlam/processed_bedlam/Training/20221013_3_250_batch01hand_orbit_bigOffice_seq_000000/rgb \
QUERY_SOURCE=bedlam_sidecar \
MAX_FRAMES=32 \
CUDA_VISIBLE_DEVICES_VALUE=6 \
bash scripts/vis/serve_nlf_hsi_vggt_sequence_viewer.sh
```

For arbitrary frame folders without BEDLAM sidecars, use:

```bash
QUERY_SOURCE=nlf_detector \
FRAMES_DIR=/path/to/frame_folder \
MAX_FRAMES=32 \
CUDA_VISIBLE_DEVICES_VALUE=6 \
bash scripts/vis/serve_nlf_hsi_vggt_sequence_viewer.sh
```

The script defaults to the rank-1 checkpoint in
`checkpoint_topk_index.json` under `STAGE2_DIR`.

Set `SMOKE_ONLY=true` on `scripts/vis/serve_nlf_hsi_vggt_sequence_viewer.sh`
when you want the same validation path to exit immediately instead of serving
the browser UI.

## Output

The viewer writes:

```text
outputs/vis/nlf_hsi_vggt_sequence_viewer/run_summary.json
```

This summary records the selected checkpoint, input shape, NLF image size,
per-frame point counts, people counts, and HSI scene affine values.
