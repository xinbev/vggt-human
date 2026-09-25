# RICH GT SMPL Cached Viewer

`scripts/vis/serve_full_sequence_viewer_cache.sh` can attach the RICH ground-truth SMPL mesh to an existing full-sequence viewer cache. It does not rerun VGGT or NLF.

When enabled, the viewer:

1. Loads `rich_test_labels.pt` and `cam2params.pt`.
2. Decodes the RICH SMPL-X parameters with the gender-specific SMPL-X model.
3. Converts the 10475 SMPL-X vertices to 6890 SMPL vertices with `smplx2smpl.pkl`.
4. Transforms the GT vertices into camera coordinates with RICH `T_w2c`, then into the cached predicted world frame with the cached `hsi_extrinsic` or `raw_extrinsic`.
5. Adds a blue `Show GT SMPL` checkbox to the Viser GUI.

The default server paths match the checked-in RICH assets:

```text
RICH_SUPPORT_ROOT=/home/zhw/xyb_space/RICH/hmr4d_support
SMPLX_MODEL_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/body_models/smplx
SMPLX_TO_SMPL=/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/utils/smplx2smpl.pkl
```

Example for a RICH cache whose image paths contain the official sequence path:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
SHOW_GT_SMPL=true \
GT_COORDINATE_SOURCE=hsi_scaled \
CUDA_VISIBLE_DEVICES_VALUE=7 \
bash scripts/vis/serve_full_sequence_viewer_cache.sh
```

For a cache whose image paths do not contain `/test/<recording>/cam_XX/`, pass the sequence explicitly:

```bash
SHOW_GT_SMPL=true \
RICH_SEQUENCE=test/Gym_011_cooking1/cam_04 \
bash scripts/vis/serve_full_sequence_viewer_cache.sh
```

Use `GT_COORDINATE_SOURCE=raw_vggt` to place GT against the raw VGGT world instead of the HSI-scaled world. GT visibility can be changed in Viser with `Show GT SMPL`; the cache's predicted Base/HSI/TRSTR switches remain independent.
