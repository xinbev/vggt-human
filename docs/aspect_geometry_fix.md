# VGGT-Omega Aspect Geometry Fix

This change replaces the training/eval default square-resize geometry with a VGGT-style aspect-preserving, patch-aligned geometry path.

## Coordinate Convention

- `image_resolution` is the target token budget reference, default `512`.
- `resize_mode: balanced` keeps aspect ratio, applies the original VGGT extreme-aspect center crop, and outputs patch-aligned `(H, W)`.
- `image_size` remains only as a legacy fallback.
- Model heads derive runtime geometry from `images.shape[-2:]`.
- Normalized boxes are `cx/W, cy/H, w/W, h/H`.
- Camera intrinsics are updated with independent `scale_x/scale_y` and padding offsets.

## Main Touch Points

- Geometry helper: `vggt_omega/data/geometry.py`
- Datasets/collate: BEDLAM, HF-BEDLAM, 3DPW, HMR4D eval
- Model geometry: `VGGTOmega`, `SMPLHead`, `HSIRefinementHead`
- Loss geometry: `HungarianSMPLLoss`
- Query/mask: sidecar patch-mask builder and SAM2 patch-mask preprocess scripts
- Visualization: SMPL/HSI diagnostic and PLY element scripts

## Server Smoke

Run after syncing to the server:

```bash
bash scripts/smoke/check_aspect_geometry.sh
python -m py_compile vggt_omega/data/geometry.py vggt_omega/models/heads/smpl_head.py vggt_omega/models/heads/hsi_refinement_head.py
```

Then run the relevant dataset diagnostic script, for example:

```bash
bash scripts/diagnostics/check_bedlam_full_system_data.sh
```

The expected fixed behavior for a `612x408` or `408x612` frame is a non-square patch-aligned model input, not a stretched square.
