# HF BEDLAM Dataset Integration Notes

## Background

This note records the project-specific handling of the HuggingFace/raw BEDLAM
layout:

```text
/home/zhw/xyb_space/bedlam/hf_bedlam/training_images/
/home/zhw/xyb_space/bedlam/all_npz_12_training/
```

The image layout is:

```text
training_images/<scene>/png/<seq_name>/<seq_name>_<frame>.png
```

The annotation layout is:

```text
all_npz_12_training/<scene>.npz
```

This is different from the older project `processed_bedlam` layout:

```text
processed_bedlam/Training/<sequence>/rgb/*.png
processed_bedlam/Training/<sequence>/smpl/*.pkl
processed_bedlam/Training/<sequence>/cam/*.npz
```

## Raw NPZ Fields Observed

One inspected NPZ contained:

```text
imgname      [N]
center       [N,2]
scale        [N]
pose_cam     [N,165]
pose_world   [N,165]
shape        [N,11]
trans_cam    [N,3]
trans_world  [N,3]
gtkps        [N,127,3]
cam_int      [N,3,3]
cam_ext      [N,4,4]
gender       [N]
proj_verts   [N,437,3]
```

Important detail: `imgname` is relative to the scene `png/` directory, for
example:

```text
seq_000000/seq_000000_0000.png
```

The loader resolves this to:

```text
training_images/<scene>/png/seq_000000/seq_000000_0000.png
```

`all_npz_12_training` is not guaranteed to be a strict one-to-one match with
`hf_bedlam/training_images`. For example, an NPZ scene can exist while the
matching image scene directory is absent. The current loader defaults to:

```yaml
data:
  skip_missing_images: true
```

and will skip those missing-image NPZ files/rows while printing:

```text
[hf_bedlam] skipped missing images: npz_files=... rows=...
```

## Translation Handling

Do not use raw `trans_cam` directly as the project `gt_transl_cam`.

Human3R's BEDLAM preprocessing does:

```python
transl = trans_cam_array[i] + H_array[i][:, 3][:3]
```

where:

```text
H_array = annot_x["cam_ext"]
```

So the project loader must use:

```text
gt_transl_cam = trans_cam + cam_ext[:3, 3]
```

Why this matters:

Raw `trans_cam` can have a near-zero or negative Z value, e.g. `z ~= -0.05`,
which is not a valid forward camera depth for the current ray/depth translation
losses. After adding `cam_ext[:3,3]`, sampled Z values became reasonable, e.g.
roughly `6m - 12m`.

Current config:

```yaml
data:
  transl_add_cam_ext: true
```

## BBox Handling

The raw NPZ may not contain an explicit `bbox` field.

Available bbox-related fields:

```text
center / scale
gtkps
proj_verts
```

Observed behavior:

```text
center/scale: large HMR crop box with substantial padding
gtkps: sometimes incomplete because keypoint visibility/coverage can miss body extent
proj_verts: tightest complete body coverage in visual checks
```

For this project's current SMPL query training, `proj_verts` is the preferred
GT box source. The loader priority is:

```text
explicit bbox
  -> proj_verts
  -> gtkps
  -> center/scale
```

The final training box is expanded by `data.bbox_expand` after the selected
source box is computed.

Current config:

```yaml
data:
  bbox_expand: 0.15
```

## Implemented Project Files

Dataset loader:

```text
vggt_omega/data/hf_bedlam.py
```

Training entry support:

```text
scripts/train/train_smpl.py
```

Path config:

```text
configs/path.yaml
```

Training config:

```text
configs/train_smpl_base_hf_bedlam_ray_refine.yaml
```

Training script:

```text
scripts/train/train_smpl_base_hf_bedlam_ray_refine.sh
```

Diagnostics:

```text
scripts/diagnostics/inspect_hf_bedlam_npz.sh
scripts/diagnostics/check_hf_bedlam_smpl_base_data.sh
scripts/vis/visualize_hf_bedlam_boxes.sh
```

## Recommended Check Flow

Inspect NPZ keys:

```bash
bash scripts/diagnostics/inspect_hf_bedlam_npz.sh
```

Smoke-check dataset tensors:

```bash
bash scripts/diagnostics/check_hf_bedlam_smpl_base_data.sh
```

Expected smoke-check fields include:

```text
images
K_scal3r
gt_intrinsics
gt_pose_6d
gt_betas
gt_transl_cam
gt_boxes
boxes_mask
smpl_mask
gt_track_ids
```

Check `transl_min_xyz` and `transl_max_xyz`; Z should be a reasonable positive
camera depth after `trans_cam + cam_ext[:3,3]`.

Visualize sampled boxes:

```bash
bash scripts/vis/visualize_hf_bedlam_boxes.sh
```

Visualization colors:

```text
yellow: current loader training box
red:    center/scale HMR crop box
green:  gtkps min/max box
blue:   proj_verts min/max box
```

The expected result is that the yellow training box closely follows the blue
`proj_verts` box with a small expansion.

## Small Training Smoke

Before full training, run a small subset:

```bash
DEVICE=cuda \
MAX_NPZ_FILES=1 \
MAX_FRAMES=200 \
EPOCHS=1 \
bash scripts/train/train_smpl_base_hf_bedlam_ray_refine.sh
```

Full training:

```bash
DEVICE=cuda \
bash scripts/train/train_smpl_base_hf_bedlam_ray_refine.sh
```

## Lessons

1. Raw BEDLAM NPZ fields are not identical to the project `processed_bedlam`
   fields.
2. `center/scale` is useful for HMR crop-based models, but too loose for the
   current query-box training path.
3. `proj_verts` is currently the best GT box source for full-body query boxes.
4. Raw `trans_cam` is incomplete for this project; add `cam_ext[:3,3]`.
5. Always run visual bbox checks and translation range checks before using a new
   BEDLAM source for SMPL/trans training.
