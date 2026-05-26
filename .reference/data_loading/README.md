# Portable Data Loading

This folder contains the reusable data-loading logic extracted from the current Scal3R workspace. It is designed to be copied as a standalone folder for another agent or workspace while preserving the batch contract used by the existing training, visualization, export, and smoke-test scripts.

## What Is Included

- `bedlam/`: preprocessed BEDLAM sequence indexing, low-level file IO, and `BedlamDataset`.
- `preprocessing/`: ImageNet normalization, direct square resize, Multi-HMR contain+pad letterbox, intrinsics transforms, and single-image Multi-HMR preprocessing.
- `smpl/`: BEDLAM SMPL-X person dict conversion to padded SMPL tensors, axis-angle/6D rotation conversion, and optional `smplx.SMPL` model construction helpers.
- `batching/`: `bedlam_collate_fn`, including the `[B, S] -> [B*S]` flattening required by the Multi-HMR branch.
- `supervision/`: `SMPLGTBuilder`, `perspective_projection`, and `resize_K` for converting loaded SMPL joints into slot UVs and patch score maps.
- `image_folder_dataset.py`: lightweight image file listing utility used by inference pipelines.

The original project import paths under `scal3r.dataloaders.datasets`, `scal3r.models.human.smpl_utils`, and `scal3r.models.human.smpl_gt_builder` are kept as compatibility wrappers around this folder.

## External Dependencies

Required for core dataset loading:

- Python 3.10+
- `torch`
- `numpy`
- `Pillow`

Optional depending on use case:

- `smplx`: required only when constructing a real SMPL model with `build_smpl_model`; without it, `BedlamDataset(..., smpl_model=None)` still returns fallback pelvis-only joints for smoke tests.
- `opencv-python`: not required by this folder, but existing visualization/export/inference scripts use it.

The project `pyproject.toml` already includes `torch`, `numpy`, and `Pillow`. If copying this folder to a fresh environment, install at least:

```bash
pip install torch numpy Pillow
```

Install `smplx` and provide local SMPL model files if real joint supervision or mesh decoding is needed.

## BEDLAM Input Layout

`BedlamDataset` expects preprocessed BEDLAM data in this layout:

```text
processed_bedlam/
  Training/
    <scene>_<seq>/
      rgb/    frame_*.png
      depth/  frame_*.npy
      cam/    frame_*.npz
      smpl/   frame_*.pkl
  Test/
    <scene>_<seq>/
      rgb/    frame_*.png
      depth/  frame_*.npy
      cam/    frame_*.npz
      smpl/   frame_*.pkl
```

Expected per-frame files:

- `rgb/frame_x.png`: RGB image.
- `depth/frame_x.npy`: optional metric depth map in metres. Missing files are replaced with zeros.
- `cam/frame_x.npz`: must contain `intrinsics` as a `[3, 3]` float matrix.
- `smpl/frame_x.pkl`: optional list of person dictionaries. Missing files are treated as no visible persons.

Expected person dictionary keys:

- `smplx_root_pose`: `[1, 3]` axis-angle.
- `smplx_body_pose`: `[21, 3]` axis-angle.
- `smplx_shape`: at least 10 values; first 10 are used as SMPL betas.
- `smplx_transl`: `[3]` camera-space translation.

## Batch Schema

Raw `BedlamDataset.__getitem__` returns one sequence with leading shape `[S, ...]`. After `bedlam_collate_fn`, the schema is:

| Key | Shape | Meaning |
| --- | --- | --- |
| `images` | `[B, S, 3, img_size, img_size]` | Scal3R branch input, direct square resize and ImageNet-normalized |
| `img_mhmr` | `[B*S, 3, mhmr_size, mhmr_size]` | Multi-HMR input, contain+pad letterbox and ImageNet-normalized |
| `gt_depth` | `[B, S, 1, img_size, img_size]` | Metric BEDLAM depth resized to `img_size` |
| `K_scal3r` | `[B, S, 3, 3]` | Intrinsics after direct square resize |
| `K_mhmr` | `[B, S, 3, 3]` | Intrinsics after Multi-HMR letterbox transform |
| `mhmr_letterbox_scale` | `[B*S, 2]` | `(scale_x, scale_y)` for letterboxed MHMR input |
| `mhmr_letterbox_pad` | `[B*S, 2]` | `(pad_x, pad_y)` for letterboxed MHMR input |
| `mhmr_orig_hw` | `[B*S, 2]` | original `(H, W)` per frame |
| `joints3d_cam` | `[B, S, M, 24, 3]` | padded SMPL joints in camera coordinates |
| `gt_pose` | `[B, S, M, 144]` | 24-joint 6D pose representation |
| `gt_betas` | `[B, S, M, 10]` | SMPL shape coefficients |
| `gt_cam_trans` | `[B, S, M, 3]` | camera-space translation |
| `smpl_mask` | `[B, S, M]` | valid-person slot mask |

`img_mhmr` and the three `mhmr_*` metadata fields are intentionally flattened to `[B*S, ...]` to match the model-facing Multi-HMR convention used by existing scripts.

## Minimal Usage

```python
from torch.utils.data import DataLoader

from portable_data_loading import BedlamDataset, bedlam_collate_fn

dataset = BedlamDataset(
    root="/path/to/processed_bedlam",
    split="Training",
    S=2,
    max_humans=10,
    img_size=518,
    mhmr_size=896,
    smpl_model=None,
)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=bedlam_collate_fn)
batch = next(iter(loader))
```

With a real SMPL model:

```python
from portable_data_loading.smpl import build_smpl_model

smpl_model = build_smpl_model("/path/to/body_models/smpl", device="cpu")
dataset = BedlamDataset(root="/path/to/processed_bedlam", smpl_model=smpl_model)
```

Keep the dataset-side SMPL model on CPU when using DataLoader workers. The training script can maintain a separate device-side SMPL model for losses or mesh decoding.

## Supervision Builder Usage

`SMPLGTBuilder` converts the loaded joints and intrinsics into slot-based supervision maps:

```python
from portable_data_loading.supervision import SMPLGTBuilder

B, S, M = batch["smpl_mask"].shape
builder = SMPLGTBuilder(mhmr_img_res=896, mhmr_patch_size=14, scal3r_patch_size=14)
gt = builder.build(
    joints3d_cam=batch["joints3d_cam"].reshape(B * S, M, 24, 3),
    K_scal3r=batch["K_scal3r"].reshape(B * S, 3, 3),
    K_mhmr=batch["K_mhmr"].reshape(B * S, 3, 3),
    scal3r_hw=(518, 518),
    mhmr_hw=(896, 896),
    smpl_mask=batch["smpl_mask"].reshape(B * S, M),
)
```

Returned keys include `smpl_mask`, `smpl_uv_scal3r`, `smpl_uv_mhmr`, `smpl_scores_scal3r`, `smpl_scores_mhmr`, `center_j3d`, and `smpl_j3d`.

## Migration Notes

To migrate this folder to another agent/workspace:

1. Copy the entire `portable_data_loading/` directory.
2. Ensure `torch`, `numpy`, and `Pillow` are installed.
3. If importing from outside the original Scal3R repo, ensure the parent directory of `portable_data_loading` is on `PYTHONPATH`.
4. Update consumers to import from `portable_data_loading` directly, or keep small compatibility wrappers in their original package paths.
5. Provide the BEDLAM processed dataset path and optional SMPL model files separately; neither data nor model assets are included here.

The folder has no import-time dependency on `scal3r`. This makes it safe to copy independently of the model code, as long as downstream code accepts the documented batch schema.

## Current Project Call Sites

The following existing files consume this data-loading contract:

- `scripts/train.py`: builds train/validation `BedlamDataset` and `DataLoader`.
- `scripts/smoke_test_training.py`: optionally loads one BEDLAM batch for end-to-end smoke testing.
- `scripts/visualize_human_predictions.py`: collates a single BEDLAM item for visualization.
- `scripts/export_human_scene_ply.py`: collates a single BEDLAM item and also reads dataset sequence metadata.
- `scal3r/pipelines/inference.py`: uses `ImageFolderDataset` for image manifest generation.

## Important Compatibility Details

- `BedlamDataset._sequences`, `BedlamDataset._index`, `BedlamDataset.S`, and `BedlamDataset.stride` are still present because export utilities currently inspect them.
- The fallback path with `smpl_model=None` sets pelvis joint 0 to `smplx_transl` and leaves other joints as zeros. This is intended for smoke tests and interface validation, not final training quality.
- `mhmr_orig_hw` stores `(H, W)`, while letterbox metadata internally stores both `orig_h` and `orig_w`.
- Direct square resize changes intrinsics by independent `sx` and `sy`; Multi-HMR letterbox uses contain+pad scaling and offset.
