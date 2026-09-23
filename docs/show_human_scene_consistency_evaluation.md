# SHOW HS-V / HS-CF evaluation integration

## Goal

Evaluate whether the predicted SMPL body and the predicted scene depth occupy a
consistent camera-space scale and placement.  The implementation is additive:
the existing 3DPW mesh metrics and model paths are unchanged.

## Reference and adaptation boundary

- Paper/project: **SHOW — Scene and Human in One World: Reconstruction in a
  Feedforward Pass**.
- Paper PDF inspected at `.paper/base_pdf/SHOW.pdf` (the repository does not
  contain the older `.paper/base/_pdf/SHOW/.pdf` path).
- Read-only reference checkout:
  `.paper/base_projects/4D-show-official-codebase`.
- Referenced code:
  `hmr4d/utils/eval/eval_utils.py`, especially
  `compute_camcoord_human_scene_synchronize_metrics`,
  `_render_visible_point_map_in_camera_frame`,
  `_flatten_valid_camera_points`, and `chamfer_distance`; and the 3DPW/EMDB
  callbacks under `hmr4d/model/gvhmr/callbacks/`.
- Integration type: **adapted rewrite**.  Nothing imports from or modifies
  `.paper/`.

The official renderer/metric stack was not copied wholesale.  The metric is a
small evaluation-only module under `vggt_omega/evaluation/`, and the 3DPW
adapter uses this project's model, SMPL layer, dataset, Hungarian association,
configurable human-region masks, configuration and output conventions.

## Human-region mask protocol

SHOW's released inference/evaluation obtains `human_masks` from bounding-box
prompted SAM2.  This project defaults to `mask_source: gt_smpl_projection`:
the annotated SMPL mesh is rendered with the evaluation camera to define the
per-person silhouette, while the body and scene being scored still come from
the project's predictions.  This removes the SAM2 preprocessing dependency and
isolates geometric consistency from segmentation noise.

The GT silhouette is rendered with the dataset camera intrinsics (`K_scal3r`),
not the project's predicted intrinsics.  SHOW metric unprojection and predicted
body visibility still use the project's predicted intrinsics, so the mask does
not move or resize to accommodate a camera-prediction error.

For HMR4D-support evaluation, the GT world-space SMPL is decoded with its
annotated male/female body model, translated in world coordinates, transformed
by `T_w2c`, and only then rasterized.  The project's prediction continues to
use its native neutral-SMPL inference path.

This is a GT-region evaluation variant, not a claim that SHOW itself used GT
masks.  It is fair for comparisons only when every compared prediction is
evaluated with the same GT-projected mask.  Set `mask_source: sam2_patch` and
`data.require_sam2_patch_masks: true` to reproduce SHOW's segmentation-based
mask source more closely.

## Table 3 protocol used by this evaluator

The default path evaluates this project's predictions with SHOW's released
3DPW evaluator protocol:

| Output | Direction and normalization |
|---|---|
| `hs_cf5`, `hs_cf10` | Released-code protocol: masked scene points to rendered body surface; divide by mean masked-scene depth; multiply by 100 |
| `hs_v5`, `hs_v10` | Released `xy_scale_mse_p5_95/p10_90`: mean absolute difference of x/y population variances |

The summary exposes both the paper-facing aliases (`hs_v5`, `hs_v10`,
`hs_cf5`, `hs_cf10`) and SHOW's exact released key names
(`xy_scale_mse_p5_95`, `xy_scale_mse_p10_90`,
`chamfer_norm_pct_p5_95`, `chamfer_norm_pct_p10_90`).

For strict comparability, the default `invalid_policy: show_zero` reproduces
the release: an empty rendered surface or empty human-region point set
contributes zero and remains in the concatenated frame mean.  Invalid counts
are still reported so an artificially improved score is visible.  The safer
but non-official `skip_nan` policy is available as an explicit audit override.

The paper's Eq. 19 differs from the released code's HS-CF direction and
normalizer.  The optional `compute_paper_audit` path exists only for diagnosis
and is disabled by default; it is not part of the project's Table 3 result.

## Tensor and coordinate contract

For every matched person-frame:

- SMPL vertices: `[V, 3]`, `float32`, camera coordinates, meters for metric
  model branches.
- Scene depth: `[H, W]`, `float32`, positive camera `z`, same scale as SMPL.
- Intrinsics: `[3, 3]`, calibrated for the depth resolution.
- Human mask: by default a dense GT SMPL projection at `[H, W]`; optionally an
  independent SAM2 patch mask resized with nearest-neighbour interpolation.
- Scene points: depth pixels inside the human mask, unprojected to camera XYZ.
- Depth validity: the human mask is intersected with the project's
  `depth_conf > 0.05`, matching SHOW's released point-extraction threshold.
- Visible body points: PyTorch3D-rasterized SMPL surface by default.

The 5/10 variants retain scene points within the 5–95% or 10–90% depth
percentile interval, matching the release's implemented filtering axis.

`vertex_zbuffer` is available only as a dependency-free diagnostic fallback.
It keeps the nearest SMPL vertex per pixel and is not numerically comparable to
the rasterized Table 3 protocol.

## Files

- Metric core: `vggt_omega/evaluation/human_scene_consistency.py`
- 3DPW evaluator: `scripts/eval/evaluate_show_human_scene_3dpw.py`
- Experiment configuration: `configs/eval_show_human_scene_3dpw.yaml`
- Server launcher: `scripts/eval/evaluate_show_human_scene_3dpw.sh`
- Numerical smoke test: `scripts/smoke/check_show_human_scene_metrics.sh`
- Unified 3DPW/EMDB-1 support evaluator:
  `scripts/eval/evaluate_show_human_scene_hmr4d.py`

## Server execution

Local repository paths:

- `C:\Users\ROG\PycharmProjects\vggt-omega\scripts\eval\evaluate_show_human_scene_3dpw.sh`
- `C:\Users\ROG\PycharmProjects\vggt-omega\scripts\smoke\check_show_human_scene_metrics.sh`

Corresponding server paths:

- `/home/zhw/lab_users/xyb/home/projects/vggt-human/scripts/eval/evaluate_show_human_scene_3dpw.sh`
- `/home/zhw/lab_users/xyb/home/projects/vggt-human/scripts/smoke/check_show_human_scene_metrics.sh`

First run the dependency-light numerical test:

```bash
bash scripts/smoke/check_show_human_scene_metrics.sh
```

No SAM2 cache is required by the default GT-projection configuration.  If the
optional `sam2_patch` mask source is selected, prepare it first:

```bash
bash scripts/preprocess/prepare_3dpw_sam2_patch_masks.sh
```

Then run a small end-to-end sample.  `CHECKPOINT` must be a checkpoint that is
loadable on top of the architecture selected by `MODEL_CONFIG`:

```bash
CHECKPOINT=/absolute/path/to/checkpoint.pt \
MAX_SAMPLES=8 \
DEVICE=cuda:0 \
bash scripts/eval/evaluate_show_human_scene_3dpw.sh
```

Full evaluation:

```bash
CHECKPOINT=/absolute/path/to/checkpoint.pt \
DEVICE=cuda:0 \
bash scripts/eval/evaluate_show_human_scene_3dpw.sh
```

For the paired Table 3 datasets, first prepare the project-local RGB frame
layout (no SAM2 run is involved).  The extractor prefers
`hmr4d_support/videos/*.mp4` when present and otherwise automatically reads the
native layouts supplied here:

- EMDB: `<emdb_root>/P*/<sequence>/images/*.{jpg,png}`
- 3DPW: `<3dpw_root>/imageFiles/<sequence>/*.{jpg,png}`

Native image sequences are not re-encoded: the script creates ordered
project-local symbolic links and falls back to copying only when symbolic links
are unavailable.  This avoids duplicating large EMDB JPEG sequences as PNGs.

```bash
DATASET=emdb1 bash scripts/preprocess/extract_hmr4d_eval_frames.sh
DATASET=3dpw bash scripts/preprocess/extract_hmr4d_eval_frames.sh
```

Then evaluate the same project checkpoint on each dataset:

```bash
DATASET=emdb1 CHECKPOINT=/absolute/path/to/checkpoint.pt DEVICE=cuda:0 \
  bash scripts/eval/evaluate_show_human_scene_hmr4d.sh

DATASET=3dpw CHECKPOINT=/absolute/path/to/checkpoint.pt DEVICE=cuda:0 \
  bash scripts/eval/evaluate_show_human_scene_hmr4d.sh
```

Or produce both datasets and the final Table-3-style JSON/CSV in one run:

```bash
CHECKPOINT=/absolute/path/to/checkpoint.pt \
DEVICE=cuda:0 \
bash scripts/eval/evaluate_show_human_scene_table3.sh
```

For a short end-to-end check before the full run, add `MAX_WINDOWS=8`.  Remove
that limit for paper numbers.

Configured server roots are `/home/zhw/xyb_space/emdb/hmr4d_support` and
`/home/zhw/xyb_space/3DPW/hmr4d_support`.  Frame outputs remain project-local
under `outputs/preprocess/hmr4d_eval_frames/`.

Useful overrides:

```bash
MODEL_CONFIG=configs/infer_smpl_hsi_v3_trstr_spatial.yaml \
SAM2_PATCH_MASKS_ROOT=outputs/preprocess/3dpw_sam2_patch_masks \
OUTPUT_DIR=outputs/eval/show_human_scene_3dpw_trstr_v3 \
CHECKPOINT=/absolute/path/to/checkpoint.pt \
bash scripts/eval/evaluate_show_human_scene_3dpw.sh
```

For an environment without PyTorch3D, a non-paper-comparable diagnostic can be
run with `VISIBILITY_BACKEND=vertex_zbuffer`.

## Outputs

All results stay under the configured `outputs/eval/...` directory:

- `show_human_scene_summary.json`: branch means, valid counts, protocol and
  invalid-reason counts.  Its top-level `table3` block is the four-number
  project result from the primary (`refined`) branch.
- `show_human_scene_rows.csv`: one row per matched person-frame and branch.
- `outputs/eval/show_human_scene_table3/show_table3_summary.json`: combined
  3DPW/EMDB-1 four-metric table plus actual sequence/frame coverage.
- `outputs/eval/show_human_scene_table3/show_table3_summary.csv`: compact table
  for direct spreadsheet/LaTeX ingestion.

The default configuration reports only `refined`: this is the project's final
prediction, using `hsi_translation_depth` (or the explicit HSI scene-affine
fallback) together with refined camera-space SMPL.  `base` can be requested as
an optional ablation, but raw VGGT depth is normally not metric-aligned and its
score must not be presented as the project's SHOW result.

## Checkpoint contract

The evaluator first loads the configured VGGT baseline and then overlays
`CHECKPOINT`, matching the project's staged training convention.  With the
default inference model, a refined evaluation requires the checkpoint to
contain both `hsi_refinement_head.*` and `hsi_trstr_head.*`.  The script checks
these prefixes and that they match the built model, and aborts instead of
silently evaluating randomly initialized heads.  The project's HSI-scoped
TRSTR checkpoints are valid when their saved prefixes contain both heads; a
checkpoint containing only one earlier stage is not sufficient for the default
configuration.

The evaluator also rejects empty/unknown branches, unknown visibility
backends, invalid percentile ranges, non-positive point/chunk settings, and
output directories outside `outputs/`.

## Remaining risks

1. PyTorch3D and the project model/checkpoints are unavailable on the Windows
   inspection machine, so the full rasterized evaluation must be run on the
   Linux server.
2. The default GT-projected mask removes segmentation noise but uses annotation
   information and therefore must be reported as a GT-region variant.  The
   optional SAM2 patch-mask route is closer to SHOW's mask source but its lower
   mask resolution can change absolute values.
3. SHOW's released code and paper Eq. 19 disagree on HS-CF.  The default JSON
   `table3_show_code` block follows the released evaluator used for comparison.
4. EMDB-1 and the HMR4D-style 3DPW evaluator use the existing support labels
   and GT mesh projection; neither requires a per-person SAM2 cache.
