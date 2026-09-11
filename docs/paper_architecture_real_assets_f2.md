# Paper Architecture Real Assets From `f2.jpg`

## Purpose

This export replaces illustrative placeholders in the method figure with
outputs from one consistent real inference run:

```text
RGB -> VGGT camera/raw depth -> NLF metric SMPL
    -> analytic coarse scale -> HSI residual affine -> TRSTR spatial refinement
```

The source image is:

```text
assets/image/f2/f2.jpg
```

Server path:

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human/assets/image/f2/f2.jpg
```

## Which figure elements should be real

| Figure element | Real exported source | Notes |
| --- | --- | --- |
| RGB input | `00_input_rgb_original.png` | Exact `f2.jpg` content |
| VGGT depth | `01_raw_vggt_depth_*` | Raw, scale-ambiguous prediction |
| Camera evidence | `02_camera_intrinsics.json` | Exact processed-plane `K` |
| NLF detection | `03_nlf_detection_overlay.png` | Real detector boxes/confidence |
| Metric SMPL | `04_*_metric_smpl.png/.ply` | Real NLF pose, shape, translation |
| 24 body anchors | `05_*_24anchors_overlay.png` | Real HSI anchor projections |
| Paired scale evidence | `06_scale_hypotheses.csv` | Real SMPL-depth correspondence ratios |
| Metric-prior token strip | `06_scale_hypothesis_strip.png` | Real ratios and median `C_coarse` |
| Local scene probes | `07_*_local_scene_probes_triplet.png` | Three real coarse-depth neighborhoods |
| Anchor-token values | `07_*_anchor_residual_tokens.png` | Real body-scene depth residuals |
| Coarse depth | `08_coarse_metric_depth_*` | `C_coarse * D_vggt` |
| TRSTR region evidence | `09_*_trstr_*_matrix.png/.csv` | Real 96-region gate, uncertainty, vote, validity |
| Segmented SMPL | `09_*_trstr_segmented_smpl.png/.ply` | Real mesh with deterministic region assignment |
| TRSTR correction | `10_*_trstr_base_refined_overlay.png` | Base/refined translation overlay |
| HSI scalars | `11_hsi_scale_values.json` | `C_coarse`, `R_hsi`, `b_hsi` |
| Metric depth | `12_final_metric_depth_*` | Final HSI metric depth consumed by TRSTR |
| Before/after depth | `13_depth_raw_coarse_metric_triptych.png` | Figure-ready comparison |

Token rectangles, arrows, module backgrounds, attention icons, braces, and
palette swatches remain abstract visual grammar. They should not be presented
as measured tensors unless the caption says so.

## Temporal limitation

The current folder contains only `f1.jpg` and `f2.jpg`. A real 9-frame temporal
window therefore cannot be produced. The exporter writes:

```text
14_temporal_status.json
14_temporal_rgb_inventory.png
```

Do not turn the two frames into a synthetic 9-frame trajectory. To export real
temporal observations later, provide at least nine consecutive RGB frames and
the accepted temporal checkpoint. The architecture's circles and arrows can
remain abstract until those inputs exist.

## Server execution

Local launcher:

```text
scripts/vis/export_paper_architecture_real_assets.sh
```

Server launcher:

```text
/home/zhw/lab_users/xyb/home/projects/vggt-human/scripts/vis/export_paper_architecture_real_assets.sh
```

Run from the server repository:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
bash scripts/vis/export_paper_architecture_real_assets.sh
```

If the accepted checkpoint has a different path:

```bash
CHECKPOINT=/absolute/path/to/checkpoint.pt \
bash scripts/vis/export_paper_architecture_real_assets.sh
```

Optional HSI-only overlay:

```bash
CHECKPOINT=/absolute/path/to/trstr_checkpoint.pt \
SCALE_CHECKPOINT=/absolute/path/to/hsi_scale_checkpoint.pt \
bash scripts/vis/export_paper_architecture_real_assets.sh
```

Output:

```text
outputs/vis/paper_figure_assets/f2/
```

The `manifest.json` file records the source image, checkpoints, selected NLF
queries, coordinate conventions, scalar values, temporal availability, and all
generated filenames.

## Validation boundaries

Windows local validation can check syntax and helper rendering only. Full
inference requires the Linux server's CUDA environment, VGGT baseline, NLF
TorchScript model, SMPL assets, and accepted HSI/TRSTR checkpoint.

