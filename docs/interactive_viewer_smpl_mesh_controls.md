# Interactive SMPL And Environment Mesh Viewer

## Task Goal

Add viewer-only controls to the existing Viser sequence viewer so a selected
SMPL body can be translated interactively, and the depth-derived environment can
be shown as a surface mesh instead of only a point cloud.

## Baseline Behavior

The baseline viewer in `scripts/vis/serve_nlf_hsi_vggt_sequence_viewer.py`:

- decodes SMPL vertices from `pred_poses`, `pred_betas`, and
  `pred_transl_cam` or HSI-refined equivalents;
- transforms SMPL vertices and depth points into the shared world frame;
- displays depth as Viser point clouds and SMPL as Viser mesh nodes.

Model predictions and `run_summary.json` geometry were not edited by the
viewer.

## New Behavior

The viewer now adds:

- `Environment Display`: `points`, `mesh`, or `both`;
- depth-grid surface mesh construction for raw and HSI depth, using
  `--env-mesh-depth-edge-rtol` to break faces across depth discontinuities;
- environment mesh nodes are built lazily when `mesh` or `both` is selected,
  so the viewer starts with the baseline point-cloud load;
- environment mesh uses Viser simple mesh nodes with a representative color for
  runtime stability;
- SMPL click/dropdown selection with `SMPL dX`, `SMPL dY`, and `SMPL dZ`
  viewer-only translation offsets;
- optional Viser transform controls when the installed Viser version supports
  `add_transform_controls`;
- edit scope: selected frame only or same track across all frames;
- `Save SMPL Offsets`, writing nonzero offsets to
  `outputs/vis/.../smpl_edit_offsets.json` by default.

## Coordinate And Shape Notes

- SMPL mesh vertices are already stored in world coordinates as `[V, 3]`.
- The viewer translation offset is a world-frame XYZ offset applied to the
  Viser scene node, not to the model output tensors.
- Environment mesh vertices are sampled from depth with the same stride and
  max-depth clipping used by point clouds.
- Faces are generated on the sampled depth grid; cells with large relative
  depth jumps are skipped.

## Server Usage

Local script path:

`scripts/vis/serve_nlf_hsi_vggt_sequence_viewer.sh`

Server project path:

`/home/zhw/lab_users/xyb/home/projects/vggt-human/scripts/vis/serve_nlf_hsi_vggt_sequence_viewer.sh`

Run on server:

```bash
bash scripts/vis/serve_nlf_hsi_vggt_sequence_viewer.sh
```

Outputs remain under the configured `outputs/vis/...` directory. Saved manual
SMPL offsets default to `smpl_edit_offsets.json` in that viewer output folder.

## Validation

Completed locally:

- `python -m py_compile scripts/vis/serve_nlf_hsi_vggt_sequence_viewer.py`
- `python -m py_compile scripts/vis/serve_nlf_roi_id_tracking_v2_viewer.py`

Not completed locally:

- full import/runtime smoke, because the Windows local environment does not
  have `torch`;
- interactive Viser verification, which needs the Linux server environment,
  checkpoints, body model assets, and viewer dependencies.

## Risks

- Environment mesh is a depth-map surface mesh, not a fused watertight scene
  mesh.
- Dense mesh mode can be heavy at low stride; use stride 4 or 6 for long
  sequences.
- Viser transform controls are version-dependent, so sliders remain the
  compatibility path.
