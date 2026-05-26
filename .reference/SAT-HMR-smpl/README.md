# SMPL Regression Reference

This folder extracts the SMPL regression design from SAT-HMR into a portable reference layout.

It is intended for AI agents or developers who want to migrate the SMPL output-head pattern into another project without reading the entire SAT-HMR codebase.

## Recommended Structure

```text
reference_smpl_regression/
  README.md
  config/
    smpl_paths.py
    README.md
  heads/
    smpl_regression_head.py
    README.md
  smpl_layer/
    smpl_layer.py
    README.md
  geometry/
    rotation_conversions.py
    camera_projection.py
    README.md
  integration_example/
    decoder_to_smpl_example.py
    README.md
  training_optional/
    matcher_reference.py
    losses_reference.py
    README.md
```

## Core Migration Path

For inference-only migration, start with these files:

```text
config/smpl_paths.py
smpl_layer/smpl_layer.py
geometry/rotation_conversions.py
geometry/camera_projection.py
heads/smpl_regression_head.py
integration_example/decoder_to_smpl_example.py
```

For training with multiple person queries, also inspect:

```text
training_optional/matcher_reference.py
training_optional/losses_reference.py
```

## Design Summary

The SAT-HMR SMPL regression design has four important ideas:

1. A decoder produces hidden states with shape `(num_decoder_layers, batch_size, num_queries, hidden_dim)`.
2. Each decoder layer owns an independent `pose_head` and `shape_head` cloned from the same MLP architecture.
3. Pose and shape are predicted as residual refinements from `mean_pose` and `mean_shape`.
4. The pose head predicts 6D rotations, then converts them to SMPL axis-angle before calling SMPL.

## Target-Project Adjustments

You usually need to modify these items after copying this folder:

- `config/smpl_paths.py`: set `SMPL_MODEL_PATH` and `SMPL_MEAN_PARAMS_PATH` for your machine or config system.
- `heads/smpl_regression_head.py`: set `num_poses`, `dim_shape`, `num_decoder_layers`, and `hidden_dim` to match your model.
- `integration_example/decoder_to_smpl_example.py`: connect your own decoder output tensor to `SMPLRegressionHead`.
- `geometry/camera_projection.py`: adapt camera scale/depth conventions if your project uses a different camera parameterization.
- `training_optional/*`: adapt target dictionary keys if your dataset uses names other than `boxes`, `j2ds`, `j2ds_mask`, `poses`, `betas`, or `depths`.

## Source Mapping

This reference is derived from these SAT-HMR locations:

```text
models/sat_model.py
  - _get_clones
  - MLP
  - pose_head / shape_head / cam_head / conf_head
  - mean_pose / mean_shape residual refinement
  - process_smpl
  - final output dictionary

models/human_models/smpl_models.py
  - SMPL_Layer wrapper around smplx

utils/transforms.py
  - rot6d_to_axis_angle and rotation conversion helpers

models/matcher.py
  - Hungarian matcher for multi-query person matching

models/criterion.py
  - L1 losses for pose, betas, 2D joints, 3D joints, depths, and confidence focal loss
```
