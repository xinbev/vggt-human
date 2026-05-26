# SMPL Regression Heads

This directory contains the portable output heads used to regress SMPL parameters from decoder hidden states.

## Input Contract

`SMPLRegressionHead.forward` expects decoder hidden states:

```text
hidden_states: (num_decoder_layers, batch_size, num_queries, hidden_dim)
```

## Output Contract

The module returns a dictionary with final-layer outputs:

```text
pred_poses:   (B, Q, 24 * 3) axis-angle pose
pred_betas:   (B, Q, 10)
pred_confs:   (B, Q, 1)
pred_cam:     (B, Q, 3)
```

If `return_aux=True`, it also returns all intermediate decoder-layer predictions.

## What To Modify

- `hidden_dim`: must match your decoder hidden dimension.
- `num_decoder_layers`: must match the first dimension of your decoder output.
- `num_poses`: usually 24 for SMPL.
- `dim_shape`: usually 10 for SMPL betas.
- `mean_pose_6d` and `mean_shape`: pass the arrays from your own `smpl_mean_params.npz`.
