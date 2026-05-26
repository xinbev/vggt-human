# Configuration

This module contains SMPL model paths and mean-parameter paths.

## What To Modify

Update `SMPL_MODEL_PATH` and `SMPL_MEAN_PARAMS_PATH` to point to your local SMPL assets.

Expected files usually include:

```text
SMPL_MODEL_PATH/
  smpl/
    SMPL_NEUTRAL.pkl or equivalent smplx-compatible files
    smpl_mean_params.npz
    body_verts_smpl.npy              # optional, used by SAT-HMR utilities
    J_regressor_h36m_correct.npy     # optional, used by evaluation utilities
```

`smpl_mean_params.npz` should contain:

```text
pose   shape: (24 * 6,) or compatible mean 6D pose vector
shape  shape: (10,)
```
