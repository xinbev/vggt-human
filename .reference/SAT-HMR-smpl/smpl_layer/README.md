# SMPL Layer Wrapper

This directory contains a minimal `smplx` wrapper compatible with the regression head.

## Dependency

Install `smplx` in the target project:

```bash
pip install smplx
```

## Expected Input Format

`SMPLLayer.forward` expects:

```text
poses:  (N, 24 * 3) or (N, 24, 3), axis-angle
betas:  (N, 10)
```

It returns:

```text
vertices: (N, 6890, 3)
joints:   (N, J, 3)
```

## What To Modify

- Set `model_path` to your SMPL asset directory.
- If your project uses SMPL-X/SMPL-H instead of SMPL, adjust the `smplx.create(..., model_type=...)` call and output dimensions.
