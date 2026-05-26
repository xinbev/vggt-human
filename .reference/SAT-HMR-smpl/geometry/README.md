# Geometry Utilities

This directory contains the geometry functions needed by the SMPL regression head.

## Files

- `rotation_conversions.py`: converts 6D rotation representation to SMPL axis-angle pose.
- `camera_projection.py`: converts predicted camera parameters plus SMPL joints/vertices into camera-space vertices, projected 2D joints, depths, and translation.

## What To Modify

- If your project already uses PyTorch3D, Kornia, or another rotation conversion library, you can replace `rotation_conversions.py` with your existing implementation.
- If your project uses weak-perspective camera, full-perspective camera, or known translation directly, adapt `camera_projection.py` accordingly.
