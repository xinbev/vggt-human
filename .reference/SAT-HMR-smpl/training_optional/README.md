# Optional Training References

This directory contains simplified references for training a multi-query SMPL regressor.

Use these only if your target project trains multiple person queries and needs Hungarian matching plus losses.

## Files

- `matcher_reference.py`: matches predicted queries to ground-truth people using confidence, bbox, GIoU, and 2D keypoint costs.
- `losses_reference.py`: computes confidence focal loss and L1 losses for pose, betas, 2D joints, 3D joints, and depth.

## Required Target Keys

The references assume each target dictionary can contain:

```text
boxes:      (N, 4), normalized cxcywh boxes
poses:      (N, 72), SMPL axis-angle pose
betas:      (N, 10), SMPL shape
j3ds:       (N, J, 3), 3D joints
j2ds:       (N, J, 2), image-space 2D joints
j2ds_mask:  (N, J, 2), visibility mask
depths:     (N, 2), depth targets
detect_all_people: bool, whether unmatched queries are valid negatives
```

## What To Modify

- If your boxes are xyxy instead of cxcywh, update the box conversion functions.
- If your 2D joints are normalized already, remove `j2ds_norm_scale` division.
- If your dataset has a different SMPL joint count or root index, adjust joint slicing and root alignment.
