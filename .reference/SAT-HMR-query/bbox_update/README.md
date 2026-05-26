# Bounding Box Refinement

This module contains the iterative bbox/reference update logic.

## Concept

SAT-HMR predicts box deltas from decoder hidden states, but applies them in inverse-sigmoid space:

```python
tmp = bbox_head(hidden)
tmp[..., :4] += inverse_sigmoid(reference)
new_reference = tmp[..., :4].sigmoid()
```

This makes the update residual in unconstrained space while final boxes stay normalized to `[0, 1]`.

## What To Modify

- Keep `output_dim=4` for normalized boxes `(cx, cy, w, h)`.
- If your model does not need explicit boxes, you can still keep reference updates internally or remove this module.
