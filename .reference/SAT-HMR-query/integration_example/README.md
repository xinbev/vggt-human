# Integration Example

This directory shows how to connect the query mechanism to a generic encoder feature tensor.

## Expected Flow

```text
encoder tokens + pos embeds
        |
QueryInitializer -> tgt/refpoint queries
        |
optional denoising queries during training
        |
QueryDecoder with reference-position encoding
        |
BBoxRefinementHeads -> pred_boxes
        |
SMPL/head/task heads consume decoder hidden states
```

## What To Modify

- Replace `memory` and `pos_embed` with your encoder outputs.
- If your encoder already outputs `(B, N, C)`, flatten it to `(B * N, C)` and set `memory_lens=[N] * B`.
- Connect `hidden_states` to your task heads, such as the SMPL head in `reference_smpl_regression`.
