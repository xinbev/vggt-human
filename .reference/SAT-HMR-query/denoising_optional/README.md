# Optional Denoising Queries

This module contains SAT-HMR/DINO-style denoising query preparation.

Use it only during training. It prepends noisy ground-truth queries before regular matching queries:

```text
[denoising queries][regular learnable queries]
```

The attention mask prevents regular matching queries from seeing denoising queries and prevents denoising groups from leaking into each other.

## Required Target Keys

Each target dictionary should contain:

```text
labels: (N,)
boxes:  (N, 4), normalized cxcywh
```

If `tgt_embed_type='params'`, each target also needs:

```text
poses: (N, 72)
betas: (N, 10)
```

## What To Modify

- Use `tgt_embed_type='labels'` for normal object detection-style denoising.
- Use `tgt_embed_type='params'` if your query content should encode SMPL pose/betas like SAT-HMR supports.
