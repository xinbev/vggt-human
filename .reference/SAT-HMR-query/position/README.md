# Reference Position Encoding

This module turns normalized reference boxes into sine embeddings.

SAT-HMR uses the reference query `(cx, cy, w, h)` as a positional prior:

```text
reference box -> sine/cosine embedding -> MLP -> query_pos
```

The decoder then uses `query_pos` in self-attention/cross-attention and uses a projected sine embedding in cross-attention.

## What To Modify

- Keep this module if your query reference is a box `(cx, cy, w, h)`.
- If your query reference is only a point `(cx, cy)`, remove the width/height encoding and update `query_dim` accordingly.
