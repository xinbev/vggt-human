# Query Initialization

This module contains the learnable query initialization used by SAT-HMR.

## Concept

SAT-HMR keeps two learnable embeddings:

```python
self.refpoint_embed = nn.Embedding(num_queries, 4)
self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
```

- `tgt_embed`: content query, consumed by decoder self-attention and cross-attention.
- `refpoint_embed`: reference box query in inverse-sigmoid space. After `.sigmoid()`, it becomes normalized `(cx, cy, w, h)`.

## What To Modify

- Change `num_queries` for your maximum number of objects/people.
- Keep `query_dim=4` if you want DAB-DETR-style box queries.
- Use `random_refpoints_xy=True` if you want random fixed xy anchors at initialization.
