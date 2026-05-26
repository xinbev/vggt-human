# Query Decoder

This module contains a portable PyTorch decoder that mirrors SAT-HMR's query flow.

The original SAT-HMR decoder uses `xformers.memory_efficient_attention`. This reference intentionally uses standard PyTorch `nn.MultiheadAttention` to make migration easier.

## Input Contract

```text
memory:            (sum_tokens, hidden_dim)
memory_lens:       list[int], number of tokens for each image
memory_pos:        (sum_tokens, hidden_dim)
tgt:               (batch_size, num_queries, hidden_dim)
refpoint_embed:    (batch_size, num_queries, 4), inverse-sigmoid reference boxes
self_attn_mask:    optional (num_queries, num_queries), bool mask
```

## Output Contract

```text
hidden_states: (num_layers, batch_size, num_queries, hidden_dim)
references:    (num_layers, batch_size, num_queries, 4)
```

## What To Modify

- Replace this decoder with your project's own transformer decoder if you already have one.
- Keep the reference-position encoding and iterative bbox update logic if you want the SAT-HMR/DAB-DETR query behavior.
