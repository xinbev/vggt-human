# Query Mechanism Reference

This folder extracts the query-based decoding design from SAT-HMR into a portable reference layout.

It is intended for AI agents or developers who want to migrate SAT-HMR's DETR/DAB-DETR-style query mechanism into another project.

## Recommended Structure

```text
reference_query_mechanism/
  README.md
  query_init/
    README.md
    query_initializer.py
  position/
    README.md
    reference_position_encoding.py
  decoder/
    README.md
    query_decoder.py
  bbox_update/
    README.md
    bbox_refinement.py
  denoising_optional/
    README.md
    denoising_queries.py
  integration_example/
    README.md
    query_pipeline_example.py
```

## Core Design

SAT-HMR follows a DAB-DETR-like query design:

1. Each object/person slot has a learnable content embedding `tgt_embed`.
2. Each object/person slot also has a learnable box reference embedding `refpoint_embed` with 4 values: `(cx, cy, w, h)` in inverse-sigmoid space.
3. The decoder converts reference boxes into sine positional embeddings and uses them as query position.
4. Each decoder layer updates hidden states through self-attention, cross-attention, and FFN.
5. BBox heads refine reference boxes layer-by-layer using residual updates in inverse-sigmoid space.
6. The final decoder hidden states can feed task heads such as SMPL pose/shape/camera heads.

## Core Migration Path

For inference-only migration, start with:

```text
query_init/query_initializer.py
position/reference_position_encoding.py
decoder/query_decoder.py
bbox_update/bbox_refinement.py
integration_example/query_pipeline_example.py
```

For training with denoising queries, also inspect:

```text
denoising_optional/denoising_queries.py
```

## Target-Project Adjustments

You usually need to modify these items after copying this folder:

- `num_queries`: maximum number of people/objects your model can represent.
- `hidden_dim`: must match your encoder feature dimension or feature projection output dimension.
- `query_dim`: SAT-HMR uses 4, meaning reference queries are boxes `(cx, cy, w, h)`.
- `memory` format: this reference expects flattened encoder features `(sum_tokens, hidden_dim)` and `memory_lens` per image.
- `pos_embed`: this reference expects flattened encoder positional embeddings with the same shape as `memory`.
- `bbox heads`: if your project does not predict boxes, you can keep reference updates only as an internal attention prior or remove `bbox_update/`.

## Source Mapping

This reference is derived from these SAT-HMR locations:

```text
models/sat_model.py
  - refpoint_embed / tgt_embed initialization
  - optional random reference xy initialization
  - concatenating denoising queries before regular matching queries
  - decoder call with flattened query tensors
  - bbox prediction from reference updates

models/decoder.py
  - TransformerDecoder
  - XformerDecoder
  - XformerDecoderLayer
  - query sine embedding from reference boxes
  - reference-box iterative update

models/position_encoding.py
  - position_encoding_xy

models/dn_components.py
  - prepare_for_cdn
  - dn_post_process
```
