# ICLR Scientific Architecture Layout Study

## Shared visual constraints

- Preserve the reference-derived muted palette.
- Use thin 2D line drawings, compact tensor strips, matrices, and formulas.
- Avoid hero illustrations, atmospheric depth, large human renders, ribbons,
  shadows, gradients, and poster-like empty space.
- Every large visual element must communicate an input, representation,
  operation, output, or constraint.
- Use modest module size differences; contributions are emphasized through
  grouping and placement rather than spectacle.

## Layout A — Dual-stream convergence

Two parallel observation streams converge into a central metric bus. The
spatial and temporal refinements continue to the right, while small mechanism
insets occupy a narrow column. This is the clearest balance between novelty
and conventional readability.

## Layout B — Three-tier hierarchy

The canvas is divided into observation extraction, metric coupling, and
spatial-temporal refinement bands. This is compact, highly scientific, and
well suited to explaining training/inference interfaces.

## Layout C — Orthogonal cross

HSI metric coupling is placed at the center, with VGGT, NLF, TRSTR, and
temporal refinement arranged around it. Orthogonal connectors make dependency
and ownership explicit without a left-to-right procession.

## Layout D — Modular matrix

A two-by-three module matrix is joined by a shared metric-geometry bus. Each
cell has a strict input/operator/output structure. This is the most restrained
and camera-ready option, but also the least visually expressive.

