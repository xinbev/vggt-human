# ICLR Mixed-Abstraction Figure Design

## Principle: focus + context

The figure should not assign equal detail to every component. Information
density follows contribution importance:

```text
Context / established components: RGB, VGGT, NLF, output
    -> abstract capsules with named outputs only

Core contributions: HSI scale, TRSTR
    -> expanded mechanisms, equations, tokens, and invariants

Post-alignment temporal module
    -> medium detail: track window, proposal, bounded fusion
```

Color saturation also encodes detail. Context modules use near-white tinted
fills; contribution modules use the full pastel palette; temporal constraints
use a compact cyan strip.

## Candidate A — Contribution focus

A shallow context band feeds two large detailed contribution panels. Temporal
fusion forms a medium-sized constraint strip beneath TRSTR.

## Candidate B — Overview with zoom callouts

An extremely compact full pipeline sits at the top. Dashed callout lines expand
only HSI and TRSTR below. This gives the clearest separation between summary
and mechanism.

