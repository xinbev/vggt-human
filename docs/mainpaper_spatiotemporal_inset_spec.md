# Main-paper Spatial–Temporal Inset

## Abstraction level

The overview contains two black boxes: `TRSTR` and `Temporal`. This inset opens
each box by exactly one level. It shows model components, but omits tensor
shapes, hidden dimensions, loss terms, and implementation-specific formulas.

## Horizontal proportions

TRSTR row:

```text
Input 12% | Region Construction 22% | Interaction Encoder 22%
| Correction Heads 18% | Robust Aggregation 18% | Output 8%
```

Temporal row:

```text
Track Window 28% | Motion Proposal 28% | Conservative Fusion 28% | Output 16%
```

## Required labels

TRSTR:

```text
Metric SMPL
Metric scene depth
Region Construction
96 body regions
multi-scale probes
Human–Scene Interaction Encoder
region geometry | local scene | owner mask
MLP | interaction token
Regional Correction Heads
vote | gate | uncertainty
Robust Aggregation
weighted fusion | person gate | 2× re-probe
refined translation
pose and shape unchanged
```

Temporal:

```text
Track Window
Motion Proposal
Masked Temporal MLP
neighbors only
Conservative Fusion
Gate MLP
alpha <= 0.5
Stable sequence
```

## Visual rules

- Dotted callouts connect the top-level TRSTR/Temporal blocks to the inset.
- TRSTR uses a pale periwinkle container; Temporal uses pale cyan.
- The interaction encoder is the only sage-highlighted component because it is
  the newly exposed mechanism.
- Use one feedback loop only: `2× re-probe` from aggregation to region
  construction.
- Do not add layer dimensions or per-tensor shapes in the main-paper version.

