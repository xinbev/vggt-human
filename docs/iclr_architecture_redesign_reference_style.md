# ICLR Architecture Redesign — Reference-Derived Visual System

## Why the previous composition failed

The previous six-column pipeline exposed the implementation order but did not
create a visual thesis. Every module had similar weight, the eye moved only
left-to-right, and the central contributions (metric calibration and regional
human-scene translation refinement) did not own the composition.

## Reference-derived visual language

The supplied paper figure uses:

- a warm white canvas with very light textured panel fills;
- thin dark outlines and rounded dashed group boundaries;
- multiple low-saturation semantic colors rather than one color per large box;
- a large overview panel plus vertically stacked mechanism insets;
- small matrices, token strips, geometric glyphs, and curved arrows to create
  local rhythm;
- asymmetry: the overview occupies most of the canvas while the detail insets
  form a narrower right column.

The redesign borrows this visual grammar without copying its method content.

## Extracted palette

Representative colors sampled from the reference:

```text
#F7F6F2  warm paper background
#D9ACA6  dusty blush
#AFCAB2  muted sage
#A9CFD8  powder cyan
#99ADD4  periwinkle blue
#CBBAD8  soft lavender
#E7DAAC  warm cream
#E9D36E  restrained yellow accent
#2F3438  charcoal ink
```

## Candidate A — Infinity convergence

The overview follows a sideways infinity rhythm rather than a row:

```text
                VGGT scene arc
             /                  \
RGB portal --                    -- HSI metric coupling core
             \                  /                 \
                NLF human arc                      TRSTR body field
                                                        \
                                                  temporal orbit -> output
```

- Upper arc: scene evidence (`K`, raw depth, scene tokens).
- Lower arc: metric human evidence (`pose`, `shape`, `translation`).
- Crossing/core: analytic coarse gauge plus HSI residual calibration.
- Right body field: 96 regional probes and gated translation votes.
- Temporal output is a curved track orbit, not another rectangular stage.

## Candidate B — Central body field with satellites

The segmented SMPL body is the dominant central object. VGGT, NLF, HSI scale,
regional probes, and temporal tracks are arranged as satellites around it.
This is more iconic and visually memorable, but the exact execution order is
slightly less immediate than Candidate A.

## Right-side mechanism insets

Both candidates use three stacked dashed insets:

1. HSI metric gauge: anchor-depth ratios, robust coarse scale, residual affine.
2. TRSTR regional consensus: 96 regions, multi-scale probes, vote/gate/variance.
3. Temporal conservative fusion: neighbour-only motion proposal, capped gate,
   identity-separated trajectories.

## Palette policy

The new reference-derived palette replaces the earlier user-provided palette
for this redesign. The final figure keeps a compact palette strip at the
bottom, as requested earlier in the conversation.

