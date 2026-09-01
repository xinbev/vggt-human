# ICLR Architecture Figure — HD Reproduction Specification

## Canvas

```text
Working canvas:  1536 × 1024 px
Paper export:    12.8 × 8.53 in at 120 dpi preview
Vector master:   1536 × 1024 logical units in Figma/Illustrator
Outer margin:    32 px
Panel gap:       16 px
Corner radius:   20 px outer panels, 12 px inner cards
```

## Layout grid

```text
Top overview panel:      x=32,  y=32,  w=1472, h=440
Bottom HSI panel:        x=32,  y=488, w=592,  h=456
Bottom refinement panel: x=640, y=488, w=864,  h=456
Palette strip:           x=32,  y=960, w=1472, h=48
```

The top panel uses a 12-column grid. The bottom uses a 5:7 split. Keep every
connector on a horizontal or vertical axis except the short `K -> NLF`
conditioning arrow.

## Top overview widths

```text
RGB input:              100 px
Parallel encoding:      300 px
Evidence interface:     190 px
HSI scale:              170 px
TRSTR spatial:          270 px
Temporal stabilizer:    190 px
Output:                 140 px
Internal gap:            18 px
```

## Visual grammar

- Outer panel border: `#2F3438`, 2 px.
- Inner module border: semantic color darkened by 18%, 1.5 px.
- Main arrows: `#2F3438`, 2 px, 9 px arrowhead.
- Auxiliary arrows: 1.25 px; dashed only for conditioning or iterative loops.
- Module fill opacity: 18–24%; avoid opaque blocks.
- No drop shadow. No gradient. No glow.
- Use 2D wireframes, token strips, matrices, depth tiles, and formulas.

## Palette

```text
Background     #F7F6F2
Dusty blush   #D9ACA6   NLF / human state
Muted sage    #AFCAB2   HSI / metric coupling
Powder cyan   #A9CFD8   camera / temporal
Periwinkle    #99ADD4   TRSTR / regional evidence
Lavender      #CBBAD8   VGGT / scene tokens
Warm cream    #E7DAAC   analytic gauge / equations
Yellow accent #E9D36E   gates / selected states
Charcoal      #2F3438   outlines / text
```

## Typography

Recommended reproducible fonts:

```text
Module headings: Inter SemiBold, 21 px
Panel headings:  Inter Bold, 23 px
Body labels:     Inter Medium, 15–17 px
Math:            STIX Two Math or Cambria Math, 17–20 px
Caption labels:  Inter Bold, 19 px
```

Do not use text smaller than 14 px on the 1536 px master.

## Redraw order

1. Draw the four outer panel rectangles.
2. Place the top-level modules using the width table; do not add arrows yet.
3. Build internal components from repeated primitives: token chip, matrix cell,
   depth tile, SMPL wireframe, gate diamond, and rounded label card.
4. Add the main dataflow arrows.
5. Add auxiliary arrows: `K -> NLF`, HSI anchor projections, TRSTR re-probe,
   and temporal identity lanes.
6. Add formulas and tensor/interface labels.
7. Apply semantic colors at low opacity.
8. Add the palette strip, then export PNG and PDF/SVG from the vector master.

## Scientific invariants

- RGB feeds VGGT and NLF in parallel.
- Camera `K` from VGGT conditions NLF.
- HSI receives raw depth and metric SMPL.
- TRSTR receives metric depth and metric SMPL and changes translation only.
- Temporal stabilization follows spatial alignment and is identity-separated.
- `θ` and `β` remain fixed through TRSTR.

