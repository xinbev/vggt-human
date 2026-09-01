# ICLR Figure Design — Overview with Two Mechanism Insets

## Structural mapping from the reference

The supplied reference uses a strong scientific hierarchy:

1. one full-width system overview;
2. one lower-left construction/detail panel;
3. one lower-right control/refinement panel.

The system is mapped into that structure as follows:

```text
Top:    RGB -> {VGGT scene evidence, NLF metric SMPL}
             -> HSI scale -> TRSTR spatial -> temporal -> output

Bottom-left: analytic coarse gauge + HSI residual metric calibration

Bottom-right: TRSTR regional voting + conservative track-wise temporal fusion
```

The design retains the muted reference-derived palette while avoiding large
hero figures, atmosphere, ribbons, and promotional rendering.

