# TRSTR–Temporal Internal Figure Specification

## Purpose

This lower-row figure is an engineering expansion of the top-level
`TRSTR Spatial Refinement -> Temporal Stabilizer` blocks. It must change
abstraction level: the top row shows system modules; the lower row shows the
actual tensor and network dataflow inside those modules.

## Panel B — TRSTR internal dataflow

Recommended width: 60%.

```text
pose6d [B,S,Q,144] --+
betas  [B,S,Q,10]  --+--> frozen SMPL decode --> V [F,Q,6890,3]
transl [B,S,Q,3]   --+                           |
                                                  v
                                        SMPL Region Bank
                                 centers [F,Q,96,3]
                                 reps    [F,Q,96,8,3]
                                                  |
metric depth [F,H,W] -----------------------------+
K [F,3,3] ----------------------------------------+--> RegionalSceneProbe
person_valid [F,Q] -------------------------------+
                                                       |
                              probe tokens [FQ,96,8,16]
                              valid ratios [FQ,96,8]
                              projected uv / owner ratios / radius
                                                       |
                                                       v
                              concatenate geometric + probe features
                                        x_r [FQ,96,178]
                                                       |
                                             MLP 178 -> 256 -> 256
                                                       |
                           +---------------------------+-------------------+
                           |                           |                   |
                    vote head                    gate head           logvar head
                    [FQ,96,3]                   [FQ,96,1]            [FQ,96,1]
                           +---------------------------+-------------------+
                                                       |
                         w_r = valid_r * sigmoid(g_r) * exp(-logvar_r)
                         delta_tau = sum(w_r * vote_r) / sum(w_r)
                                                       |
                         pooled person hidden -> person gate [FQ,1]
                                                       |
                              tau^(k+1) = tau^k + gate_person * delta_tau
                                                       |
                                          shared 2-iteration loop
                                                       v
                                             tau* [B,S,Q,3]
```

The `RegionalSceneProbe` inset should show only the essential engineering
logic: project region centers/representatives, rasterize human depth and owner,
split self-surface versus environment samples, reject other-human samples, and
attention-pool eight probe tokens (four scales x two channels).

## Panel C — Temporal stabilizer internal dataflow

Recommended width: 40%.

Translation branch:

```text
O [B,9,3] + valid [B,9]
      |
      +--> select t-2,t-1,t+1,t+2; center t is masked
      |
      +--> midpoint + relative neighbours [B,12]
      |        |
      |        +--> Proposal MLP 12 -> 128 -> 128 -> 3
      |                  tanh * 0.25 m
      |                  -> M_t [B,3]
      |
      +--> gate features:
               [O_t-M_t, abs(O_t-M_t), neighbour velocity] [B,9]
                         |
                         +--> Gate MLP 9 -> 64 -> 64 -> 1
                                  sigmoid * 0.5 -> alpha_t
                         |
                         v
               X*_t = O_t + alpha_t (M_t - O_t)
```

Invalid context and boundary frames are strict no-ops. Track IDs are separated
before the stabilizer and never mixed. If the pose branch is included, show it
as a small parallel SO(3) branch with shared per-joint weights; do not expand it
to the same visual weight as the translation branch.

## Visual relation to the overview

Place two small source chips above the panels:

```text
[TRSTR Spatial Refinement]           [Temporal Stabilizer]
           | zoom                               | zoom
           v                                    v
  (B) TRSTR internal dataflow        (C) Temporal internal dataflow
```

This is the only callout needed. The lower panels should not repeat input
images, output scene renders, or high-level system arrows.

