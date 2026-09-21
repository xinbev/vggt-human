# RICH Physical-Grounding Evaluation Protocol

## Goal

Reproduce only the physical-grounding half of UniCon3R Table 3 on RICH:

- Collision Ratio (`Coll.`, %)
- Penetrate (`Pen.`, cm)
- Float (cm)
- Penetration Max (`P.Max`, cm)

The existing VGGT-Omega paths remain unchanged. The evaluator must be added as
an optional evaluation path and must not replace the current baseline metrics.

## References

- UniCon3R, Table 3 and Section 5.2:
  `.paper/base_pdf/UniCon3R.pdf`
- HuMoS, Section 3.3 and Section 4.2:
  `.paper/base_pdf/HuMoS.pdf`
- HuMoS reference implementation:
  `.paper/base_projects/humos/humos/src/model/metrics.py`

The files under `.paper/` are read-only references. Production code must not
import from them.

## Confirmed From HuMoS

For each frame, HuMoS first computes the height of the lowest body-mesh vertex:

```text
h_t = min_v z[t, v]
```

HuMoS assumes a horizontal ground at `z = 0` and uses a `0.005 m` tolerance.
Its released implementation computes:

```text
penetrating frames: h_t < -0.005
floating frames:    h_t >= 0.005
Penetrate:           mean(abs(h_t)) over penetrating frames
Float:               mean(abs(h_t)) over floating frames
```

Frames inside the 5 mm dead zone contribute to neither conditional mean. Both
reported distances are converted from metres to centimetres.

## UniCon3R Adaptation

UniCon3R explicitly states that it adapts the HuMoS grounding protocol, uses a
robustly estimated ground height, and retains the 5 mm tolerance. It additionally
reports Collision Ratio and Penetration Max.

The most direct extension consistent with HuMoS is:

```text
d_t = min_v z[t, v] - ground_height
Coll. = 100 * count(d_t < -0.005) / count(valid frames)
Pen.  = 100 * mean(-d_t | d_t < -0.005)
Float = 100 * mean( d_t | d_t >= 0.005)
P.Max = 100 * max(-d_t | d_t < -0.005)
```

This extension is a reproduction hypothesis, not a formula printed by the
UniCon3R paper. The evaluator must label it as such until author code or an
official clarification is available.

## Details Not Published

The available UniCon3R paper does not specify:

1. How the robust ground height is estimated from the reconstructed scene.
2. Which reconstructed scene points are retained before ground estimation.
3. Whether one ground height is fitted per frame, window, camera view, or action
   recording.
4. Whether final numbers are pooled over all frames or averaged per sequence.
5. The exact list of the 40 moving-camera RICH recordings used by UniCon3R.

The paper states that source code and models will be released upon acceptance,
so the exact private evaluation implementation is not currently available in
the supplied reference material.

## Available Data

Server roots:

```text
/home/zhw/xyb_space/RICH/official
/home/zhw/xyb_space/RICH/hmr4d_support
```

The asset audit currently reports:

```text
RICH camera views: 191
Unique recordings: 50
Selected frames: 106446
Sequences with issues: 0
```

This proves that the downloaded files are complete for the current adapter. It
does not establish which 40 recordings were used for UniCon3R Table 3.

## Evaluation Model Path

The current project inference path is a two-checkpoint cascade rather than a
single standalone checkpoint:

```text
VGGT + NLF body initialization
  -> analytic coarse scene scale
  -> v3 HSI residual scale and bias overlay
  -> Stage-2 human-scene translation alignment
```

The accepted server checkpoints are:

```text
outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt
outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt
```

The physical-grounding evaluator must reproduce this load order. Evaluating
only the Stage-2 checkpoint without the scale overlay would not match the
current inference system.

## Implementation Order

1. Verify the evaluation checkpoint and SMPL body-model assets.
2. Run one camera view as an inference smoke test and save world-frame body
   vertices plus the reconstructed scene representation.
3. Validate the coordinate convention, gravity axis, metre scale, and body/scene
   alignment visually and numerically.
4. Implement both frame-pooled and sequence-mean aggregation, clearly named in
   the output, while keeping the 5 mm tolerance fixed.
5. Compare multiple robust ground estimators on the smoke sequence before
   launching the full RICH evaluation.
6. Run all 191 camera views only after the 40-recording subset is identified or
   report the evaluated recording list explicitly as a protocol deviation.

All evaluation outputs must be written under
`outputs/eval/rich_physical_grounding/`.
