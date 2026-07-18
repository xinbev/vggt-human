# HSI Geometry-First Grounding

## Objective

Correct global person float/penetration after the accepted Stage2 alignment,
without retraining VGGT, NLF, Stage1 scene scale, Stage2 translation alignment,
SMPL pose, or SMPL betas.

The previous `HSIContactRefineHead` remains available as a baseline. The new
mainline is the independent `HSIGroundingHead`.

## Forward Contract

GT curriculum:

```text
GT RGB features + GT depth + GT K + contact-root perturbed GT SMPL
    -> local support plane
    -> analytic root-normal candidate
    -> learned apply gate
```

Real bridge:

```text
VGGT depth/K -> frozen Stage1 scale -> NLF SMPL -> frozen Stage2 align
    -> local support plane
    -> analytic root-normal candidate
    -> learned apply gate
```

The analytic candidate is:

```text
support_signed = robust common-sign average of valid foot signed distances
delta_scalar = clamp(-support_signed, -0.12, 0.12)
candidate_delta = delta_scalar * support_normal
```

The strict teacher defines `abs(signed_distance) <= 2.5 cm` as normal contact.
The online candidate therefore uses the same 2.5 cm deadzone and produces zero
displacement inside it. A person-level correction is applied only when all
valid feet agree on the signed direction; a normal support foot plus a swing
foot is left unchanged.

The learned head cannot change candidate direction or magnitude. It predicts
only whether the candidate should be applied. Its target is positive only when
the candidate is measurably closer to GT translation than the base.

## Tensor Shapes

```text
pose_6d, betas, transl:       [B, S, Q, 144|10|3], fp32 on model device
depth:                       [B, S, H, W], metric in GT mode
intrinsics:                  [B, S, 3, 3]
sole/support values:         [B, S, Q, 2]
candidate/refined transl:    [B, S, Q, 3]
gate probability:            [B, S, Q, 1]
```

Camera coordinates use +Y down. Fitted plane normals are oriented upward, so
their Y component is negative. Camera intrinsics are never scaled with depth.

## Safety Properties

- Projected SMPL samples are dilated into an exclusion mask before plane fit.
- Foot centers must remain inside the corresponding person box.
- Plane RMSE, point count, depth range, slot confidence, and optional VGGT
  depth confidence control candidate validity.
- GT and real inference geometry are never mixed within a batch.
- Old Stage1/Stage2/contact checkpoints are read-only inputs. New outputs use
  separate `outputs/debug/hsi_stage3_grounding_*` and
  `outputs/train/hsi_stage3_grounding_*` directories.
- Real bridge checkpoint loading uses an explicit Stage2 resume plus a
  grounding-only overlay; saved checkpoints contain the combined frozen HSI
  modules for later inference.

## Gates

1. G0 analytic audit, no optimization: p95 reduction >= 70%, valid coverage
   >= 80%, clean contact displacement p95 <= 1 mm.
2. G1 fixed-64 gate overfit: gate accuracy and improvement rate >= 90%, refined
   p95 <= 35% of base, clean displacement p95 <= 5 mm.
3. G2 full-distribution 500-step gate: refined p95 <= 60% of base and no clean
   regression.
4. G3 real 500-step bridge: real validation refined p95 improves by at least
   10% before formal training is allowed.

Each shell script runs its metric checker and exits non-zero on failure, so a
failed gate cannot silently proceed to the next phase.
