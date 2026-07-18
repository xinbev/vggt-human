# HSI Stage2 Robust Gate V3

## Goal

Stage2 V3 keeps the successful Stage1 metric scene scale frozen and trains a
new human-scene translation aligner without modifying any legacy Stage2
checkpoint. It addresses the V2 failure mode where the align head corrected
large ray-depth perturbations but also moved clean GT people by about 9-14 mm.

## Legacy Preservation

- `legacy_scale_bias_v0` reproduces the original 25-D feature order used by the
  archived full Stage2 checkpoint, including scene scale/bias.
- `legacy_mean_v1` remains the current 23-D baseline default and omits those
  domain-specific scalars.
- V3 uses `robust_basis_v2` and a separate output root.
- V3 initializes from the Stage1 checkpoint, never from a legacy Stage2
  checkpoint.
- `scripts/tools/backup_hsi_stage2_legacy_checkpoint.sh` creates a no-clobber,
  read-only copy under `outputs/checkpoint_backups/` and verifies SHA256.

## Robust Feature Contract

The legacy head uses translation, confidence, valid ratio, residual means,
camera basis, and projected center. V3 appends robust residual statistics in
the per-person camera basis:

```text
[ray, tangent-x, tangent-y] x [P10, P50, P90, MAD]
```

These statistics distinguish a coherent whole-body translation error from
localized clothing, silhouette, occlusion, and depth-boundary residuals. The
statistics use only inference-time geometry and do not expose GT noise values
to the model.

V3 uses `robust_sign_v3` for the ray coefficient. The MLP still predicts the
correction magnitude, but its ray sign is constrained by the signed median
scene residual and receives a conservative `1.10` magnitude gain. Tangent
coefficients remain fully learned. This preserves checkpoint tensor shapes and
the legacy `learned_v1` path while preventing an unconstrained MLP from
discarding the reliable depth-residual direction.

## Gate Supervision

The gate target is continuous:

```text
gate_target = clamp(base_translation_error / 0.10 m, 0, 1)
```

Clean and perturbed groups contribute equally to the BCE. Clean people also
receive a direct identity loss. Noise metadata is used only to group losses
and metrics, never as a model input.

V3 applies the raw sigmoid gate through a differentiable dead-zone:

```text
effective_gate = relu(raw_gate - 0.55) / 0.45
```

The raw gate remains supervised. A confidently clean person therefore receives
exactly zero translation update, while noisy people retain a gradient above the
threshold. Legacy modes continue to apply the original sigmoid gate directly.

Because the magnitude target intentionally ignores tiny corrections, overfit
improvement metrics are evaluated on perturbed people with at least 5 cm base
translation error. Exact clean people retain the separate 5 mm identity gate.

## Frozen Components

- VGGT aggregator, camera head, and depth head
- NLF
- Stage1 `hsi_refinement_head`, including scene scale/bias
- SMPL pose and betas
- contact head

Only `hsi_human_scene_align_head` is trainable.

## Gates

Run smoke, fixed-64 overfit, and 500-step distribution checks in that order.
The fixed-64 gate requires:

- active-noisy improvement rate at least 90%
- active-noisy refined median no more than 30% of base
- clean displacement no more than 5 mm
- clean gate lower than noisy gate
- tangent degradation no more than 0.5 mm

Full Stage2-A/B must not start until all gates pass. V3 writes only to
`outputs/debug/hsi_stage2_robust_gate_v3_*` and
`outputs/train/hsi_stage2_transl_robust_gate_v3/`.

## Remaining Risk

The robust statistics should improve clean/noisy identifiability, but this is
not verified on Windows because the local checkout lacks Torch, checkpoints,
SMPL assets, and BEDLAM. The server smoke must verify tensor shapes, finite
quantiles, trainable prefixes, and frozen hashes before overfit training.

## Residual-Signal Diagnosis

The fixed-64 zero-LR diagnosis measured `95.52%` signed-residual ray accuracy
and `93.85%` analytic residual improvement, while the learned effective delta
reached only about `60%` ray-sign accuracy. This rules out a global sign error
in GT perturbation, camera basis, or ray supervision. With `robust_sign_v3`
enabled but before fine-tuning, active-noisy improvement remained `68.41%`
because the existing soft dead-zone still suppressed roughly one third of
active corrections. The next gate is therefore a 500-step continuation from
the preserved V3 checkpoint under the new parameterization, not another
geometry redesign or Stage1 restart.
