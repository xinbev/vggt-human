# HSI Stage3 CI-A: Foot Contact Intent

## Purpose

The previous grounding Gate could not separate a true global root-translation float from a legitimate airborne pose. CI-A therefore learns only whether each foot is intended to support the body. It never changes pose, shape, scene scale, or translation.

## Data contract

- Existing `hsi_contact_teachers_v3_strict` sidecars are reused; no preprocessing rerun is required.
- `contact_teacher_valid` masks unreliable feet.
- `contact_label` is the per-foot target.
- All sequence windows are loaded. Contact-only filtering must remain disabled so jumping and other no-contact states are present as hard negatives.
- Clean GT SMPL is used. Translation perturbation is disabled in CI-A.
- Sequence length is five and GT track IDs provide clip-consistent temporal association.
- Classification is supervised only on the center frame, matching the teacher's two-sided temporal context. Sliding windows still cover almost all physical frames.

## Inputs and outputs

The `camera_motion_v3_joint5` head uses a bidirectional temporal encoder over five track-aligned frames. It jointly reads both feet, lower-body pose, root-relative sole locations, camera-space root/foot velocities, accelerations, and teacher-aligned mean step distances. Its target is person-level support intent: any reliable contacting foot is positive; a negative requires both feet to be reliable and neither to contact. This avoids contradictory left/right supervision and gives the gate enough context to reject jump and airborne states. The `world_v1` and `camera_motion_v2` paths remain available for checkpoint compatibility.

The model contract is `pose [B,5,Q,24,6]`, `betas [B,5,Q,10]`, and camera translation `[B,5,Q,3]`. Track alignment builds context `[B,5,Q,5,90]`; the V3 output is one person-support logit `[B,5,Q]`. Features are float tensors on the model device. Absolute scene height is deliberately excluded: the frozen analytic grounding candidate owns scene-plane geometry, while this head only decides whether applying that candidate is behaviorally appropriate.

Checkpoint scope is exactly `hsi_foot_contact_intent_head.`. Existing Stage1, Stage2, G1, G2, and audit checkpoints are neither loaded nor overwritten.

## Gates

1. `smoke`: two train and two validation batches; validates interfaces, target presence, temporal features, gradients, and trainable prefixes.
2. `overfit`: fixed 64-window subset; requires recall >= 98%, precision >= 95%, negative FPR <= 2%, and airborne FPR <= 2%.
3. `gate500`: a freshly initialized V3 head runs for 500 full-distribution steps and 100 validation batches. The fixed-64 checkpoint is deliberately not used as initialization. This gate requires recall >= 90%, precision >= 80%, negative FPR <= 5%, and airborne FPR <= 3%.

Translation grounding remains disabled until all CI-A gates pass. CI-B will combine the frozen intent probability with the analytic two-foot severe-float candidate.

Use `PHASE=pipeline` to run smoke, fixed-64 overfit, and the fresh full-distribution gate in order. The launcher stops at the first failed gate and never reuses a fixed-subset checkpoint as the distribution initializer.
