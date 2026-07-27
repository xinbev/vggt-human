# HSI Stage3 CI-A: Foot Contact Intent

## Purpose

The previous grounding Gate could not separate a true global root-translation float from a legitimate airborne pose. CI-A therefore learns only whether each foot is intended to support the body. It never changes pose, shape, scene scale, or translation.

## Data contract

- Existing `hsi_contact_teachers_v3_strict` sidecars are reused; no preprocessing rerun is required.
- `contact_teacher_valid` masks unreliable feet.
- `contact_label` is the per-foot target.
- All sequence windows are loaded. Contact-only filtering must remain disabled so jumping and other no-contact states are present as hard negatives.
- Clean GT SMPL is used. Translation perturbation is disabled in CI-A.
- Sequence length is three and GT track IDs provide clip-consistent temporal association.
- Classification is supervised only on the center frame, matching the teacher's two-sided temporal context. Sliding windows still cover almost all physical frames.

## Inputs and outputs

The `camera_motion_v2` head uses lower-body pose, root-relative sole locations, and track-aligned camera-space root/foot velocities, accelerations, and mean step distances. Camera-space mean step distance exactly matches the V3 teacher's velocity contract while temporal differences avoid absolute translation as a shortcut. The legacy `world_v1` feature path remains available only for checkpoint compatibility.

Checkpoint scope is exactly `hsi_foot_contact_intent_head.`. Existing Stage1, Stage2, G1, G2, and audit checkpoints are neither loaded nor overwritten.

## Gates

1. `smoke`: two train and two validation batches; validates interfaces, target presence, temporal features, gradients, and trainable prefixes.
2. `overfit`: fixed 64-window subset; requires recall >= 98%, precision >= 95%, negative FPR <= 2%, and airborne FPR <= 2%.
3. `gate500`: 500 full-distribution steps and 100 validation batches; requires recall >= 90%, precision >= 80%, negative FPR <= 5%, and airborne FPR <= 3%.

Translation grounding remains disabled until all CI-A gates pass. CI-B will combine the frozen intent probability with the analytic two-foot severe-float candidate.
