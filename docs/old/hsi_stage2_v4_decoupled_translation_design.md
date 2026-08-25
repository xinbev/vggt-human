# HSI Stage2 V4: Decoupled Translation Correction

## 1. Objective

Stage2 V4 learns to align SMPL root translation to the metric human-scene
geometry produced after Stage1. It does not train pose, betas, VGGT, NLF, or
scene scale. V4 replaces the V3 `soft_gate * delta` optimization with two
separately trainable decisions:

```text
candidate_delta = correction_head(geometry_features)
apply = gate_head(geometry_features, candidate_features) > threshold
refined_transl = base_transl + apply * candidate_delta
```

The gate is a selector only. It never scales correction magnitude.

## 2. Why V3 Is Not Continued

The fixed-64 diagnosis established:

```text
signed residual ray accuracy       95.52%
analytic residual improvement      93.85%
effective learned ray accuracy     about 60%
active-noisy combined improvement  about 68%
```

V3's geometry signal is usable. Its failure comes from coupling confidence and
step size through a soft dead-zone. Increasing the dead-zone protects clean
people but suppresses active corrections; lowering it improves active people
but moves clean people. More V3 fine-tuning is not part of the V4 plan.

## 3. Preservation Rules

- Stage1 checkpoint and hash remain unchanged.
- VGGT aggregator, camera head, dense/depth head, and NLF remain frozen.
- Legacy Stage2 and V3 checkpoints are read-only diagnostic baselines.
- V4 uses a new module prefix, config, scripts, and output root.
- V4 initializes from Stage1, not from V2/V3 translation-head weights.
- Existing `HSIHumanSceneAlignHead` and its baseline config paths remain.

Proposed V4 prefix:

```text
hsi_translation_refine_v4_head.
```

## 4. Geometry Contract

Every batch must use one coherent mode:

| Mode | Base SMPL | Geometry depth | Camera K | Target |
|---|---|---|---|---|
| `gt_metric` | perturbed GT | GT metric depth | GT `K_scal3r` | clean GT translation |
| `real_inference` | NLF | Stage1-scaled VGGT depth | VGGT K | matched GT translation |

Forbidden geometry inputs remain:

- GT SMPL with VGGT K
- NLF SMPL with GT K
- GT metric depth passed through Stage1 scale a second time

Stage2-A uses `gt_metric`. Stage2-B introduces `real_inference` only after all
Stage2-A gates pass.

## 5. Fixed Perturbation Contract

Validation perturbations must be pre-generated and immutable. Training noise
is deterministic from `(seed, epoch, sequence, frame, person)` and balanced by
category. Noise is constant for one person across a sampled clip.

Categories:

```text
20% exact clean
20% ray-only
20% tangent-only
40% ray+tangent
```

Ray levels are sampled equally from:

```text
-15%, -10%, -5%, -3%, +3%, +5%, +10%, +15%
```

Tangent magnitudes are sampled equally from:

```text
2 cm, 5 cm, 8 cm
```

Tangent direction is uniform in the camera tangent plane. Positive and
negative ray signs, near/far depth bins, and person-count bins must be balanced
in the fixed-64 and validation manifests.

Supervision groups use actual metric base error, not the sampled percentage:

```text
clean:      exactly 0 m
transition: 0 < error < 0.05 m
active:     error >= 0.05 m
```

Transition samples train the correction candidate but are excluded from gate
BCE and strict improvement-rate gates.

## 6. V4 Module Design

### 6.1 Geometry Feature Extractor

The extractor has no trainable parameters and only uses inference-available
inputs. It projects self-visible SMPL surface samples into the corresponding
depth map and computes robust signed statistics in the camera ray/tangent
basis.

Required features per person:

```text
base translation and base depth
SMPL confidence
valid correspondence count and ratio
ray/tangent residual P10, P50, P90, MAD
ray residual sign-agreement ratio
ray residual inlier ratio
mean absolute residual
projected center and normalized person-box size
depth-confidence summaries when available
```

Scene scale/bias values, GT noise values, clean/noisy labels, and GT
translation must never be model inputs.

Correspondence selection keeps the current self-visibility and box checks, but
the nearest-point residual is treated as a feature only. It is not an
authoritative translation target and does not receive a competing point loss
in Stage2-A.

### 6.2 Correction Network

The correction network has its own MLP trunk and does not share trainable
layers with the gate network.

Outputs:

```text
ray magnitude ratio
tangent-x delta in meters
tangent-y delta in meters
```

Ray direction and scale are anchored to the observed signed median residual
when residual coherence is valid. The MLP predicts a positive gain rather than
an unconstrained depth-sized magnitude:

```text
ray_delta = ray_residual_median * predicted_positive_gain
```

This retains the reliable geometric sign, learns the biased residual's missing
magnitude, and prevents near-zero ray evidence from producing a large
depth-proportional correction. Tangent coefficients remain signed learned
outputs. The gain is bounded by `4.0` and tangent coefficients by `12 cm`.

If correspondence support or sign coherence is below a threshold fixed by the
training audit, the candidate is marked geometry-ineligible and the inference
gate is forced closed. Eligibility coverage is a required metric; ineligible
people cannot simply be removed from the overall success rate.

The candidate correction is always computed, including for clean people. The
correction-only loss is masked to non-clean samples, so the candidate network
cannot reduce its loss by manipulating the gate.

### 6.3 Independent Gate Network

The gate network has a separate MLP trunk. It receives detached geometry
features plus detached candidate magnitude/coherence features. Gate training
cannot change the correction candidate.

Targets:

```text
clean  -> 0
active -> 1
transition -> ignored
```

Clean and active BCE terms are averaged separately and then weighted equally.
The gate reports probability for calibration, but application is binary:

```text
hard_apply = gate_probability >= threshold
```

Combined fine-tuning may use a straight-through binary estimator, whose
forward value is still exactly zero or one. There is no soft amplitude gate.
The threshold defaults to `0.5`; any calibration must use the training split
and is frozen before validation. Validation metrics may not tune it.

## 7. Training Phases

### V4-A1: Correction Only

```text
input: GT depth + GT K + noisy GT SMPL
gate: bypassed, apply=1 for metric evaluation
trainable: correction trunk and correction outputs
frozen: gate, Stage1, old HSI heads, VGGT, NLF
```

Losses:

```text
candidate translation Huber       8.0
normalized ray magnitude Huber    4.0
tangent-vector Huber              4.0
candidate magnitude regularizer   0.01
```

No `align_point`, scene `no_worse`, gate BCE, clean identity, joints3d, or
vertices loss is active. Pose is fixed, so joints/vertices would duplicate the
same translation supervision and are metrics only.

Default overfit: batch 24, two views, LR `2e-5`, at most 1000 steps.

Gate to A2 on fixed-64 active samples:

```text
candidate improvement rate >= 95%
candidate refined median <= 25% of base median
ray sign accuracy >= 95%
tangent error does not exceed base by 0.5 mm
finite gradients only in correction parameters
```

### V4-A2: Gate Only

```text
correction: frozen and always computed
trainable: independent gate trunk/head only
transition samples: ignored by gate loss
```

Losses:

```text
balanced clean/active BCE         1.0
optional calibration Brier loss   0.1
```

Default overfit: batch 24, two views, LR `1e-5`, at most 500 steps.

Gate to A3:

```text
active true-open rate >= 95%
clean false-open rate <= 5%
active/clean AUROC >= 0.98
clean hard-applied displacement p95 <= 1 mm
correction checkpoint hash unchanged
```

### V4-A3: Combined Calibration

The hard selector is enabled. Correction and gate are both trainable at low
LR, but candidate losses remain present before gating so the gate cannot hide
a bad correction.

Losses:

```text
pre-gate candidate translation    4.0
post-gate final translation       8.0
ray/tangent candidate losses      2.0 / 2.0
balanced gate BCE                 1.0
clean hard identity               4.0
```

Default overfit: batch 24, two views, LR `2e-6`, at most 500 steps.

Final fixed-64 Stage2-A gate:

```text
active improvement rate >= 90%
active refined median <= 30% of base
clean false-open rate <= 5%
clean displacement p95 <= 5 mm
tangent degradation <= 0.5 mm
Stage1 hash unchanged
```

### V4-B: Real-Inference Bridge

Only after A3 passes, mix complete geometry modes by batch:

```text
epoch 1: gt_metric / real_inference = 75 / 25
epoch 2: gt_metric / real_inference = 50 / 50
epoch 3: gt_metric / real_inference = 25 / 75
```

The real branch is strictly:

```text
RGB -> frozen VGGT depth/K -> frozen Stage1 scale -> frozen NLF -> V4
```

GT translation is supervision only. NLF pose/betas are not replaced by GT in
the geometry path. Missing or unmatched NLF slots are excluded.

Real validation requires translation median and p90 improvement without MPJPE,
PVE, or projected-joint regression. Gate calibration is reported separately
for GT and real branches.

## 8. Metric Implementation Requirements

V4 validation writes per-person records and computes global statistics after
the epoch. Averaging batch medians is forbidden.

Required groups:

```text
clean / transition / active
ray-only / tangent-only / combined
positive-ray / negative-ray
0-5 m / 5-10 m / 10-20 m
1 person / 2-3 people / 4+ people
gt_metric / real_inference
```

Required metrics include mean, global median, p90, p95, correction improvement
rate, gate TPR/FPR, AUROC, clean displacement p95, correspondence support, and
failure sample identifiers.

Both eligible-only and all-person active metrics are written. The validation
gate additionally requires geometry-eligibility coverage of at least `95%`, or
the stage fails regardless of eligible-only correction quality.

## 9. Checkpoint Contract

Separate output roots are mandatory:

```text
outputs/debug/hsi_stage2_v4_a1_*
outputs/debug/hsi_stage2_v4_a2_*
outputs/debug/hsi_stage2_v4_a3_*
outputs/train/hsi_stage2_v4_decoupled/
```

Loading requirements:

- A1 requires complete `hsi_refinement_head.` from Stage1.
- A2 requires complete Stage1 and V4 correction prefixes from A1 top01.
- A3 requires complete Stage1 and all V4 prefixes from A2 top01.
- B requires A3 top01.
- Missing required tensors are fatal.
- Every phase resets epoch/global step and uses a new output directory.
- Stage1 hash is checked before and after every phase.

## 10. Implementation And Test Order

1. Add deterministic fixed-noise manifest generator and audit script.
2. Add V4 feature extractor and independent correction/gate trunks.
3. Add phase-specific freezing and loss masks.
4. Add per-person validation recorder and global metric reducer.
5. Add shape/gradient/unit smoke tests.
6. Run two-batch interface smoke.
7. Run A1 fixed-64 correction-only gate once.
8. Do not implement or run A2 until A1 passes.

Required server scripts:

```text
scripts/preprocess/prepare_hsi_stage2_v4_noise_manifests.sh
scripts/vis/vis_hsi_stage2_v4_noise_audit.sh
scripts/smoke/check_hsi_stage2_v4_interface.sh
scripts/train/train_smpl_hsi_stage2_v4.sh
scripts/eval/eval_smpl_hsi_stage2_v4.sh
```

## 11. Explicit Non-Goals

- Stage3 contact training
- pose or betas refinement
- identity tracking or temporal memory
- dense full-scene depth supervision
- VGGT or NLF fine-tuning
- reusing V3 soft gate or V3 top checkpoint as V4 initialization

## 12. A1 Implementation Status

Implemented for the first feasibility gate:

```text
vggt_omega/models/heads/hsi_translation_refine_v4_head.py
vggt_omega/training/hsi_stage2_v4_noise.py
configs/train_smpl_hsi_stage2_v4_a1_correction.yaml
scripts/preprocess/prepare_hsi_stage2_v4_noise_manifests.sh
scripts/train/train_smpl_hsi_stage2_v4_a1_correction.sh
scripts/smoke/check_hsi_stage2_v4_interface.sh
scripts/smoke/check_hsi_stage2_v4_a1_overfit.sh
```

The implementation adds no Stage2-A2/A3 training path. The A1 config enforces
trainable and non-zero-gradient prefix contracts for the correction trunk and
head, records per-person validation rows, reduces global metrics, and keeps the
Stage1 hash frozen. Dynamic validation still requires the Linux server because
the Windows checkout has neither Torch nor SMPL/checkpoint assets.
