# Track-aware Support Manifold Refinement

## 1. Objective

The current mainline already provides two useful priors:

1. persistent person identities across frames; and
2. a Stage2 SMPL translation aligned to a metric VGGT scene.

The remaining target is narrower than general human mesh recovery: reduce unsupported
floating and scene penetration without moving already-correct people or destroying the
accepted Stage2 geometry.

The proposed Stage3 is **Track-aware Support Manifold Refinement (TSMR)**. It is a new,
optional post-Stage2 module. Stage1, Stage2, tracking, and all existing checkpoints remain
frozen and unchanged.

TSMR does not learn a binary `pull_to_ground` gate. It constructs several physically valid
translation hypotheses from local metric scene geometry, predicts which support hypothesis
best explains the current human-scene state, applies a bounded correction, and then probes
the scene again. Identity-conditioned temporal memory stabilizes the support state over time.

## 2. Diagnosis of the failed route

The completed distribution experiments rule out insufficient optimization as the main cause.
The full V3 support-intent run reached 6101 steps but obtained recall 0.6870, precision 0.6608,
negative FPR 0.2872, and static-negative FPR 0.4804. Pose and kinematics alone cannot separate
standing support from seated, lying, elevated, static, or legitimately airborne states.

The current `HSIGroundingHead` therefore asks an underdetermined question:

```text
pose + kinematics -> should this person be pulled to the local plane?
```

The older `HSIRefinementHead` also differs from a true iterative geometric refiner. It builds
the SMPL anchors, geometric probes, and local scene tokens once. Its `num_iters` loop repeatedly
processes the same tokens; after updating SMPL parameters it does not decode a new mesh and
re-probe the scene. Repetition without state-dependent observations is not geometric rollout.

There is also a supervision issue. BEDLAM provides GT scene depth and GT SMPL parameters, but
this does not guarantee that every human-scene pair is a physically correct contact example.
GRAFT explicitly notes unrealistic BEDLAM interactions, including humans floating above
furniture. The existing strict contact sidecars are useful visibility and geometry audits, but
must not be treated as universally reliable contact semantics.

## 3. Evidence from the five references

| Reference | Useful mechanism | Boundary for this project |
| --- | --- | --- |
| UniSH | Coarse synthetic metric alignment followed by visible-SMPL-to-human-point-cloud one-way Chamfer and depth-order regularization | Solves global metric placement, not support/contact; its non-parametric geometry may still contain floaters |
| Crowd4D | Confidence-filtered Scene Interaction Point Cloud/Surface, gravity-aligned support proxy, feasible support range, ray consistency, and temporal trajectory smoothing | Uses costly per-sequence optimization and assumes the lowest vertex is the relevant terrain support |
| GRAFT | Body-anchored geometric probes containing nearest-scene displacement and normal; recurrent mesh update and re-probing; mixed realistic/perturbed queries; per-step rollout supervision | Single-image, broad SMPL-X refinement; depends on upstream geometry; official repository currently says code is coming soon |
| MetricHMSR | Explicit camera-ray metric cues and human-guided local affine depth refinement with anchor, TV, and variance regularization | Refines the scene from the human; an incorrect human can become a bad anchor and it does not model contact intent |
| UniCon3R | Current scene context, metric geometry, temporal momentum, and contact-guided residual feedback into the human latent | Dense binary contact remains sensitive to scene errors and rigid-support assumptions; contact supervision alone is insufficient |

The common conclusion is that **metric alignment is necessary but insufficient**, and that
interaction information must be corrective rather than an auxiliary classifier. GRAFT further
shows that the correction must change the next geometric observation.

## 4. Proposed method

### 4.1 Frozen baseline and integration point

The Stage3 input is the accepted Stage2 output:

```text
Stage2 pose/shape/translation + metric depth/pointmap + K/camera + track ID
    -> TSMR
    -> support-refined translation
```

The first implementation changes root translation only. Pose, shape, scene scale, Stage2
translation head, NLF, VGGT, and tracking remain frozen. Lower-leg pose refinement is an optional
later ablation, not part of the first claim.

### 4.2 Tensor and coordinate contract

For batch `B`, sequence length `S`, queries `Q`, body anchors `A`, scene samples `P`, and support
hypotheses `K_h`:

| Tensor | Shape | Coordinate system |
| --- | --- | --- |
| Stage2 translation | `[B,S,Q,3]` | metric camera coordinates |
| Stage2 pose | `[B,S,Q,24,6]` | SMPL 6D rotations |
| decoded vertices | `[B,S,Q,6890,3]` | metric camera coordinates after translation |
| scene pointmap | `[B,S,H,W,3]` or depth + `K` | metric camera coordinates |
| track IDs/mask | `[B,S,Q]` | integer identity / Boolean validity |
| body probes | `[B,S,Q,A,D]` | body-relative vectors plus camera/world geometry |
| candidate corrections | `[B,S,Q,K_h,3]` | metric camera translation deltas |
| hypothesis logits | `[B,S,Q,K_h]` | dimensionless |
| refined translation | `[B,S,Q,3]` | metric camera coordinates |

All distances are in meters. Gravity and support calculations are performed in a gravity-aligned
world basis when reliable camera poses are available, then transformed back to camera coordinates.
The camera-only fallback uses local surface normals and ray consistency and is explicitly marked
lower confidence.

### 4.3 Confidence-aware support surface

Adapt Crowd4D's SIPC/SIS idea locally rather than fitting one plane under each projected sole:

1. remove human pixels using the available person masks/boxes and projected body exclusion mask;
2. back-project metric depth to 3D and reject low-confidence, discontinuity, and excessive-depth points;
3. aggregate a short temporal neighborhood in a common world frame to reveal floor hidden by feet;
4. voxelize the local horizontal plane and retain confidence-weighted lower-envelope samples;
5. estimate local normals, roughness, support extent, and cross-frame consistency;
6. expose an abstain flag when geometry is incomplete or mutually inconsistent.

This surface is only an interaction proxy. It does not overwrite the accepted VGGT scene.

### 4.4 Body-anchored geometric probes

Adapt GRAFT conceptually, without importing reference code. Probe a compact set of body anchors:

- left/right heel and toe groups;
- ankles, knees, pelvis, hands, shoulders, and torso;
- a small deterministic set of full-body surface vertices.

For anchor `a`, its feature contains:

```text
body-relative anchor position
nearest scene displacement vector
distance and direction
local surface normal and gravity cosine
surface roughness and support extent
depth/point confidence and visibility
Stage2 pose context
track velocity/acceleration and previous support state
```

The full-body anchors are essential even when only foot translation is corrected: they tell the
model that a seated or lying person is already supported elsewhere and should not be pulled by a
nearby floor hypothesis.

### 4.5 Analytic support hypothesis bank

For each person-frame construct a small candidate set:

```text
H0: no-op
HL: left-foot support
HR: right-foot support
HLR: bilateral robust foot support
HB: nearest reliable non-foot body support (diagnostic/abstention in phase 1)
```

Each non-zero candidate is computed by robustly aligning the corresponding body anchors to a
locally consistent support patch. Corrections are constrained to the span of gravity and the local
support normal, clipped to a configured metric range, and rejected when plane/surface confidence,
support extent, ray consistency, or left-right agreement is poor.

The network therefore cannot invent an arbitrary XYZ translation. It predicts a categorical
distribution over physically generated candidates plus a small bounded scalar residual along the
selected candidate direction. At inference, high entropy or candidate disagreement selects `H0`.
This replaces the failed binary gate with **selection among explicit geometric explanations**.

### 4.6 True recurrent re-probing

Use `R=3` shared-weight refinement steps:

```text
for r in 1..R:
    decode current SMPL mesh
    rebuild body anchors
    query the current local support surface
    rebuild probe tokens and candidate corrections
    fuse current scene evidence with ID-keyed temporal memory
    select/abstain and apply one bounded correction
```

The probes and candidates must be recomputed from the updated mesh at every step. This is the
minimum structural correction required before another training run is meaningful.

### 4.7 ID-conditioned temporal memory

For each persistent track, retain a detached compact state containing the previous support
hypothesis, correction, support normal, support height, confidence, and pooled interaction token.
Current-frame fusion follows UniCon3R's temporal-momentum principle, but matching uses the existing
track IDs rather than row truncation.

Temporal behavior is asymmetric:

- support onset requires consistent geometric evidence;
- established support receives hysteresis against one-frame dropout;
- abrupt vertical velocity/acceleration weakens support persistence;
- ID gaps or low tracking confidence clear memory rather than borrowing another person's state.

## 5. Supervision and preprocessing

### 5.1 Do not discard the current preprocessing

Keep `hsi_contact_teachers_v3_strict`. Reuse its plane validity, visibility, depth residual, and
track-aware velocity fields as geometry-quality metadata. Do not use `contact_label` as the sole
ground-truth decision target.

### 5.2 New support-manifold sidecar

Create a versioned `hsi_support_manifold_v1` sidecar under `outputs/preprocess/` containing:

```text
support points/normals/confidence/roughness/extent
per-anchor nearest point, offset, visibility, and validity
candidate translation vectors and candidate validity
GT-to-candidate residuals
clean/support/airborne/ambiguous supervision mask
sequence/frame/person/track identity
```

Only high-confidence examples receive support labels. Ambiguous BEDLAM interactions remain usable
for geometry robustness or no-change consistency but are masked out of support classification.

### 5.3 Query mixture

Train from a mixture inspired by GRAFT:

1. `20%` clean, strictly verified GT states with a zero-update target;
2. `40%` verified GT states perturbed along gravity/support normals by `+/-1,2,4,6,8,12 cm`;
3. `25%` frozen Stage2 predictions paired with GT translation where the human-scene geometry is reliable;
4. `15%` unsupported/airborne/ambiguous hard negatives with an explicit no-op or abstain target.

The exact mixture is a configuration and an ablation. Stage2 predictions are cached or generated
with the frozen checkpoint; the Stage2 checkpoint itself is never modified.

### 5.4 Losses

Apply losses at every rollout step:

```text
L = lambda_hyp  * candidate-selection CE
  + lambda_trans * robust translation residual
  + lambda_mono  * monotonic rollout loss
  + lambda_clean * clean-state invariance
  + lambda_pen   * one-sided penetration loss on reliable probes
  + lambda_temp  * ID-conditioned support/correction consistency
  + lambda_cal   * confidence/abstention calibration
```

`L_mono` penalizes an iteration only when it increases GT translation error or reliable support
residual. Contact-distance losses are masked by strict geometry reliability. Pose and shape losses
remain disabled in phase 1 because the first claim is grounding through translation.

## 6. Validation ladder

No full training should start until the previous gate passes.

### G0: candidate-oracle audit, no training

Verify that at least one candidate can recover synthetic perturbations while `H0` exactly preserves
clean and unsupported samples. Report by perturbation magnitude and sign. Failure here means the
support surface/candidate generator must be fixed; a network cannot repair missing candidates.

### G1: deterministic forward and gradient smoke

Check all shapes, finite values, candidate validity, non-zero gradients, per-step re-probe changes,
frozen Stage1/Stage2 hashes, ID-memory isolation, and compact JSON logging.

### G2: fixed-subset overfit

Use the same fixed subset for training and validation only to verify model capacity. Require candidate
selection separation, monotonically falling rollout error, and zero clean movement.

### G3: held-out distribution gate

Evaluate complete held-out sequences, not a random batch subset. Selection is based on physical
improvement relative to frozen Stage2, with confidence intervals, not only classification accuracy.

### G4: real-sequence visual and benchmark evaluation

Use the accepted early Stage2 Viser path and display base/refined meshes, selected support mode,
candidate vector, support points, and abstentions. Quantitative evaluation should include RICH when
its scene geometry is available; BEDLAM alone is not sufficient evidence for a contact paper.

## 7. Metrics and ablations

Primary physical metrics:

- floating distance/rate;
- penetration depth/rate and maximum penetration;
- collision ratio;
- foot sliding and temporal jitter;
- clean-person displacement;
- candidate coverage and oracle/model gap;
- support selection precision/recall and abstention calibration;
- WA-MPJPE, W-MPJPE, RTE, and local pose metrics to detect collateral damage;
- runtime per person/frame.

Required ablations:

1. frozen Stage2;
2. old analytic plane candidate;
3. old learned binary gate;
4. candidate bank with deterministic selection;
5. candidate bank plus learned selector;
6. no recurrent re-probing;
7. no body-wide probes;
8. no ID temporal memory;
9. no abstention/uncertainty guard;
10. unconstrained XYZ residual instead of candidate-constrained correction;
11. single-frame surface instead of temporally fused support surface.

## 8. Publication claim and relationship to prior work

The defensible contribution is not "we added contact loss." It is:

> A plug-in, track-aware recurrent support refiner for frozen feed-forward human-scene
> reconstruction that converts uncertain metric scene geometry into an explicit support
> hypothesis manifold, selects or abstains among physically constrained corrections, and
> re-probes the scene after every update.

Relative to UniSH, it adds physical support after metric alignment. Relative to Crowd4D, it
amortizes terrain-aware support reasoning instead of running long per-sequence optimization.
Relative to GRAFT, it limits correction to an analytic support manifold, adds video identity memory
and abstention, and initially preserves pose/shape. Relative to UniCon3R, it replaces descriptive
dense binary contact with explicit corrective hypotheses and guards against scene-error propagation.
Relative to MetricHMSR, it uses human geometry to validate local support evidence without allowing
the human to overwrite the whole scene.

## 9. Risks and stopping rules

- If G0 oracle coverage is poor, improve the support surface or add hypotheses; do not train.
- If clean examples move under deterministic candidates, fix candidate validity/deadzone; do not tune a classifier.
- If G2 passes and G3 fails with a large oracle/model gap, the selector inputs or training distribution are insufficient.
- If both oracle and model fail on real data, upstream scene geometry is the limiting factor; report abstention rather than forcing contact.
- If RICH scene scans/contact annotations are unavailable, restrict the claim to synthetic grounding and real qualitative results instead of claiming general contact reconstruction.

## 10. Implementation order

1. Freeze and hash the accepted Stage2 and tracking checkpoints.
2. Implement the support surface and candidate bank as standalone utilities.
3. Implement a no-training G0 oracle audit and visualization.
4. Add a new optional TSMR head with true three-step re-probing and no pose/shape branch.
5. Add support-manifold preprocessing and masked query generation.
6. Add compact smoke/overfit/distribution shell launchers and JSON gate reports.
7. Train only after G0 and G1 pass.
8. Evaluate against frozen Stage2 using the full physical metric and ablation suite.

## 11. Reference implementation status

UniSH contains local source code and was inspected for its AlignNet/pipeline integration pattern.
The local GRAFT repository currently contains the project page assets and a README whose roadmap
marks the code release as pending; therefore the GRAFT portion of TSMR is a concept-level rewrite
from the paper equations and algorithms, not a direct code port. Crowd4D, MetricHMSR, and
UniCon3R were used from their local PDFs. No main-project module imports code from `.paper/`.
