# HSI Stage2 Track-Aware Regional Surface Refiner

> Archived superseded draft. Do not implement the pose/joint decoder described
> below. The current reviewed design is
> `docs/hsi_stage2_trsr_full_design_review_zh.md` and is strictly
> translation-only: NLF/HMR pose, global orientation, and betas are read-only;
> regional queries only estimate one shared `delta_transl_cam` per person.

## 1. Objective

Assume the Stage1 scene-scale branch has converged and is frozen. Stage2 should
reduce the remaining human-scene mismatch caused by residual scale/camera/depth
error and imperfect NLF SMPL reconstruction, while preserving already-correct
people and avoiding temporal flicker.

The proposed module is the **Track-Aware Regional Surface Refiner (TRSR)**:

```text
scaled metric scene + NLF SMPL + camera + persistent track state
    -> non-uniform SMPL surface-region queries
    -> multi-scale local scene patch attention
    -> continuous region correction field
    -> kinematic region-to-parameter decoding
    -> bounded SMPL root and joint updates
    -> mesh decode and true scene re-probing
    -> track-aware temporal fusion
```

This is a new optional post-scale module. It does not replace VGGT, NLF, the
accepted scale branch, or the baseline inference path.

## 2. Critical Model Boundary

SMPL has 6890 surface vertices, but those vertices are controlled by only 24
joints, root translation, global orientation, and shape. Standard SMPL has no
articulated fingers.

Therefore:

- 6890 vertices are useful as the finest measurement surface;
- regional queries are compact measurements and constraints;
- regional queries must not output independent vertex displacement;
- all accepted corrections must be projected back to SMPL root/pose parameters;
- betas should remain fixed in Stage2 and stable over the complete track.

For example, several left-hand surface queries can independently observe a
wall, but their corrections must jointly route to the left wrist, elbow, and
shoulder chain. This preserves the SMPL manifold and prevents torn or
non-anatomical meshes.

## 3. Current Project Audit

### 3.1 Reusable components

| Existing component | Reuse | Boundary |
| --- | --- | --- |
| Stage1 `hsi_refinement_head.scale_delta/bias_delta` | Freeze and use its metric depth | Stage2 must not relearn scene scale through pose deformation |
| `HSIHumanSceneAlignHead` | Reuse local scene sampling, robust residual statistics, camera-basis parameterization, and as a root-only baseline | Uniform FPS points are collapsed to one person feature; output is root translation only |
| `HSITranslationRefineV4Head` | Reuse no-worse metrics and as another root baseline | Its hand-designed candidate route is not reused by TRSR |
| `HSIContactRefineHead` | Reuse sole/support geometry, bounded lower-limb deltas, and contact metrics | Covers only root normal plus hip/knee/ankle; not full-body collision refinement |
| `HSIGroundingHead` | Reuse deadzone, reliability gate, and abstention metrics as a baseline | Its analytic candidate-selection architecture is explicitly not the TRSR mainline |
| `contact_geometry.py` | Reuse robust local plane fit and depth-confidence checks | Generalize from two feet to arbitrary regions and non-horizontal surfaces |
| HSI temporal losses | Reuse track-aligned velocity, acceleration, no-worse, and foot-sliding losses | Loss-only smoothing cannot provide causal online state |
| `BaseSMPLTrackAssigner` | Reuse box/transl/betas/ID matching score | It currently resets inside each call and must run before refinement with persistent stream state |
| `HSITrackMemory` | Reuse the track-state ownership concept | Do not reuse naive element-wise EMA of pose parameter lists |
| `apply_hsi_scene_affine_mode` | Reuse clip median offline and EMA idea online | Current EMA is clip-local; streaming needs persistent scene state |

### 3.2 Components not suitable as the new mainline

The existing `HSIRefinementHead` has 24 anchors: 21 body joints, two hand
midpoints, and one average of 27 full-body FPS vertices. It repeats transformer
blocks for `num_iters`, but does not decode the updated mesh and re-query scene
geometry after each update. Its unrestricted pose head also predicts all 144
6D pose values from every pooled person token.

The old Stage2 align head decodes 24 joints plus 96 uniform FPS vertices, then
reduces all valid residuals to person-level means or quantiles. It cannot retain
the distinction between a local hand collision and a globally shifted body.

Both remain valuable baselines. Neither should be extended by simply increasing
the number of FPS samples.

### 3.3 Tracking gap in current inference

The NLF-detector Viser currently sets model tracking to `none`, runs HSI, and
adds display-only IDs afterward. Those IDs cannot condition model refinement.

TRSR requires this causal order:

```text
NLF detector/SMPL
    -> persistent base-SMPL tracking
    -> read per-track memory
    -> regional refinement
    -> write refined state back to memory
```

Training may use GT track IDs. Inference must use detector-produced boxes plus
base SMPL geometry/identity features. Sidecar boxes and the retired HMR query
prior are not reintroduced.

## 4. Non-Uniform Surface Region Bank

### 4.1 Deterministic construction

Build the region bank once from the neutral SMPL template, mesh faces, and LBS
weights already exposed by the SMPL layer:

1. assign every vertex to its dominant skinning joint;
2. preserve mesh-connected components using face adjacency;
3. assign a configurable query budget per anatomical risk group;
4. subdivide each group with geodesic FPS or adjacency-constrained clustering;
5. assign every one of the 6890 vertices to exactly one region;
6. record normalized pooling weights and kinematic influence masks.

No external body-part repository is required for the first version. Exported
audit artifacts belong under `outputs/debug/hsi_region_bank_v1/`; deterministic
construction remains in project code/config rather than a hidden binary asset.

### 4.2 Risk-weighted query budget

Use a configurable default near `A=96`, then ablate `48/72/96`. A reasonable
initial allocation is:

```text
head + neck + torso + pelvis: 20 coarse regions
upper limbs excluding hands: 16 medium regions
left/right hands:             24 fine regions total
lower limbs excluding feet:  16 medium regions
left/right feet:              20 fine regions total
```

The exact count is not a semantic claim. Accept a bank only after auditing:

- every vertex is assigned exactly once;
- left/right symmetry is preserved;
- region geodesic radius and vertex-count imbalance are bounded;
- hands, soles, toes, heels, knees, elbows, and buttocks have adequate coverage;
- each region has a valid controlling joint chain.

### 4.3 Region metadata

For region `a`, store:

```text
vertex indices and pooling weights
representative surface vertices
canonical centroid/normal/radius
mean LBS weights over 24 joints
dominant joint and allowed kinematic chain
risk level and update bounds
left/right and semantic group IDs
```

At runtime, decode the current mesh and aggregate current region centroid,
normal, covariance, extent, and representative vertices. High-risk regions use
more representative samples than low-risk torso regions even when each region
still produces one query token.

## 5. Scene Observation And Reliability

### 5.1 Do not attract every body region to the nearest scene point

Nearest-point attraction would pull visible body surfaces toward background,
furniture, or foreground occluders. TRSR uses one-sided, reliability-gated
constraints:

- scene clearly in front of a body region may indicate penetration/occlusion;
- a reliable support surface below a foot may indicate float/contact;
- scene behind a visible body is not automatically a contact target;
- missing or single-layer-occluded geometry produces abstention, not motion.

### 5.2 Human/self-depth handling

VGGT depth may include the person. Build a sparse projected SMPL exclusion mask
from the current decoded mesh and depth order. For each region:

1. project representative vertices with the active camera;
2. compare mesh z and scaled depth to classify likely self-depth;
3. sample both a compact local patch and an annulus outside the projected body;
4. reject depth edges, low confidence, invalid normals, and inconsistent points;
5. optionally fuse nearby frames in world coordinates to reveal surfaces hidden
   in the current frame;
6. mark truly unobservable contacts invalid.

This can adapt the existing projected-human exclusion and local plane utilities
without restoring sidecar person masks.

### 5.3 Region query features

Each query contains only inference-available values:

```text
region body-relative and camera/world position
surface normal, radius, covariance, and LBS/joint-chain embedding
projected location and camera ray basis
local scene point/normal/roughness/extent
signed residuals along ray, region normal, and gravity
P10/P50/P90/MAD residual statistics
valid ratio, depth confidence, edge score, and self-depth score
current NLF confidence and scale confidence
root/pose motion prior and previous track-region state
```

Recommended tensors for `B,S,Q,A,R,C`:

```text
region_vertices_cam:       [B,S,Q,A,R,3]
region_tokens:             [B,S,Q,A,C]
region_valid:              [B,S,Q,A]
region_joint_weights:      [A,24]
region_allowed_joint_mask: [A,24]
```

### 5.4 Metric camera and world-coordinate contract

Single-frame regional probing uses metric camera-space SMPL vertices, scaled
depth, and image intrinsics `K`. Intrinsics and camera rotation are never scaled.

Cross-frame scene fusion and temporal memory need an additional contract. For a
pure multiplicative scene correction `s`, the VGGT camera translation/center
must be transformed by the same `s` before constructing a common world frame:

```text
depth_metric = s * depth_raw
camera_translation_metric = s * camera_translation_raw
camera_rotation_metric = camera_rotation_raw
K_metric = K_raw
```

An additive depth bias is ray-dependent and is not a rigid camera transform.
It cannot be added to camera translation. If Stage1 predicts a material bias,
temporally fused scene points must be built from the corrected depth per frame
and explicitly checked for cross-frame consistency; otherwise TRSR should use
camera-space local probes and abstain from world-scene fusion.

The current BEDLAM loader exposes `K_scal3r` but does not load camera
extrinsics from the camera files. Before Stage2-C, audit the real camera NPZ
schema and add an explicit, tested camera-from-world tensor when available.
Do not silently call camera-space SMPL translations world-space. If reliable GT
extrinsics are unavailable, temporal GT training can compare prediction and GT
motion residuals in camera space, but world-memory claims must be validated in
the VGGT bridge using decoded and scale-consistent extrinsics.

## 6. Hierarchical Correction Model

### 6.1 Continuous regional correction field

TRSR does not enumerate correction candidates or classify a hand-written error
mode. Each regional query attends to its multi-scale local scene patch and
directly predicts a continuous desired displacement for that body region:

```text
region displacement vote: [B,S,Q,A,3]
region reliability gate:   [B,S,Q,A,1]
region uncertainty:        [B,S,Q,A,1]
```

The three-vector is expressed in a stable local basis such as camera ray,
region normal, and a tangent direction, then converted to camera/world XYZ. A
bounded `tanh` scale limits each vote, but there is no discrete `2/5/8 cm`
catalog and no candidate-selection classification loss.

The region gate is learned from visibility, local patch consistency, scene
confidence, and temporal state. Invalid or unobservable regions contribute no
vote. Clean examples train every valid vote toward zero.

### 6.2 Continuous common/local decomposition

Region votes are jointly interpreted rather than independently applied to mesh
vertices. A robust confidence-weighted set aggregator estimates the component
shared across many regions:

```text
common_delta = robust_set_aggregate(region_votes, reliability)
local_vote[a] = region_vote[a] - common_delta
```

The common component proposes a bounded root translation. The residual local
field contains evidence that cannot be explained by translating the whole
person and is sent to the joint decoder. Coherence and uncertainty remain
continuous scalars; there is no `global/local/mixed` class label.

This decomposition protects against residual Stage1 scale error: if most body
regions agree along the camera ray, the common component absorbs the evidence
and the local residual is small. Stage2 still does not update scene scale in the
first version; it logs the coherent residual for later scale diagnostics.

### 6.3 Kinematic region-to-joint decoding

Use masked cross-attention from 24 joint tokens to the local regional votes
allowed by the region bank. A hand query can influence wrist/elbow/shoulder,
while a foot query can influence ankle/knee/hip. Torso evidence cannot directly
rotate an ankle. The decoder learns how multiple continuous region constraints
should be reconciled when they agree or conflict.

Predict:

```text
root translation delta: [B,S,Q,3]
joint SO(3) log delta:   [B,S,Q,24,3]
joint update gate:       [B,S,Q,24,1]
person uncertainty:      [B,S,Q,1]
```

Apply pose updates by composition:

```text
R_new[j] = Exp(delta_omega[j]) @ R_current[j]
```

Do not add or EMA axis-angle/6D rotation components directly. Use joint-specific
angle limits, sparse delta regularization, and optional root-orientation bounds.
Betas remain the track-level NLF estimate or a robust track aggregate. After
decoding the updated SMPL mesh, supervision compares the realized regional
vertex motion with the continuous region displacement field; the field is a
constraint representation, not a free-form mesh offset.

### 6.4 True recurrent re-probing

Use two or three shared-weight iterations:

```text
decode current SMPL
build current regional anchors
query scaled scene and reliability
read track memory and update continuous region votes
decode common root and local joint updates
compose SMPL parameters
decode and query again
```

Every iteration must observe a newly decoded mesh. Step bounds should decay
across iterations. The rollout stops early when valid residuals are small,
uncertainty is high, or a proposed update worsens reliable geometry.

## 7. Track-Aware Temporal State

### 7.1 Separate scene state and person state

Maintain one stream-level scene state:

```text
stable log scale/bias, confidence, camera continuity, last frame
```

Maintain one detached state per persistent person ID:

```text
last refined root in world coordinates
root velocity/acceleration
24 local rotation matrices or relative SO(3) motion
stable betas
pooled regional interaction token
per-region contact/collision state and uncertainty
previous common/local correction confidence, tracking quality, last-seen frame,
missing count
```

### 7.2 Temporal fusion

Temporal memory is a prior, not a replacement observation. Fuse the current
region token with a constant-velocity/root prediction and previous interaction
token through a confidence gate. Supervise prediction dynamics against GT
dynamics when available; do not simply force velocity toward zero.

Use asymmetric hysteresis:

- a new contact/collision correction needs repeated reliable evidence;
- an established support may survive one-frame depth dropout;
- strong motion, camera discontinuity, low track quality, or contradictory
  geometry weakens memory;
- ID change, excessive gap, or implausible jump clears memory;
- one person's state is never borrowed by another query slot.

For offline clips, process frames causally during evaluation even if VGGT sees
the complete clip. For streaming, persist tracker, scene state, and person
memory across overlapping chunks.

### 7.3 Multi-person interaction

Version 1 keeps memories independent by track. A later optional branch may use
broad-phase region proximity and cross-person attention for human-human
penetration. It must use a permutation-invariant set operation and cannot assume
query index stability.

## 8. Training Curriculum

### Stage2-A: region-bank and probe observability audit

Use GT SMPL, GT camera, and GT metric depth without training the refiner. Audit
region coverage, self-depth rejection, multi-scale patch validity, and whether
known continuous SMPL perturbations produce consistent changes in the relevant
regional observations.

This is an **observability audit**: it asks whether the local measurements carry
enough information to infer a correction. It does not generate or select a set
of correction candidates. For each synthetic root/joint perturbation, report:

```text
affected-region response magnitude and sign consistency
unaffected-region leakage
valid 3x3/7x7/annulus patch coverage
finite-difference observation sensitivity by controlling joint
self-depth/background rejection rate
```

### Stage2-B1: single-frame continuous root correction

Freeze Stage1 and all upstream models. Train the regional displacement votes,
continuous common-component aggregator, and bounded root translation from a
mixture of clean GT and synthetic ray/tangent/support perturbations. This
validates continuous correction recovery and no-op safety.

### Stage2-B2: single-frame regional articulation

Add kinematic joint deltas. Perturb only valid controlling chains, with more
hand/foot/limb examples than torso examples. Include clean identity, one-sided
penetration, rotation geodesic, joint/vertex, reprojection no-worse, sparse
delta, and per-rollout monotonic losses.

Supervision is direct and continuous: the perturbed SMPL parameters are input,
and the clean GT root/rotations/vertices are targets. No artificial movement
catalog or correction-class label is constructed.

Do not treat every BEDLAM human-scene pair as physically correct contact. Only
reliable geometric regions receive contact/collision supervision; ambiguous
regions still contribute GT parameter recovery and no-op consistency.

### Stage2-C: causal temporal training

Increase sequence length gradually (`4 -> 8 -> 12`). Use GT track IDs first and
simulate detector behavior:

```text
temporary missed detections
track gaps and quality drops
small base-SMPL jitter
occluded region dropout
occasional ID reset/swap negatives
```

Train root/joint velocity and acceleration residuals, temporal no-worse,
contact hysteresis, foot sliding, and memory confidence. Betas consistency is a
track-level constraint, not per-frame shape regression.

### Stage2-D: real-inference bridge

Use the exact inference front end:

```text
RGB -> VGGT camera/depth/features
RGB + VGGT K -> NLF detector + SMPL
frozen Stage1 -> scaled depth
persistent base-SMPL tracker -> track IDs
TRSR -> refined SMPL
```

Cache or run frozen NLF/VGGT/Stage1 outputs under a versioned
`outputs/preprocess/` directory. Match detector people to BEDLAM GT only for
supervision. Mix synthetic GT examples with this bridge so the model does not
overfit either perfect geometry or one frozen predictor's errors.

## 9. Losses

Apply relevant losses at every rollout step:

```text
L = lambda_vote    * continuous regional displacement consistency
  + lambda_root    * robust root translation loss
  + lambda_rot     * SO(3) geodesic pose loss
  + lambda_joint   * camera/world joint loss
  + lambda_vertex  * sampled/full vertex loss
  + lambda_region  * reliable one-sided regional collision/support loss
  + lambda_clean   * clean-state identity loss
  + lambda_sparse  * bounded sparse joint-delta regularization
  + lambda_mono    * rollout no-worse/monotonic improvement
  + lambda_temp    * track-aligned velocity/acceleration residual loss
  + lambda_slide   * contact-conditioned foot sliding loss
  + lambda_cal     * uncertainty/abstention calibration
```

Important safeguards:

- local pose update is suppressed by continuous common/local decomposition when
  residual evidence is globally coherent;
- unsupported or invisible regions cannot generate an attraction loss;
- clean examples must have near-zero root, pose, and vertex displacement;
- Stage1 scale and VGGT/NLF hashes remain unchanged;
- old align/contact/grounding heads are mutually exclusive with TRSR in the
  mainline to prevent repeated overwrite of `hsi_refined_*` outputs.

## 10. Evaluation And Gates

### G0: deterministic region bank

Check 6890-vertex exact coverage, topology connectivity, symmetry, region
radius, controlling chain validity, and Viser coloring of all regions.

### G1: probe observability and controllability audit

No refiner training. Measure reliable-region coverage, patch-scale ablations,
self-depth rejection, affected-joint observation sensitivity, unaffected-region
leakage, and whether continuous root/joint perturbations produce distinguishable
regional feature changes. Here **controllability** means that the available SMPL
root/joint parameters can move the observed regions in the required directions;
it does not mean choosing from hand-written movement candidates.

### G2: forward/backward and fixed-subset overfit

Verify finite shapes/gradients, frozen hashes, true per-step re-probe changes,
continuous correction recovery, common/local decomposition, clean invariance,
and monotonic rollout error.

### G3: held-out synthetic distribution

Require improvement over frozen Stage1 plus NLF/GT seed, report separately for
root, hands, feet, limbs, torso, visibility, and uncertainty bins.

### G4: temporal/track corruption gate

Evaluate root/joint acceleration and jerk, foot sliding, memory reset behavior,
ID-switch contamination, gaps, and clean-sequence displacement.

### G5: NLF/VGGT/Stage1 bridge

Use complete held-out sequences and NLF detector. Compare:

```text
base NLF + scaled scene
old root-only align head
old contact/grounding baseline
TRSR without temporal memory
TRSR with temporal memory
```

### G6: real-video Viser

Display base/refined meshes, region colors, valid probes, local normals,
continuous region votes, root/joint deltas, uncertainty, track ID, and memory
reset events.

Primary metrics:

```text
MPJPE/PVE/root error and no-worse rate
penetration depth/rate and collision ratio by body region
float/support distance and foot sliding
root/joint acceleration, jerk, and frame-to-frame vertex jitter
clean-person displacement
probe coverage, perturbation-recovery error, and abstention calibration
ID switches and memory-contamination rate
runtime and peak memory per person/frame
```

## 11. Implementation Boundaries

Recommended new ownership:

```text
vggt_omega/models/heads/hsi_regional_surface_refiner.py
vggt_omega/models/geometry/smpl_region_bank.py
vggt_omega/models/geometry/regional_scene_probe.py
vggt_omega/tracking/hsi_regional_track_memory.py
configs/train_smpl_hsi_stage2_regional_*.yaml
scripts/train/train_smpl_hsi_stage2_regional_*.sh
scripts/smoke/check_hsi_stage2_regional_*.sh
scripts/vis/serve_hsi_stage2_regional_refiner.sh
```

Baseline behavior remains the default. The new head needs an explicit enable
flag and a startup contract that rejects simultaneous overwrite by legacy
align/contact/grounding heads.

Checkpoint composition should be explicit:

```text
VGGT baseline
+ accepted box-free Stage1 scale checkpoint (frozen and hashed)
+ new TRSR checkpoint
```

Stage2 outputs should save the scale and TRSR prefixes required for deployment,
without saving VGGT/NLF weights or per-epoch duplicates.

## 12. Recommended First Implementation Slice

Do not implement the complete temporal pose refiner in one change. The first
safe slice is:

1. deterministic 96-region bank and Viser audit;
2. box-free multi-scale regional patch attention using GT depth/K;
3. no-training G0/G1 observability/controllability report for continuous root
   and joint perturbations;
4. continuous region-vote plus root-only common-component decoder with true
   two-step re-probing;
5. only after those pass, add kinematic joint routing;
6. only after single-frame geometry passes, move tracking before refinement and
   add causal memory.

This sequence tests whether the observations contain enough information before
committing to a large trainable architecture.

## 13. Main Risks And Stopping Rules

- If G1 regional observability is poor, improve patch sampling, self-depth
  rejection, or region allocation before training a larger network.
- If self-depth is frequently mistaken for environment, fix exclusion/depth
  ordering before adding pose capacity.
- If globally coherent residuals trigger local joint updates, fix the
  continuous common/local decomposition; otherwise Stage2 will hide scale
  errors by deforming humans.
- If GT succeeds but the NLF/VGGT bridge fails, address the teacher-forcing
  domain gap rather than increasing epochs.
- If tracking IDs are unavailable before refinement, temporal memory remains
  disabled; posthoc display IDs are insufficient.
- If old heads and TRSR both overwrite refined SMPL, reject the configuration.
- If clean people move, rollout worsens reliable geometry, or memory crosses
  IDs, stop the experiment regardless of average loss.
