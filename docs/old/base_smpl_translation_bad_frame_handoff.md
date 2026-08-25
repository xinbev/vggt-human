# Base SMPL Translation Bad-Frame Handoff

This note is for a parallel agent working on the base SMPL translation problem while the main thread trains the HSI temporal no-worse route.

## Current Conclusion

The remaining bad-frame issue is not only an HSI temporal/refinement issue.

For at least one inspected failure case, the earliest single-frame SMPL output is already in the wrong 3D position before later HSI / scene / contact refinement. In that case, HSI is trying to refine from a bad initialization and may not be able to fully recover the person.

Therefore, the problem must be split into two independent tracks:

1. HSI video stabilization:
   - clip-level scene affine
   - temporal momentum
   - temporal no-worse guard
   - no contact for now
2. Base SMPL translation repair:
   - improve `pred_transl_cam` quality from the SMPL head or a compatible translation refinement module
   - keep pose/betas quality and previous checkpoints compatible

The current `noworse` training handles track 1. It cannot solve base SMPL bad translation by itself.

## Important Observations

### 121 HSI checkpoint is still the best single-frame baseline

Validated checkpoint:

```text
outputs/train/smpl_hsi_refine_20q/checkpoint_epoch_0121.pt
```

This checkpoint is effective:

```text
3D joints MPJPE:
base 0.07453m -> HSI 0.06160m

Vertices PVE:
base 0.07888m -> HSI 0.06732m

Translation L2:
base 0.06544m -> HSI 0.04886m

Projected joints:
base 6.74px -> HSI 4.93px
```

It also fixes most depth scale:

```text
raw VGGT depth median L1 -> HSI aligned depth median L1
5.826m -> 0.259m
```

So HSI itself is useful. The bad frame does not invalidate HSI.

### 0122 and later HSI checkpoints are unsafe

`checkpoint_epoch_0122.pt` suddenly collapses compared with `0121`.

Known failure:

```text
hsi_scene_scale saturates around exp(3) = 20.0855
environment PLY becomes broken / incomplete
humans can disappear in visualization
```

Do not use `checkpoint_latest.pt` blindly for old HSI runs. Use explicitly verified checkpoints.

### Clip-level scene affine is validated

For video inference, the per-frame HSI scene affine should not be used as the final scene scale.

Validated behavior:

```text
raw depth median L1      ~4.50m
per-frame HSI median L1 ~0.30m
clip-median median L1   ~0.30m
```

`clip_median` keeps almost the same depth accuracy while eliminating clip-level scene scale/bias jitter.

This belongs to the HSI/video track, not the base SMPL translation track.

## Bad-Frame Diagnosis

The user inspected bad-frame PLY outputs, especially under a path like:

```text
ply/0020_seq_000000_0100/
```

Visual finding:

```text
The bad person position is already wrong in the earliest/base single-frame SMPL reconstruction.
The later HSI/environment correction is not the first source of the error.
```

Interpretation:

```text
If base `pred_transl_cam` is wrong, HSI starts from a bad person location.
Temporal momentum/no-worse can reduce jitter, but it may also preserve a bad base if used too strongly.
Contact losses can pull feet toward a plane, but they cannot reliably fix a globally wrong root translation.
```

So the next translation work should focus on the base SMPL translation estimate, not only HSI residual post-processing.

## Failed Translation Experiment To Avoid Repeating

There was a later raw-depth / geometry translation residual route that was rolled back.

Known output directory from that failed route:

```text
outputs/train/smpl_geo_translation_depth_residual
```

Observed training signs:

```text
smpl_geo_gate_mean approximately 1e-13
smpl_geo_residual_l1 approximately 1e-6
smpl_geo_anchor_depth_l1 around 3m
smpl_geo_anchor_depth loss large
```

Meaning:

```text
The residual/gate effectively shut itself off.
The geometry/depth anchor loss remained large.
The method did not learn a useful translation correction.
```

Also, the run filled disk with large checkpoints:

```text
checkpoint_epoch_0001.pt ... checkpoint_epoch_0019.pt ~= 4.89GB each
checkpoint_epoch_0020.pt was partial/corrupted due to disk full
```

User planned or completed git rollback to:

```text
87d718b bad ply vis
```

Do not continue that exact raw-depth residual implementation.

## Why The Raw-Depth Translation Route Likely Failed

The failure is not proof that translation refinement is impossible. It suggests the specific design was too brittle.

Likely causes:

1. Raw VGGT depth is not metrically stable globally before HSI/clip affine.
2. Depth contains far/outlier regions and local sampling can be noisy.
3. If the translation correction is gated too strongly or regularized against base too hard, it learns to do nothing.
4. If the anchor-depth objective dominates, it can conflict with GT SMPL translation and 3D joints.
5. A direct geometry residual after an already-trained HSI head may not expose the right features to infer missing root translation.

User preference / assumption for future design:

```text
For the next design, camera and depth outputs from the current project can be trusted as available cues.
Use them if helpful.
But the goal is still best final 3D effect, not preserving any failed implementation.
```

## Recommended Direction For The Parallel Agent

Goal:

```text
Improve base SMPL `pred_transl_cam`, especially bad-frame root translation, without breaking pose/betas and without requiring a new incompatible checkpoint format.
```

Preferred design:

```text
Add a config-gated base SMPL translation refinement module, or strengthen the existing SMPL head translation branch.
Do not replace the full HSI pipeline.
Do not directly import `.reference` code.
Keep old checkpoints loadable with strict=false or missing new module keys only.
```

The new design should be compatible with existing checkpoints:

```text
checkpoint_epoch_0121.pt should still load.
New module keys can be missing when loading older checkpoints.
Baseline path must stay available behind config switches.
```

### Input Cues To Consider

Use existing project signals:

```text
SMPL query token / decoder output
pred_pose_6d
pred_betas
pred_transl_cam initial estimate
pred_boxes / GT box prior during training
VGGT pose_enc / camera intrinsics
VGGT depth
local human ROI scene tokens
local depth samples around projected body anchors
```

A reasonable module shape:

```text
BaseSMPLTranslationRefiner
  inputs:
    initial SMPL query features
    current pose/betas/transl
    projected joints / anchors
    local depth statistics in human ROI
    camera ray / intrinsics features
  output:
    delta_transl_cam [B, S, Q, 3]
    optional confidence/gate [B, S, Q, 1]
```

Then:

```text
pred_transl_cam_refined = pred_transl_cam + gated_delta
```

Expose both:

```text
pred_transl_cam              # original/base
pred_transl_cam_refined      # optional refined branch
```

Or, if integrating into existing loss path is easier:

```text
use refined transl as `pred_transl_cam` only when config enables the branch
keep original under `base_pred_transl_cam` for diagnostics
```

### Camera / Plucker Ray Idea

The user asked whether representing camera parameters as Plucker rays may help.

Recommendation:

```text
It can help if used as a feature encoding for translation/depth reasoning.
It should not be a new dependency or a major architectural rewrite.
```

Practical use:

```text
For each body anchor projection:
  compute normalized image coordinate
  unproject with camera intrinsics to a ray direction
  encode ray direction and possibly ray moment/origin if camera extrinsic convention is available
  concatenate with depth residual and anchor 3D features
```

Since the current project works in camera coordinates, a simple ray-direction encoding may be enough:

```text
ray_dir = normalize(K^-1 [u, v, 1])
```

Full Plucker line may be useful later if multi-view/world-frame camera extrinsics become central. For the immediate single-frame camera-space translation problem, ray direction + depth statistics is the safer minimum.

## Loss Design For Translation Repair

The user explicitly wants 3D-space correctness. 2D constraints are allowed but should not dominate.

Primary losses:

```text
transl_cam L1 / SmoothL1
joints3d absolute L1
vertices or joints absolute L1, if affordable
root depth/Z loss
```

Auxiliary losses:

```text
projected joints 2D with small weight
projected bbox with small weight
human ROI depth consistency with robust clipping
anchor depth consistency only as auxiliary, not dominant
```

Protection losses:

```text
pose/betas no-worse or frozen
translation delta regularization, but not so strong that correction gate collapses to zero
bad-frame upweighting if diagnostics identify hard samples
```

Avoid:

```text
large raw depth anchor loss before robust clipping / scene affine
hard contact loss at this stage
training a gate that can collapse to exactly zero without pressure to fix hard translation errors
letting 2D projection dominate 3D translation
```

Suggested loss weights to start conceptually:

```text
transl_cam_weight: high
joints3d_weight: high
vertices_weight: medium
projected_joints2d_weight: very low
depth_anchor_weight: low, robust clipped
delta_reg_weight: low-to-medium
gate_reg_weight: very low or initially disabled
```

## Suggested Experimental Plan

### Stage T0: Diagnostics First

Before training, produce or reuse diagnostics that compare:

```text
base pred_transl_cam vs GT transl_cam
hsi refined transl vs GT transl_cam
per-person / per-frame translation error
depth at projected body anchors
camera-ray consistency
```

Purpose:

```text
Identify whether errors are mostly Z-depth, lateral X/Y, or full 3D offset.
Do not infer this only from image overlay.
```

### Stage T1: Translation-only baseline repair

Freeze:

```text
VGGT aggregator
camera head
dense head
pose/betas branches if possible
HSI head
```

Train:

```text
only translation refiner or SMPL head translation branch
```

Use:

```text
GT box prior for the first controlled experiment
small dataset first
121-compatible initialization or best pre-HSI SMPL checkpoint depending on integration point
```

Success criteria:

```text
base transl L2 improves
base MPJPE/PVE improves or does not regress
pose/betas do not regress
bad-frame PLY shows base SMPL is closer before HSI
```

### Stage T2: Reconnect HSI

After base translation improves:

```text
load improved base SMPL head
load or retrain HSI from 121-style setup
use clip-level scene affine
then temporal momentum/no-worse
```

The idea is:

```text
Fix foundation first, then use HSI/video modules as refiners.
```

## Parallel Work Boundary

While the main thread trains:

```text
scripts/train/train_smpl_hsi_scene_then_temporal_noworse_from0121.sh
```

the parallel trans agent should not modify those files unless necessary.

Prefer adding new files, e.g.:

```text
configs/train_smpl_translation_refine.yaml
scripts/train/train_smpl_translation_refine.sh
scripts/eval/eval_smpl_translation_refine.sh
scripts/vis/vis_smpl_translation_refine.sh
```

If modifying shared model/loss code:

```text
add config switches
keep defaults disabled
preserve baseline predictions
keep checkpoint loading tolerant of missing new keys
```

## What Not To Claim

Do not claim:

```text
temporal no-worse solves base bad-frame translation
contact loss can fix wrong root translation
raw VGGT depth alone is enough for metric root recovery
the failed residual experiment proves translation refinement is impossible
```

Correct framing:

```text
temporal no-worse is a video stabilization guard.
base SMPL translation repair is a separate foundation problem.
Both are needed before final contact micro-finetune and large-scale training.
```

