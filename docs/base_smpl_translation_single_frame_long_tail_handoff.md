# Base SMPL Translation Single-Frame Long-Tail Handoff

This handoff is for the parallel/side agent that will continue solving the
remaining single-frame SMPL translation failures after the translation-ray-refine
branch has already been merged into the main HSI route.

## Short Version

The current bad single-frame visualization is **not** caused by forgetting to
enable the translation-ray-refine branch.

The inspected PLY was generated with the post-translation-refine mainline:

```text
outputs/train/smpl_hsi_temporal_after_translation_ray_refine/stage2_human_momentum_no_worse/checkpoint_latest.pt
```

and the visualization scripts explicitly set:

```text
model.smpl_enable_translation_refine=true
```

Therefore, the remaining issue is:

```text
The camera-ray translation refiner improves average translation quality, but
some long-tail single frames still have wrong person root/translation after the
refiner. HSI then starts from this still-wrong location and does not recover it.
```

## Current Observed Frame

User inspected a single-frame PLY folder for:

```text
sequence: 20221013_3_250_batch01hand_orbit_bigOffice_seq_000000
frame:    seq_000000_0085
```

The folder contains nine PLY-style comparison files, including:

```text
seq_000000_0085_gt_mesh_top03.ply
seq_000000_0085_pred_mesh_top03.ply
seq_000000_0085_hsi_refined_mesh_top03.ply
seq_000000_0085_vggt_depth_mesh_hsi_aligned.ply
```

Visual observation:

```text
gt_mesh_top03 and hsi_refined_mesh_top03 do not overlap.
pred_mesh_top03 and hsi_refined_mesh_top03 are almost in the same position,
with only small pose differences.
gt_mesh_top03 and vggt_depth_mesh_hsi_aligned overlap very well.
```

Interpretation:

```text
The scene/depth affine is good for this frame.
The environment is already in the right metric space.
The person mesh is not at the GT person location.
HSI is not introducing a new large translation error; it mostly follows the
current base/refined SMPL location.
The remaining error is in the base SMPL translation/root location after the
translation refiner.
```

## Important Correction To The Previous Diagnosis

Earlier wording may have made it sound like "base SMPL" meant the old raw 0121
SMPL output. For the current after-translation-ray-refine visualization, that is
not true.

In the current code, when the refiner is enabled:

```text
pred_mesh_top03 = decoded predictions["pred_transl_cam"]
```

and `pred_transl_cam` has already been replaced by the camera-ray translation
refiner output.

The unrepaired value is preserved separately as:

```text
base_pred_transl_cam
```

but the current PLY export does **not** write a separate
`base_pred_transl_cam_mesh` file.

So in this context:

```text
pred_mesh_top03 means translation-refined base SMPL mesh.
```

## Code Paths That Confirm This

SMPL head:

```text
vggt_omega/models/heads/smpl_head.py
```

Relevant behavior:

```python
base_pred_transl_cam = pred_transl_cam
translation_refine_outputs = self.translation_refiner(...)
pred_transl_cam = translation_refine_outputs["pred_transl_cam"]

outputs["pred_transl_cam"] = pred_transl_cam
outputs["base_pred_transl_cam"] = base_pred_transl_cam
```

PLY export:

```text
scripts/vis/visualize_smpl_inference.py
```

Relevant behavior:

```python
transl_key = "hsi_refined_pred_transl_cam" if use_hsi_refined else "pred_transl_cam"
```

Therefore:

```text
seq_*.ply pred_mesh_top03 uses predictions["pred_transl_cam"].
If model.smpl_enable_translation_refine=true, this is already the refined
translation, not the pre-refiner translation.
```

Current wrapper scripts that enable the refiner:

```text
scripts/vis/vis_hsi_bad_frame_ply_after_translation_ray_refine.sh
scripts/vis/vis_hsi_bad_frame_sequence_ply_after_translation_ray_refine.sh
scripts/eval/eval_hsi_bad_frame_person_diagnostics_after_translation_ray_refine.sh
scripts/eval/scan_hsi_bad_translation_frames_after_translation_ray_refine.sh
```

They set:

```bash
SMPL_ENABLE_TRANSLATION_REFINE="${SMPL_ENABLE_TRANSLATION_REFINE:-true}"
--override "model.smpl_enable_translation_refine=${SMPL_ENABLE_TRANSLATION_REFINE}"
```

## Reproduce The Current Single-Frame PLY

Server command, using the current mainline after translation-ray-refine:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

CUDA_VISIBLE_DEVICES_VALUE=6 \
PLY_FRAME_STEMS=seq_000000_0085 \
OUTPUT_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/vis/hsi_bad_frame_0085_after_translation_ray_refine_ply \
bash scripts/vis/vis_hsi_bad_frame_ply_after_translation_ray_refine.sh
```

Expected key console lines:

```text
Checkpoint  : .../outputs/train/smpl_hsi_temporal_after_translation_ray_refine/stage2_human_momentum_no_worse/checkpoint_latest.pt
SMPL ray ref: true
Scene affine: clip_median
Use HSI     : true
GT prior    : true
```

Expected PLY directory:

```text
outputs/vis/hsi_bad_frame_0085_after_translation_ray_refine_ply/ply/0017_seq_000000_0085/
```

The frame index is `0017` because this clip starts at `seq_000000_0000` and the
frame stems in this dataset increase by 5:

```text
0000, 0005, ..., 0085
```

## Related Dataset-Wide Scan

A scanner was added to count how many similar long-tail cases exist in the
training split:

```text
scripts/eval/scan_hsi_bad_translation_frames.py
scripts/eval/scan_hsi_bad_translation_frames_after_translation_ray_refine.sh
```

Run smoke:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

CUDA_VISIBLE_DEVICES_VALUE=6 \
MAX_SAMPLES=100 \
BATCH_SIZE=1 \
bash scripts/eval/scan_hsi_bad_translation_frames_after_translation_ray_refine.sh
```

Run full scan:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

CUDA_VISIBLE_DEVICES_VALUE=6 \
MAX_SAMPLES=0 \
BATCH_SIZE=1 \
bash scripts/eval/scan_hsi_bad_translation_frames_after_translation_ray_refine.sh
```

Outputs:

```text
outputs/eval/hsi_bad_translation_scan_after_translation_ray_refine/
  hsi_bad_translation_scan_summary.json
  bad_frame_person_rows.csv
  bad_frame_summary.csv
  bad_sequence_summary.csv
  all_frame_person_translation_rows.csv
```

Default bad thresholds:

```text
base/hsi transl_l2 > 0.50m  => bad
base/hsi transl_l2 > 0.80m  => severe
base/hsi MPJPE    > 0.50m  => bad
HSI worse than base by > 0.05m => hsi_worse
```

Note: in this scanner, "base" also means the post-refiner
`pred_transl_cam` when `model.smpl_enable_translation_refine=true`.

## What The Side Agent Should Solve

The previous translation-ray-refine branch was a successful average-case
improvement, but it is not yet robust enough for long-tail single frames.

Known average improvement from the closeout:

```text
Original 0121 base transl L2: 0.06544m
T2 transl-heads + ray-refiner: 0.04956m
relative improvement: ~24.26%
```

New issue:

```text
Some individual frames still show large residual translation/root error after
the refiner. HSI cannot reliably fix these because HSI tokens attend around the
current SMPL anchors; if the anchor is already far from the GT person, the HSI
module sees the wrong local scene context.
```

Primary goal for the side agent:

```text
Reduce long-tail post-refiner SMPL translation failures, not just improve mean
translation metrics.
```

Recommended success metrics:

```text
1. Preserve or improve mean/median transl_l2, MPJPE, PVE.
2. Reduce p90/p95/p99 transl_l2.
3. Reduce count of frames with transl_l2 > 0.50m and > 0.80m.
4. Specifically improve known frames such as seq_000000_0085 and seq_000000_0100.
5. Do not break HSI downstream: HSI refined mesh should remain at least no worse
   than the post-refiner base on average.
```

## Recommended Next Diagnostics

Before designing another training route, first add/export direct comparisons for
the same frame:

```text
pre-refiner base_pred_transl_cam mesh
post-refiner pred_transl_cam mesh
hsi_refined_pred_transl_cam mesh
GT mesh
HSI aligned scene
```

This will answer:

```text
Did the refiner move the person in the right direction but not enough?
Did the refiner move the person in the wrong direction?
Did the refiner do almost nothing on the long-tail case?
Did HSI preserve or worsen the post-refiner translation?
```

Suggested implementation:

```text
Add optional PLY export for predictions["base_pred_transl_cam"] when it exists.
Name it seq_*_pre_refine_base_mesh_top03.ply.
Also write a per-person JSON row with:
  base_pred_transl_cam
  pred_transl_cam
  hsi_refined_pred_transl_cam
  gt_transl_cam
  L2 errors for each
```

## Possible Design Directions

Do not repeat the failed raw-depth residual route unchanged.

The failed route had symptoms like:

```text
smpl_geo_gate_mean ~ 1e-13
smpl_geo_residual_l1 ~ 1e-6
anchor_depth_l1 around 3m
```

That route effectively learned to shut itself off.

More promising directions:

1. Long-tail-aware translation training

   Increase weight on hard frames/persons where current refined `pred_transl_cam`
   has large error. Use robust hard mining, not only mean L1.

2. Quantile / tail loss

   Add p90/p95-style translation penalty or top-k hard example loss inside a
   batch so the network cannot hide long-tail failures behind good average
   metrics.

3. Explicit pre-vs-post refiner supervision

   Train the ray refiner with direct supervision on:

   ```text
   refined pred_transl_cam -> gt_transl_cam
   ```

   while monitoring how far it moves from `base_pred_transl_cam`.

4. Camera-ray + box geometry consistency

   Keep the ray basis idea, but strengthen constraints that GT/root must lie on
   the camera ray implied by the GT/prior box center. Avoid over-trusting raw
   VGGT depth as an absolute target.

5. Multi-hypothesis translation refinement

   For ambiguous frames, predict several bounded translation candidates along
   the camera ray and select/supervise the best during training. This may help
   when one residual estimate is trapped near a bad base.

6. Use scene/depth as a cue, not as an absolute teacher

   Current visual evidence says HSI aligned scene is often metrically correct.
   Depth can help, but do not use raw unaligned VGGT depth globally as the direct
   training target. If using scene, prefer human-ROI / near-person aligned depth
   and robust clipped losses.

## Visualization Caveat To Verify

Current PLY export selects top predictions by confidence and uses `query_index`
to pick the corresponding GT mesh:

```python
query_idx = int(item["query_index"])
gt_meshes.append(gt_points["vertices"][query_idx])
```

This is usually acceptable when using GT box prior and query slots correspond to
GT slots, but it should be verified for any non-default setup.

For a rigorous side-agent diagnostic, export the matched GT slot from Hungarian
matching or GT-prior slot identity explicitly, instead of assuming
`query_idx == gt_slot`.

This does not change the current core conclusion, because the environment/GT
scene alignment and pred-vs-HSI relation still indicate a post-refiner
translation issue. But it is worth fixing to make future PLY inspection
unambiguous.

## Files To Read First

Core model:

```text
vggt_omega/models/heads/smpl_head.py
vggt_omega/models/heads/hsi_refinement_head.py
vggt_omega/models/vggt_omega.py
```

Loss/training:

```text
vggt_omega/training/hungarian_losses.py
scripts/train/train_smpl.py
configs/train_smpl_hsi_temporal_momentum_noworse_after_scene.yaml
configs/train_smpl_hsi_after_translation_ray_refine.yaml
configs/train_smpl_translation_ray_refine.yaml
```

Diagnostics / visualization:

```text
scripts/eval/evaluate_hsi_sequence_person_diagnostics.py
scripts/eval/scan_hsi_bad_translation_frames.py
scripts/vis/visualize_smpl_inference.py
scripts/vis/visualize_hsi_clip_scene_affine_video.py
scripts/vis/vis_hsi_bad_frame_ply_after_translation_ray_refine.sh
scripts/vis/vis_hsi_bad_frame_sequence_ply_after_translation_ray_refine.sh
```

Prior handoffs:

```text
docs/base_smpl_translation_ray_refine_closeout_handoff.md
docs/base_smpl_translation_bad_frame_handoff.md
docs/hsi_mainline_after_translation_ray_refine.md
```

## What Not To Do

Do not assume the single-frame failure means HSI or scene affine failed.

Do not assume the trans-refine branch was disabled. It was enabled in the
current after-translation-ray-refine visualization scripts.

Do not optimize only visual contact/feet before fixing root translation. A
person can appear to touch the floor while still being globally offset from GT.

Do not report only mean translation. The current problem is a long-tail failure,
so p95/p99 and bad-frame counts are required.

