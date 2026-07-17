# HSI Scale / Translation / Contact Curriculum V2

## Purpose

This is the reproducible mainline after the successful Stage1 metric-scale run.
It keeps VGGT and NLF frozen and trains only project-local HSI modules:

```text
Stage1 checkpoint (frozen scene affine)
  -> Stage2-A GT metric translation denoising
  -> Stage2-B coherent GT / real NLF bridge
  -> Stage3-A1 root contact denoising
  -> Stage3-A2 lower-leg contact denoising
  -> Stage3-B coherent GT / real contact bridge
```

The old `train_smpl_hsi_nlf_four_stage.sh`, `stage2_abc_transl_refine`, and the
direct-real Stage3 contact scripts are retained as experiment history but are
not part of this mainline.

## Geometry Contract

Only two geometry modes are valid:

```text
gt_metric:
  GT perturbed SMPL + GT depth + K_scal3r
  GT depth bypasses Stage1 affine because it is already metric.

real_inference:
  NLF SMPL + raw VGGT depth/camera + frozen Stage1 affine
  Align/contact heads see Stage1-scaled metric depth and VGGT K.
```

`mixed` selects one complete mode per batch. It never combines GT SMPL with
VGGT K or NLF SMPL with GT K. The frozen HSI scene-affine head always reads raw
VGGT depth; `hsi_depth_override` is only consumed by downstream geometry heads.

The Stage2 align head uses bounded ray/tangent translation and robust local
correspondences constrained by the person box, max 3D distance, and per-person
MAD filtering. Scene scale/bias values are not align-head features.

The Stage3 contact head is a separate module after Stage2. It can only predict:

- root translation along an estimated support-plane normal;
- bounded SO(3) residuals for left/right hip, knee, and ankle;
- left/right contact logits.

It cannot change betas or scene scale.

## Server Preparation

All commands run from:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
```

Create deterministic sequence-disjoint train/val manifests:

```bash
BEDLAM_ROOT=/home/zhw/xyb_space/bedlam/processed_bedlam \
bash scripts/preprocess/prepare_hsi_sequence_manifests.sh
```

Outputs:

```text
outputs/preprocess/hsi_sequence_split_v2/train_sequences.txt
outputs/preprocess/hsi_sequence_split_v2/val_sequences.txt
outputs/preprocess/hsi_sequence_split_v2/overfit64_indices.csv
```

Before the full pass, generate a 256-window strict pilot in an isolated output
directory. Partial mode writes all three frames needed by each sampled clip and
marks the summary with `partial=true`:

```bash
SEQUENCE_MANIFEST=outputs/preprocess/hsi_sequence_split_v2/val_sequences.txt \
OUTPUT_ROOT=outputs/debug/hsi_contact_teachers_v3_strict_pilot256 \
MAX_WINDOWS=256 \
CUDA_VISIBLE_DEVICES_VALUE=7 \
bash scripts/preprocess/prepare_hsi_contact_teachers.sh
```

Validate every strict field in the pilot sidecars:

```bash
CONTACT_TEACHER_ROOT=outputs/debug/hsi_contact_teachers_v3_strict_pilot256 \
bash scripts/smoke/check_hsi_contact_teacher_strict.sh
```

After the pilot passes, precompute the full strict contact teachers on GPU 7:

```bash
BEDLAM_ROOT=/home/zhw/xyb_space/bedlam/processed_bedlam \
BOXES_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/preprocess/bedlam_boxes \
CUDA_VISIBLE_DEVICES_VALUE=7 \
bash scripts/preprocess/prepare_hsi_contact_teachers.sh
```

Outputs are under `outputs/preprocess/hsi_contact_teachers_v3_strict`. Each
sidecar contains local plane center/normal/RMSE, signed sole distance, foot
velocity, contact label, and validity for both feet of every visible-person
slot. Final teacher validity also requires the sole center to remain inside its
visible-person box and enough sole vertices to agree with GT depth within
`0.20m`; the raw plane and geometry validity fields are retained separately.

Audit the GT projection, boxes, depth, and synthetic contact shift before any
training:

```bash
CUDA_VISIBLE_DEVICES_VALUE=7 \
NUM_SAMPLES=24 \
bash scripts/vis/vis_hsi_curriculum_data_audit.sh
```

Output: `outputs/vis/hsi_curriculum_v2_data_audit`.

Run the depth-visible endpoint/contact audit before accepting the contact
sidecars. This only samples validation frames and does not regenerate teachers:

```bash
CUDA_VISIBLE_DEVICES_VALUE=7 \
NUM_SAMPLES=24 \
DEPTH_VISIBILITY_TOLERANCE_M=0.20 \
MIN_SOLE_VISIBLE_RATIO=0.25 \
bash scripts/vis/vis_hsi_contact_visibility_audit.sh
```

Output: `outputs/vis/hsi_contact_visibility_audit_v2`. The overview images
separate all projected vertices from GT-depth-consistent vertices. Per-person
four-panel images expose hand/foot endpoints, sole vertices, support samples,
plane normals, and synthetic float/penetration directions. `audit_summary.json`
reports existing-teacher rejection rates; `worst_feet.json` lists the feet that
most strongly violate the sole visibility and person-box checks.

To audit the previously generated unfiltered V2 teachers before regenerating,
set `CONTACT_TEACHER_ROOT=outputs/preprocess/hsi_contact_teachers_v2`
explicitly. Stage3 training defaults only to the strict V3 directory.

## Gates

Two-batch interface smoke:

```bash
GATE_STAGE=2A GATE_MODE=smoke CUDA_VISIBLE_DEVICES_VALUE=7 \
bash scripts/smoke/check_hsi_curriculum_v2.sh
```

Fixed 64-clip overfit gate:

```bash
GATE_STAGE=2A GATE_MODE=overfit CUDA_VISIBLE_DEVICES_VALUE=7 \
bash scripts/smoke/check_hsi_curriculum_v2.sh
```

Full-distribution 500-step gate:

```bash
GATE_STAGE=2A GATE_MODE=distribution CUDA_VISIBLE_DEVICES_VALUE=7 \
bash scripts/smoke/check_hsi_curriculum_v2.sh
```

Do not launch the full run unless the smoke is finite, the overfit improvement
rate exceeds 90%, and the distribution improvement rate exceeds 50%. Stage3
gates use `GATE_STAGE=3A1` after a Stage2-B top checkpoint exists; set
`STAGE2B_CKPT=/absolute/path/checkpoint_top01.pt` when using a separate gate
output root.

## Full Training

Recommended gated end-to-end launcher (automatically stops at the first failed
numeric gate):

```bash
CUDA_VISIBLE_DEVICES_VALUE=7 \
bash scripts/train/run_hsi_curriculum_v2_with_gates.sh
```

The lower-level launcher below is intended for resuming from an already passed
gate or running selected stages.

```bash
DATA_ROOT=/home/zhw/xyb_space \
BEDLAM_ROOT=/home/zhw/xyb_space/bedlam/processed_bedlam \
PREPROCESSED_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/preprocess/bedlam_boxes \
STAGE1_CKPT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/stage1_scale_linear_b20_gpu7/checkpoint_top_train_epoch_0003_loss_total_0.171740.pt \
CUDA_VISIBLE_DEVICES_VALUE=7 \
BATCH_SIZE_2A=24 BATCH_SIZE_2B=24 \
BATCH_SIZE_3A1=16 BATCH_SIZE_3A2=12 BATCH_SIZE_3B=12 \
bash scripts/train/train_smpl_hsi_scale_trans_contact_curriculum.sh
```

Run selected stages with `RUN_STAGES=2A`, `RUN_STAGES=2B`, or a comma-separated
list. `MAX_STEPS_PER_EPOCH` and `MAX_VAL_STEPS` remain exposed for diagnostics.

Output root:

```text
outputs/train/hsi_scale_trans_contact_v2/
```

Every stage writes `resolved_config.json`, `metrics_latest.json`, per-epoch
metrics, HSI-only `checkpoint_latest.pt`, and validation-selected stable
`checkpoint_top01/02/03.pt`. The next stage always loads `top01`.

## Safety Checks

- The curriculum launcher explicitly zeros dense-depth, anchor-depth, scene-scale,
  betas, and legacy gate losses. The smoke gate reads `resolved_config.json` and
  aborts if a generic provider default re-enables any of them.
- Checkpoint loading validates every tensor under required HSI prefixes.
- Stage2 hashes `hsi_refinement_head`; Stage3 hashes both scene refinement and
  translation alignment. Any frozen-weight change aborts training.
- VGGT, NLF, optimizer, ID tracking, temporal memory, foot sliding, full-scene
  depth loss, and far/sky depth are outside this curriculum.
- Stage2 top-k uses `metric_stage2_selection`; Stage3 uses
  `metric_stage3_selection`, both from the sequence-disjoint validation set.
- Windows local to one sequence may overlap inside train or val, but no sequence
  appears in both sets.

## Required Server Validation

Windows has no local torch/smplx/checkpoints. The following remain server-only:

1. SMPL/contact plane tensor forward and gradient checks.
2. NLF mixed-mode forward with real VGGT K/depth.
3. Two-batch smoke and 64-clip overfit.
4. GPU memory confirmation for the default batch sizes.
5. Visual confirmation that GT SMPL, GT depth, boxes, and teacher planes agree.

Final walking-sequence Viser:

```bash
CUDA_VISIBLE_DEVICES_VALUE=7 PORT=8080 \
bash scripts/vis/serve_hsi_curriculum_v2_walking.sh
```
