# HSI Stage3 Contact Refinement Design

## Goal

Stage1 has learned metric scene scale and Stage2 has learned small human-scene translation alignment. Stage3 focuses on contact quality:

- reduce foot/sole penetration into the support surface
- reduce visible foot floating
- keep translation and SMPL quality from regressing
- add temporal foot-sliding control after single-frame contact is stable

This stage does not fine-tune VGGT aggregator, camera head, dense depth head, or NLF. Checkpoints remain HSI-only.

## Main Files

- Config: `configs/train_smpl_hsi_nlf_stage3_contact_refine.yaml`
- Stage3-A train: `scripts/train/train_smpl_hsi_nlf_stage3_contact_refine.sh`
- Stage3-B train: `scripts/train/train_smpl_hsi_nlf_stage3_temporal_contact.sh`
- Smoke: `scripts/smoke/check_hsi_stage3_contact_refine.sh`
- Metrics: `scripts/eval/eval_smpl_hsi_nlf_stage3_contact_diagnostics.sh`
- Visual debug: `scripts/vis/vis_smpl_hsi_nlf_stage3_contact_debug.sh`

## Model Setup

Stage3 uses the real inference path:

```text
VGGT depth + VGGT camera + NLF SMPL + Stage1 scale + Stage2 human-scene align + HSI contact refinement
```

Frozen:

- VGGT aggregator
- VGGT camera head
- VGGT dense depth head
- HSI scene affine scale/bias
- HSI beta delta by default

Trainable:

- HSI pose/transl/contact heads
- last two HSI transformer blocks
- HSI human-scene align head

The learned scene scale is treated as fixed geometry. Contact loss should not relearn scale.

Contact teacher generation uses `loss.hsi_contact_teacher_camera_source=gt` by default. This means GT SMPL contact labels are projected with BEDLAM `K_scal3r`, while predicted HSI feet still sample the scaled VGGT scene using the active HSI/VGGT camera path. This avoids using a biased VGGT K to decide whether a GT foot is truly touching GT depth.

## Contact Losses

The key losses are already implemented in `vggt_omega/training/hungarian_losses.py`.

### Sole Contact

`loss_hsi_foot_sole_contact`

Uses a set of low template-y SMPL sole vertices. GT SMPL + GT depth decides which sole points are true contact points. HSI refined sole points are compared to scaled VGGT depth at their projected locations.

Metrics:

- `metric_hsi_foot_sole_float_m`
- `metric_hsi_foot_sole_penetration_m`
- `metric_hsi_foot_sole_contact_count`

### Local Support Plane

`loss_hsi_support_plane_contact`

Uses sole vertices and samples a local depth window around each projected foot point. This is more stable near foot boundaries than sampling one pixel. It estimates signed distance from the predicted sole point to local support geometry.

Metrics:

- `metric_hsi_support_plane_float_m`
- `metric_hsi_support_plane_penetration_m`
- `metric_hsi_support_plane_signed_m`
- `metric_hsi_support_plane_contact_count`

### Foot Sliding

`loss_hsi_foot_sliding`

Used only in Stage3-B with multiple frames. When GT contact says a foot is in contact in adjacent frames, the predicted contact foot residual should not slide over time.

Metrics:

- `metric_hsi_foot_sliding_l1`
- `metric_hsi_foot_sliding_contact_count`

## Stage3-A: Contact Detail

Purpose: single-frame contact cleanup.

Default command:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

DATA_ROOT=/home/zhw/xyb_space \
BEDLAM_ROOT=/home/zhw/xyb_space/bedlam/processed_bedlam \
PREPROCESSED_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/preprocess/bedlam_boxes \
STAGE2_CKPT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt \
CUDA_VISIBLE_DEVICES_VALUE=7 \
BATCH_SIZE=16 \
NUM_WORKERS=12 \
NLF_INTERNAL_BATCH_SIZE=128 \
MAX_HUMANS=20 \
NUM_VIEWS=2 \
EPOCHS=3 \
LR=2e-6 \
bash scripts/train/train_smpl_hsi_nlf_stage3_contact_refine.sh
```

Smoke first:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

DATA_ROOT=/home/zhw/xyb_space \
BEDLAM_ROOT=/home/zhw/xyb_space/bedlam/processed_bedlam \
PREPROCESSED_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/preprocess/bedlam_boxes \
STAGE2_CKPT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt \
CUDA_VISIBLE_DEVICES_VALUE=7 \
BATCH_SIZE=12 \
MAX_STEPS_PER_EPOCH=200 \
bash scripts/smoke/check_hsi_stage3_contact_refine.sh
```

Proceed only if contact counts are non-zero and translation/joint deltas do not degrade.

## Stage3-B: Temporal Contact

Purpose: reduce foot sliding after Stage3-A is stable.

Default command:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

DATA_ROOT=/home/zhw/xyb_space \
BEDLAM_ROOT=/home/zhw/xyb_space/bedlam/processed_bedlam \
PREPROCESSED_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/preprocess/bedlam_boxes \
STAGE3A_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage3_contact_refine \
CUDA_VISIBLE_DEVICES_VALUE=7 \
BATCH_SIZE=8 \
NUM_WORKERS=12 \
NLF_INTERNAL_BATCH_SIZE=128 \
MAX_HUMANS=20 \
NUM_VIEWS=4 \
EPOCHS=3 \
LR=1e-6 \
bash scripts/train/train_smpl_hsi_nlf_stage3_temporal_contact.sh
```

## Diagnostics

Run before and after Stage3 training:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

DATA_ROOT=/home/zhw/xyb_space \
BEDLAM_ROOT=/home/zhw/xyb_space/bedlam/processed_bedlam \
PREPROCESSED_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/preprocess/bedlam_boxes \
SMPL_CKPT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage3_contact_refine/checkpoint_latest.pt \
CUDA_VISIBLE_DEVICES_VALUE=7 \
NUM_FRAMES=16 \
bash scripts/eval/eval_smpl_hsi_nlf_stage3_contact_diagnostics.sh
```

Visual debug:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

DATA_ROOT=/home/zhw/xyb_space \
BEDLAM_ROOT=/home/zhw/xyb_space/bedlam/processed_bedlam \
PREPROCESSED_ROOT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/preprocess/bedlam_boxes \
SMPL_CKPT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/smpl_hsi_nlf_stage3_contact_refine/checkpoint_latest.pt \
CUDA_VISIBLE_DEVICES_VALUE=7 \
NUM_FRAMES=16 \
PLY_FRAME_INDICES=0 \
bash scripts/vis/vis_smpl_hsi_nlf_stage3_contact_debug.sh
```

Use `PLY_FRAME_INDICES=7,20` or `PLY_FRAME_STEMS=seq_...` to inspect specific bad frames.

## Success Criteria

Primary:

- `metric_hsi_support_plane_penetration_m` decreases
- `metric_hsi_foot_sole_penetration_m` decreases
- `metric_hsi_foot_sliding_l1` decreases in Stage3-B
- contact counts are non-zero and stable

Guardrails:

- `metric_hsi_transl_l1_delta` should not become strongly positive
- `metric_hsi_joint_error_delta` should not become strongly positive
- visible projection quality should not drift
- person should not be pulled unnaturally toward a wrong floor surface

## Important Risk

Contact loss must stay gated by GT contact. Do not force all foot vertices to the depth surface. Swing feet should remain free, otherwise walking motion becomes stiff and the model may pull the whole body to satisfy an incorrect contact target.
