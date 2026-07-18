# HSI Stage3 Contact-Only From Frozen Stage2

## Purpose

This experiment keeps the accepted Stage2 human-scene alignment checkpoint and
trains only the contact refinement head. It targets visible foot floating and
penetration against the strict BEDLAM support-plane teacher.

The V4 translation experiments are not part of this path.

## Geometry Path

```text
RGB -> frozen VGGT depth/K -> frozen Stage1 scene affine
    -> frozen Stage2 human-scene align head -> NLF SMPL
    -> contact refinement head -> refined SMPL
```

Frozen components:

- VGGT aggregator, camera head, and depth head
- Stage1 `hsi_refinement_head`
- accepted Stage2 `hsi_human_scene_align_head`
- NLF provider
- HSI betas and the HSI backbone

Trainable component:

- `hsi_contact_refine_head.` only

The root-normal branch is trained first. The lower-body pose branch is frozen
in this first contact-only run. Swing-foot no-pull and contact classification
losses protect non-contact feet from being pulled toward the support plane.

The contact classifier also uses inference-available temporal features. For
each foot, the head converts sole centers from camera coordinates to world
coordinates using the VGGT pose encoding, matches neighboring frames using
`assigned_track_ids`, and computes the neighboring sole displacement. GT
`contact_foot_velocity_m` is kept as a teacher diagnostic only and is never
passed as a model input.

## Teacher Contract

The data loader must read:

```text
outputs/preprocess/hsi_contact_teachers_v3_strict
```

The teacher contains GT-K/GT-depth support planes, contact labels, signed foot
distances, visibility validity, and low-speed contact filtering. Missing teacher
files are fatal for this experiment.

## Checkpoint Contract

Input checkpoint:

```text
outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt
```

The input Stage2 checkpoint is read-only. Contact outputs are written to:

```text
outputs/train/hsi_stage3_contact_from_stage2_full
```

The output checkpoint saves the frozen Stage1/Stage2 HSI prefixes and the new
`hsi_contact_refine_head.` prefix. The Stage1 and Stage2 prefixes are hashed
before and after training.

## Validation Order

1. Run `scripts/smoke/check_hsi_stage3_contact_from_stage2_full.sh` with three
   views, 200 train steps, and 20 validation steps.
2. Confirm non-zero contact teacher counts, finite contact gradients, unchanged
   frozen hashes, and non-zero contact-plane loss.
3. Inspect the Stage2-vs-contact Viser output on the walking sequence.
4. Only then run the three-epoch contact-only training.

Do not enable temporal foot sliding or lower-body pose refinement until root
normal contact correction reduces floating without increasing penetration or
translation/joint error.
