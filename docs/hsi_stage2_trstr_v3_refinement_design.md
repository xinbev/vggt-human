# TRSTR V3 Refinement Design

## Motivation From V2

V2 is effective but incomplete. Across seven epochs:

```text
refined translation L1: 0.1883 -> 0.1430 m
refined L2 p90:         1.034  -> 0.884 m
refined camera-XY p90:  0.807  -> 0.710 m
strong refined mean:    0.716  -> 0.565 m
clean displacement:     0.0220 -> 0.0124 m
monotonic violation:    29.6%  -> 26.3%
```

V2 still leaves approximately half a meter on strong samples, moves clean
people by about 1.2 cm, and worsens the second iteration in roughly one quarter
of cases. Its `loss_total` is also obscured by a constant default GIoU term.

## Independence

```text
source:  v2 train-selected top-1
config:  configs/train_smpl_hsi_stage2_trstr_v3_refine.yaml
script:  scripts/train/train_smpl_hsi_stage2_trstr_v3_refine.sh
output:  outputs/train/smpl_hsi_stage2_trstr_v3_refine
W&B:     hsi_stage2_trstr_v3_refine
```

V3 uses a fresh AdamW optimizer and does not write to v1 or v2 directories.

## Loss Isolation

Every non-TRSTR Hungarian weight is explicitly zero, including pose, betas,
base translation, confidence, bbox, and GIoU. `loss_total` therefore represents
only the active TRSTR objective.

Active terms:

```text
region vote:              1.0
final translation:        4.0
region gate:              0.05
delta regularization:     0.0005
iteration monotonic:      3.0
strong translation:       2.0
clean identity:           2.0
final no-worse:           2.0
```

The strong term targets base error >= 0.5 m. Clean identity directly penalizes
movement on exact clean samples. Final no-worse penalizes any final error above
the input error plus 5 mm. Increased monotonic weight targets v2's 26% second
iteration violation rate.

## Training Distribution

V3 retains v2's coupled pseudo-GT construction and 2 m capacity. The case mix
is adjusted to:

```text
20% clean
30% scale-only
20% NLF-only
30% mixed
```

This increases clean protection and keeps 60% environment-scale exposure while
preserving substantial NLF/mixed strong correction.

## Fixed Evaluation

V3 evaluates the fixed BEDLAM Training-index range `90000..90255`. This is a
deterministic diagnostic subset, not an independent generalization benchmark.

Each dataset index deterministically cycles through clean, scale-only,
NLF-only, and mixed cases. Scale uses fixed 0.90/1.10 alternatives; NLF noise is
generated from a fixed seed. The same sample therefore has identical inputs
and targets every epoch. Validation batch size is four, so every batch contains
one sample from each case instead of writing zero-valued absent-subset metrics.

Checkpoint selection uses validation:

```text
metric_hsi_trstr_v3_selection =
  XY refined p90
  + 0.5 * strong refined mean
  + 5.0 * clean displacement
  + 0.25 * monotonic violation rate
  + 0.25 * final worsening rate
```

Only val-selected top-1 and an overwriting latest checkpoint are retained.

## Optimization

```text
initialization: v2 epoch-7 top-1
learning rate: 2e-6
epochs: 5
batch size: 2
temporal: disabled
v3 HSI scale: frozen
```

## Server Gates

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
bash scripts/smoke/run_hsi_stage2_trstr_v3_refine_one_step.sh
```

The one-step gate performs one optimizer step and four deterministic validation
steps, checks v2 prefix loading and gradients, and saves an isolated debug
checkpoint.

Formal training:

```bash
bash scripts/train/train_smpl_hsi_stage2_trstr_v3_refine.sh
```
