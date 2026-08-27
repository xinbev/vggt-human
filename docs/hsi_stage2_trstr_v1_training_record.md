# TRSTR V1 Spatial Training Record

## Status

This document freezes the first completed TRSTR spatial-only experiment as
`TRSTR v1`. Existing v1 config, launcher, output directory, W&B run, and
checkpoints must not be reused as v2 output targets.

## Artifacts

```text
config:
  configs/train_smpl_hsi_stage2_trstr.yaml

launcher:
  scripts/train/train_smpl_hsi_stage2_trstr.sh

output:
  outputs/train/smpl_hsi_stage2_trstr_v3_scale_spatial

accepted top-1:
  checkpoint_top_train_epoch_0005_loss_total_0.001113.pt

W&B group:
  hsi_stage2_trstr_spatial
```

The `v3_scale` phrase in the output name refers to the frozen HSI environment
scale overlay. This experiment is TRSTR v1.

## Training Contract

```text
GT pose/betas: read-only
GT transl_cam: perturbed online
GT metric depth and K: geometry supervision
TRSTR iterations: 2
temporal: disabled
regions: 96
```

Translation disturbance:

```text
ray: zero-mean Gaussian, std 0.075 m, clipped to +/-0.15 m
tangent x/y: zero-mean Gaussian, std 0.025 m, clipped to +/-0.05 m each
clean identity samples: 20%
```

Output bounds per iteration:

```text
ray vote: 0.20 m
tangent vote: 0.08 m
person aggregate per component: 0.22 m
```

## Real-Inference Observation

On the Human3R walking sequence, v1 produced:

```text
valid people: 88
mean translation correction: 0.0146 m
p90 translation correction: 0.0309 m
maximum translation correction: 0.0517 m
region-valid ratio: 0.9929
```

The newly identified case requires approximately `0.7 m` translation
correction. This is far outside v1's training disturbance and practical output
distribution. The insufficient correction is therefore recorded as a v1
coverage limitation, not treated as a reason to overwrite v1.

## Preservation Rule

TRSTR v2 may initialize from the v1 top-1 checkpoint, but must use separate
config, launcher, W&B group, output directory, top-k index, and checkpoint
files. No v2 command may write under the v1 output directory.
