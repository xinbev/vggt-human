# TRSTR V2 Training Analysis

## Verdict

The completed seven-epoch v2 run is effective but not converged to the desired
strong-correction quality. Step curves are noisy by construction; epoch-level
TRSTR metrics show consistent improvement.

Source artifacts:

```text
outputs/train/smpl_hsi_stage2_trstr_v2_strong/metrics_epoch_0001.json
...
outputs/train/smpl_hsi_stage2_trstr_v2_strong/metrics_epoch_0007.json
outputs/train/smpl_hsi_stage2_trstr_v2_strong/resolved_config.json
```

Accepted train-selected checkpoint:

```text
checkpoint_top_train_epoch_0007_metric_hsi_trstr_xy_refined_l2_p90_m_0.709676.pt
```

## Epoch Trend

| Metric | Epoch 1 | Epoch 7 | Change |
| --- | ---: | ---: | ---: |
| TRSTR translation loss | 0.0623 | 0.0478 | -23.2% |
| refined translation L1 | 0.1883 m | 0.1430 m | -24.1% |
| overall improvement rate | 72.3% | 75.7% | +3.4 pp |
| refined L2 p90 | 1.034 m | 0.884 m | -14.5% |
| refined camera-XY L2 p90 | 0.807 m | 0.710 m | -12.0% |
| strong refined L2 mean | 0.716 m | 0.565 m | -21.1% |
| clean displacement L1 | 0.0220 m | 0.0124 m | -43.6% |
| monotonic violation rate | 29.6% | 26.3% | -3.3 pp |

The input distribution stayed stable: base L1 remained approximately `0.30 m`
and base L2 p90 approximately `1.43 m` across epochs. The refined improvements
therefore reflect learning rather than easier later samples.

## Why Step Curves Oscillate

Batch size is two frame samples. Each frame independently selects clean,
scale-only, NLF-only, or mixed perturbation. NLF perturbations additionally
select fine, medium, or strong Gaussian components. Consecutive steps can thus
change from centimeter-level clean samples to meter-level mixed samples. Raw
step loss is not expected to be monotonic.

Strong-subset metrics also write zero when a batch contains no qualifying
strong sample. Their raw W&B step curves are therefore coverage-dependent and
should not be interpreted as continuous optimization curves.

## Misleading Total Loss

The v2 YAML did not explicitly zero all legacy Hungarian base-loss defaults.
In the resolved run:

```text
giou_weight default: 2.0
loss_giou: approximately 1.0
```

This contributes an approximately constant `2.0` floor to `loss_total` for the
box-free GT path. Consequently:

```text
loss_total: 2.317 -> 2.248  (visually almost flat)
TRSTR translation loss: 0.0623 -> 0.0478
```

The constant term does not explain the learned TRSTR improvement and should be
removed in any continuation config. The completed run's resolved config and
metrics must remain unchanged for reproducibility.

## Remaining Risk

V2 reduces strong mean error from about `1.13 m` input to `0.57 m`, but does not
fully solve strong correction. Epoch 7 still improves over epoch 6, so the run
has not shown a hard plateau. However, there is no fixed validation set and all
reported values are training-distribution metrics. Real-sequence inference is
required before accepting v2.

Do not select a checkpoint from `loss_total`. The current top-1 selection by
`metric_hsi_trstr_xy_refined_l2_p90_m` correctly chose epoch 7, but it is still
a train-selected checkpoint.

## Recommended Next Gate

1. Run real inference with the epoch-7 v2 checkpoint and record actual TRSTR
   displacement and visual alignment.
2. Build a deterministic fixed evaluation mixture separated into clean,
   scale-only, NLF-only, and mixed subsets.
3. If continuation is needed, create an independent v2.1 config that zeros all
   non-TRSTR loss weights and monitors fixed-evaluation strong/clean metrics.
