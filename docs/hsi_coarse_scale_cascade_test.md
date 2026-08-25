# HSI Analytic Coarse-Scale Cascade Test

## Hypothesis

Use NLF SMPL as a metric ruler before learned HSI correction:

```text
coarse scale = median(z_smpl / z_vggt)
coarse depth = VGGT depth * coarse scale
final depth = coarse depth * HSI residual scale + HSI residual bias
```

The estimator projects subsampled NLF SMPL vertices with VGGT intrinsics,
keeps only the nearest SMPL anchor at each pixel, filters invalid/out-of-range
ratios, and uses the median. This is a coarse robust estimate rather than a
single-point division.

## Test Variants

The diagnostic compares:

```text
raw_vggt
analytic_coarse
direct_hsi
coarse_then_hsi
```

For every variant it estimates the additional residual scale still required to
align sampled depth with NLF SMPL. A good correction should have residual scale
near `1` and lower anchor depth error.

## Server Command

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

CUDA_VISIBLE_DEVICES_VALUE=7 \
bash scripts/eval/evaluate_hsi_coarse_scale_cascade.sh
```

Output:

```text
outputs/eval/hsi_coarse_scale_cascade/summary.json
```

Interpret the aggregate fields:

- `coarse_scale`: analytic scale inferred directly from NLF SMPL and raw VGGT depth.
- `direct_hsi.required_residual_scale`: missing multiplier after the current model.
- `analytic_coarse.required_residual_scale`: should be near one by construction;
  anchor error indicates whether the estimate is geometrically useful.
- `coarse_then_hsi.required_residual_scale`: should remain near one and ideally
  improve anchor error over analytic coarse alone.
- `cascade_effective_scale`: product of analytic coarse scale and learned residual scale.

This is a two-forward diagnostic and is intentionally kept outside the baseline
inference path until its behavior is verified across several sequences.
