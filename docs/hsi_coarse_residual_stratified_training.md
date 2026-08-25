# HSI Coarse-Residual Stratified Training

## Pipeline

```text
clean GT depth
-> divide by stratified absolute scale S_extra
-> traditional SMPL/depth coarse estimator
-> multiply coarse estimate by stratified coarse-error factor
-> coarse-corrected depth enters HSI
-> HSI predicts residual scale/bias
-> final depth supervised by clean GT depth
```

Formally:

```text
D_disturbed = D_gt / S_extra
C_used = C_algorithm * E
D_coarse = D_disturbed * C_used
R_teacher = S_extra / C_used
D_final = D_coarse * R_pred + B_pred
```

## Non-Gaussian Sampling

Absolute correction buckets:

```text
10%  exactly 1
20%  log-uniform [0.25, 2]
50%  log-uniform [2, 12]
20%  log-uniform [12, 20]
```

Coarse residual-error buckets:

```text
30%  exactly 1
40%  log-uniform [0.67, 1.50]
25%  log-uniform [0.40, 2.50]
5%   log-uniform [0.25, 4.00]
```

The identity bucket teaches HSI to output residual scale one when coarse is
already correct. The hard buckets cover large VGGT-like absolute scales while
keeping HSI responsible for residual correction rather than the entire gauge.

## Coarse Failure Mask

A frame is valid only when the traditional estimator has enough projected
anchor pixels, finite in-range ratios, and acceptable relative MAD. Invalid
frames are excluded from both SMPL scale teacher and dense depth teacher. If an
entire batch has no supervised scale points, the optimizer step is skipped.
Fallback `coarse=1` is never treated as a teacher target.

## W&B

Only configured useful scalar metrics are uploaded. Runtime W&B Media charts
and Tables are disabled because W&B 0.28.2 can segfault while serializing them
inside the training process. The following scalar namespaces are uploaded from
the first optimizer step:

```text
scale_compare/residual_teacher
scale_compare/residual_pred

scale_compare/absolute_GT
scale_compare/traditional_coarse
scale_compare/final_pred
scale_compare/final_log_l1
```

In the W&B workspace, create one line panel with the two residual keys and one
line panel with the three absolute-scale keys. W&B then overlays teacher and
prediction without requiring Media objects in the trainer. Per-person behavior
is monitored through scalar min/mean/max/std and weighted-contribution std;
full person Tables remain available in code but are disabled by default for
process stability.

## Server Commands

One-step gate:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
bash scripts/smoke/run_hsi_coarse_residual_one_step.sh
```

Full continuation:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

CUDA_VISIBLE_DEVICES_VALUE=7 \
bash scripts/train/train_smpl_hsi_coarse_residual_stratified.sh
```

Output:

```text
outputs/train/smpl_hsi_coarse_residual_stratified
```

The run initializes from the completed box-free epoch-5 scale checkpoint,
uses learning rate `5e-6`, and preserves the bounded latest plus top-1
checkpoint policy.
