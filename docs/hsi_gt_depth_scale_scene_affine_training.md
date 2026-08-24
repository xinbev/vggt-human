# HSI GT-Depth Scale Scene-Affine Training

## Goal

The existing Stage2 `smpl_hsi_nlf_stage2_human_scene_align_full` checkpoint
trains `hsi_human_scene_align_head`, which corrects SMPL root translation. It
does not directly learn an environment-scale correction. The direct training
target for environment scale is the HSI scene affine branch:

```text
hsi_refinement_head.scale_delta
hsi_refinement_head.bias_delta
```

This experiment trains that branch with controlled GT geometry:

- GT SMPL
- GT camera intrinsics
- GT metric depth
- artificial multiplicative perturbation on GT depth

`smpl_provider=gt_perturbed` is configured with all SMPL perturbations at zero,
so NLF is not called and the head receives clean dataset SMPL. Dataset
`K_scal3r` overrides the VGGT camera intrinsics, and perturbed dataset depth
overrides VGGT predicted depth. BEDLAM SMPL is already stored in camera
coordinates for this training path, so no separate camera extrinsic is needed.

This training path is box-free. It does not load BEDLAM box sidecars, inject a
box prior into aggregator SMPL queries, filter GT SMPL with `boxes_mask`, or use
Hungarian box matching. `data.box_free_gt_slots=true` also disables fallback
box construction from legacy bbox/j2d annotation fields; zero box tensors remain
only as a compatibility placeholder for the shared batch schema. Raw dataset
SMPL slots are filtered online by projecting
subsampled GT vertices with `K_scal3r` and comparing their z values with clean
GT depth. A person is retained when at least 32 vertices pass the configured
image-bound and depth-visibility checks.

`model.smpl_use_aggregator_queries=false` also removes the retired HMR SMPL
query tokens from VGGT attention. `num_smpl_queries=20` remains only as the
maximum GT/NLF person capacity. This keeps both training and NLF-detector
inference on the same original RGB-to-VGGT token path.

The frozen VGGT aggregator still produces the RGB feature tensors expected by
the existing HSI architecture. Its predicted camera and depth are not used as
the geometry inputs. Removing the aggregator itself would change the module
architecture and would no longer be fine-tuning the existing checkpoint.

For every randomly sampled perturbation, the target scene scale is its inverse.
No particular correction value is treated as a special target.

## Baseline Preserved

The original Stage2 human-scene alignment checkpoint and config are not
modified. This is a new optional experiment:

```text
configs/train_smpl_hsi_gt_depth_scale_scene_affine.yaml
scripts/train/train_smpl_hsi_gt_depth_scale_scene_affine.sh
```

Default output:

```text
outputs/train/smpl_hsi_gt_depth_scale_scene_affine_boxfree
```

## Implementation

The training code adds an optional GT-depth perturbation branch in:

```text
scripts/train/train_smpl.py
```

The experiment activates it with a log-scale Gaussian schedule:

```text
training_prior.hsi_gt_depth_log_scale_std_schedule
```

When unset or set to `0.0`, existing training behavior is unchanged.

For `model.hsi_geometry_mode=gt_metric`, the trainer now can replace
`batch["gt_depth"]` with:

```text
perturbed_depth = gt_depth * sampled_scale
target_scene_scale = 1 / sampled_scale
```

The original `batch["gt_depth"]` remains unchanged and is still used by losses
as the metric target.

The model now records:

```text
hsi_refinement_depth_input
```

so HSI scene-affine losses use the same depth tensor that the HSI refinement
head saw.

## Trained Parameters

The config uses:

```text
model.smpl_provider: gt_perturbed
model.hsi_geometry_mode: gt_metric
model.gt_smpl_box_free: true
model.gt_smpl_online_visibility: true
model.smpl_query_box_prior: false
model.smpl_use_aggregator_queries: false
model.train_hsi_scene_affine_only: true
model.freeze_hsi_backbone: true
model.freeze_hsi_scene_affine: false
model.enable_hsi_human_scene_align: false
```

Optimizer contracts require only:

```text
hsi_refinement_head.scale_delta.
hsi_refinement_head.bias_delta.
```

to be trainable and receive gradients.

Person correspondence uses `matching.mode=gt_slots`. Valid dataset SMPL slot
`q` is supervised against prediction slot `q`; boxes do not participate in the
assignment. The existing Hungarian path remains unchanged for baseline configs.

## Empty-Person Robustness

The trainer explicitly handles missing or unusable people after online
visibility filtering:

- A sequence sample with no visible GT person in any frame is removed from the
  batch before VGGT/HSI forward.
- If every sample in a loader batch is removed, forward, backward, optimizer
  step, and `global_step` are all skipped.
- If people pass the initial visibility mask but the scale teacher later reports
  zero `metric_hsi_smpl_scale_teacher_valid_points`, backward and optimizer step
  are skipped for that batch.
- An empty frame inside a sequence that has another valid frame receives no
  direct person/scale supervision, but remains as temporal RGB context. With the
  current `per_frame` affine mode and disabled temporal HSI momentum, it does not
  contribute a scale target to the valid frame.
- If an entire epoch produces no valid optimizer step, training raises an error
  instead of writing a misleading zero-loss checkpoint.

Skipped items are excluded from epoch loss averages. W&B, terminal summaries,
and `metrics_latest.json` include:

```text
metric_train_dropped_no_visible_samples
metric_train_dropped_no_visible_sample_rate
metric_train_empty_visible_frames
metric_train_skipped_no_visible_batches
metric_train_skipped_no_supervision_batches
metric_train_skipped_batch_rate
metric_train_optimizer_steps
```

## Initialization

Default resume checkpoint:

```text
outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt
```

Only `hsi_refinement_head.` is required from that checkpoint. The align head is
not enabled in this experiment, because the goal is scale correction rather than
translation correction.

## Key Config Values

Default training perturbation:

```text
hsi_gt_depth_log_scale_std_schedule: 0.30
hsi_gt_depth_log_scale_mean: 0.0
hsi_gt_depth_scale_noise_mode: lognormal
hsi_gt_depth_scale_noise_unit: sequence
hsi_gt_depth_scale_clean_prob: 0.0
hsi_gt_depth_scale_min: 0.0
hsi_gt_depth_scale_max: 0.0
gt_smpl_visibility_max_points: 512
gt_smpl_visibility_decode_batch_size: 128
gt_smpl_visibility_window: 3
gt_smpl_visibility_tolerance_m: 0.20
gt_smpl_visibility_max_z_m: 20.0
gt_smpl_visibility_min_points: 32
```

For noisy samples, training uses:

```text
log(depth_scale) ~ Normal(0, 0.30^2)
```

This is a multiplicative log-normal perturbation. It is always positive and
has no configured hard minimum or maximum. About 95% of samples naturally lie
between `0.55` and `1.82`; rarer values remain possible. There is no separate
fixed clean-sample spike; the continuous Gaussian density around log-scale zero
already supplies near-identity examples.

This run does not create a validation loader or use a fixed validation scale.
Training health is judged from the W&B training curves and downstream Viser
visualization. Useful training signals are:

```text
loss_total -> decreases
metric_hsi_smpl_scale_teacher_pred_scale -> metric_hsi_smpl_scale_teacher_scale
metric_hsi_smpl_scale_teacher_log_l1 -> decreases
```

Main loss:

```text
hsi_smpl_scale_teacher_weight: 4.0
hsi_smpl_scale_teacher_log_loss: true
hsi_depth_teacher_weight: 0.10
```

The dense depth teacher is a small auxiliary. The primary supervision is the
SMPL-visible scale teacher. In the box-free run the dense auxiliary explicitly
uses all valid GT-depth pixels (`hsi_depth_teacher_use_human_roi=false`); it does
not attempt to build an ROI from zero compatibility boxes. It is restricted to
frames containing at least one online-visible matched GT person
(`hsi_depth_teacher_require_matched_frame=true`), so empty frames cannot add a
constant, non-learnable depth error to `loss_total`.

The synthetic corruption is multiplicative, so non-zero `scale_delta` gradient
is mandatory while `bias_delta` may legitimately have zero gradient when its
prediction is already zero. Only the scale branch is therefore part of the
hard first-step gradient contract. Both gradient norms are still logged as
`metric_grad_norm_hsi_scale_delta` and `metric_grad_norm_hsi_bias_delta`.
The dense auxiliary uses no hard metric-error clamp
(`hsi_depth_teacher_error_clip_m=0`); Smooth L1 already bounds its gradient,
whereas clamping large warm-start errors would create a dead gradient region.

The HSI log-scale clamp remains `[-5, 5]`, matching the source checkpoint. A
narrow clamp around the new perturbation range is unsafe during warm start:
the old scale branch can initially predict a much larger log-scale, and a hard
clamp would zero its scale gradient before it can adapt.

## W&B Monitoring

W&B is enabled by default under `logging.wandb`. Step-level training losses are
logged every 20 optimizer steps, and a complete training summary is logged
after each epoch. The perturbation mean, standard deviation, log-scale standard
deviation, and inverse target mean are also logged.

The launch script accepts `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_NAME`,
`WANDB_GROUP`, and `WANDB_MODE`. Authentication remains external to the repo;
the server environment must already contain `WANDB_API_KEY` or a previous
`wandb login`. Set `WANDB_MODE=offline` when the server cannot reach W&B.

## Checkpoint Policy

This run continues the existing scale branch from model weights in:

```text
outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt
```

The checkpoint's `hsi_refinement_head` tensors are loaded before optimization,
and the same `scale_delta` and `bias_delta` parameters continue training. The
old optimizer state is not required and is not available, so AdamW is created
again. Epoch/global step are reset only for logging this run; model parameters
are not reinitialized.

Storage is bounded to at most two model files: one overwriting
`checkpoint_latest.pt` and one dynamically named best checkpoint. Epoch
checkpoints, final duplicates, and stable top-k copies are disabled. Top-1 is
selected from epoch-average training `loss_total`; when a new best checkpoint
is produced, the previous best file is deleted.

## Training And Inference Boundary

The GT perturbation run is teacher-forced by design:

```text
training:  RGB features + GT SMPL + GT K + perturbed GT depth -> HSI scale
inference: RGB features + NLF SMPL + VGGT K + VGGT depth      -> HSI scale
```

NLF is not called during this training stage. For real inference, NLF receives
the processed RGB frames and intrinsics decoded from VGGT `pose_enc`; it does
not consume VGGT depth. NLF's internal `detect_smpl_batched` path supplies its
own detections, so inference does not require BEDLAM sidecars or the retired
aggregator box-prior path.

Use this dedicated inference/Viser wrapper after training:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
bash scripts/smoke/check_hsi_gt_depth_scale_nlf_detector.sh
bash scripts/vis/serve_hsi_gt_depth_scale_nlf_detector.sh
```

The detector smoke verifies that the configured NLF TorchScript checkpoint
actually exports `detect_smpl_batched`; having only `estimate_smpl_batched`
would still require external boxes.

It forces `QUERY_SOURCE=nlf_detector`, which resolves to:

```text
model.nlf_use_detector=true
model.nlf_require_boxes=false
model.smpl_query_box_prior=false
```

## Server Commands

Smoke check:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
bash scripts/smoke/check_hsi_gt_depth_scale_scene_affine.sh
bash scripts/smoke/run_hsi_gt_depth_scale_one_step.sh
```

The second command loads the real baseline and continuation checkpoint, finds
one valid filtered batch, runs forward/backward/gradient-contract/optimizer
step, and writes metrics under
`outputs/debug/hsi_gt_depth_scale_boxfree_one_step`. It disables W&B and all
checkpoint files for this gate, so it does not consume model-storage space.

Short gate:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

MAX_STEPS_PER_EPOCH=200 \
CUDA_VISIBLE_DEVICES_VALUE=7 \
bash scripts/train/train_smpl_hsi_gt_depth_scale_scene_affine.sh
```

Full run:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

CUDA_VISIBLE_DEVICES_VALUE=7 \
bash scripts/train/train_smpl_hsi_gt_depth_scale_scene_affine.sh
```

Output:

```text
outputs/train/smpl_hsi_gt_depth_scale_scene_affine_boxfree
```

Important artifacts:

```text
resolved_config.json
metrics_latest.json
checkpoint_latest.pt
checkpoint_topk_index.json
```

The best checkpoint filename is recorded in `checkpoint_topk_index.json`.

## Local Validation Status

Local Windows validation is limited to syntax and smoke checks because the full
server environment and checkpoints are not present locally.

Run locally or on server:

```bash
bash scripts/smoke/check_hsi_gt_depth_scale_scene_affine.sh
```

Expected smoke summary:

```text
clean_eval_depth_scale = 1.0
clean_eval_target_scale = 1.0
sampled_log_scale_std ~= 0.30
smpl_override_clean = true
box_free_gt_slots = true
```

The same smoke command scans eight real BEDLAM windows by default without a
sidecar root, runs online GT-SMPL visibility, reports empty-sample/frame rates,
and writes:

```text
outputs/debug/hsi_gt_depth_scale_boxfree_data_smoke/summary.json
```

## Risks

- This trains scale/bias only, not the Stage2 translation align head.
- If later inference still applies the old align head, the new scale checkpoint
  may need to be merged or resumed with the align head enabled/frozen.
- GT-depth perturbation is synthetic. It tests whether the model can learn the
  scale-correction mechanism, not whether real VGGT/NLF inference errors have
  the exact same distribution.
- Training uses GT SMPL/K/depth while inference uses NLF/VGGT predictions. This
  teacher-forcing domain gap must be judged with the NLF-detector Viser wrapper;
  decreasing training loss alone is insufficient.
- There is no held-out validation metric in this run. Final selection still
  requires downstream inference and Viser inspection.
