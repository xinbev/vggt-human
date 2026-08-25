# NLF ID Tracking Experiment

## Objective

Keep NLF as the camera-consistent SMPL provider and add an identity association branch. The baseline NLF pose, beta, camera translation, and projection path remain unchanged.

## Implementation

The NLF provider is inference-only and returns dense tensors with shape `[B,S,Q,*]`. It does not expose a trainable identity feature. `VGGTOmega` therefore reads the final aggregator SMPL query tokens `[B,S,Q,2*embed_dim]` and applies `nlf_id_head`, producing normalized `pred_id_embed` with shape `[B,S,Q,id_embed_dim]`.

Training uses the existing Hungarian matcher on NLF boxes. The existing identity contrastive loss then pulls embeddings from the same BEDLAM `gt_track_ids` together across the clip and pushes different IDs apart. Only `nlf_id_head.*` is trainable in the first experiment.

At inference, `smpl_track_assignment_mode=base_smpl` enables the geometry-aware `BaseSMPLTrackAssigner`. Its hard gates remain:

- box center distance;
- camera translation distance;
- beta L1 distance.

After those gates, the score combines the existing geometry/confidence score with cosine similarity of the ID embeddings. `smpl_track_assign_id_weight` controls the embedding contribution and `smpl_track_assign_max_id_distance` rejects an embedding whose cosine distance is too large. With ID weight zero, the previous assigner behavior is preserved.

## Experiment

Configuration: `configs/train_nlf_id_tracking.yaml`

Training output: `outputs/train/nlf_id_tracking_gpu5/`

Evaluation output: `outputs/eval/nlf_id_tracking_gpu5/summary.json`

The server checkout is `/home/zhw/lab_users/xyb/home/projects/vggt-human`. Run:

```bash
bash scripts/smoke/run_nlf_id_tracking_smoke.sh
bash scripts/train/train_nlf_id_tracking_gpu5.sh
bash scripts/eval/run_nlf_id_tracking_eval_gpu5.sh outputs/train/nlf_id_tracking_gpu5/checkpoint_latest.pt Training 0.35 0.70 200 pilot_id
# Geometry-only ablation using the same checkpoint.
bash scripts/eval/run_nlf_id_tracking_eval_gpu5.sh outputs/train/nlf_id_tracking_gpu5/checkpoint_latest.pt Training 0.0 0.70 200 pilot_geometry
```

The evaluator reports temporal ID switch rate, majority association accuracy, positive/negative embedding cosine, and their margin. Because the training loader uses short clips, switches are measured within each evaluated clip; a long-sequence evaluation should use a sequence-length configuration or a persistent track-memory wrapper before claiming full-video IDF1.

## Risks

NLF parameters are frozen, so this experiment cannot repair NLF pose or beta accuracy. BEDLAM IDs are used only as supervision for the embedding branch. The first result should be treated as an association ablation: compare NLF geometry-only assignment against geometry plus the learned embedding, while checking that projection metrics remain unchanged.

## V2 Redesign After Pilot

The first query-only ID head increased both positive and negative cosine similarity and reduced tracking accuracy. It is retained as an ablation, but the recommended method is V2.

V2 pools the final VGGT patch tokens inside each processed-image person box. The ROI feature contains only mean and max appearance pooling; box coordinates, validity, and area are used for pooling validity but are excluded from the identity embedding to prevent a position shortcut. It is fused with the corresponding SMPL query token by `SMPLROIIdentityHead`. The NLF SMPL outputs remain frozen and unchanged.

The V2 identity loss is supervised contrastive loss plus a batch-hard cosine margin term. The training log now exposes positive cosine, negative cosine, and their margin. During association, the embedding weight is reduced to `0.10` and the maximum ID distance is `2.0`, so appearance acts as a soft tie-breaker and cannot reject a geometrically valid match.

V2 pilot commands on GPU5:

```bash
bash scripts/train/train_nlf_roi_id_tracking_v2_pilot_gpu5.sh
bash scripts/eval/run_nlf_roi_id_tracking_v2_eval_gpu5.sh \
  outputs/train/nlf_roi_id_tracking_v2_pilot_gpu5/checkpoint_latest.pt \
  Training 0.10 2.0 200 pilot_id
bash scripts/eval/run_nlf_roi_id_tracking_v2_eval_gpu5.sh \
  outputs/train/nlf_roi_id_tracking_v2_pilot_gpu5/checkpoint_latest.pt \
  Training 0.0 2.0 200 pilot_geometry
```

Only after the V2 pilot improves over geometry-only should the full run be started with `bash scripts/train/train_nlf_roi_id_tracking_v2_gpu5.sh`.

## Epoch 14 Viser Check

The dedicated V2 viewer loads the VGGT baseline first and then the partial epoch 14 ID-head checkpoint. BEDLAM sidecar boxes are used as NLF person proposals, but sidecar identity labels are deliberately not passed into the model. `BaseSMPLTrackAssigner` therefore produces every displayed track ID from bbox, `transl_cam`, beta, confidence, and the learned ROI identity embedding.

Server command:

```bash
bash scripts/vis/serve_nlf_roi_id_tracking_v2_viewer_gpu5.sh
```

The default run uses physical GPU5, 32 frames, `ID_WEIGHT=0.10`, `MAX_ID_DISTANCE=2.0`, and checkpoint `outputs/train/nlf_roi_id_tracking_v2_gpu5/checkpoint_epoch_0014.pt`. Viser displays a stable color and an `ID n` label for each assigned track. The run summary is written under `outputs/vis/nlf_roi_id_tracking_v2_gpu5/`.

This check measures identity association with known person boxes. It does not yet validate a detector-to-ID end-to-end pipeline on arbitrary video, because the V2 ROI head was trained with BEDLAM box proposals.
