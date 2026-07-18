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
