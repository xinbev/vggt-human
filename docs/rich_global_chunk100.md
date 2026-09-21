# RICH chunk100: RGB-VGGT-NLF

## Protocol

- Dataset: `rich_test_labels.pt` plus `rich_test_preproc.pt`, labeled RICH test camera views (`hmr4d_test_camera_views`). The nine `cam_10` moving-camera recordings in the grounding protocol do not provide per-frame person GT and cannot yield these metrics.
- All labeled frames in each view, no frame subsampling. VGGT receives up to 100 consecutive labeled frames per call, with eight repeated frames between neighboring calls for prediction-only SE(3) camera stitching. GT is never used in stitching.
- VGGT camera and depth plus NLF detector and pose; no HSI Scale, Stage2 alignment, or TRSTR checkpoint. The previously used analytic SMPL/depth median ratio gives a metric camera-translation gauge, estimated only from predictions, shared across each inference chunk. If a chunk has no valid coarse-scale anchors, the run fails instead of silently substituting GT or unit scale.
- GT: gender-specific RICH world-coordinate SMPL-X vertices, mapped to SMPL vertices by the supplied `[6890,10475]` matrix, then projected through the same neutral SMPL 24-joint regressor as prediction. `hmr4d_support/cam2params.pt` projects GT joints to image coordinates **only** for evaluation-time person matching. Predictions use VGGT intrinsics and NLF camera-space SMPL; no GT camera/intrinsics enters inference.
- Metrics: W-MPJPE uses a Sim(3) fitted on the first two matched frames of each non-overlapping 100-position evaluation window; WA-MPJPE fits all matched joints in that window. RTE uses the entire stitched root track and one rigid SE(3), divided by total GT root motion. Units: mm/mm/percent. Missing detections are excluded and coverage is reported; sparse labeled frame IDs or missed detections create temporal gaps, so RTE is conditional on matched labeled frames.
- Outputs: `outputs/eval/rich_global_chunk100/sequence_metrics.csv`, `summary.json` (frame-weighted aggregate), `run.log`. This is a labeled-view subset protocol, **not** the official moving-camera benchmark.

## Server execution

Local launcher: `benchmarks/rich_global/run_chunk100.sh`; server: `/home/zhw/lab_users/xyb/home/projects/vggt-human/benchmarks/rich_global/run_chunk100.sh`.

The RICH paths in `docs/rich_evaluation_handoff.md` confirm `official/test/<recording>/cam_XX/*.jpg` and all three HMR4D support files (`rich_test_labels.pt`, `rich_test_preproc.pt`, `cam2params.pt`). The separate moving-camera test-9 manifest and its physical-grounding results are unrelated to these global-human metrics. The server model paths are:

```text
SMPL_MODEL_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/body_models/smpl
  model files: ${SMPL_MODEL_DIR}/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl or .npz
SMPLX_MODEL_DIR=/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/body_models/smplx
  model files: ${SMPLX_MODEL_DIR}/smplx/SMPLX_{MALE,FEMALE}.npz or .pkl
SMPLX_TO_SMPL=/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/utils/smplx2smpl.pkl
```

Both `smplx.create` and the project's `SMPLLayer` receive the **parent** of the `smplx/` or `smpl/` file directory. The launcher prechecks these files; overrides remain available through the environment. Ensure `smplx` is installed in the server environment. RICH roots are in `configs/path.yaml` and may be overridden by `RICH_OFFICIAL_ROOT` and `RICH_SUPPORT_ROOT`.

Smoke (one view, first 100 labeled frames):

```bash
MAX_SEQUENCES=1 MAX_FRAMES_PER_SEQUENCE=100 CUDA_VISIBLE_DEVICES_VALUE=7 bash benchmarks/rich_global/run_chunk100.sh
```

Full run (use a different output directory from the smoke run):

```bash
OUTPUT_DIR=outputs/eval/rich_global_chunk100_full CUDA_VISIBLE_DEVICES_VALUE=7 bash benchmarks/rich_global/run_chunk100.sh
```

The code is adapted from this project's EMDB-2 `chunk100` window stitching and global metrics, using the RICH HMR4D labeled-view format and Human3R's SMPL-X-to-SMPL evaluation idea. Neither Human3R nor GVHMR code is imported at runtime; EMDB-2 paths are unchanged. Windows-local verification cannot execute model inference or check server-side RICH annotation contents and conversion assets.
