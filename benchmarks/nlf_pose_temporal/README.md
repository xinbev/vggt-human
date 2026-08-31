# NLF Pose Temporal Benchmark

This standalone benchmark measures local-body SMPL quality with:

```text
RGB + dataset ground-truth K -> NLF detector -> optional PoseTemporalStabilizer V2
```

VGGT, HSI and TRSTR are intentionally absent. NLF requires an intrinsic matrix, so this is a **GT-intrinsics NLF benchmark**, not an intrinsic-free or VGGT-camera benchmark.

Primary metrics: SMPL-24 pelvis-aligned PA-MPJPE, MPJPE and PVE (mm). The benchmark writes `*_metrics.json` and `*_rows.csv` under `outputs/eval/`.

For 3DPW test, use the raw `imageFiles` parent root and physical GPU 7 as:

```bash
CUDA_VISIBLE_DEVICES_VALUE=7 \
DATASET=3dpw \
FRAMES_ROOT=/home/zhw/xyb_space/3DPW/imageFiles \
TEMPORAL_CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/train/<v2-run>/checkpoint_best.pt \
OUTPUT_DIR=outputs/eval/nlf_pose_temporal/3dpw_test \
bash benchmarks/nlf_pose_temporal/run.sh
```

`CUDA_VISIBLE_DEVICES_VALUE=7` makes physical GPU 7 appear as `cuda:0` inside the process. Do not set `DEVICE=cuda:7` at the same time.
