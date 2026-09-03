# EMDB-2 Chunk-100 Evaluation

## Goal

Evaluate the three pipeline stages on all EMDB-2 protocol-valid frames while
keeping the maximum VGGT/NLF input at 100 frames:

```text
RGB -> VGGT camera/depth -> NLF SMPL -> HSI Scale v3 -> TRSTR
```

The older stride-7 evaluator is preserved and writes to a different output
directory.

## Protocol

- Dataset: native EMDB-2 annotations and `good_frames_mask`.
- Temporal sampling: none after selecting protocol good frames.
- Inference: windows of at most 100 good frames.
- Window overlap: 8 good frames by default.
- Matching: Human3R-style projected 2D matching, used only to select the
  predicted person query.
- Metric windows: `chunk_length=100` on the merged original-frame sequence.
- Units: meters in prediction archives; W-MPJPE and WA-MPJPE are reported in
  millimeters; RTE is reported in percent.

## Coordinate Handling

Each VGGT window has a local predicted camera/world gauge. The exporter uses
overlapping predicted camera poses to estimate a prediction-only SE(3)
transform (rotation plus translation) for each subsequent window. EMDB GT and
predicted human joints are not used to estimate this stitch. HSI and TRSTR
share the same camera-world stitch; TRSTR cannot change the stitch transform.

The final archive stores one prediction per original good-frame ID, a shared
valid mask, three `[F, 24, 3]` world-joint arrays, and stitch metadata. The
existing evaluator then computes per-sequence rows and frame-weighted dataset
summary rows.

## Tensor Contracts

- Model images: `[1, L, 3, H, W]`, where `L <= 100`.
- Predicted camera joints: `[1, L, 24, 3]` after query selection.
- Predicted world joints per stage: `[L, 24, 3]`, meters.
- Camera-to-world poses: `[L, 4, 4]`.
- Archive `valid`: `[F]`, where `F` is the number of EMDB good frames.

## Server Command

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
bash benchmarks/emdb2_global/run_chunk100_full.sh
```

Outputs:

```text
outputs/eval/emdb2_global_chunk100/predictions/
outputs/eval/emdb2_global_chunk100/metrics/summary.json
outputs/eval/emdb2_global_chunk100/metrics/sequence_metrics.csv
outputs/eval/emdb2_global_chunk100/metrics/stage_metrics.csv
outputs/eval/emdb2_global_chunk100/metrics/frame_metrics.csv
```

## Risks

The overlap stitch is a memory-management protocol, not a replacement for a
native recurrent VGGT state. If neighbouring predicted camera poses cannot be
aligned from the overlap, export fails instead of producing silently invalid
RTE. Final paper numbers should still be checked against a full-frame run and
reported with the exact chunk/stitch protocol.
