# Human3R-Style 3DPW Test Benchmark

This folder is a project-native reimplementation of the relevant Human3R local-human evaluation ideas. It does **not** import Human3R code.

```text
raw 3DPW test pkl + original RGB
  -> gender-specific, camera-space SMPL GT
  -> NLF detector using GT K transformed to the processed image plane
  -> projected 2D common-joint association
  -> NLF base and optional Pose Temporal Stabilizer V2
  -> PA-MPJPE / pelvis-MPJPE / pelvis-PVE / diagnostics
```

The protocol follows Human3R for: original test pkl input, gender-specific GT SMPL, camera-space conversion, pelvis `[1,2]`, and projection-first 2D association. NLF output semantics differ from Human3R, so this is named **Human3R-style**, not an official Human3R evaluator.

Association accepts a 2D joint pair only when the predicted/GT common-joint bounding-box IoU is at least `0.05`, matching Human3R's `match_2d_greedy` guard. PA metrics use a row-vector similarity Procrustes solve and must satisfy `PA-MPJPE <= MPJPE`; a result that violates this is invalid and indicates a metric implementation bug.

Use `run.sh` after setting the test root and V2 checkpoint. The first required validation is that **NLF base** approaches its own published 3DPW baseline under a documented matching/model protocol; do not claim V2 improvement before that baseline audit passes.

## Single-sequence and error-source diagnostics

For a fast run on one official test sequence, use `SEQUENCE_FILTER` and `MAX_FRAMES`. This is diagnostic only, not a reportable benchmark number:

```bash
SEQUENCE_FILTER=downtown_arguing_00 MAX_FRAMES=100 COMPONENT_DIAGNOSTICS=true bash benchmarks/human3r_style_3dpw/run.sh
```

`COMPONENT_DIAGNOSTICS=true` additionally writes `component_rows.csv` and summary aggregates for four non-additive counterfactual local-body proxies:

| Name | Meaning |
| --- | --- |
| `full_pred_neutral` | predicted pose + predicted beta, neutral SMPL |
| `pred_pose_gt_beta` | predicted pose + GT beta: shape/beta oracle proxy |
| `gt_pose_pred_beta` | GT pose + predicted beta: pose oracle proxy |
| `gt_pose_gt_beta_neutral` | GT pose + GT beta decoded by neutral SMPL: neutral-vs-gender representation floor |

If `pred_pose_gt_beta` reduces error strongly, beta/shape is a primary source. If `gt_pose_pred_beta` reduces it strongly, pose is the primary source. These are diagnostics, not additive contributions; inspect `component_rows.csv` sorted by a relevant error column to locate difficult sequence/frame/person cases.
