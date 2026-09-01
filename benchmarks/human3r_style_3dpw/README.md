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

Use `run.sh` after setting the test root and V2 checkpoint. The first required validation is that **NLF base** approaches its own published 3DPW baseline under a documented matching/model protocol; do not claim V2 improvement before that baseline audit passes.
