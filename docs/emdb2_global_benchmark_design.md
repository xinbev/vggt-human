# EMDB-2 Global Human Benchmark Design

## Objective

Implement the Human3R global-human EMDB-2 protocol for the paper metrics:

```text
W-MPJPE
WA-MPJPE
RTE
```

The implementation lives under `benchmarks/emdb2_global`. Human3R's code under
`.paper/base_projects/human3r/Human3R/eval/global_human` is read-only reference
material and is neither imported nor modified.

## Reference Mapping

```text
Human3R wa2_mpjpe -> paper W-MPJPE
Human3R waa_mpjpe -> paper WA-MPJPE
Human3R rte        -> paper RTE
```

W-MPJPE splits a matched sequence into chunks of at most 100 frames, estimates
one Sim(3) from the first two frames' 24 joints, applies it to the whole chunk,
and reports per-frame MPJPE in millimeters.

WA-MPJPE uses the same chunks but estimates Sim(3) from all joints in each
chunk. RTE rigid-aligns the complete predicted root trajectory without scale,
divides each frame's root error by total GT root displacement, and reports
percent.

## GT Protocol

The exact 25-sequence Human3R EMDB-2 list is frozen in `data.py`. Native EMDB
pickles provide gender, `good_frames_mask`, world SMPL root/body pose, shape,
translation, intrinsics, and camera extrinsics. Metrics use only good frames.

GT world joints are decoded with gender-specific SMPL and this project's
SMPL-24 joint convention. Prediction archives must use the same convention.

## Prediction Boundary

Metrics accept already continuous world-space joints, or camera-space joints
paired with the model-predicted continuous `T_c2w`. Using EMDB GT camera to
transform predictions is explicitly forbidden because it injects oracle camera
motion into a global-human benchmark.

## Known Problems Requiring User Decision

1. Native EMDB is configured as `datasets.emdb_root=/home/zhw/xyb_space/emdb/`.
   The server must still pass the benchmark data smoke before formal inference.
2. Current VGGT-Omega full-sequence inference cannot process 700-3300 frames in
   one forward pass. Independent chunk inference creates independent Sim(3)
   worlds. A validated stateful camera path or overlap-stitching protocol is
   required before exporting paper-comparable predictions.
3. Overlap-based chunk stitching is not automatically equivalent to Human3R's
   persistent global state and must be reported as an approximation if used.
4. Prediction joints must be neutral-SMPL regressor SMPL-24 in meters. Another
   24-joint ordering can produce plausible but invalid numbers.
5. Human3R's textual summary labels the global section as centimeters, while
   its implementation multiplies W/WA error by 1000. This benchmark follows
   the computation and reports millimeters.
6. RTE is normalized by GT path length. Low-motion sequences and missing-frame
   gaps can make it unstable, so sequence CSV reports GT displacement, coverage,
   and maximum matched frame gap.
7. EMDB-2 is single-person, but detector misses still matter. Metrics use only
   matched valid frames and always report coverage; missing frames are never
   silently copied or interpolated.

## Implemented S7 Baseline

The independent-world blocker is avoided for the first baseline by fixed
stride-7 sampling. Every selected sequence has at most 500 frames and is passed
through one unchunked two-pass cascade, yielding one final pass-two predicted
world gauge. The
exporter uses the model's `pose_enc` W2C, scales camera translation by the final
shared HSI effective scale, inverts it, and transforms TRSTR SMPL-24 camera
joints to world.

This is a valid internally consistent unchunked benchmark named
`EMDB-2-S7`, but it is not the official Human3R `subsample=1` protocol. W/WA use
chunk length 14 and RTE uses the stride-7 root path.

## Multi-Stage Attribution

Every sequence archive contains three world predictions using the same
selected NLF query and final-pass VGGT camera rotation:

```text
A. RGB-VGGT-NLF + shared analytic coarse metric gauge
B. A + HSI Scale v3 residual
C. B + TRSTR translation
```

The coarse gauge is included in A because raw VGGT camera translation and NLF
metric SMPL do not otherwise share units. B changes only the shared camera
translation scale. C uses the same metric camera as B and changes only human
translation. The evaluator reports A, B, C and error reductions A-B, B-C and
A-C, with positive values denoting improvement.

All stages use the same person selected by Human3R-style 2D association. GT
gender SMPL-24 world joints are projected with EMDB GT camera only for matching;
prediction world coordinates continue to use predicted VGGT camera exclusively.
Reports distinguish stride sampling rate from detector prediction coverage.

## Outputs

```text
summary.json          frame-weighted paper metrics and coverage
sequence_metrics.csv per-sequence metrics, displacement and frame gaps
frame_metrics.csv    per-frame W/WA/RTE arrays
```

## Verification

`test_metrics.py` verifies zero error, removal of one global similarity,
W-vs-WA sensitivity to drift, and rejection of undefined static-trajectory RTE.
