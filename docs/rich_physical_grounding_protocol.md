# RICH Physical-Grounding Evaluation Protocol

## Goal

Reproduce only the physical-grounding half of UniCon3R Table 3 on RICH:

- Collision Ratio (`Coll.`, %)
- Penetrate (`Pen.`, cm)
- Float (cm)
- Penetration Max (`P.Max`, cm)

The existing VGGT-Omega paths remain unchanged. The evaluator must be added as
an optional evaluation path and must not replace the current baseline metrics.
The currently runnable benchmark is deliberately reduced to the nine moving-
camera recordings in the official RICH test split; it is not presented as the
full 40-recording Table 3 result.

## Dedicated Evaluator

The metric path is intentionally independent of all Viser/viewer modules:

- `vggt_omega/data/rich_physical_grounding.py` reads all frames from `cam_10`
  for the nine official test recordings marked `moving_cam=V`. Its retained
  `hmr4d_test_camera_views` mode still supports label-selected static views.
- `vggt_omega/evaluation/rich_physical_grounding.py` runs the two-pass metric
  cascade and implements the grounding metrics.
- `scripts/eval/evaluate_rich_physical_grounding.py` handles sequence/chunk
  iteration and writes JSON/CSV results.
- `scripts/eval/evaluate_rich_physical_grounding.sh` is the server entry point.

NLF runs in detector mode with eight candidate slots, matching the accepted
inference script. The untracked `cam_10` view has no entry in HMR4D's
`rich_test_labels.pt`, so the moving-camera protocol scores the highest-
confidence valid person detection in each frame. These nine recordings contain
one captured subject. The retained static-view protocol continues to match the
detector output to the RICH `bbx_xys` target by maximum IoU.

The inference path is:

```text
VGGT + NLF target query
-> analytic SMPL/depth coarse scale
-> v3 HSI residual scale and bias
-> Stage-2 human-scene alignment
-> target SMPL vertices + metric reconstructed depth in one VGGT world frame
```

No function is imported from `scripts/vis/`.

Checkpoint loading and construction settings follow
`scripts/vis/serve_stage2_walking_coarse_scale_hsi_cascade.sh`: the Stage-2
checkpoint is loaded over the VGGT baseline, the v3 scale checkpoint overlays
only `hsi_refinement_head.*`, and the model uses
`hsi_align_feature_version=legacy_scale_bias_v0`,
`hsi_scene_affine_mode=per_frame`, `smpl_use_aggregator_queries=false`, and
eight NLF detector slots. The evaluator then adds protocol-specific target-
person selection, which the general-purpose visualization script does not
perform.

## References

- UniCon3R, Table 3 and Section 5.2:
  `.paper/base_pdf/UniCon3R.pdf`
- HuMoS, Section 3.3 and Section 4.2:
  `.paper/base_pdf/HuMoS.pdf`
- HuMoS reference implementation:
  `.paper/base_projects/humos/humos/src/model/metrics.py`

The files under `.paper/` are read-only references. Production code must not
import from them.

## Confirmed From HuMoS

For each frame, HuMoS first computes the height of the lowest body-mesh vertex:

```text
h_t = min_v z[t, v]
```

HuMoS assumes a horizontal ground at `z = 0` and uses a `0.005 m` tolerance.
Its released implementation computes:

```text
penetrating frames: h_t < -0.005
floating frames:    h_t >= 0.005
Penetrate:           mean(abs(h_t)) over penetrating frames
Float:               mean(abs(h_t)) over floating frames
```

Frames inside the 5 mm dead zone contribute to neither conditional mean. Both
reported distances are converted from metres to centimetres.

## UniCon3R Adaptation

UniCon3R explicitly states that it adapts the HuMoS grounding protocol, uses a
robustly estimated ground height, and retains the 5 mm tolerance. It additionally
reports Collision Ratio and Penetration Max.

The most direct extension consistent with HuMoS is:

```text
d_t = min_v z[t, v] - ground_height
Coll. = 100 * count(d_t < -0.005) / count(valid frames)
Pen.  = 100 * mean(-d_t | d_t < -0.005)
Float = 100 * mean( d_t | d_t >= 0.005)
P.Max = 100 * max(-d_t | d_t < -0.005)
```

This extension is a reproduction hypothesis, not a formula printed by the
UniCon3R paper. The evaluator must label it as such until author code or an
official clarification is available.

In the current project coordinates, `-Y` is up. Therefore the physical lowest
body vertex has the largest Y coordinate, and the signed clearance is:

```text
d_t = robust_ground_y - max(body_vertex_y)
```

The current ground estimator is an explicit reproduction hypothesis because
UniCon3R does not publish its implementation details. Per frame, it removes the
target image box from reconstructed scene points, keeps points close to the
body's horizontal XZ footprint and lowest vertex, and takes a configurable high
Y quantile. The chunk ground height is the median of those frame estimates.
Every estimator parameter is saved in `run_config.json` and `summary.json`.

The downloaded RICH scans, calibration XMLs, and multicam-to-world transforms
are resolved and recorded for provenance, but are not used as the ground source:
Table 3 explicitly evaluates grounding in the reconstructed scene. This keeps
the body and scene in the model's common coordinate frame without introducing
an unpublished prediction-to-reference-scene alignment.

## Details Not Published

The available UniCon3R paper does not specify:

1. How the robust ground height is estimated from the reconstructed scene.
2. Which reconstructed scene points are retained before ground estimation.
3. Whether one ground height is fitted per frame, window, camera view, or action
   recording.
4. Whether final numbers are pooled over all frames or averaged per sequence.
5. Whether UniCon3R aggregates its 40 moving-camera recordings by frame or by
   sequence.

The paper states that source code and models will be released upon acceptance,
so the exact private evaluation implementation is not currently available in
the supplied reference material.

## Available Data

Server roots:

```text
/home/zhw/xyb_space/RICH/official
/home/zhw/xyb_space/RICH/hmr4d_support
```

The original HMR4D asset audit reports:

```text
RICH camera views: 191
Unique recordings: 50
Selected frames: 106446
Sequences with issues: 0
```

Those 191 entries are calibrated static-camera views and are not the current
moving-camera protocol. RICH's official metadata contains exactly 40 recordings
marked `moving_cam=V`: 22 train, 9 validation, and 9 test. The current reduced
protocol intentionally evaluates only the nine test recordings, using their
untracked `cam_10` directories:

```text
ParkingLot2_017_burpeejump2
ParkingLot2_017_burpeejump1
ParkingLot2_017_overfence1
ParkingLot2_017_overfence2
ParkingLot2_017_eating1
ParkingLot2_017_pushup2
Gym_011_cooking1
Gym_011_cooking2
Gym_012_cooking2
```

The canonical manifest is `configs/eval/rich_test_moving_camera_9.txt`. This
reduced result must not be reported as the full UniCon3R Table 3 protocol.

## Evaluation Model Path

The current project inference path is a two-checkpoint cascade rather than a
single standalone checkpoint:

```text
VGGT + NLF body initialization
  -> analytic coarse scene scale
  -> v3 HSI residual scale and bias overlay
  -> Stage-2 human-scene translation alignment
```

The accepted server checkpoints are:

```text
outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_latest.pt
outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt
```

The physical-grounding evaluator must reproduce this load order. Evaluating
only the Stage-2 checkpoint without the scale overlay would not match the
current inference system.

## Implementation Order

1. Verify the evaluation checkpoint and SMPL body-model assets.
2. Run one `cam_10` recording as an inference smoke test and save world-frame body
   vertices plus the reconstructed scene representation.
3. Validate the coordinate convention, gravity axis, metre scale, and body/scene
   alignment visually and numerically.
4. Implement both frame-pooled and sequence-mean aggregation, clearly named in
   the output, while keeping the 5 mm tolerance fixed.
5. Compare multiple robust ground estimators on the smoke sequence before
   launching the full RICH evaluation.
6. Run the nine-recording test subset and report it explicitly as a reduced
   protocol. Add the 31 train/validation moving-camera recordings only after
   those images are downloaded.

All full-evaluation outputs for this reduced protocol are written under
`outputs/eval/rich_physical_grounding_test_moving9_cam10/`.

## Server Commands

First run the evaluator checks inside the server environment:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
bash scripts/smoke/check_rich_physical_grounding_evaluator.sh
```

Then run four frames from a known moving-camera recording:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
CUDA_VISIBLE_DEVICES_VALUE=0 \
MODE=smoke \
RICH_SEQUENCE=ParkingLot2_017_burpeejump2/cam_10 \
SMOKE_FRAMES=4 \
WINDOW_SIZE=100 \
bash scripts/eval/evaluate_rich_physical_grounding.sh
```

Smoke outputs are written to
`outputs/debug/rich_physical_grounding/test_moving9_cam10_smoke/`. A full run
uses `MODE=full`, automatically loads
`configs/eval/rich_test_moving_camera_9.txt`, and writes to
`outputs/eval/rich_physical_grounding_test_moving9_cam10/`.

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
CUDA_VISIBLE_DEVICES_VALUE=0 \
MODE=full \
WINDOW_SIZE=100 \
MAX_FRAMES_PER_SEQUENCE=0 \
bash scripts/eval/evaluate_rich_physical_grounding.sh
```

Full mode enables resume by default. Re-running the same command skips completed
recordings only when their saved protocol and window size still match.

Primary output files are:

- `summary.json` and `summary.csv`: pooled-frame and sequence-mean metrics.
- `per_window.csv`: one row per non-overlapping 100-frame evaluation window;
  the final shorter window of each recording is retained.
- `per_sequence.csv`: one row per moving-camera recording.
- `per_frame.csv`: signed clearance and validity for every source frame.
- `sequences/*.json`: resumable per-recording results and ground diagnostics.
