# Human3R-style TUM-Dynamics ATE

This benchmark reimplements the camera-pose ATE protocol used by Human3R on
the TUM-Dynamics subset.  It is project-native and does not import code from
`.paper/` or from Human3R.

The attached ATE-vs-number-of-views plot corresponds to the curve protocol:
each requested prefix (`50, 100, ..., 1000`) is evaluated as an independent
sequence, and the per-sequence ATE values are averaged.  ATE is the RMSE of
translation errors after a similarity (Sim(3)) alignment, equivalent to
Human3R's `evo.main_ape.ape(..., pose_relation=translation_part,
align=True, correct_scale=True)`.

## 1. Download the official TUM-Dynamics data

The download script fetches the eight Freiburg3 sequences used by the
Human3R/TTT3R camera-pose benchmark:

```text
freiburg3_sitting_{static,xyz,halfsphere,rpy}
freiburg3_walking_{static,xyz,halfsphere,rpy}
```

The data are provided by the [TUM RGB-D dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset);
check its license/terms before downloading.  On the Linux server:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
RAW_ROOT=/home/zhw/xyb_space/tum_dynamics_raw \
bash benchmarks/tum_dynamics_ate/download_tum_dynamics.sh
```

The script keeps the downloaded `.tgz` archives and extracts the sequence
folders below `RAW_ROOT`.  It does not place data in the Git repository.

## 2. Prepare Human3R/TTT3R prefixes

Associate `rgb.txt` and `groundtruth.txt` with a 20 ms tolerance, then create
`rgb_90` plus `rgb_50`, ..., `rgb_1000` and their matching GT files:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
RAW_ROOT=/home/zhw/xyb_space/tum_dynamics_raw \
PREPARED_ROOT=/home/zhw/xyb_space/tum_dynamics_long_s1 \
bash benchmarks/tum_dynamics_ate/prepare_tum_dynamics.sh
```

The prepared layout is:

```text
tum_dynamics_long_s1/
  rgbd_dataset_freiburg3_walking_xyz/
    rgb_500/<timestamp>.png
    groundtruth_500.txt
    ...
  manifest.json
```

`manifest.json` records the matched-pair count and actual prefix length.  If a
sequence has fewer than a requested number of associated frames, the prefix is
truncated and the actual count is recorded; no frames are fabricated.

## 3. Run your VGGT-Omega system and export trajectories

You do **not** need to run Human3R.  Human3R is only the source of the metric
protocol.  For the released/default VGGT-Omega model, use the project-native
inference wrapper:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human

DATASET_ROOT=/home/zhw/xyb_space/tum_dynamics_long_s1 \
CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/vggt_omega_1b_512.pt \
PRED_PARENT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/eval/tum_dynamics_predictions \
LENGTHS=50,100,150,200,300,400,500,600,700,800,900,1000 \
MODEL=vggt \
bash benchmarks/tum_dynamics_ate/infer_vggt.sh
```

This wrapper reads the RGB prefixes generated in step 2, runs
`VGGTOmega -> pose_enc -> camera extrinsics`, converts the camera-from-world
extrinsics into camera centers, and writes:

```text
outputs/eval/tum_dynamics_predictions/
  tum_50_vggt/<sequence>/pred_traj.txt
  tum_100_vggt/<sequence>/pred_traj.txt
  ...
```

The script uses one forward pass per sequence prefix.  This is intentional:
each input length is an independent experiment.  Start with a small smoke run
because the 1B model's memory grows quickly with the number of input views:

```bash
DATASET_ROOT=/home/zhw/xyb_space/tum_dynamics_long_s1 \
CHECKPOINT=/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/vggt_omega_1b_512.pt \
PRED_PARENT=outputs/eval/tum_dynamics_predictions_smoke \
LENGTHS=50 \
MAX_SEQUENCES=1 \
bash benchmarks/tum_dynamics_ate/infer_vggt.sh
```

For a custom VGGT-Omega checkpoint whose constructor differs from the released
`VGGTOmega()` default, keep the same export contract—one
`<sequence>/pred_traj.txt` per prefix—but adapt the model construction in
`infer_vggt.py`.  The ATE evaluator is independent of the model code.

## 4. Compute one ATE value

Now evaluate the trajectories produced by **your VGGT system**:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
DATASET_ROOT=/home/zhw/xyb_space/tum_dynamics_long_s1 \
PRED_PARENT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/eval/tum_dynamics_predictions \
MODEL=vggt \
LENGTHS=500 \
OUTPUT_DIR=outputs/eval/tum_dynamics_ate/vggt \
bash benchmarks/tum_dynamics_ate/run_ate.sh
```

Outputs are `length_500/summary.json` and
`length_500/sequence_metrics.csv` below `OUTPUT_DIR`.  The reportable
Human3R-style number is `length_500/summary.json` →
`ate_rmse_m_mean_over_sequences` in meters.  `sequence_metrics.csv` is useful
for finding sequences that fail or have very few associated poses.

## 5. Generate the ATE curve in the attached figure's style

Once all VGGT prefix runs exist:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
DATASET_ROOT=/home/zhw/xyb_space/tum_dynamics_long_s1 \
PRED_PARENT=/home/zhw/lab_users/xyb/home/projects/vggt-human/outputs/eval/tum_dynamics_predictions \
MODEL=vggt \
OUTPUT_DIR=outputs/eval/tum_dynamics_ate/vggt \
bash benchmarks/tum_dynamics_ate/run_ate.sh
```

The curve table is written to:

```text
outputs/eval/tum_dynamics_ate/vggt/curve.csv
outputs/eval/tum_dynamics_ate/vggt/curve_summary.json
outputs/eval/tum_dynamics_ate/vggt/length_<N>/summary.json
```

`curve.csv` contains one row per requested length and the sequence-macro mean
ATE in meters.  Convert to centimeters for a plot by multiplying
`ate_rmse_m_mean_over_sequences` by 100; do not mix this with a
frame-weighted mean unless you explicitly report a different protocol.

## Association and reproducibility notes

Before downloading data or running a checkpoint, the data-free implementation
test can be run on the server:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
bash benchmarks/tum_dynamics_ate/test.sh
```

* If prediction and GT counts are equal, `--association auto` pairs them by
  index.  This matches Human3R, which replaces synthetic prediction timestamps
  with GT timestamps before calling `evo`.
* If counts differ, `--association auto` performs one-to-one nearest timestamp
  matching with a 20 ms maximum difference.  Use `--association timestamp` to
  force this behavior.
* The evaluator requires at least three associated poses for a non-degenerate
  Sim(3) solve and fails loudly instead of emitting a misleading ATE.
* This implementation reports only ATE.  Human3R's relpose runner also logs
  RPE; adding RPE would be a separate metric protocol and is intentionally not
  mixed into the ATE output.
