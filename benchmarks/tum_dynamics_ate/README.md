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

## 3. Produce Human3R predictions

This benchmark only computes the metric; it does not silently download model
weights or run Human3R.  Run Human3R's released relpose evaluator in its own
checkout, pointing its TUM metadata to the prepared tree.  Its output layout
should be:

```text
<human3r-root>/eval_results/relpose/
  tum_50_human3r/<sequence>/pred_traj.txt
  tum_100_human3r/<sequence>/pred_traj.txt
  ...
```

In the Human3R checkout, edit `eval/relpose/metadata.py` so the `tum` entry
and every generated `tum_50` ... `tum_1000` entry use the same prepared root
(the `img_path` value).  The released metadata already expects the subfolder
names `rgb_90`/`groundtruth_90` and `rgb_<N>`/`groundtruth_<N>` created above.
Then launch its evaluator from the Human3R checkout, for example:

```bash
cd /path/to/Human3R
CUDA_VISIBLE_DEVICES=0 bash eval/relpose/run.sh
```

This produces one `pred_traj.txt` per sequence and prefix under
`eval_results/relpose/`.  Human3R weights and its CUDA environment are not
part of this repository; if those are not already present on the server, stop
at this step and prepare the Human3R environment/checkpoint first.

The released Human3R `eval/relpose/run.sh` currently looks for
`src/human3r.pth`, while the download command names the file
`human3r_896L.pth`.  Change the `ckpt_name` variable in that script to
`human3r_896L`, or place the downloaded checkpoint under the name expected by
the script; do not evaluate with a missing/placeholder checkpoint.

Human3R's `save_tum_poses` writes quaternion columns as `qw qx qy qz`; the
metric script therefore defaults to `--prediction-quaternion-order wxyz`.
Quaternion values do not enter translation ATE, but the option documents the
file convention.  If a different predictor writes standard TUM
`qx qy qz qw`, pass `PREDICTION_QUATERNION_ORDER=xyzw`.

## 4. Compute one ATE value

For one Human3R length, use the supplied wrapper (the repository convention is
to launch server evaluation through a `.sh` file):

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
DATASET_ROOT=/home/zhw/xyb_space/tum_dynamics_long_s1 \
PRED_PARENT=/path/to/human3r/eval_results/relpose \
MODEL=human3r \
LENGTHS=500 \
OUTPUT_DIR=outputs/eval/tum_dynamics_ate/human3r \
bash benchmarks/tum_dynamics_ate/run_ate.sh
```

Outputs are `length_500/summary.json` and
`length_500/sequence_metrics.csv` below `OUTPUT_DIR`.  The reportable
Human3R-style number is `length_500/summary.json` →
`ate_rmse_m_mean_over_sequences` in meters.  `sequence_metrics.csv` is useful
for finding sequences that fail or have very few associated poses.

## 5. Generate the ATE curve in the attached figure's style

Once all Human3R prefix runs exist:

```bash
cd /home/zhw/lab_users/xyb/home/projects/vggt-human
DATASET_ROOT=/home/zhw/xyb_space/tum_dynamics_long_s1 \
PRED_PARENT=/path/to/human3r/eval_results/relpose \
MODEL=human3r \
OUTPUT_DIR=outputs/eval/tum_dynamics_ate/human3r \
bash benchmarks/tum_dynamics_ate/run_ate.sh
```

The curve table is written to:

```text
outputs/eval/tum_dynamics_ate/human3r/curve.csv
outputs/eval/tum_dynamics_ate/human3r/curve_summary.json
outputs/eval/tum_dynamics_ate/human3r/length_<N>/summary.json
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
