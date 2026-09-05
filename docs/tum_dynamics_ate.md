# TUM-Dynamics ATE benchmark record

## Goal

Provide a reproducible Human3R-compatible camera-pose ATE evaluator for the
TUM-Dynamics Freiburg3 subset without changing the VGGT baseline or importing
the read-only Human3R reference tree.

## Protocol

1. Download the eight official Freiburg3 dynamic sequences.
2. Associate `rgb.txt` and `groundtruth.txt` with a 20 ms timestamp tolerance.
3. Prepare independent prefixes `50, 100, ..., 1000` and the Human3R-compatible
   `90` prefix.
4. Read Human3R `pred_traj.txt` files and pair by index when prediction/GT
   lengths agree (Human3R's in-memory behavior); otherwise use one-to-one
   nearest timestamp matching.
5. Estimate one Sim(3) from estimated camera centers to GT camera centers and
   report translation RMSE in meters.  The dataset summary is the arithmetic
   mean of sequence RMSE values, matching Human3R's `calculate_averages`.

## Files

The implementation lives in `benchmarks/tum_dynamics_ate/`:

- `download_tum_dynamics.sh`: official archive download/extraction.
- `prepare_tum_dynamics.py` and `.sh`: timestamp association and prefix tree.
- `evaluate_ate.py`: one prediction-root evaluator.
- `evaluate_curve.py` and `run_ate.sh`: all-prefix curve evaluator.
- `test_ate.py` and `test.sh`: data-free Sim(3)/association smoke tests.

All metric outputs go to `outputs/eval/`.  Raw/prepared datasets remain outside
the repository by default.

## Verification status and risks

Local validation completed: Python compilation, synthetic Sim(3) recovery,
timestamp one-to-one matching, Human3R synthetic-timestamp index matching, and
the data-free unit test suite.  A real server run still requires the TUM data,
Human3R weights/environment, and generated `pred_traj.txt` files.  The
evaluator intentionally reports ATE only; Human3R RPE is a separate metric.

