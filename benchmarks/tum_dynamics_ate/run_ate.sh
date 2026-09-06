#!/usr/bin/env bash
set -euo pipefail

# Evaluate project-native per-length camera trajectories.  The expected
# directories are, for example, outputs/eval/tum_dynamics_predictions/tum_500_vggt.
REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/long_tum_s1}"
PRED_PARENT="${PRED_PARENT:-${REPO_ROOT}/outputs/eval/tum_dynamics_predictions}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval/tum_dynamics_ate}"
MODEL="${MODEL:-vggt}"
LENGTHS="${LENGTHS:-50,100,150,200,300,400,500,600,700,800,900,1000}"
PRED_DIR_PATTERN="${PRED_DIR_PATTERN:-}"
if [[ -z "${PRED_DIR_PATTERN}" ]]; then
  # Assign this separately: nesting the literal ``{length}`` inside Bash's
  # ${VAR:-default} expansion makes its closing brace terminate the outer
  # expansion and produces the invalid Python placeholder ``{length_vggt}``.
  PRED_DIR_PATTERN="tum_{length}_${MODEL}"
fi

python benchmarks/tum_dynamics_ate/evaluate_curve.py \
  --dataset-root "${DATASET_ROOT}" \
  --pred-parent "${PRED_PARENT}" \
  --pred-dir-pattern "${PRED_DIR_PATTERN}" \
  --model "${MODEL}" \
  --lengths "${LENGTHS}" \
  --output-dir "${OUTPUT_DIR}" \
  --prediction-quaternion-order "${PREDICTION_QUATERNION_ORDER:-wxyz}" \
  --association "${ASSOCIATION:-auto}" \
  --max-time-difference "${MAX_TIME_DIFFERENCE:-0.02}"
