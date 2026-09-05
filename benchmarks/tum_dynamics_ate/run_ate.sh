#!/usr/bin/env bash
set -euo pipefail

# Evaluate Human3R's per-length relpose outputs.  The expected prediction
# directories are, for example, eval_results/relpose/tum_500_human3r.
REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/long_tum_s1}"
PRED_PARENT="${PRED_PARENT:-${REPO_ROOT}/eval_results/relpose}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval/tum_dynamics_ate}"
MODEL="${MODEL:-human3r}"
LENGTHS="${LENGTHS:-50,100,150,200,300,400,500,600,700,800,900,1000}"

python benchmarks/tum_dynamics_ate/evaluate_curve.py \
  --dataset-root "${DATASET_ROOT}" \
  --pred-parent "${PRED_PARENT}" \
  --pred-dir-pattern "${PRED_DIR_PATTERN:-tum_{length}_${MODEL}}" \
  --model "${MODEL}" \
  --lengths "${LENGTHS}" \
  --output-dir "${OUTPUT_DIR}" \
  --prediction-quaternion-order "${PREDICTION_QUATERNION_ORDER:-wxyz}" \
  --association "${ASSOCIATION:-auto}" \
  --max-time-difference "${MAX_TIME_DIFFERENCE:-0.02}"

