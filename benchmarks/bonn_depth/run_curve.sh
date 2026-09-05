#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the Bonn dataset directory}"
PRED_ROOT="${PRED_ROOT:-/home/zhw/xyb_space/vggt_bonn_curve_predictions}"
STAGE="${STAGE:-vggt_traditional_hsi_scale}"
PREFIX_LENGTHS="${PREFIX_LENGTHS:-50 100 150 200 250 300 350 400 450 500}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/bonn_depth_curve}"
START_FRAME="${START_FRAME:-30}"
ALLOW_SHORT="${ALLOW_SHORT:-true}"

EXTRA_ARGS=()
if [[ "${ALLOW_SHORT}" == "true" ]]; then
  EXTRA_ARGS+=(--allow-short)
fi

python "${REPO_ROOT}/benchmarks/bonn_depth/evaluate_curve.py" \
  --dataset-root "${DATASET_ROOT}" \
  --prediction-root "${PRED_ROOT}/${STAGE}" \
  --stage-name "${STAGE}" \
  --prefix-lengths ${PREFIX_LENGTHS} \
  --start-frame "${START_FRAME}" \
  "${EXTRA_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}"
