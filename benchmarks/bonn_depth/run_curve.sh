#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the Bonn dataset directory}"
PRED_ROOT="${PRED_ROOT:-/home/zhw/xyb_space/vggt_bonn_curve_predictions}"
STAGE="${STAGE:-vggt_traditional_hsi_scale}"
PREFIX_LENGTHS="${PREFIX_LENGTHS:-100 200 300 400 500}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/bonn_depth_curve}"

python "${REPO_ROOT}/benchmarks/bonn_depth/evaluate_curve.py" \
  --dataset-root "${DATASET_ROOT}" \
  --prediction-root "${PRED_ROOT}/${STAGE}" \
  --stage-name "${STAGE}" \
  --prefix-lengths ${PREFIX_LENGTHS} \
  --output-dir "${OUTPUT_DIR}"
