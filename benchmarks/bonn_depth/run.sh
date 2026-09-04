#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the Bonn dataset directory}"
PRED_ROOT="${PRED_ROOT:?Set PRED_ROOT to the model prediction directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/bonn_depth}"
ALIGNMENT="${ALIGNMENT:-scale}"

python "${REPO_ROOT}/benchmarks/bonn_depth/evaluate.py" \
  --dataset-root "${DATASET_ROOT}" \
  --pred-root "${PRED_ROOT}" \
  --alignment "${ALIGNMENT}" \
  --output-dir "${OUTPUT_DIR}"
