#!/usr/bin/env bash
set -euo pipefail

# Evaluate one already-generated pure VGGT prediction tree twice:
# (1) raw absolute metric depth, and (2) the historical per-sequence GT-scale
# diagnostic used by the UniSH/Pi3-style Bonn evaluation.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the Bonn dataset directory}"
PRED_ROOT="${PRED_ROOT:?Set PRED_ROOT to the pure VGGT curve prediction root}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/eval/bonn_depth_pure_vggt}"
PREFIX_LENGTHS="${PREFIX_LENGTHS:-50 100 150 200 250 300 350 400 450 500}"
START_FRAME="${START_FRAME:-30}"
ALLOW_SHORT="${ALLOW_SHORT:-true}"

for alignment in metric scale; do
  output_dir="${OUTPUT_ROOT}/${alignment}"
  args=(
    --dataset-root "${DATASET_ROOT}"
    --prediction-root "${PRED_ROOT}/pure_vggt"
    --stage-name "pure_vggt"
    --prefix-lengths ${PREFIX_LENGTHS}
    --start-frame "${START_FRAME}"
    --alignment "${alignment}"
    --output-dir "${output_dir}"
  )
  if [[ "${ALLOW_SHORT}" == "true" ]]; then
    args+=(--allow-short)
  fi
  python "${REPO_ROOT}/benchmarks/bonn_depth/evaluate_curve.py" "${args[@]}"
done
