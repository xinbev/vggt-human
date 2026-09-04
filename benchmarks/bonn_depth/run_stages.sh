#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the Bonn dataset directory}"
PURE_VGGT_ROOT="${PURE_VGGT_ROOT:?Set PURE_VGGT_ROOT to pure VGGT predictions}"
TRADITIONAL_HSI_ROOT="${TRADITIONAL_HSI_ROOT:?Set TRADITIONAL_HSI_ROOT to VGGT+traditional+HSI predictions}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/bonn_depth_stages}"
ALIGNMENT="${ALIGNMENT:-scale}"

python "${REPO_ROOT}/benchmarks/bonn_depth/evaluate_stages.py" \
  --dataset-root "${DATASET_ROOT}" \
  --stage "pure_vggt=${PURE_VGGT_ROOT}" \
  --stage "vggt_traditional_hsi_scale=${TRADITIONAL_HSI_ROOT}" \
  --alignment "${ALIGNMENT}" \
  --output-dir "${OUTPUT_DIR}"
