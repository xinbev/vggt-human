#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the Bonn dataset directory}"
PRED_ROOT="${PRED_ROOT:-/home/zhw/xyb_space/vggt_bonn_predictions}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/bonn_depth_stages}"

bash "${REPO_ROOT}/benchmarks/bonn_depth/infer_stages.sh"

DATASET_ROOT="${DATASET_ROOT}" \
PURE_VGGT_ROOT="${PRED_ROOT}/pure_vggt" \
TRADITIONAL_HSI_ROOT="${PRED_ROOT}/vggt_traditional_hsi_scale" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash "${REPO_ROOT}/benchmarks/bonn_depth/run_stages.sh"
