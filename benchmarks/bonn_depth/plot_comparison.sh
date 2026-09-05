#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CURVE_DIR="${CURVE_DIR:-${REPO_ROOT}/outputs/eval/bonn_depth_curve}"
ENHANCED_CSV="${ENHANCED_CSV:-${CURVE_DIR}/vggt_traditional_hsi_scale_curve_points.csv}"
PURE_VGGT_CSV="${PURE_VGGT_CSV:-${CURVE_DIR}/pure_vggt_curve_points.csv}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/outputs/vis/bonn_depth_curve/human3r_figure9b_with_ours_and_pure_vggt.png}"

python "${REPO_ROOT}/benchmarks/bonn_depth/plot_human3r_figure9b_comparison.py" \
  --curve-csv "${ENHANCED_CSV}" \
  --pure-vggt-csv "${PURE_VGGT_CSV}" \
  --output "${OUTPUT}"
