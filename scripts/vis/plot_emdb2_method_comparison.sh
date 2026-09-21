#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
DATA_CSV="${DATA_CSV:-${PROJECT_ROOT}/configs/vis/emdb2_method_comparison.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/vis/emdb2_method_comparison}"

cd "${PROJECT_ROOT}"
python scripts/vis/plot_emdb2_method_comparison.py \
  --csv "${DATA_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

printf 'Figures written under %s\n' "${OUTPUT_DIR}"
