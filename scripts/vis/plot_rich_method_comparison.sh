#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
DATA_CSV="${DATA_CSV:-${PROJECT_ROOT}/configs/vis/rich_wa_mpjpe100_fps_comparison.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/vis/rich_wa_mpjpe100_fps_comparison}"

cd "${PROJECT_ROOT}"
python scripts/vis/plot_emdb2_method_comparison.py \
  --csv "${DATA_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --y-metric fps \
  --y-scale log \
  --wa-layout linear \
  --y-label "Run-Time Efficiency (FPS)" \
  --output-name rich_wa_mpjpe100_vs_fps \
  --formats png,pdf,svg \
  "$@"

printf 'Figures written under %s\n' "${OUTPUT_DIR}"
