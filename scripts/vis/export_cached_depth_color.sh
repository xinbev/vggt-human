#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
FRAME_PKL="${FRAME_PKL:-}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/outputs/vis/cached_depth/frame_depth.png}"
SOURCE="${SOURCE:-hsi}"
COLORMAP="${COLORMAP:-turbo}"
DEPTH_PERCENTILE_LOW="${DEPTH_PERCENTILE_LOW:-2}"
DEPTH_PERCENTILE_HIGH="${DEPTH_PERCENTILE_HIGH:-98}"
MIN_DEPTH="${MIN_DEPTH:-}"
MAX_DEPTH="${MAX_DEPTH:-}"

cd "${REPO_ROOT}"
[[ -n "${FRAME_PKL}" ]] || { echo "[ERROR] Set FRAME_PKL=/path/to/frame_XXXX.pkl" >&2; exit 1; }
[[ -f "${FRAME_PKL}" ]] || { echo "[ERROR] Missing frame pickle: ${FRAME_PKL}" >&2; exit 1; }

ARGS=(
  --frame-pkl "${FRAME_PKL}"
  --output "${OUTPUT}"
  --source "${SOURCE}"
  --colormap "${COLORMAP}"
  --percentiles "${DEPTH_PERCENTILE_LOW}" "${DEPTH_PERCENTILE_HIGH}"
)
[[ -n "${MIN_DEPTH}" ]] && ARGS+=(--min-depth "${MIN_DEPTH}")
[[ -n "${MAX_DEPTH}" ]] && ARGS+=(--max-depth "${MAX_DEPTH}")

python scripts/vis/export_cached_depth_color.py "${ARGS[@]}"
