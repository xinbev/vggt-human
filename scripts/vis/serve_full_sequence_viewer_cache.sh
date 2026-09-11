#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/outputs/vis/stage2_walking_coarse_residual_v3/full_viewer_cache}"
PORT="${PORT:-8080}"
SMPL_EDIT_OUTPUT="${SMPL_EDIT_OUTPUT:-}"
POINT_SIZE="${POINT_SIZE:-}"
HUMAN_MASK_DILATION_PX="${HUMAN_MASK_DILATION_PX:-}"
FILTER_HUMAN_POINTS="${FILTER_HUMAN_POINTS:-}"

cd "${REPO_ROOT}"
[[ -f "${CACHE_DIR}/manifest.json" ]] || { echo "[ERROR] Missing full viewer cache: ${CACHE_DIR}/manifest.json" >&2; exit 1; }

ARGS=(
  --cache-dir "${CACHE_DIR}"
  --port "${PORT}"
)
if [[ -n "${SMPL_EDIT_OUTPUT}" ]]; then
  ARGS+=(--smpl-edit-output "${SMPL_EDIT_OUTPUT}")
fi
if [[ -n "${POINT_SIZE}" ]]; then
  ARGS+=(--point-size "${POINT_SIZE}")
fi
if [[ -n "${HUMAN_MASK_DILATION_PX}" ]]; then
  ARGS+=(--human-mask-dilation-px "${HUMAN_MASK_DILATION_PX}")
fi
case "${FILTER_HUMAN_POINTS}" in
  "") ;;
  0|false|FALSE|False|no|NO|No|off|OFF|Off) ARGS+=(--no-filter-human-points) ;;
  *) ARGS+=(--filter-human-points) ;;
esac

echo "========== Full cached SequenceViewer: no inference =========="
echo "Cache      : ${CACHE_DIR}"
echo "Port       : ${PORT}"
echo "SMPL edits : ${SMPL_EDIT_OUTPUT:-<cached output setting>}"
echo "Point size : ${POINT_SIZE:-<cached value>}"
echo "Mask pixels: ${HUMAN_MASK_DILATION_PX:-<cached value>}"
echo "Filter     : ${FILTER_HUMAN_POINTS:-<cached value>}"

python scripts/vis/serve_full_sequence_viewer_cache.py "${ARGS[@]}"
