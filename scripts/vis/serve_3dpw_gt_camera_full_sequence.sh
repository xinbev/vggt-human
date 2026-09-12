#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/3dpw_downtown_walkBridge_01_gt_camera_full}"
CACHE_DIR="${CACHE_DIR:-${OUTPUT_DIR}/full_viewer_cache_gt_camera}"
PORT="${PORT:-8090}"
POINT_SIZE="${POINT_SIZE:-0.006}"
SMPL_EDIT_OUTPUT="${SMPL_EDIT_OUTPUT:-${OUTPUT_DIR}/smpl_edit_offsets.json}"
HUMAN_MASK_DILATION_PX="${HUMAN_MASK_DILATION_PX:-}"
FILTER_HUMAN_POINTS="${FILTER_HUMAN_POINTS:-}"
VIEWER_MODE="${VIEWER_MODE:-}"
SMPL_DISPLAY_FRAMES="${SMPL_DISPLAY_FRAMES:-}"
DISPLAY_PEOPLE="${DISPLAY_PEOPLE:-}"

cd "${REPO_ROOT}"
[[ -f "${CACHE_DIR}/manifest.json" ]] || { echo "[ERROR] Missing merged cache: ${CACHE_DIR}/manifest.json" >&2; exit 1; }
ARGS=(
  --cache-dir "${CACHE_DIR}"
  --port "${PORT}"
  --point-size "${POINT_SIZE}"
  --smpl-edit-output "${SMPL_EDIT_OUTPUT}"
)
if [[ -n "${HUMAN_MASK_DILATION_PX}" ]]; then
  ARGS+=(--human-mask-dilation-px "${HUMAN_MASK_DILATION_PX}")
fi
if [[ -n "${VIEWER_MODE}" ]]; then ARGS+=(--viewer-mode "${VIEWER_MODE}"); fi
if [[ -n "${SMPL_DISPLAY_FRAMES}" ]]; then ARGS+=(--smpl-display-frames "${SMPL_DISPLAY_FRAMES}"); fi
if [[ -n "${DISPLAY_PEOPLE}" ]]; then ARGS+=(--display-people "${DISPLAY_PEOPLE}"); fi
case "${FILTER_HUMAN_POINTS}" in
  0|false|FALSE|False|no|NO|No|off|OFF|Off) ARGS+=(--no-filter-human-points) ;;
  1|true|TRUE|True|yes|YES|Yes|on|ON|On) ARGS+=(--filter-human-points) ;;
esac
echo "========== Serve merged 3DPW GT-camera cache =========="
echo "Cache      : ${CACHE_DIR}"
echo "Port       : ${PORT}"
python scripts/vis/serve_full_sequence_viewer_cache.py "${ARGS[@]}"
