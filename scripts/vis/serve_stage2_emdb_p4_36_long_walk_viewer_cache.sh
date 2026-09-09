#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/outputs/vis/stage2_emdb_p4_36_long_walk_coarse_residual_v3/viewer_cache}"
PORT="${PORT:-8081}"
POINT_SIZE="${POINT_SIZE:-0.006}"
SMPL_DISPLAY_FRAMES="${SMPL_DISPLAY_FRAMES:-50}"
INITIAL_TIMESTEP="${INITIAL_TIMESTEP:--1}"
VIEWER_MODE="${VIEWER_MODE:-3d}"
SHOW_TRACK_IDS="${SHOW_TRACK_IDS:-true}"

case "${VIEWER_MODE}" in
  4d|"4D current frame") VIEWER_MODE_ARG="4D current frame" ;;
  3d|"3D accumulate") VIEWER_MODE_ARG="3D accumulate" ;;
  hybrid|Hybrid) VIEWER_MODE_ARG="Hybrid" ;;
  *) echo "[ERROR] VIEWER_MODE must be one of: 4d, 3d, hybrid. Got: ${VIEWER_MODE}" >&2; exit 1 ;;
esac

cd "${REPO_ROOT}"
[[ -f "${CACHE_DIR}/manifest.json" ]] || { echo "[ERROR] Missing viewer cache: ${CACHE_DIR}/manifest.json" >&2; exit 1; }

ARGS=(
  --cache-dir "${CACHE_DIR}"
  --port "${PORT}"
  --point-size "${POINT_SIZE}"
  --smpl-display-frames "${SMPL_DISPLAY_FRAMES}"
  --initial-timestep "${INITIAL_TIMESTEP}"
  --viewer-mode "${VIEWER_MODE_ARG}"
)
case "${SHOW_TRACK_IDS}" in
  0|false|FALSE|False|no|NO|No|off|OFF|Off) ARGS+=(--no-show-track-ids) ;;
  *) ARGS+=(--show-track-ids) ;;
esac

echo "========== Stage2 cached Viser viewer (no inference) =========="
echo "Cache       : ${CACHE_DIR}"
echo "Port        : ${PORT}"
echo "Viewer mode : ${VIEWER_MODE_ARG}"
echo "SMPL frames : ${SMPL_DISPLAY_FRAMES}"
echo "Start frame : ${INITIAL_TIMESTEP} (-1 means final frame)"

python scripts/vis/serve_stage2_viewer_cache.py "${ARGS[@]}"
