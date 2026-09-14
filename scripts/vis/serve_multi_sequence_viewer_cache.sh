#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
CACHE_DIRS="${CACHE_DIRS:-}"
PORT="${PORT:-8080}"
POINT_SIZE="${POINT_SIZE:-0.006}"
CAMERA_SCALE="${CAMERA_SCALE:-0.05}"
SMPL_DISPLAY_FRAMES="${SMPL_DISPLAY_FRAMES:-50}"
INITIAL_TIMESTEP="${INITIAL_TIMESTEP:-0}"
VIEWER_MODE="${VIEWER_MODE:-Hybrid}"
SHOW_CAMERAS="${SHOW_CAMERAS:-false}"
SHOW_TRAJECTORIES="${SHOW_TRAJECTORIES:-true}"
CACHE_NAMES="${CACHE_NAMES:-}"
LAYOUT_JSON="${LAYOUT_JSON:-${REPO_ROOT}/outputs/vis/multi_cache_alignment.json}"
ALIGNMENT_PREVIEW="${ALIGNMENT_PREVIEW:-false}"

if [[ -z "${CACHE_DIRS}" && -z "${LAYOUT_JSON}" ]]; then
  echo "[ERROR] CACHE_DIRS or LAYOUT_JSON is required" >&2
  exit 1
fi

cd "${REPO_ROOT}"
IFS=':' read -r -a CACHE_ARRAY <<< "${CACHE_DIRS}"
ARGS=()
for cache_dir in "${CACHE_ARRAY[@]}"; do
  [[ -n "${cache_dir}" ]] || continue
  [[ -f "${cache_dir}/manifest.json" ]] || { echo "[ERROR] Missing full viewer cache: ${cache_dir}/manifest.json" >&2; exit 1; }
  ARGS+=(--cache-dir "${cache_dir}")
done
if [[ -z "${LAYOUT_JSON}" && "${#ARGS[@]}" -eq 0 ]]; then
  echo "[ERROR] CACHE_DIRS did not contain a usable cache" >&2
  exit 1
fi

if [[ -n "${CACHE_NAMES}" ]]; then
  IFS=':' read -r -a NAME_ARRAY <<< "${CACHE_NAMES}"
  for name in "${NAME_ARRAY[@]}"; do ARGS+=(--cache-name "${name}"); done
fi
if [[ -n "${LAYOUT_JSON}" ]]; then ARGS+=(--layout-json "${LAYOUT_JSON}"); fi

case "${ALIGNMENT_PREVIEW}" in
  0|false|FALSE|no|NO|off|OFF) ALIGNMENT_ARGS=(--no-alignment-preview) ;;
  *) ALIGNMENT_ARGS=(--alignment-preview) ;;
esac
case "${SHOW_CAMERAS}" in
  0|false|FALSE|no|NO|off|OFF) CAMERA_ARGS=(--no-show-cameras) ;;
  *) CAMERA_ARGS=(--show-cameras) ;;
esac
case "${SHOW_TRAJECTORIES}" in
  0|false|FALSE|no|NO|off|OFF) TRAJECTORY_ARGS=(--no-show-trajectories) ;;
  *) TRAJECTORY_ARGS=(--show-trajectories) ;;
esac

echo "========== Multi-cache SequenceViewer =========="
echo "Caches        : ${CACHE_DIRS}"
echo "Port          : ${PORT}"
echo "Mode          : ${VIEWER_MODE}"
echo "SMPL frames   : ${SMPL_DISPLAY_FRAMES}"
echo "Point size    : ${POINT_SIZE}"
echo "Camera scale  : ${CAMERA_SCALE}"
echo "Layout        : ${LAYOUT_JSON:-<none>}"
echo "Align preview : ${ALIGNMENT_PREVIEW}"

python scripts/vis/serve_multi_sequence_viewer_cache.py \
  "${ARGS[@]}" \
  --port "${PORT}" \
  --point-size "${POINT_SIZE}" \
  --camera-scale "${CAMERA_SCALE}" \
  --smpl-display-frames "${SMPL_DISPLAY_FRAMES}" \
  --initial-timestep "${INITIAL_TIMESTEP}" \
  --viewer-mode "${VIEWER_MODE}" \
  "${CAMERA_ARGS[@]}" \
  "${TRAJECTORY_ARGS[@]}" \
  "${ALIGNMENT_ARGS[@]}"
