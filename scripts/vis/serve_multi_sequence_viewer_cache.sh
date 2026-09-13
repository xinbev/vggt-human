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

[[ -n "${CACHE_DIRS}" ]] || { echo "[ERROR] CACHE_DIRS is required (colon-separated full cache directories)" >&2; exit 1; }

cd "${REPO_ROOT}"
IFS=':' read -r -a CACHE_ARRAY <<< "${CACHE_DIRS}"
ARGS=()
for cache_dir in "${CACHE_ARRAY[@]}"; do
  [[ -n "${cache_dir}" ]] || continue
  [[ -f "${cache_dir}/manifest.json" ]] || { echo "[ERROR] Missing full viewer cache: ${cache_dir}/manifest.json" >&2; exit 1; }
  ARGS+=(--cache-dir "${cache_dir}")
done
[[ "${#ARGS[@]}" -gt 0 ]] || { echo "[ERROR] CACHE_DIRS did not contain a usable cache" >&2; exit 1; }

if [[ -n "${CACHE_NAMES}" ]]; then
  IFS=':' read -r -a NAME_ARRAY <<< "${CACHE_NAMES}"
  for name in "${NAME_ARRAY[@]}"; do ARGS+=(--cache-name "${name}"); done
fi

echo "========== Multi-cache SequenceViewer =========="
echo "Caches        : ${CACHE_DIRS}"
echo "Port          : ${PORT}"
echo "Mode          : ${VIEWER_MODE}"
echo "SMPL frames   : ${SMPL_DISPLAY_FRAMES}"
echo "Point size    : ${POINT_SIZE}"
echo "Camera scale  : ${CAMERA_SCALE}"

python scripts/vis/serve_multi_sequence_viewer_cache.py \
  "${ARGS[@]}" \
  --port "${PORT}" \
  --point-size "${POINT_SIZE}" \
  --camera-scale "${CAMERA_SCALE}" \
  --smpl-display-frames "${SMPL_DISPLAY_FRAMES}" \
  --initial-timestep "${INITIAL_TIMESTEP}" \
  --viewer-mode "${VIEWER_MODE}" \
  "$(case "${SHOW_CAMERAS}" in 0|false|FALSE|no|off) echo --no-show-cameras ;; *) echo --show-cameras ;; esac)" \
  "$(case "${SHOW_TRAJECTORIES}" in 0|false|FALSE|no|off) echo --no-show-trajectories ;; *) echo --show-trajectories ;; esac)"
