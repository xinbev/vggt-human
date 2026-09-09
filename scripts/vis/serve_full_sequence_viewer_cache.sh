#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/outputs/vis/stage2_walking_coarse_residual_v3/full_viewer_cache}"
PORT="${PORT:-8080}"
SMPL_EDIT_OUTPUT="${SMPL_EDIT_OUTPUT:-}"

cd "${REPO_ROOT}"
[[ -f "${CACHE_DIR}/manifest.json" ]] || { echo "[ERROR] Missing full viewer cache: ${CACHE_DIR}/manifest.json" >&2; exit 1; }

ARGS=(
  --cache-dir "${CACHE_DIR}"
  --port "${PORT}"
)
if [[ -n "${SMPL_EDIT_OUTPUT}" ]]; then
  ARGS+=(--smpl-edit-output "${SMPL_EDIT_OUTPUT}")
fi

echo "========== Full cached SequenceViewer: no inference =========="
echo "Cache      : ${CACHE_DIR}"
echo "Port       : ${PORT}"
echo "SMPL edits : ${SMPL_EDIT_OUTPUT:-<cached output setting>}"

python scripts/vis/serve_full_sequence_viewer_cache.py "${ARGS[@]}"
