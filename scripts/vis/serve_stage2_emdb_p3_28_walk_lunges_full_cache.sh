#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/stage2_walking_coarse_residual_v3}"
CACHE_DIR="${CACHE_DIR:-${OUTPUT_DIR}/full_viewer_cache}"
PORT="${PORT:-8088}"
SMPL_EDIT_OUTPUT="${SMPL_EDIT_OUTPUT:-${OUTPUT_DIR}/smpl_edit_offsets.json}"

cd "${REPO_ROOT}"
[[ -f "${CACHE_DIR}/manifest.json" ]] || { echo "[ERROR] Missing full viewer cache: ${CACHE_DIR}/manifest.json" >&2; exit 1; }

echo "========== Full cached SequenceViewer: no inference =========="
echo "Cache      : ${CACHE_DIR}"
echo "Port       : ${PORT}"
echo "SMPL edits : ${SMPL_EDIT_OUTPUT}"

python scripts/vis/serve_full_sequence_viewer_cache.py \
  --cache-dir "${CACHE_DIR}" \
  --port "${PORT}" \
  --smpl-edit-output "${SMPL_EDIT_OUTPUT}"
