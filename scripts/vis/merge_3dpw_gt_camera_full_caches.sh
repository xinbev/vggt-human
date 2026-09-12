#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
INPUT_ROOT="${INPUT_ROOT:-${REPO_ROOT}/outputs/vis/3dpw_downtown_walkBridge_01_gt_camera_full}"
GT_PKL="${GT_PKL:-/home/zhw/xyb_space/3DPW/sequenceFiles/test/downtown_walkBridge_01.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-${INPUT_ROOT}/full_viewer_cache_gt_camera}"
TOTAL_FRAMES="${TOTAL_FRAMES:-1371}"
FRAME_INDEX_OFFSET="${FRAME_INDEX_OFFSET:-0}"
ALLOW_MISSING_FRAMES="${ALLOW_MISSING_FRAMES:-false}"

cd "${REPO_ROOT}"
[[ -f "${GT_PKL}" ]] || { echo "[ERROR] Missing 3DPW GT pickle: ${GT_PKL}" >&2; exit 1; }
ARGS=(
  scripts/vis/merge_3dpw_gt_camera_full_caches.py
  --input-root "${INPUT_ROOT}"
  --gt-pkl "${GT_PKL}"
  --output-dir "${OUTPUT_DIR}"
  --expected-frames "${TOTAL_FRAMES}"
  --frame-index-offset "${FRAME_INDEX_OFFSET}"
)
case "${ALLOW_MISSING_FRAMES}" in
  1|true|TRUE|True|yes|YES|Yes|on|ON|On) ARGS+=(--allow-missing) ;;
esac
python "${ARGS[@]}"
