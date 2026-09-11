#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/outputs/vis/stage2_emdb_p8_68_outdoor_handstand_300_1200_500f/full_viewer_cache}"
GT_PKL="${GT_PKL:-/home/zhw/xyb_space/emdb/P8/68_outdoor_handstand/P8_68_outdoor_handstand_data.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/emdb_camera_trajectory/p8_68_outdoor_handstand_300_1200_500f}"
PLOT_AXES="${PLOT_AXES:-xz}"
FRAME_INDEX_OFFSET="${FRAME_INDEX_OFFSET:-0}"
GOOD_FRAMES_ONLY="${GOOD_FRAMES_ONLY:-true}"
TITLE="${TITLE:-P8_68_outdoor_handstand}"

cd "${REPO_ROOT}"
[[ -f "${CACHE_DIR}/manifest.json" ]] || { echo "[ERROR] Missing full viewer cache: ${CACHE_DIR}/manifest.json" >&2; exit 1; }
[[ -f "${GT_PKL}" ]] || { echo "[ERROR] Missing EMDB GT: ${GT_PKL}" >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}"

ARGS=(
  --cache-dir "${CACHE_DIR}"
  --gt-pkl "${GT_PKL}"
  --output-dir "${OUTPUT_DIR}"
  --plot-axes "${PLOT_AXES}"
  --frame-index-offset "${FRAME_INDEX_OFFSET}"
  --title "${TITLE}"
)
case "${GOOD_FRAMES_ONLY}" in
  0|false|FALSE|False|no|NO|No|off|OFF|Off) ARGS+=(--no-good-frames-only) ;;
  *) ARGS+=(--good-frames-only) ;;
esac

echo "========== EMDB camera trajectory comparison =========="
echo "Cache       : ${CACHE_DIR}"
echo "GT          : ${GT_PKL}"
echo "Output      : ${OUTPUT_DIR}"
echo "Plot axes   : ${PLOT_AXES}"
echo "Good frames : ${GOOD_FRAMES_ONLY}"

python scripts/vis/plot_emdb_camera_trajectory_comparison.py "${ARGS[@]}"
