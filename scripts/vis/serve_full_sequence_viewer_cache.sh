#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/outputs/vis/stage2_walking_coarse_residual_v3/full_viewer_cache}"
PORT="${PORT:-8080}"
SMPL_EDIT_OUTPUT="${SMPL_EDIT_OUTPUT:-}"
SMPL_VISUAL_SCALE="${SMPL_VISUAL_SCALE:-}"
POINT_SIZE="${POINT_SIZE:-}"
HUMAN_MASK_DILATION_PX="${HUMAN_MASK_DILATION_PX:-}"
FILTER_HUMAN_POINTS="${FILTER_HUMAN_POINTS:-}"
SHOW_GT_SMPL="${SHOW_GT_SMPL:-0}"
SHOW_PINNED_ROOT_TRAJECTORY="${SHOW_PINNED_ROOT_TRAJECTORY:-0}"
RICH_SEQUENCE="${RICH_SEQUENCE:-}"
RICH_SUPPORT_ROOT="${RICH_SUPPORT_ROOT:-/home/zhw/xyb_space/RICH/hmr4d_support}"
SMPLX_MODEL_DIR="${SMPLX_MODEL_DIR:-${REPO_ROOT}/checkpoints/body_models/smplx}"
SMPLX_TO_SMPL="${SMPLX_TO_SMPL:-${REPO_ROOT}/checkpoints/utils/smplx2smpl.pkl}"
GT_COORDINATE_SOURCE="${GT_COORDINATE_SOURCE:-hsi_scaled}"
GT_DEVICE="${GT_DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-}"

cd "${REPO_ROOT}"
if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
fi
[[ -f "${CACHE_DIR}/manifest.json" ]] || { echo "[ERROR] Missing full viewer cache: ${CACHE_DIR}/manifest.json" >&2; exit 1; }

ARGS=(
  --cache-dir "${CACHE_DIR}"
  --port "${PORT}"
)
if [[ -n "${SMPL_EDIT_OUTPUT}" ]]; then
  ARGS+=(--smpl-edit-output "${SMPL_EDIT_OUTPUT}")
fi
if [[ -n "${SMPL_VISUAL_SCALE}" ]]; then
  ARGS+=(--smpl-visual-scale "${SMPL_VISUAL_SCALE}")
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
case "${SHOW_GT_SMPL}" in
  1|true|TRUE|True|yes|YES|Yes|on|ON|On) ARGS+=(--show-gt-smpl); GT_ENABLED=1 ;;
  0|false|FALSE|False|no|NO|No|off|OFF|Off|"") GT_ENABLED=0 ;;
  *) echo "[ERROR] SHOW_GT_SMPL must be a boolean, got: ${SHOW_GT_SMPL}" >&2; exit 1 ;;
esac
case "${SHOW_PINNED_ROOT_TRAJECTORY}" in
  1|true|TRUE|True|yes|YES|Yes|on|ON|On) ARGS+=(--show-pinned-root-trajectory) ;;
  0|false|FALSE|False|no|NO|No|off|OFF|Off|"") ARGS+=(--no-show-pinned-root-trajectory) ;;
  *) echo "[ERROR] SHOW_PINNED_ROOT_TRAJECTORY must be a boolean, got: ${SHOW_PINNED_ROOT_TRAJECTORY}" >&2; exit 1 ;;
esac
if [[ "${GT_ENABLED}" -eq 1 ]]; then
  ARGS+=(
    --rich-support-root "${RICH_SUPPORT_ROOT}"
    --smplx-model-dir "${SMPLX_MODEL_DIR}"
    --smplx-to-smpl "${SMPLX_TO_SMPL}"
    --gt-coordinate-source "${GT_COORDINATE_SOURCE}"
    --gt-device "${GT_DEVICE}"
  )
  if [[ -n "${RICH_SEQUENCE}" ]]; then
    ARGS+=(--rich-sequence "${RICH_SEQUENCE}")
  fi
fi

echo "========== Full cached SequenceViewer: no inference =========="
echo "Cache      : ${CACHE_DIR}"
echo "Port       : ${PORT}"
echo "SMPL edits : ${SMPL_EDIT_OUTPUT:-<cached output setting>}"
echo "SMPL scale : ${SMPL_VISUAL_SCALE:-1.0} (viewer-only, mesh-centered)"
echo "Point size : ${POINT_SIZE:-<cached value>}"
echo "Mask pixels: ${HUMAN_MASK_DILATION_PX:-<cached value>}"
echo "Filter     : ${FILTER_HUMAN_POINTS:-<cached value>}"
echo "GT SMPL    : ${SHOW_GT_SMPL} (${RICH_SEQUENCE:-auto sequence}, source=${GT_COORDINATE_SOURCE})"
echo "Root path  : ${SHOW_PINNED_ROOT_TRAJECTORY} (pin-panel checkbox initial state)"
echo "GPU        : ${CUDA_VISIBLE_DEVICES_VALUE:-<default>}"

python scripts/vis/serve_full_sequence_viewer_cache.py "${ARGS[@]}"
