#!/usr/bin/env bash
set -euo pipefail

# Project SMPL meshes saved by a previous viewer-cache inference onto RGB frames.
REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/outputs/vis/stage2_walking_coarse_residual_v3/full_viewer_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/cached_smpl_projection}"
IMAGE_ROOT="${IMAGE_ROOT:-}"
FRAME_INDEX="${FRAME_INDEX:-0}"
FRAME_ID="${FRAME_ID:-}"
RENDER_ALL="${RENDER_ALL:-false}"
MESH_SOURCE="${MESH_SOURCE:-hsi}"
CAMERA_SOURCE="${CAMERA_SOURCE:-hsi}"
RESIZE_MODE="${RESIZE_MODE:-balanced}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"
PATCH_SIZE="${PATCH_SIZE:-16}"
FACE_STRIDE="${FACE_STRIDE:-1}"
LINE_WIDTH="${LINE_WIDTH:-1}"
FILL_ALPHA="${FILL_ALPHA:-92}"
EDGE_ALPHA="${EDGE_ALPHA:-235}"

cd "${REPO_ROOT}"
[[ -f "${CACHE_DIR}/manifest.json" ]] || { echo "[ERROR] Missing cache manifest: ${CACHE_DIR}/manifest.json" >&2; exit 1; }

ARGS=(
  --cache-dir "${CACHE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --mesh-source "${MESH_SOURCE}"
  --camera-source "${CAMERA_SOURCE}"
  --resize-mode "${RESIZE_MODE}"
  --image-resolution "${IMAGE_RESOLUTION}"
  --patch-size "${PATCH_SIZE}"
  --face-stride "${FACE_STRIDE}"
  --line-width "${LINE_WIDTH}"
  --fill-alpha "${FILL_ALPHA}"
  --edge-alpha "${EDGE_ALPHA}"
)
if [[ -n "${IMAGE_ROOT}" ]]; then ARGS+=(--image-root "${IMAGE_ROOT}"); fi
if [[ -n "${FRAME_ID}" ]]; then
  ARGS+=(--frame-id "${FRAME_ID}")
elif [[ "${RENDER_ALL}" == "true" ]]; then
  ARGS+=(--all)
else
  ARGS+=(--frame-index "${FRAME_INDEX}")
fi

echo "========== Cached SMPL RGB projection =========="
echo "Cache       : ${CACHE_DIR}"
echo "Image root  : ${IMAGE_ROOT:-<cache source paths>}"
echo "Mesh/camera : ${MESH_SOURCE}/${CAMERA_SOURCE}"
echo "Output      : ${OUTPUT_DIR}"
python scripts/vis/project_cached_smpl_on_rgb.py "${ARGS[@]}"
