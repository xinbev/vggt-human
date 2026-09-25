#!/usr/bin/env bash
set -euo pipefail

# Generate SAM2 masks and RGB overlays from an existing viewer cache.
REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/outputs/vis/stage2_walking_coarse_residual_v3/full_viewer_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/cached_sam2_masks}"
EXPORT_CACHE_DIR="${EXPORT_CACHE_DIR:-}"
IMAGE_ROOT="${IMAGE_ROOT:-}"
FRAME_INDEX="${FRAME_INDEX:-0}"
FRAME_ID="${FRAME_ID:-}"
RENDER_ALL="${RENDER_ALL:-false}"
SAM2_ROOT="${SAM2_ROOT:-${REPO_ROOT}/third_party/sam2}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${REPO_ROOT}/third_party/weights/sam/sam2.1_hiera_large.pt}"
SAM2_MODEL_CFG="${SAM2_MODEL_CFG:-configs/sam2.1/sam2.1_hiera_l.yaml}"
DEVICE="${DEVICE:-cuda}"
MESH_SOURCE="${MESH_SOURCE:-hsi}"
CAMERA_SOURCE="${CAMERA_SOURCE:-hsi}"
RESIZE_MODE="${RESIZE_MODE:-balanced}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"
PATCH_SIZE="${PATCH_SIZE:-16}"
BOX_EXPAND_RATIO="${BOX_EXPAND_RATIO:-0.05}"
MASK_ALPHA="${MASK_ALPHA:-105}"
DRAW_BOXES="${DRAW_BOXES:-false}"
SINGLE_MASK="${SINGLE_MASK:-false}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-}"
DISABLE_CV2="${DISABLE_CV2:-false}"

cd "${REPO_ROOT}"
if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
fi
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
if [[ "${DISABLE_CV2}" == "true" ]]; then
  export VGGT_OMEGA_DISABLE_CV2=1
fi
[[ -f "${CACHE_DIR}/manifest.json" ]] || { echo "[ERROR] Missing cache manifest: ${CACHE_DIR}/manifest.json" >&2; exit 1; }
[[ -d "${SAM2_ROOT}" ]] || { echo "[ERROR] Missing SAM2 root: ${SAM2_ROOT}" >&2; exit 1; }
[[ -f "${SAM2_CHECKPOINT}" ]] || { echo "[ERROR] Missing SAM2 checkpoint: ${SAM2_CHECKPOINT}" >&2; exit 1; }

ARGS=(
  --cache-dir "${CACHE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --sam2-root "${SAM2_ROOT}"
  --sam2-checkpoint "${SAM2_CHECKPOINT}"
  --sam2-model-cfg "${SAM2_MODEL_CFG}"
  --device "${DEVICE}"
  --mesh-source "${MESH_SOURCE}"
  --camera-source "${CAMERA_SOURCE}"
  --resize-mode "${RESIZE_MODE}"
  --image-resolution "${IMAGE_RESOLUTION}"
  --patch-size "${PATCH_SIZE}"
  --box-expand-ratio "${BOX_EXPAND_RATIO}"
  --mask-alpha "${MASK_ALPHA}"
)
if [[ -n "${EXPORT_CACHE_DIR}" ]]; then ARGS+=(--export-cache-dir "${EXPORT_CACHE_DIR}"); fi
if [[ -n "${IMAGE_ROOT}" ]]; then ARGS+=(--image-root "${IMAGE_ROOT}"); fi
if [[ -n "${FRAME_ID}" ]]; then
  ARGS+=(--frame-id "${FRAME_ID}")
elif [[ "${RENDER_ALL}" == "true" ]]; then
  ARGS+=(--all)
else
  ARGS+=(--frame-index "${FRAME_INDEX}")
fi
if [[ "${DRAW_BOXES}" == "true" ]]; then ARGS+=(--draw-boxes); fi
if [[ "${SINGLE_MASK}" == "true" ]]; then ARGS+=(--single-mask); fi

echo "========== Cached SAM2 mask visualization =========="
echo "Cache      : ${CACHE_DIR}"
echo "Image root : ${IMAGE_ROOT:-<cache source paths>}"
echo "SAM2       : ${SAM2_CHECKPOINT}"
echo "Prompt     : projected ${MESH_SOURCE} SMPL vertices (${CAMERA_SOURCE} camera)"
echo "Output     : ${OUTPUT_DIR}"
echo "Disable cv2: ${DISABLE_CV2}"
python scripts/vis/visualize_cached_sam2_masks.py "${ARGS[@]}"
