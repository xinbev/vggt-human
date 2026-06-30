#!/usr/bin/env bash

set -euo pipefail

# Generate paper-figure elements for image patches -> person region -> query pooling.
#
# Example:
#   IMAGE_PATH=/path/to/frame.png bash scripts/vis/create_arch_patch_pooling_elements.sh
#
# Optional reuse of existing SAM2 masks:
#   IMAGE_PATH=/path/to/frame.png PERSON_MASK=/path/to/sam2_masks.npz MASK_KEY=person_1 \
#   bash scripts/vis/create_arch_patch_pooling_elements.sh

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
IMAGE_PATH="${IMAGE_PATH:?Set IMAGE_PATH to the input image}"
PERSON_BBOX="${PERSON_BBOX:-}"
PERSON_MASK="${PERSON_MASK:-}"
MASK_KEY="${MASK_KEY:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/paper_arch_patch_pooling_elements}"
PATCH_SIZE="${PATCH_SIZE:-16}"
LONG_SIDE="${LONG_SIDE:-768}"
MIN_OVERLAP="${MIN_OVERLAP:-0.12}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
DEVICE="${DEVICE:-cuda}"
YOLO_CHECKPOINT="${YOLO_CHECKPOINT:-}"
SAM2_ROOT="${SAM2_ROOT:-}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-}"
SAM2_MODEL_CFG="${SAM2_MODEL_CFG:-configs/sam2.1/sam2.1_hiera_l.yaml}"
DET_CONF="${DET_CONF:-0.25}"
DET_IOU="${DET_IOU:-0.7}"
AUTO_PERSON_INDEX="${AUTO_PERSON_INDEX:-0}"
AUTO_TOP_K="${AUTO_TOP_K:-2}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "========== Architecture patch pooling elements =========="
echo "Image      : ${IMAGE_PATH}"
echo "Person bbox: ${PERSON_BBOX:-<none>}"
echo "Person mask: ${PERSON_MASK:-<none>}"
echo "Mask key   : ${MASK_KEY:-<all>}"
echo "Patch size : ${PATCH_SIZE}"
echo "Output     : ${OUTPUT_DIR}"
if [[ -z "${PERSON_MASK}" && -z "${PERSON_BBOX}" ]]; then
  echo "Auto mode  : YOLO TorchScript person detection + SAM2 mask"
fi

REGION_ARGS=()
if [[ -n "${PERSON_MASK}" ]]; then
  REGION_ARGS+=(--mask "${PERSON_MASK}")
  if [[ -n "${MASK_KEY}" ]]; then
    REGION_ARGS+=(--mask-key "${MASK_KEY}")
  fi
fi
if [[ -n "${PERSON_BBOX}" ]]; then
  REGION_ARGS+=(--bbox "${PERSON_BBOX}")
fi
if [[ -n "${YOLO_CHECKPOINT}" ]]; then
  REGION_ARGS+=(--yolo-checkpoint "${YOLO_CHECKPOINT}")
fi
if [[ -n "${SAM2_ROOT}" ]]; then
  REGION_ARGS+=(--sam2-root "${SAM2_ROOT}")
fi
if [[ -n "${SAM2_CHECKPOINT}" ]]; then
  REGION_ARGS+=(--sam2-checkpoint "${SAM2_CHECKPOINT}")
fi

python scripts/vis/create_arch_patch_pooling_elements.py \
  --image "${IMAGE_PATH}" \
  "${REGION_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --patch-size "${PATCH_SIZE}" \
  --long-side "${LONG_SIDE}" \
  --min-overlap "${MIN_OVERLAP}" \
  --path-config "${PATH_CONFIG}" \
  --device "${DEVICE}" \
  --sam2-model-cfg "${SAM2_MODEL_CFG}" \
  --det-conf "${DET_CONF}" \
  --det-iou "${DET_IOU}" \
  --auto-person-index "${AUTO_PERSON_INDEX}" \
  --auto-top-k "${AUTO_TOP_K}"

echo "========== Architecture patch pooling elements finished =========="
echo "Output root: ${OUTPUT_DIR}"
