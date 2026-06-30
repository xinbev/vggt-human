#!/usr/bin/env bash

set -euo pipefail

# Generate paper-figure elements for image patches -> person region -> query pooling.
#
# Example:
#   IMAGE_PATH=/path/to/frame.png \
#   PERSON_MASK=/path/to/sam2_masks.npz MASK_KEY=person_1 \
#   bash scripts/vis/create_arch_patch_pooling_elements.sh

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
IMAGE_PATH="${IMAGE_PATH:?Set IMAGE_PATH to the input image}"
PERSON_BBOX="${PERSON_BBOX:-}"
PERSON_MASK="${PERSON_MASK:-}"
MASK_KEY="${MASK_KEY:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/paper_arch_patch_pooling_elements}"
PATCH_SIZE="${PATCH_SIZE:-32}"
LONG_SIDE="${LONG_SIDE:-768}"
MIN_OVERLAP="${MIN_OVERLAP:-0.12}"

if [[ -z "${PERSON_MASK}" && -z "${PERSON_BBOX}" ]]; then
  echo "[ERROR] Set PERSON_MASK to a SAM2 mask file or PERSON_BBOX as x1,y1,x2,y2." >&2
  exit 1
fi

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "========== Architecture patch pooling elements =========="
echo "Image      : ${IMAGE_PATH}"
echo "Person bbox: ${PERSON_BBOX:-<none>}"
echo "Person mask: ${PERSON_MASK:-<none>}"
echo "Mask key   : ${MASK_KEY:-<all>}"
echo "Patch size : ${PATCH_SIZE}"
echo "Output     : ${OUTPUT_DIR}"

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

python scripts/vis/create_arch_patch_pooling_elements.py \
  --image "${IMAGE_PATH}" \
  "${REGION_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --patch-size "${PATCH_SIZE}" \
  --long-side "${LONG_SIDE}" \
  --min-overlap "${MIN_OVERLAP}"

echo "========== Architecture patch pooling elements finished =========="
echo "Output root: ${OUTPUT_DIR}"
