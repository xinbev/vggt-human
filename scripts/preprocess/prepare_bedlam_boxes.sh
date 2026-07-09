#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
SPLITS="${SPLITS:-Training}"
MAX_HUMANS="${MAX_HUMANS:-20}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
REQUIRE_BOXES="${REQUIRE_BOXES:-true}"
USE_SMPL_PROJECTION="${USE_SMPL_PROJECTION:-false}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-/home/zhw/lab_users/xyb/home/projects/vggt-human/checkpoints/body_models}"
PROJECTION_SOURCE="${PROJECTION_SOURCE:-vertices}"
VISIBLE_ONLY="${VISIBLE_ONLY:-true}"
MIN_VISIBLE_JOINTS="${MIN_VISIBLE_JOINTS:-4}"
MIN_BOX_AREA="${MIN_BOX_AREA:-100}"
REQUIRE_J2D_VISIBILITY="${REQUIRE_J2D_VISIBILITY:-false}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }

ARGS=(
  --dataset-root "${BEDLAM_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --splits ${SPLITS}
  --image-size "${IMAGE_SIZE}"
  --max-humans "${MAX_HUMANS}"
  --min-visible-joints "${MIN_VISIBLE_JOINTS}"
  --min-box-area "${MIN_BOX_AREA}"
)

if [[ "${REQUIRE_BOXES}" == "true" ]]; then
  ARGS+=(--require-boxes)
fi

if [[ "${USE_SMPL_PROJECTION}" == "true" ]]; then
  [[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }
  ARGS+=(--use-smpl-projection --smpl-model-dir "${SMPL_MODEL_DIR}" --projection-source "${PROJECTION_SOURCE}")
fi

if [[ "${VISIBLE_ONLY}" == "true" ]]; then
  ARGS+=(--visible-only)
fi

if [[ "${REQUIRE_J2D_VISIBILITY}" == "true" ]]; then
  ARGS+=(--require-j2d-visibility)
fi

echo "========== Prepare BEDLAM box sidecars =========="
echo "BEDLAM root : ${BEDLAM_ROOT}"
echo "Output root : ${OUTPUT_ROOT}"
echo "Splits      : ${SPLITS}"
echo "Max humans  : ${MAX_HUMANS}"
echo "Projection  : ${USE_SMPL_PROJECTION}"
echo "Visible only: ${VISIBLE_ONLY}"
echo "Min joints  : ${MIN_VISIBLE_JOINTS}"
echo "Min box area: ${MIN_BOX_AREA}"

python scripts/preprocess/prepare_bedlam_boxes.py "${ARGS[@]}"

echo "========== BEDLAM box sidecars ready =========="
echo "Summary: ${OUTPUT_ROOT}/summary.json"
