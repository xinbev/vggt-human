#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/bedlam_annotation_inspection}"
SPLITS="${SPLITS:-Training}"
MAX_SEQUENCES="${MAX_SEQUENCES:-20}"
MAX_FRAMES_PER_SEQUENCE="${MAX_FRAMES_PER_SEQUENCE:-20}"
MAX_SAMPLES="${MAX_SAMPLES:-80}"
MIN_VISIBLE_JOINTS="${MIN_VISIBLE_JOINTS:-4}"
MIN_BOX_AREA="${MIN_BOX_AREA:-100}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }

echo "========== Inspect BEDLAM annotations =========="
echo "BEDLAM root : ${BEDLAM_ROOT}"
echo "Output dir  : ${OUTPUT_DIR}"
echo "Splits      : ${SPLITS}"
echo "Sequences   : ${MAX_SEQUENCES}"
echo "Frames/seq  : ${MAX_FRAMES_PER_SEQUENCE}"
echo "Min joints  : ${MIN_VISIBLE_JOINTS}"
echo "Min box area: ${MIN_BOX_AREA}"

python scripts/diagnostics/inspect_bedlam_annotations.py \
  --dataset-root "${BEDLAM_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --splits ${SPLITS} \
  --max-sequences "${MAX_SEQUENCES}" \
  --max-frames-per-sequence "${MAX_FRAMES_PER_SEQUENCE}" \
  --max-samples "${MAX_SAMPLES}" \
  --min-visible-joints "${MIN_VISIBLE_JOINTS}" \
  --min-box-area "${MIN_BOX_AREA}"

echo "========== BEDLAM annotation inspection ready =========="
echo "Summary: ${OUTPUT_DIR}/summary.json"
