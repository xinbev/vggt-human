#!/usr/bin/env bash

set -euo pipefail

# Collect the top bad SMPL Translation V2 input RGB frames into one folder.
# Run this on the server where BEDLAM RGB images exist, then sync the output.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
FRAME_CSV="${FRAME_CSV:-${REPO_ROOT}/outputs/eval/smpl_translation_v2_longseq_80g/dedup_frame_person_report/dedup_frame_translation_summary.csv}"
PERSON_CSV="${PERSON_CSV:-${REPO_ROOT}/outputs/eval/smpl_translation_v2_longseq_80g/dedup_frame_person_report/dedup_frame_person_translation_metrics.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/smpl_translation_v2_bad_input_images_top10}"
TOP_K="${TOP_K:-10}"
SORT_KEY="${SORT_KEY:-refined_max_transl_l2_m}"
SPLIT="${SPLIT:-Training}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${FRAME_CSV}" ]] || { echo "[ERROR] Missing frame CSV: ${FRAME_CSV}" >&2; exit 1; }
[[ -f "${PERSON_CSV}" ]] || { echo "[ERROR] Missing person CSV: ${PERSON_CSV}" >&2; exit 1; }

echo "========== Collect bad translation input images =========="
echo "Frame CSV : ${FRAME_CSV}"
echo "Person CSV: ${PERSON_CSV}"
echo "BEDLAM    : ${BEDLAM_ROOT}"
echo "Output    : ${OUTPUT_DIR}"
echo "Top K     : ${TOP_K}"
echo "Sort key  : ${SORT_KEY}"

python scripts/vis/collect_bad_translation_input_images.py \
  --frame-csv "${FRAME_CSV}" \
  --person-csv "${PERSON_CSV}" \
  --path-config "${PATH_CONFIG}" \
  --bedlam-root "${BEDLAM_ROOT}" \
  --split "${SPLIT}" \
  --output-dir "${OUTPUT_DIR}" \
  --top-k "${TOP_K}" \
  --sort-key "${SORT_KEY}" \
  --copy-images \
  --contact-sheet

echo "========== Bad translation image collection finished =========="
echo "Output: ${OUTPUT_DIR}"
