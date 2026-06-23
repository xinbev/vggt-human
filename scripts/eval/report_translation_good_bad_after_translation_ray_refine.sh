#!/usr/bin/env bash

set -euo pipefail

# Build an overall good/bad frame report for SMPL translation after the
# translation-ray-refine + HSI route.  By default this reads an existing scan
# CSV.  Set RUN_SCAN=1 to run the dataset-wide model scan first.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-6}"

SCAN_OUTPUT_DIR="${SCAN_OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/hsi_bad_translation_scan_after_translation_ray_refine}"
INPUT_CSV="${INPUT_CSV:-${SCAN_OUTPUT_DIR}/all_frame_person_translation_rows.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/translation_good_bad_after_translation_ray_refine}"

RUN_SCAN="${RUN_SCAN:-0}"
BAD_TRANSL_M="${BAD_TRANSL_M:-0.50}"
SEVERE_TRANSL_M="${SEVERE_TRANSL_M:-0.80}"
FOCUS_FRAMES="${FOCUS_FRAMES:-seq_000000_0085,seq_000000_0100}"
SOURCES="${SOURCES:-base,hsi}"
TOP_K="${TOP_K:-30}"
DEDUPE_FRAME_PERSON="${DEDUPE_FRAME_PERSON:-true}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "========== Translation good/bad report =========="
echo "Repo       : ${REPO_ROOT}"
echo "Run scan   : ${RUN_SCAN}"
echo "Input CSV  : ${INPUT_CSV}"
echo "Output     : ${OUTPUT_DIR}"
echo "Thresholds : bad>${BAD_TRANSL_M}m severe>${SEVERE_TRANSL_M}m"
echo "Focus      : ${FOCUS_FRAMES}"

if [[ "${RUN_SCAN}" == "1" ]]; then
  OUTPUT_DIR="${SCAN_OUTPUT_DIR}" \
  BAD_BASE_TRANSL_M="${BAD_TRANSL_M}" \
  BAD_HSI_TRANSL_M="${BAD_TRANSL_M}" \
  SEVERE_BASE_TRANSL_M="${SEVERE_TRANSL_M}" \
  SEVERE_HSI_TRANSL_M="${SEVERE_TRANSL_M}" \
  CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
  bash scripts/eval/scan_hsi_bad_translation_frames_after_translation_ray_refine.sh
fi

[[ -f "${INPUT_CSV}" ]] || {
  echo "[ERROR] Missing input CSV: ${INPUT_CSV}" >&2
  echo "Run with RUN_SCAN=1 first, or set INPUT_CSV=/path/to/all_frame_person_translation_rows.csv" >&2
  exit 1
}

DEDUPE_ARGS=()
if [[ "${DEDUPE_FRAME_PERSON}" == "true" ]]; then
  DEDUPE_ARGS+=(--dedupe-frame-person)
else
  DEDUPE_ARGS+=(--no-dedupe-frame-person)
fi

python scripts/eval/report_translation_good_bad_frames.py \
  --input-csv "${INPUT_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --bad-transl-m "${BAD_TRANSL_M}" \
  --severe-transl-m "${SEVERE_TRANSL_M}" \
  --focus-frames "${FOCUS_FRAMES}" \
  --sources "${SOURCES}" \
  --top-k "${TOP_K}" \
  "${DEDUPE_ARGS[@]}"

echo "========== Translation good/bad report finished =========="
echo "Summary : ${OUTPUT_DIR}/translation_good_bad_summary.json"
echo "Frames  : ${OUTPUT_DIR}/translation_good_bad_frame_summary.csv"
echo "Focus   : ${OUTPUT_DIR}/focused_frame_person_translation_errors.csv"
echo "Markdown: ${OUTPUT_DIR}/translation_good_bad_report.md"
