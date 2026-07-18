#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
STAGE_OUTPUT_DIR="${STAGE_OUTPUT_DIR:?Set STAGE_OUTPUT_DIR to the completed V4-A1 output directory}"
EPOCH="${EPOCH:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/hsi_stage2_v4_a1_failure_analysis}"

cd "${REPO_ROOT}"
RECORDS_JSONL="${STAGE_OUTPUT_DIR}/v4_val_people_epoch_$(printf '%04d' "${EPOCH}").jsonl"
[[ -f "${RECORDS_JSONL}" ]] || { echo "[ERROR] Missing V4 per-person records: ${RECORDS_JSONL}" >&2; exit 1; }

python scripts/eval/analyze_hsi_stage2_v4_a1_failures.py \
  --records-jsonl "${RECORDS_JSONL}" \
  --output-dir "${OUTPUT_DIR}"
