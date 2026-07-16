#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/preprocess/hsi_sequence_split_v2}"
SPLIT="${SPLIT:-Training}"
VAL_RATIO="${VAL_RATIO:-0.10}"
SEED="${SEED:-42}"

cd "${REPO_ROOT}"
python scripts/preprocess/prepare_hsi_sequence_manifests.py \
  --bedlam-root "${BEDLAM_ROOT}" \
  --split "${SPLIT}" \
  --output-dir "${OUTPUT_DIR}" \
  --val-ratio "${VAL_RATIO}" \
  --seed "${SEED}"
