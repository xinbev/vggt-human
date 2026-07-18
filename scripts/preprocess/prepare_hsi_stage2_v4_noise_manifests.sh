#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
SUBSET_INDICES_CSV="${SUBSET_INDICES_CSV:-${REPO_ROOT}/outputs/preprocess/hsi_sequence_split_v2/overfit64_indices.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/preprocess/hsi_stage2_v4_noise}"
MAX_HUMANS="${MAX_HUMANS:-20}"
SEED="${SEED:-42}"

cd "${REPO_ROOT}"
python scripts/preprocess/prepare_hsi_stage2_v4_noise_manifest.py \
  --subset-indices-csv "${SUBSET_INDICES_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-humans "${MAX_HUMANS}" \
  --seed "${SEED}" \
  --epoch 0
