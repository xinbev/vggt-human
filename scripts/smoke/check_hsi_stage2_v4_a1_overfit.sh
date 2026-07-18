#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
SPLIT_ROOT="${SPLIT_ROOT:-${REPO_ROOT}/outputs/preprocess/hsi_sequence_split_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/hsi_stage2_v4_a1_overfit64}"

cd "${REPO_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR}" \
SUBSET_INDICES_CSV="${SPLIT_ROOT}/overfit64_indices.csv" \
SUBSET_REPEAT="${SUBSET_REPEAT:-400}" \
SUBSET_APPLY_TO_VAL=true \
VAL_SEQUENCE_MANIFEST="${SPLIT_ROOT}/train_sequences.txt" \
BATCH_SIZE="${BATCH_SIZE:-24}" \
EPOCHS=1 \
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-1000}" \
MAX_VAL_STEPS="${MAX_VAL_STEPS:-3}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}" \
bash scripts/train/train_smpl_hsi_stage2_v4_a1_correction.sh

python scripts/smoke/check_hsi_stage2_v4_a1_metrics.py --output-dir "${OUTPUT_DIR}" --mode overfit
