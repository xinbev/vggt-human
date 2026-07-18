#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
SPLIT_ROOT="${SPLIT_ROOT:-${REPO_ROOT}/outputs/preprocess/hsi_sequence_split_v2}"
V4_CKPT="${V4_CKPT:?Set V4_CKPT to a completed V4-A1 checkpoint_top01.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/hsi_stage2_v4_a1_fixed64}"

cd "${REPO_ROOT}"
STAGE1_CKPT="${V4_CKPT}" \
RESUME_REQUIRED_PREFIXES="hsi_refinement_head.,hsi_translation_refine_v4_head." \
FROZEN_HASH_PREFIXES="hsi_refinement_head.,hsi_translation_refine_v4_head." \
OUTPUT_DIR="${OUTPUT_DIR}" \
SUBSET_INDICES_CSV="${SPLIT_ROOT}/overfit64_indices.csv" \
SUBSET_REPEAT=1 \
SUBSET_APPLY_TO_VAL=true \
VAL_SEQUENCE_MANIFEST="${SPLIT_ROOT}/train_sequences.txt" \
BATCH_SIZE="${BATCH_SIZE:-24}" \
EPOCHS=1 \
LR=0 \
MAX_STEPS_PER_EPOCH=1 \
MAX_VAL_STEPS=3 \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}" \
bash scripts/train/train_smpl_hsi_stage2_v4_a1_correction.sh

python scripts/smoke/check_hsi_stage2_v4_a1_metrics.py --output-dir "${OUTPUT_DIR}" --mode overfit
