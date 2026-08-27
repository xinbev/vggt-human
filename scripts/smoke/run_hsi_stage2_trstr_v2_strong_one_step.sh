#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"

REPO_ROOT="${REPO_ROOT}" \
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/hsi_stage2_trstr_v2_strong_one_step}" \
BATCH_SIZE="${BATCH_SIZE:-1}" \
NUM_WORKERS="${NUM_WORKERS:-2}" \
ALIGNMENT_CASE_PROBABILITIES="[0.0,0.0,0.0,1.0]" \
ALIGNMENT_SCALE_LOG_STD="${ALIGNMENT_SCALE_LOG_STD:-0.12}" \
EPOCHS=1 \
MAX_STEPS_PER_EPOCH=1 \
WANDB_ENABLED=false \
bash "${REPO_ROOT}/scripts/train/train_smpl_hsi_stage2_trstr_v2_strong.sh"
