#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/hsi_stage2_v4_a1_interface_smoke}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"

cd "${REPO_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
NUM_WORKERS="${NUM_WORKERS:-4}" \
EPOCHS=1 \
MAX_STEPS_PER_EPOCH=2 \
MAX_VAL_STEPS=2 \
bash scripts/train/train_smpl_hsi_stage2_v4_a1_correction.sh

python scripts/smoke/check_hsi_stage2_v4_a1_metrics.py --output-dir "${OUTPUT_DIR}" --mode smoke
