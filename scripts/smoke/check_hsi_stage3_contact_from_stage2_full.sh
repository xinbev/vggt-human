#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
export REPO_ROOT
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/hsi_stage3_contact_from_stage2_smoke}"
export CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
export BATCH_SIZE="${BATCH_SIZE:-12}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export NLF_INTERNAL_BATCH_SIZE="${NLF_INTERNAL_BATCH_SIZE:-128}"
export NUM_VIEWS="${NUM_VIEWS:-2}"
export EPOCHS="${EPOCHS:-1}"
export LR="${LR:-2e-6}"
export MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-200}"
export MAX_VAL_STEPS="${MAX_VAL_STEPS:-20}"

echo "========== Smoke: HSI Stage3 contact-only from frozen Stage2 =========="
echo "Output       : ${OUTPUT_DIR}"
echo "GPU          : ${CUDA_VISIBLE_DEVICES_VALUE}"
echo "Batch/views  : ${BATCH_SIZE} / ${NUM_VIEWS}"
echo "Train steps  : ${MAX_STEPS_PER_EPOCH}"
echo "Val steps    : ${MAX_VAL_STEPS}"

bash "${REPO_ROOT}/scripts/train/train_smpl_hsi_stage3_contact_from_stage2_full.sh"

echo "========== Smoke finished =========="
echo "Required checks:"
echo "  trainable module is only hsi_contact_refine_head"
echo "  contact teacher counts are non-zero"
echo "  required contact gradient is finite and non-zero"
echo "  frozen Stage1/Stage2 hashes remain unchanged"
echo "  contact plane loss and contact metrics are non-zero"
