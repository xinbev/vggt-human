#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
export REPO_ROOT

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/smoke_stage3_contact_refine}"
export CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
export BATCH_SIZE="${BATCH_SIZE:-12}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export NLF_INTERNAL_BATCH_SIZE="${NLF_INTERNAL_BATCH_SIZE:-128}"
export NUM_VIEWS="${NUM_VIEWS:-2}"
export EPOCHS="${EPOCHS:-1}"
export LR="${LR:-2e-6}"
export MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-200}"

echo "========== Smoke: HSI Stage3 contact refinement =========="
echo "Output      : ${OUTPUT_DIR}"
echo "GPU         : ${CUDA_VISIBLE_DEVICES_VALUE}"
echo "Batch/views : ${BATCH_SIZE} / ${NUM_VIEWS}"
echo "Max steps   : ${MAX_STEPS_PER_EPOCH}"

bash "${REPO_ROOT}/scripts/train/train_smpl_hsi_nlf_stage3_contact_refine.sh"

echo "========== Smoke finished =========="
echo "Check the progress line for:"
echo "  metric_hsi_support_plane_contact_count > 0"
echo "  metric_hsi_foot_sole_contact_count > 0"
echo "  metric_hsi_support_plane_penetration_m not increasing badly"
echo "  metric_hsi_transl_l1_delta and metric_hsi_joint_error_delta not obviously worse"
