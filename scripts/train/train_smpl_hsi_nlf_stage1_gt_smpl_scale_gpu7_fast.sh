#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"

# One-command launcher for the current Stage1 scale-teacher experiment on the
# 80G GPU7 machine. Set FAST_DEBUG=true to run only a short sanity check.
FAST_DEBUG="${FAST_DEBUG:-false}"

export REPO_ROOT
export CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
export BATCH_SIZE="${BATCH_SIZE:-20}"
export NUM_WORKERS="${NUM_WORKERS:-16}"
export NLF_INTERNAL_BATCH_SIZE="${NLF_INTERNAL_BATCH_SIZE:-192}"
export NUM_VIEWS="${NUM_VIEWS:-2}"
export LR="${LR:-5e-6}"
export DEPTH_MAX_M="${DEPTH_MAX_M:-20.0}"
export HSI_SMPL_SCALE_TEACHER_MAX_Z_M="${HSI_SMPL_SCALE_TEACHER_MAX_Z_M:-20.0}"
export HSI_SMPL_SCALE_TEACHER_LOG_LOSS="${HSI_SMPL_SCALE_TEACHER_LOG_LOSS:-false}"
export HSI_SCENE_LOG_SCALE_MAX="${HSI_SCENE_LOG_SCALE_MAX:-5.0}"

export RESUME_CKPT="${RESUME_CKPT-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_gt_smpl_scale_stage1_b12/checkpoint_top_train_epoch_0002_loss_total_0.021235.pt}"
export RESET_EPOCH="${RESET_EPOCH:-true}"

if [[ "${FAST_DEBUG}" == "true" ]]; then
  export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/stage1_scale_linear_b20_gpu7_debug}"
  export EPOCHS="${EPOCHS:-1}"
  export MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-300}"
else
  export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/stage1_scale_linear_b20_gpu7}"
  export EPOCHS="${EPOCHS:-3}"
  export MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-0}"
fi

echo "========== Stage1 GT-SMPL scale teacher GPU7 fast launcher =========="
echo "Repo        : ${REPO_ROOT}"
echo "GPU visible : ${CUDA_VISIBLE_DEVICES_VALUE}"
echo "Batch       : ${BATCH_SIZE}"
echo "Workers     : ${NUM_WORKERS}"
echo "NLF batch   : ${NLF_INTERNAL_BATCH_SIZE}"
echo "Views       : ${NUM_VIEWS}"
echo "Epochs      : ${EPOCHS}"
echo "Max steps   : ${MAX_STEPS_PER_EPOCH}"
echo "Resume      : ${RESUME_CKPT}"
echo "Output      : ${OUTPUT_DIR}"

bash "${REPO_ROOT}/scripts/train/train_smpl_hsi_nlf_stage1_gt_smpl_scale.sh"
