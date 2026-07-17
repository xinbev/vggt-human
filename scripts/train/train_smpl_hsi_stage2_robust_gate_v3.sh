#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
STAGE2_TRAIN_CONFIG="${STAGE2_TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_stage2_transl_robust_gate_v3.yaml}"
STAGE1_CKPT="${STAGE1_CKPT:-${REPO_ROOT}/outputs/train/stage1_scale_linear_b20_gpu7/checkpoint_top_train_epoch_0003_loss_total_0.171740.pt}"
LEGACY_STAGE2_CKPT="${LEGACY_STAGE2_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_top_train_epoch_0003_loss_total_4.120546.pt}"
BACKUP_DIR="${BACKUP_DIR:-${REPO_ROOT}/outputs/checkpoint_backups/hsi_stage2_legacy_20260718}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/train/hsi_stage2_transl_robust_gate_v3}"
RUN_STAGES="${RUN_STAGES:-2A,2B}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"

SOURCE_CKPT="${LEGACY_STAGE2_CKPT}" \
BACKUP_DIR="${BACKUP_DIR}" \
BACKUP_NAME=stage2_human_scene_align_full_top03.pt \
bash "${REPO_ROOT}/scripts/tools/backup_hsi_stage2_legacy_checkpoint.sh"

STAGE2_TRAIN_CONFIG="${STAGE2_TRAIN_CONFIG}" \
STAGE1_CKPT="${STAGE1_CKPT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
RUN_STAGES="${RUN_STAGES}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
NUM_WORKERS="${NUM_WORKERS:-16}" \
NLF_INTERNAL_BATCH_SIZE="${NLF_INTERNAL_BATCH_SIZE:-192}" \
MAX_HUMANS="${MAX_HUMANS:-20}" \
BATCH_SIZE_2A="${BATCH_SIZE_2A:-24}" \
BATCH_SIZE_2B="${BATCH_SIZE_2B:-24}" \
EPOCHS_2A="${EPOCHS_2A:-3}" \
EPOCHS_2B="${EPOCHS_2B:-3}" \
LR_2A="${LR_2A:-2e-5}" \
LR_2B="${LR_2B:-5e-6}" \
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-0}" \
bash "${REPO_ROOT}/scripts/train/train_smpl_hsi_scale_trans_contact_curriculum.sh"
