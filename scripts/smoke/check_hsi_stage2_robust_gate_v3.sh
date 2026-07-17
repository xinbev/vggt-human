#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
GATE_MODE="${GATE_MODE:-smoke}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
STAGE2_TRAIN_CONFIG="${STAGE2_TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_stage2_transl_robust_gate_v3.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/debug/hsi_stage2_robust_gate_v3_${GATE_MODE}}"
LEGACY_STAGE2_CKPT="${LEGACY_STAGE2_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full/checkpoint_top_train_epoch_0003_loss_total_4.120546.pt}"
BACKUP_DIR="${BACKUP_DIR:-${REPO_ROOT}/outputs/checkpoint_backups/hsi_stage2_legacy_20260718}"

SOURCE_CKPT="${LEGACY_STAGE2_CKPT}" \
BACKUP_DIR="${BACKUP_DIR}" \
BACKUP_NAME=stage2_human_scene_align_full_top03.pt \
bash "${REPO_ROOT}/scripts/tools/backup_hsi_stage2_legacy_checkpoint.sh"

STAGE2_TRAIN_CONFIG="${STAGE2_TRAIN_CONFIG}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
GATE_STAGE=2A \
GATE_MODE="${GATE_MODE}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
bash "${REPO_ROOT}/scripts/smoke/check_hsi_curriculum_v2.sh"

