#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
SOURCE_DIR="${SOURCE_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_gt_depth_scale_scene_affine_boxfree}"
INIT_CKPT="${INIT_CKPT:-${SOURCE_DIR}/checkpoint_top_train_epoch_0005_loss_total_0.010738.pt}"

[[ -f "${INIT_CKPT}" ]] || { echo "[ERROR] Missing completed scale checkpoint: ${INIT_CKPT}" >&2; exit 1; }

REPO_ROOT="${REPO_ROOT}" \
INIT_CKPT="${INIT_CKPT}" \
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_gt_depth_scale_scene_affine_log10_wide}" \
WANDB_RUN_NAME="${WANDB_RUN_NAME:-smpl_hsi_gt_depth_scale_scene_affine_log10_wide}" \
WANDB_GROUP="${WANDB_GROUP:-hsi_scene_scale_log10_wide}" \
NOISE_MODE="${NOISE_MODE:-log10_normal}" \
LOG10_SCALE_STD_SCHEDULE="${LOG10_SCALE_STD_SCHEDULE:-0.30}" \
LR="${LR:-5e-6}" \
EPOCHS="${EPOCHS:-5}" \
bash "${REPO_ROOT}/scripts/train/train_smpl_hsi_gt_depth_scale_scene_affine.sh"
