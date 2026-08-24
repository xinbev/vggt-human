#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"

REPO_ROOT="${REPO_ROOT}" \
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/hsi_gt_depth_scale_boxfree_one_step}" \
BATCH_SIZE="${BATCH_SIZE:-2}" \
NUM_WORKERS="${NUM_WORKERS:-0}" \
EPOCHS=1 \
MAX_STEPS_PER_EPOCH=1 \
WANDB_ENABLED=false \
SAVE_TOP_K=0 \
SAVE_LATEST=false \
bash "${REPO_ROOT}/scripts/train/train_smpl_hsi_gt_depth_scale_scene_affine.sh"
