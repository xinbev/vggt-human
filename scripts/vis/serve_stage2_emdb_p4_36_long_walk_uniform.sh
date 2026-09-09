#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
FRAMES_DIR="${FRAMES_DIR:-/home/zhw/xyb_space/emdb/P4/36_outdoor_long_walk/images}"
STAGE2_DIR="${STAGE2_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full}"
CHECKPOINT="${CHECKPOINT:-${STAGE2_DIR}/checkpoint_latest.pt}"
SCALE_CHECKPOINT="${SCALE_CHECKPOINT:-${REPO_ROOT}/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/stage2_emdb_p4_36_long_walk_coarse_residual_v3}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
PORT="${PORT:-8080}"
MAX_FRAMES="${MAX_FRAMES:-500}"
FRAME_SAMPLING="${FRAME_SAMPLING:-uniform}"
SMPL_DISPLAY_FRAMES="${SMPL_DISPLAY_FRAMES:-50}"
VIEWER_MODE="${VIEWER_MODE:-3d}"
SMOKE_ONLY="${SMOKE_ONLY:-false}"

REPO_ROOT="${REPO_ROOT}" \
FRAMES_DIR="${FRAMES_DIR}" \
STAGE2_DIR="${STAGE2_DIR}" \
CHECKPOINT="${CHECKPOINT}" \
SCALE_CHECKPOINT="${SCALE_CHECKPOINT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
PORT="${PORT}" \
MAX_FRAMES="${MAX_FRAMES}" \
FRAME_STRIDE=1 \
FRAME_SAMPLING="${FRAME_SAMPLING}" \
SMPL_DISPLAY_FRAMES="${SMPL_DISPLAY_FRAMES}" \
VIEWER_MODE="${VIEWER_MODE}" \
SMOKE_ONLY="${SMOKE_ONLY}" \
bash "${REPO_ROOT}/scripts/vis/serve_stage2_walking_coarse_scale_hsi_cascade.sh"
