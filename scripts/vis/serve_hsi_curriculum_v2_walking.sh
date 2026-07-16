#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
STAGE_DIR="${STAGE_DIR:-${REPO_ROOT}/outputs/train/hsi_scale_trans_contact_v2/stage3b_real_contact}"
CHECKPOINT="${CHECKPOINT:-${STAGE_DIR}/checkpoint_top01.pt}"
FRAMES_DIR="${FRAMES_DIR:-/home/zhw/lab_users/xyb/home/projects/Human3R-master/outputs/walking/color}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/hsi_curriculum_v2_walking}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
PORT="${PORT:-8080}"
MAX_FRAMES="${MAX_FRAMES:-0}"
DEPTH_POINT_STRIDE="${DEPTH_POINT_STRIDE:-6}"
MAX_SCENE_DEPTH="${MAX_SCENE_DEPTH:-30.0}"
POINT_SIZE="${POINT_SIZE:-0.012}"

[[ -f "${CHECKPOINT}" ]] || { echo "[ERROR] Missing V2 checkpoint: ${CHECKPOINT}" >&2; exit 1; }

REPO_ROOT="${REPO_ROOT}" \
STAGE2_DIR="${STAGE_DIR}" \
CHECKPOINT="${CHECKPOINT}" \
TRAIN_CONFIG="${REPO_ROOT}/configs/train_smpl_hsi_stage3_contact_curriculum_v2.yaml" \
FRAMES_DIR="${FRAMES_DIR}" \
QUERY_SOURCE=nlf_detector \
OUTPUT_DIR="${OUTPUT_DIR}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
PORT="${PORT}" \
MAX_FRAMES="${MAX_FRAMES}" \
DEPTH_POINT_STRIDE="${DEPTH_POINT_STRIDE}" \
MAX_SCENE_DEPTH="${MAX_SCENE_DEPTH}" \
POINT_SIZE="${POINT_SIZE}" \
bash "${REPO_ROOT}/scripts/vis/serve_nlf_hsi_vggt_sequence_viewer.sh"
