#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
FRAMES_DIR="${FRAMES_DIR:-/home/zhw/lab_users/xyb/home/projects/Human3R-master/outputs/walking/color}"
STAGE2_DIR="${STAGE2_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full}"
CHECKPOINT="${CHECKPOINT:-${STAGE2_DIR}/checkpoint_latest.pt}"
SCALE_CHECKPOINT="${SCALE_CHECKPOINT:-${REPO_ROOT}/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/stage2_walking_coarse_residual_v3}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
PORT="${PORT:-8080}"
MAX_FRAMES="${MAX_FRAMES:-20}"

[[ -d "${FRAMES_DIR}" ]] || { echo "[ERROR] Missing walking frames: ${FRAMES_DIR}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "[ERROR] Missing Stage2 checkpoint: ${CHECKPOINT}" >&2; exit 1; }
[[ -f "${SCALE_CHECKPOINT}" ]] || { echo "[ERROR] Missing current scale checkpoint: ${SCALE_CHECKPOINT}" >&2; exit 1; }

echo "========== Walking Viser: analytic coarse -> current HSI -> Stage2 align =========="
echo "Frames            : ${FRAMES_DIR}"
echo "Stage2 checkpoint : ${CHECKPOINT}"
echo "Scale overlay     : ${SCALE_CHECKPOINT}"
echo "Output            : ${OUTPUT_DIR}"
echo "Port              : ${PORT}"

REPO_ROOT="${REPO_ROOT}" \
FRAMES_DIR="${FRAMES_DIR}" \
QUERY_SOURCE=nlf_detector \
TRAIN_CONFIG="${REPO_ROOT}/configs/train_smpl_hsi_nlf_stage2_human_scene_align.yaml" \
STAGE2_DIR="${STAGE2_DIR}" \
CHECKPOINT="${CHECKPOINT}" \
HSI_OVERLAY_CHECKPOINT="${SCALE_CHECKPOINT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
PORT="${PORT}" \
MAX_FRAMES="${MAX_FRAMES}" \
FRAME_STRIDE=1 \
MAX_HUMANS=8 \
CONF_THRESHOLD=0.05 \
SMPL_USE_AGGREGATOR_QUERIES=false \
HSI_SCENE_AFFINE_MODE=per_frame \
SCENE_SCALE_PREALIGN=smpl_median \
COARSE_SCALE_MIN="${COARSE_SCALE_MIN:-0.10}" \
COARSE_SCALE_MAX="${COARSE_SCALE_MAX:-10.0}" \
COARSE_ANCHOR_STRIDE="${COARSE_ANCHOR_STRIDE:-8}" \
COARSE_MIN_ANCHOR_PIXELS="${COARSE_MIN_ANCHOR_PIXELS:-32}" \
COARSE_FALLBACK=sequence_median \
DEPTH_POINT_STRIDE=2 \
MAX_SCENE_DEPTH=80 \
VIEWER_MODE="${VIEWER_MODE:-hybrid}" \
ENVIRONMENT_DISPLAY="${ENVIRONMENT_DISPLAY:-points}" \
HSI_VISUAL_SCALE=1.0 \
HUMAN_MASK_DILATION_PX="${HUMAN_MASK_DILATION_PX:-12}" \
FILTER_HUMAN_POINTS="${FILTER_HUMAN_POINTS:-true}" \
POINT_SIZE=0.006 \
SMPL_EDIT_OUTPUT="${SMPL_EDIT_OUTPUT:-${OUTPUT_DIR}/smpl_edit_offsets.json}" \
HSI_ALIGN_FEATURE_VERSION=legacy_scale_bias_v0 \
TRACKING_OVERLAY=base_smpl \
SHOW_TRACK_IDS="${SHOW_TRACK_IDS:-true}" \
TRACK_MAX_AGE=90 \
TRACK_MIN_QUALITY=0.25 \
TRACK_MAX_CENTER_DISTANCE=0.25 \
TRACK_MAX_TRANSL_DISTANCE=1.50 \
TRACK_MAX_BETA_L1=0.30 \
SMOKE_ONLY="${SMOKE_ONLY:-false}" \
bash "${REPO_ROOT}/scripts/vis/serve_nlf_hsi_vggt_sequence_viewer.sh"
