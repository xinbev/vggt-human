#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
FRAMES_DIR="${FRAMES_DIR:-/home/zhw/lab_users/xyb/home/projects/Human3R-master/outputs/walking/color}"
STAGE2_DIR="${STAGE2_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_stage2_human_scene_align_full}"
CHECKPOINT="${CHECKPOINT:-${STAGE2_DIR}/checkpoint_latest.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/stage2_human_scene_align_walking_viewer_id_overlay}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
PORT="${PORT:-8080}"
MAX_FRAMES="${MAX_FRAMES:-64}"

[[ -d "${FRAMES_DIR}" ]] || { echo "[ERROR] Missing walking frames: ${FRAMES_DIR}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "[ERROR] Missing Stage2 checkpoint: ${CHECKPOINT}" >&2; exit 1; }

echo "========== Early Stage2 Viser + post-HSI ID overlay =========="
echo "Geometry checkpoint: ${CHECKPOINT}"
echo "Align compatibility : legacy_scale_bias_v0 (25 input features)"
echo "Tracking effect    : display IDs/colors only"
echo "Output             : ${OUTPUT_DIR}"

REPO_ROOT="${REPO_ROOT}" \
FRAMES_DIR="${FRAMES_DIR}" \
QUERY_SOURCE=nlf_detector \
TRAIN_CONFIG="${REPO_ROOT}/configs/train_smpl_hsi_nlf_stage2_human_scene_align.yaml" \
STAGE2_DIR="${STAGE2_DIR}" \
CHECKPOINT="${CHECKPOINT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
PORT="${PORT}" \
MAX_FRAMES="${MAX_FRAMES}" \
FRAME_STRIDE=1 \
MAX_HUMANS=8 \
CONF_THRESHOLD=0.05 \
DEPTH_POINT_STRIDE=2 \
MAX_SCENE_DEPTH=80 \
POINT_SIZE=0.006 \
HSI_ALIGN_FEATURE_VERSION=legacy_scale_bias_v0 \
TRACKING_OVERLAY=base_smpl \
TRACK_MAX_AGE=90 \
TRACK_MIN_QUALITY=0.25 \
TRACK_MAX_CENTER_DISTANCE=0.25 \
TRACK_MAX_TRANSL_DISTANCE=1.50 \
TRACK_MAX_BETA_L1=0.30 \
bash "${REPO_ROOT}/scripts/vis/serve_nlf_hsi_vggt_sequence_viewer.sh"
