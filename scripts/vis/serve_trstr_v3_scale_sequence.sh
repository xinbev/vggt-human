#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
FRAMES_DIR="${FRAMES_DIR:-/home/zhw/lab_users/xyb/home/projects/Human3R-master/outputs/walking/color}"
TRSTR_DIR="${TRSTR_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_stage2_trstr_v3_scale_spatial}"
CHECKPOINT="${CHECKPOINT:-${TRSTR_DIR}/checkpoint_latest.pt}"
SCALE_CHECKPOINT="${SCALE_CHECKPOINT:-${REPO_ROOT}/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/trstr_v3_scale_spatial_sequence}"

[[ -d "${FRAMES_DIR}" ]] || { echo "[ERROR] Missing frames: ${FRAMES_DIR}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "[ERROR] Missing TRSTR checkpoint: ${CHECKPOINT}" >&2; exit 1; }
[[ -f "${SCALE_CHECKPOINT}" ]] || { echo "[ERROR] Missing HSI scale v3 checkpoint: ${SCALE_CHECKPOINT}" >&2; exit 1; }

echo "========== TRSTR spatial inference: analytic coarse -> HSI scale v3 -> TRSTR =========="
echo "Frames          : ${FRAMES_DIR}"
echo "TRSTR checkpoint: ${CHECKPOINT}"
echo "Scale v3        : ${SCALE_CHECKPOINT}"
echo "Output          : ${OUTPUT_DIR}"

REPO_ROOT="${REPO_ROOT}" \
FRAMES_DIR="${FRAMES_DIR}" \
QUERY_SOURCE=nlf_detector \
TRAIN_CONFIG="${REPO_ROOT}/configs/train_smpl_hsi_stage2_trstr.yaml" \
STAGE2_DIR="${TRSTR_DIR}" \
CHECKPOINT="${CHECKPOINT}" \
HSI_OVERLAY_CHECKPOINT="${SCALE_CHECKPOINT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}" \
PORT="${PORT:-8080}" \
MAX_FRAMES="${MAX_FRAMES:-20}" \
MAX_HUMANS="${MAX_HUMANS:-8}" \
CONF_THRESHOLD="${CONF_THRESHOLD:-0.05}" \
SMPL_USE_AGGREGATOR_QUERIES=false \
HSI_SCENE_AFFINE_MODE=per_frame \
SCENE_SCALE_PREALIGN=smpl_median \
COARSE_SCALE_MIN="${COARSE_SCALE_MIN:-0.10}" \
COARSE_SCALE_MAX="${COARSE_SCALE_MAX:-10.0}" \
COARSE_ANCHOR_STRIDE="${COARSE_ANCHOR_STRIDE:-8}" \
COARSE_MIN_ANCHOR_PIXELS="${COARSE_MIN_ANCHOR_PIXELS:-32}" \
COARSE_FALLBACK=sequence_median \
TRACKING_OVERLAY=none \
SHOW_TRACK_IDS=true \
SMOKE_ONLY="${SMOKE_ONLY:-false}" \
bash "${REPO_ROOT}/scripts/vis/serve_nlf_hsi_vggt_sequence_viewer.sh"
