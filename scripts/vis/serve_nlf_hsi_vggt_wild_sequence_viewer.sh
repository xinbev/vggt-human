#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
FRAMES_DIR="${FRAMES_DIR:-/home/zhw/lab_users/xyb/home/projects/Human3R-master/outputs/walking/color}"
STAGE2_DIR="${STAGE2_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_nlf_full_b12_20260710/stage2_anchor_transl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/nlf_hsi_vggt_wild_sequence_viewer/walking}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-6}"

QUERY_SOURCE="nlf_detector"
MAX_FRAMES="${MAX_FRAMES:-32}"
START_INDEX="${START_INDEX:-0}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_HUMANS="${MAX_HUMANS:-8}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.05}"
DEPTH_POINT_STRIDE="${DEPTH_POINT_STRIDE:-4}"
MAX_SCENE_DEPTH="${MAX_SCENE_DEPTH:-30.0}"
POINT_SIZE="${POINT_SIZE:-0.012}"
PORT="${PORT:-8080}"
SMOKE_ONLY="${SMOKE_ONLY:-false}"

cd "${REPO_ROOT}"

echo "========== Wild NLF-HSI VGGT sequence viewer =========="
echo "Repo        : ${REPO_ROOT}"
echo "Frames      : ${FRAMES_DIR}"
echo "Query source: ${QUERY_SOURCE}"
echo "Stage2 dir  : ${STAGE2_DIR}"
echo "Output      : ${OUTPUT_DIR}"
echo "Max frames  : ${MAX_FRAMES}"
echo "GPU visible : ${CUDA_VISIBLE_DEVICES_VALUE}"
echo "Smoke only  : ${SMOKE_ONLY}"

QUERY_SOURCE="${QUERY_SOURCE}" \
FRAMES_DIR="${FRAMES_DIR}" \
STAGE2_DIR="${STAGE2_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
MAX_FRAMES="${MAX_FRAMES}" \
START_INDEX="${START_INDEX}" \
FRAME_STRIDE="${FRAME_STRIDE}" \
MAX_HUMANS="${MAX_HUMANS}" \
CONF_THRESHOLD="${CONF_THRESHOLD}" \
DEPTH_POINT_STRIDE="${DEPTH_POINT_STRIDE}" \
MAX_SCENE_DEPTH="${MAX_SCENE_DEPTH}" \
POINT_SIZE="${POINT_SIZE}" \
PORT="${PORT}" \
SMOKE_ONLY="${SMOKE_ONLY}" \
bash scripts/vis/serve_nlf_hsi_vggt_sequence_viewer.sh
