#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
FRAMES_DIR="${FRAMES_DIR:-/home/zhw/lab_users/xyb/home/projects/Human3R-master/outputs/walking/color}"
STAGE1_DIR="${STAGE1_DIR:-${REPO_ROOT}/outputs/train/stage1_scale_linear_b20_gpu7}"
CHECKPOINT="${CHECKPOINT:-${STAGE1_DIR}/checkpoint_latest.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/stage1_scale_walking_vggt_nlf_viewer}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"

QUERY_SOURCE="${QUERY_SOURCE:-nlf_detector}"
MAX_FRAMES="${MAX_FRAMES:-48}"
START_INDEX="${START_INDEX:-0}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_HUMANS="${MAX_HUMANS:-8}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.05}"
DEPTH_POINT_STRIDE="${DEPTH_POINT_STRIDE:-4}"
MAX_SCENE_DEPTH="${MAX_SCENE_DEPTH:-30.0}"
POINT_SIZE="${POINT_SIZE:-0.012}"
ALIGNMENT_VERTEX_STRIDE="${ALIGNMENT_VERTEX_STRIDE:-16}"
PORT="${PORT:-8080}"
SMOKE_ONLY="${SMOKE_ONLY:-false}"

cd "${REPO_ROOT}"

[[ -d "${FRAMES_DIR}" ]] || { echo "[ERROR] Missing walking frames dir: ${FRAMES_DIR}" >&2; exit 1; }
[[ -d "${STAGE1_DIR}" ]] || { echo "[ERROR] Missing Stage1 dir: ${STAGE1_DIR}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "[ERROR] Missing Stage1 checkpoint: ${CHECKPOINT}" >&2; exit 1; }

echo "========== Stage1 scale walking VGGT+NLF Viser viewer =========="
echo "Repo        : ${REPO_ROOT}"
echo "Frames      : ${FRAMES_DIR}"
echo "Query source: ${QUERY_SOURCE}"
echo "Stage1 dir  : ${STAGE1_DIR}"
echo "Checkpoint  : ${CHECKPOINT}"
echo "Output      : ${OUTPUT_DIR}"
echo "GPU visible : ${CUDA_VISIBLE_DEVICES_VALUE}"
echo "Max frames  : ${MAX_FRAMES}"
echo "Port        : ${PORT}"
echo "Smoke only  : ${SMOKE_ONLY}"

FRAMES_DIR="${FRAMES_DIR}" \
QUERY_SOURCE="${QUERY_SOURCE}" \
STAGE2_DIR="${STAGE1_DIR}" \
CHECKPOINT="${CHECKPOINT}" \
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
ALIGNMENT_VERTEX_STRIDE="${ALIGNMENT_VERTEX_STRIDE}" \
PORT="${PORT}" \
SMOKE_ONLY="${SMOKE_ONLY}" \
bash scripts/vis/serve_nlf_hsi_vggt_sequence_viewer.sh
