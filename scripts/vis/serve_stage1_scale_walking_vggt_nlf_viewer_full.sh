#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
export REPO_ROOT

# Full-sequence launcher: all RGB frames under FRAMES_DIR are processed with
# stride 1 in one VGGT forward pass, preserving one shared predicted camera
# frame for the whole sequence. If the sequence is too long for memory, rerun
# with MAX_FRAMES or FRAME_STRIDE overrides.
export FRAMES_DIR="${FRAMES_DIR:-/home/zhw/lab_users/xyb/home/projects/Human3R-master/outputs/walking/color}"
export STAGE1_DIR="${STAGE1_DIR:-${REPO_ROOT}/outputs/train/stage1_scale_linear_b20_gpu7}"
export CHECKPOINT="${CHECKPOINT:-${STAGE1_DIR}/checkpoint_latest.pt}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/stage1_scale_walking_vggt_nlf_viewer_full}"
export CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"

export QUERY_SOURCE="${QUERY_SOURCE:-nlf_detector}"
export MAX_FRAMES="${MAX_FRAMES:-0}"
export START_INDEX="${START_INDEX:-0}"
export FRAME_STRIDE="${FRAME_STRIDE:-1}"
export MAX_HUMANS="${MAX_HUMANS:-8}"
export CONF_THRESHOLD="${CONF_THRESHOLD:-0.05}"
export MAX_SCENE_DEPTH="${MAX_SCENE_DEPTH:-30.0}"

# Full sequences can create very large point clouds. These defaults keep the
# viewer responsive while retaining enough geometry for alignment inspection.
export DEPTH_POINT_STRIDE="${DEPTH_POINT_STRIDE:-6}"
export POINT_SIZE="${POINT_SIZE:-0.014}"
export ALIGNMENT_VERTEX_STRIDE="${ALIGNMENT_VERTEX_STRIDE:-16}"
export PORT="${PORT:-8080}"
export SMOKE_ONLY="${SMOKE_ONLY:-false}"

echo "========== Full walking sequence VGGT+NLF+HSI viewer =========="
echo "Frames      : ${FRAMES_DIR}"
echo "Checkpoint  : ${CHECKPOINT}"
echo "Output      : ${OUTPUT_DIR}"
echo "GPU visible : ${CUDA_VISIBLE_DEVICES_VALUE}"
echo "Max frames  : ${MAX_FRAMES} (0 means all)"
echo "Frame stride: ${FRAME_STRIDE}"
echo "Depth stride: ${DEPTH_POINT_STRIDE}"
echo "Port        : ${PORT}"
echo "Smoke only  : ${SMOKE_ONLY}"

bash "${REPO_ROOT}/scripts/vis/serve_stage1_scale_walking_vggt_nlf_viewer.sh"
