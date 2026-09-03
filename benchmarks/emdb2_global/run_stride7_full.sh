#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-${REPO_ROOT}/outputs/eval/emdb2_global_stride7/predictions}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/emdb2_global_stride7/metrics}"

REPO_ROOT="${REPO_ROOT}" \
PREDICTIONS_ROOT="${PREDICTIONS_ROOT}" \
SUBSAMPLE_STRIDE=7 \
MAX_INPUT_FRAMES=500 \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-}" \
TRSTR_DIR="${TRSTR_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_stage2_trstr_v3_refine}" \
TRSTR_CHECKPOINT="${TRSTR_CHECKPOINT:-}" \
SCALE_CHECKPOINT="${SCALE_CHECKPOINT:-${REPO_ROOT}/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt}" \
MAX_SEQUENCES="${MAX_SEQUENCES:-0}" \
SEQUENCE_FILTER="${SEQUENCE_FILTER:-}" \
bash "${REPO_ROOT}/benchmarks/emdb2_global/export_stride7.sh"

REPO_ROOT="${REPO_ROOT}" \
PREDICTIONS_ROOT="${PREDICTIONS_ROOT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
SUBSAMPLE_STRIDE=7 \
CHUNK_LENGTH=14 \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-}" \
MAX_SEQUENCES="${MAX_SEQUENCES:-0}" \
SEQUENCE_FILTER="${SEQUENCE_FILTER:-}" \
bash "${REPO_ROOT}/benchmarks/emdb2_global/run.sh"

echo "Summary: ${OUTPUT_DIR}/summary.json"
