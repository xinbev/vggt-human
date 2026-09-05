#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-${REPO_ROOT}/outputs/eval/emdb2_global_chunk100/predictions}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/emdb2_global_chunk100/metrics}"
EXPORT_LOG="${EXPORT_LOG:-${REPO_ROOT}/outputs/eval/emdb2_global_chunk100/export.log}"
mkdir -p "$(dirname "${EXPORT_LOG}")"

REPO_ROOT="${REPO_ROOT}" \
PREDICTIONS_ROOT="${PREDICTIONS_ROOT}" \
CHUNK_SIZE="${CHUNK_SIZE:-100}" \
CHUNK_OVERLAP="${CHUNK_OVERLAP:-8}" \
MAX_INPUT_FRAMES="${MAX_INPUT_FRAMES:-100}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-}" \
TRSTR_DIR="${TRSTR_DIR:-${REPO_ROOT}/outputs/train/smpl_hsi_stage2_trstr_v3_refine}" \
TRSTR_CHECKPOINT="${TRSTR_CHECKPOINT:-}" \
SCALE_CHECKPOINT="${SCALE_CHECKPOINT:-${REPO_ROOT}/outputs/train/smpl_hsi_coarse_residual_stratified_v3/checkpoint_top_train_epoch_0005_loss_total_0.009242.pt}" \
MAX_SEQUENCES="${MAX_SEQUENCES:-0}" \
SEQUENCE_FILTER="${SEQUENCE_FILTER:-}"
if ! bash "${REPO_ROOT}/benchmarks/emdb2_global/export_chunk100.sh" >"${EXPORT_LOG}" 2>&1; then
  tail -n 120 "${EXPORT_LOG}" >&2
  exit 1
fi

REPO_ROOT="${REPO_ROOT}" \
PREDICTIONS_ROOT="${PREDICTIONS_ROOT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
SUBSAMPLE_STRIDE=1 \
CHUNK_LENGTH=100 \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-}" \
MAX_SEQUENCES="${MAX_SEQUENCES:-0}" \
SEQUENCE_FILTER="${SEQUENCE_FILTER:-}" \
METRICS_ONLY_OUTPUT=true \
bash "${REPO_ROOT}/benchmarks/emdb2_global/run.sh"
