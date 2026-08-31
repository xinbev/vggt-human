#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${CHECKPOINT:?Set CHECKPOINT to the VGGT checkpoint used for runtime camera.}"

DATASET="${DATASET:-3dpw}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/eval_nlf_pose_stabilizer_v2.yaml}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/outputs/preprocess/nlf_vggt_temporal_cache}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"
DEVICE="${DEVICE:-cuda:0}"
FRAMES_ROOT="${FRAMES_ROOT:-}"
SEQUENCE_FILTER="${SEQUENCE_FILTER:-}"
NUM_WORKERS="${NUM_WORKERS:-2}"
OVERWRITE="${OVERWRITE:-false}"

[[ -f "${CONFIG}" ]] || { echo "[ERROR] Missing config: ${CONFIG}" >&2; exit 1; }
[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${CHECKPOINT}" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ARGS=(
  --dataset "${DATASET}"
  --checkpoint "${CHECKPOINT}"
  --config "${CONFIG}"
  --path-config "${PATH_CONFIG}"
  --cache-root "${CACHE_ROOT}"
  --device "${DEVICE}"
  --num-workers "${NUM_WORKERS}"
)
if [[ -n "${FRAMES_ROOT}" ]]; then ARGS+=(--frames-root "${FRAMES_ROOT}"); fi
if [[ -n "${SEQUENCE_FILTER}" ]]; then ARGS+=(--sequence-filter "${SEQUENCE_FILTER}"); fi
if [[ "${OVERWRITE}" == "true" ]]; then ARGS+=(--overwrite); fi

python scripts/preprocess/cache_nlf_vggt_temporal_inputs.py "${ARGS[@]}"
