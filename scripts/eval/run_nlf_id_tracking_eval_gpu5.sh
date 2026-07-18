#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/home/zhw/lab_users/xyb/home/projects/vggt-human"
PATH_CONFIG="${REPO_ROOT}/configs/path.yaml"
TRAIN_CONFIG="${REPO_ROOT}/configs/train_nlf_id_tracking.yaml"
CHECKPOINT="${1:?Usage: bash scripts/eval/run_nlf_id_tracking_eval_gpu5.sh CHECKPOINT [SPLIT]}"
SPLIT="${2:-Test}"
TRACK_ID_WEIGHT="${3:-0.35}"
MAX_ID_DISTANCE="${4:-0.70}"
OUTPUT_DIR="${REPO_ROOT}/outputs/eval/nlf_id_tracking_gpu5"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=5
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python -u scripts/eval/eval_nlf_id_tracking.py \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --split "${SPLIT}" \
  --track-id-weight "${TRACK_ID_WEIGHT}" \
  --max-id-distance "${MAX_ID_DISTANCE}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda
