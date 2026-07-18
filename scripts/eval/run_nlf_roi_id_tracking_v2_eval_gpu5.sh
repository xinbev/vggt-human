#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/home/zhw/lab_users/xyb/home/projects/vggt-human"
PATH_CONFIG="${REPO_ROOT}/configs/path.yaml"
TRAIN_CONFIG="${REPO_ROOT}/configs/train_nlf_roi_id_tracking_v2.yaml"
CHECKPOINT="${1:?Usage: bash scripts/eval/run_nlf_roi_id_tracking_v2_eval_gpu5.sh CHECKPOINT [SPLIT] [ID_WEIGHT] [MAX_ID_DISTANCE] [MAX_BATCHES] [RUN_NAME]}"
SPLIT="${2:-Training}"
TRACK_ID_WEIGHT="${3:-0.10}"
MAX_ID_DISTANCE="${4:-2.0}"
MAX_BATCHES="${5:-200}"
RUN_NAME="${6:-${SPLIT}_idw${TRACK_ID_WEIGHT}_batches${MAX_BATCHES}}"
OUTPUT_DIR="${REPO_ROOT}/outputs/eval/nlf_roi_id_tracking_v2_gpu5/${RUN_NAME}"

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
  --max-batches "${MAX_BATCHES}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda
