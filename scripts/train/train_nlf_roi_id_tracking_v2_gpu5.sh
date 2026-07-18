#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/home/zhw/lab_users/xyb/home/projects/vggt-human"
CONFIG_PATH="${REPO_ROOT}/configs/train_nlf_roi_id_tracking_v2.yaml"
PATH_CONFIG="${REPO_ROOT}/configs/path.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/nlf_roi_id_tracking_v2_gpu5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
NLF_INTERNAL_BATCH_SIZE="${NLF_INTERNAL_BATCH_SIZE:-256}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -f "${CONFIG_PATH}" ]] || { echo "[ERROR] Missing train config: ${CONFIG_PATH}" >&2; exit 1; }
[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=5
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "========== NLF ROI ID tracking V2 training =========="
echo "GPU       : physical GPU 5"
echo "Config    : ${CONFIG_PATH}"
echo "Output    : ${OUTPUT_DIR}"
echo "Batch     : ${BATCH_SIZE}"
echo "Workers   : ${NUM_WORKERS}"
echo "Prefetch  : ${PREFETCH_FACTOR}"
echo "NLF batch : ${NLF_INTERNAL_BATCH_SIZE}"

python -u scripts/train/train_smpl.py \
  --path-config "${PATH_CONFIG}" \
  --train-config "${CONFIG_PATH}" \
  --device cuda \
  --override "optim.batch_size=${BATCH_SIZE}" \
  --override "data.num_workers=${NUM_WORKERS}" \
  --override "data.prefetch_factor=${PREFETCH_FACTOR}" \
  --override "data.persistent_workers=true" \
  --override "model.nlf_internal_batch_size=${NLF_INTERNAL_BATCH_SIZE}"

echo "========== NLF ROI ID tracking V2 training finished =========="
