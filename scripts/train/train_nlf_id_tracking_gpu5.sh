#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/home/zhw/lab_users/xyb/home/projects/vggt-human"
CONFIG_PATH="${REPO_ROOT}/configs/train_nlf_id_tracking.yaml"
PATH_CONFIG="${REPO_ROOT}/configs/path.yaml"
OUTPUT_DIR="${REPO_ROOT}/outputs/train/nlf_id_tracking_gpu5"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -f "${CONFIG_PATH}" ]] || { echo "[ERROR] Missing train config: ${CONFIG_PATH}" >&2; exit 1; }
[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=5
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "========== NLF ID tracking training =========="
echo "GPU       : physical GPU 5"
echo "Config    : ${CONFIG_PATH}"
echo "Output    : ${OUTPUT_DIR}"

python -u scripts/train/train_smpl.py \
  --path-config "${PATH_CONFIG}" \
  --train-config "${CONFIG_PATH}" \
  --device cuda

echo "========== NLF ID tracking training finished =========="
