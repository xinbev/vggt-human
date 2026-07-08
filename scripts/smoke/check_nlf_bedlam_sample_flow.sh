#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_nlf_provider.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/nlf_bedlam_sample_flow}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"
MAX_HUMANS="${MAX_HUMANS:-20}"
NUM_VIEWS="${NUM_VIEWS:-2}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-false}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }

echo "========== NLF BEDLAM sample flow =========="
echo "Repo        : ${REPO_ROOT}"
echo "BEDLAM root : ${BEDLAM_ROOT}"
echo "Boxes root  : ${PREPROCESSED_ROOT}"
echo "Output      : ${OUTPUT_DIR}"

bash scripts/smoke/check_nlf_runtime_requirements.sh
bash scripts/smoke/check_nlf_provider_interface.sh

if [[ "${FORCE_PREPROCESS}" == "true" || ! -f "${PREPROCESSED_ROOT}/summary.json" ]]; then
  BEDLAM_ROOT="${BEDLAM_ROOT}" \
  OUTPUT_ROOT="${PREPROCESSED_ROOT}" \
  SPLITS="Training" \
  MAX_HUMANS="${MAX_HUMANS}" \
  bash scripts/preprocess/prepare_bedlam_boxes.sh
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/diagnostics/check_nlf_hsi_forward.py \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --output-dir "${OUTPUT_DIR}/nlf_hsi_forward" \
  --max-batches 1 \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
  --override "data.train_split=Training" \
  --override "data.val_split=" \
  --override "data.sequence_length=${NUM_VIEWS}" \
  --override "data.max_humans=${MAX_HUMANS}" \
  --override "data.require_boxes=true" \
  --override "data.require_depth=true" \
  --override "data.num_workers=0" \
  --override "data.pin_memory=false" \
  --override "model.smpl_provider=nlf" \
  --override "model.nlf_use_detector=false" \
  --override "model.nlf_require_boxes=true" \
  --override "model.num_smpl_queries=${MAX_HUMANS}" \
  --override "optim.batch_size=1"

echo "========== NLF BEDLAM sample flow passed =========="
echo "Forward summary: ${OUTPUT_DIR}/nlf_hsi_forward/summary.json"
