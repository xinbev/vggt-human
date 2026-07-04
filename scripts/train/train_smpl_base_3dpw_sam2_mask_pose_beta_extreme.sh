#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

REQUESTED_DEVICE="${DEVICE:-cuda}"
TRAIN_DEVICE="${REQUESTED_DEVICE}"

if [[ "${REQUESTED_DEVICE}" =~ ^cuda:([0-9]+)$ ]]; then
  REQUESTED_GPU_ID="${BASH_REMATCH[1]}"
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${CUDA_VISIBLE_DEVICES_VALUE:-}" && -z "${GPU_ID:-}" ]]; then
    GPU_ID="${REQUESTED_GPU_ID}"
  fi
  TRAIN_DEVICE="cuda"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES_VALUE:-${GPU_ID:-0}}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PATH_CONFIG="${PATH_CONFIG:-configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_smpl_base_3dpw_sam2_mask_pose_beta_extreme.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/stageE_3dpw_sam2_mask_pose_beta_extreme_from_roi_pool}"
VAL_OUT_DIR="${VAL_OUT_DIR:-outputs/eval/stageE_3dpw_validation_sam2_mask_pose_beta_extreme}"
TEST_OUT_DIR="${TEST_OUT_DIR:-outputs/eval/stageE_3dpw_test_sam2_mask_pose_beta_extreme}"
SAM2_MASK_ROOT="${SAM2_MASK_ROOT:-outputs/preprocess/3dpw_sam2_patch_masks}"

BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-30}"
NUM_WORKERS="${NUM_WORKERS:-28}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"

RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

INIT_CKPT="${INIT_CKPT:-}"
if [[ -z "${INIT_CKPT}" ]]; then
  for candidate in \
    "outputs/train/stageB_3dpw_smpl_base_roi_pool_from_hf_bedlam/checkpoint_top01.pt" \
    "outputs/train/stageB_3dpw_smpl_base_roi_pool_from_hf_bedlam/checkpoint_latest.pt"; do
    if [[ -f "${candidate}" ]]; then
      INIT_CKPT="${candidate}"
      break
    fi
  done
fi
if [[ -z "${INIT_CKPT}" ]]; then
  echo "[ERROR] Missing ROI-pool init checkpoint. Set INIT_CKPT=/path/to/checkpoint.pt" >&2
  exit 1
fi

echo "========== 3DPW SAM2 mask-intersection pose/beta fine-tune =========="
echo "CUDA devices    : ${CUDA_VISIBLE_DEVICES}"
echo "Torch device    : ${TRAIN_DEVICE}"
echo "Config          : ${TRAIN_CONFIG}"
echo "Init checkpoint : ${INIT_CKPT}"
echo "Output dir      : ${OUTPUT_DIR}"
echo "SAM2 mask root  : ${SAM2_MASK_ROOT}"
echo "Epochs/batch    : ${EPOCHS} / ${BATCH_SIZE}"
echo "Workers         : ${NUM_WORKERS}"

if [[ "${RUN_PREPROCESS}" == "1" ]]; then
  echo "========== Stage 0: prepare 3DPW SAM2 patch masks =========="
  OUTPUT_ROOT="${SAM2_MASK_ROOT}" \
  SPLITS="${SAM2_SPLITS:-train validation test}" \
  DEVICE="${TRAIN_DEVICE}" \
  MAX_FRAMES="${SAM2_MAX_FRAMES:-}" \
  OVERWRITE="${SAM2_OVERWRITE:-0}" \
  LOG_INTERVAL="${SAM2_LOG_INTERVAL:-100}" \
  bash scripts/preprocess/prepare_3dpw_sam2_patch_masks.sh
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  echo "========== Stage E: train pose/beta with SAM2 mask-intersection pooling =========="
  PATH_CONFIG="${PATH_CONFIG}" \
  TRAIN_CONFIG="${TRAIN_CONFIG}" \
  OUT_DIR="${OUTPUT_DIR}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  EPOCHS="${EPOCHS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  PREFETCH_FACTOR="${PREFETCH_FACTOR}" \
  PERSISTENT_WORKERS="${PERSISTENT_WORKERS}" \
  RESUME="${INIT_CKPT}" \
  RESET_EPOCH=true \
  RESUME_OPTIMIZER=false \
  SAVE_TOP_K=3 \
  CHECKPOINT_MONITOR=loss_total \
  CHECKPOINT_MONITOR_MODE=min \
  SAM2_PATCH_MASKS_ROOT="${SAM2_MASK_ROOT}" \
  DEVICE="${TRAIN_DEVICE}" \
  bash scripts/train/train_smpl_base_3dpw_ray_refine.sh
fi

CKPT="${CHECKPOINT:-${OUTPUT_DIR}/checkpoint_top01.pt}"
if [[ ! -f "${CKPT}" ]]; then
  if [[ -f "${OUTPUT_DIR}/checkpoint_latest.pt" ]]; then
    CKPT="${OUTPUT_DIR}/checkpoint_latest.pt"
  else
    echo "[ERROR] Missing trained checkpoint: ${CKPT}" >&2
    exit 1
  fi
fi
echo "Eval checkpoint  : ${CKPT}"

if [[ "${RUN_EVAL}" == "1" ]]; then
  echo "========== Stage E-val: 3DPW validation eval with SAM2 masks =========="
  CHECKPOINT="${CKPT}" \
  TRAIN_CONFIG="${TRAIN_CONFIG}" \
  PATH_CONFIG="${PATH_CONFIG}" \
  OUT_DIR="${VAL_OUT_DIR}" \
  SPLIT=validation \
  BATCH_SIZE=1 \
  NUM_WORKERS="${NUM_WORKERS}" \
  SAM2_PATCH_MASKS_ROOT="${SAM2_MASK_ROOT}" \
  DEVICE="${TRAIN_DEVICE}" \
  bash scripts/eval/evaluate_3dpw_smpl_base_metrics.sh

  echo "========== Stage E-test: 3DPW test eval with SAM2 masks =========="
  CHECKPOINT="${CKPT}" \
  TRAIN_CONFIG="${TRAIN_CONFIG}" \
  PATH_CONFIG="${PATH_CONFIG}" \
  OUT_DIR="${TEST_OUT_DIR}" \
  SPLIT=test \
  BATCH_SIZE=1 \
  NUM_WORKERS="${NUM_WORKERS}" \
  SAM2_PATCH_MASKS_ROOT="${SAM2_MASK_ROOT}" \
  DEVICE="${TRAIN_DEVICE}" \
  bash scripts/eval/evaluate_3dpw_smpl_base_metrics.sh
fi

echo "========== SAM2 mask pose/beta fine-tune finished =========="
echo "Checkpoint      : ${CKPT}"
echo "Validation eval : ${VAL_OUT_DIR}"
echo "Test eval       : ${TEST_OUT_DIR}"
