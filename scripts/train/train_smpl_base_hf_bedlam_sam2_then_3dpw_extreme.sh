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
HF_CONFIG="${HF_CONFIG:-configs/train_smpl_base_hf_bedlam_sam2_mask_pose_beta_extreme.yaml}"
THREEDPW_CONFIG="${THREEDPW_CONFIG:-configs/train_smpl_base_3dpw_sam2_mask_pose_beta_extreme.yaml}"

HF_SAM2_MASK_ROOT="${HF_SAM2_MASK_ROOT:-outputs/preprocess/hf_bedlam_sam2_patch_masks}"
THREEDPW_SAM2_MASK_ROOT="${THREEDPW_SAM2_MASK_ROOT:-outputs/preprocess/3dpw_sam2_patch_masks}"

HF_OUT_DIR="${HF_OUT_DIR:-outputs/train/stageF_hf_bedlam_sam2_mask_pose_beta_extreme}"
THREEDPW_OUT_DIR="${THREEDPW_OUT_DIR:-outputs/train/stageG_3dpw_sam2_mask_pose_beta_extreme_from_hf_bedlam}"
VAL_OUT_DIR="${VAL_OUT_DIR:-outputs/eval/stageG_3dpw_validation_sam2_mask_from_hf_bedlam}"
TEST_OUT_DIR="${TEST_OUT_DIR:-outputs/eval/stageG_3dpw_test_sam2_mask_from_hf_bedlam}"

HF_BATCH_SIZE="${HF_BATCH_SIZE:-40}"
THREEDPW_BATCH_SIZE="${THREEDPW_BATCH_SIZE:-12}"
HF_EPOCHS="${HF_EPOCHS:-12}"
THREEDPW_EPOCHS="${THREEDPW_EPOCHS:-40}"
NUM_WORKERS="${NUM_WORKERS:-28}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"

RUN_HF_PREPROCESS="${RUN_HF_PREPROCESS:-1}"
RUN_3DPW_PREPROCESS="${RUN_3DPW_PREPROCESS:-1}"
RUN_HF_TRAIN="${RUN_HF_TRAIN:-1}"
RUN_3DPW_FINETUNE="${RUN_3DPW_FINETUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

INIT_CKPT="${INIT_CKPT:-}"
if [[ -z "${INIT_CKPT}" ]]; then
  for candidate in \
    "outputs/train/stageB_3dpw_smpl_base_roi_pool_from_hf_bedlam/checkpoint_top01.pt" \
    "outputs/train/stageB_3dpw_smpl_base_roi_pool_from_hf_bedlam/checkpoint_latest.pt" \
    "outputs/train/stageA_hf_bedlam_smpl_base_roi_pool/checkpoint_latest.pt"; do
    if [[ -f "${candidate}" ]]; then
      INIT_CKPT="${candidate}"
      break
    fi
  done
fi
if [[ -z "${INIT_CKPT}" ]]; then
  echo "[ERROR] Missing ROI-pool initialization checkpoint. Set INIT_CKPT=/path/to/checkpoint.pt" >&2
  exit 1
fi

echo "========== HF BEDLAM SAM2 -> 3DPW SAM2 pose/beta extreme pipeline =========="
echo "CUDA devices       : ${CUDA_VISIBLE_DEVICES}"
echo "Torch device       : ${TRAIN_DEVICE}"
echo "Initial checkpoint : ${INIT_CKPT}"
echo "HF config          : ${HF_CONFIG}"
echo "3DPW config        : ${THREEDPW_CONFIG}"
echo "HF SAM2 masks      : ${HF_SAM2_MASK_ROOT}"
echo "3DPW SAM2 masks    : ${THREEDPW_SAM2_MASK_ROOT}"
echo "HF out             : ${HF_OUT_DIR}"
echo "3DPW out           : ${THREEDPW_OUT_DIR}"
echo "HF epochs/batch    : ${HF_EPOCHS} / ${HF_BATCH_SIZE}"
echo "3DPW epochs/batch  : ${THREEDPW_EPOCHS} / ${THREEDPW_BATCH_SIZE}"
echo "Workers            : ${NUM_WORKERS}"

if [[ "${RUN_HF_PREPROCESS}" == "1" ]]; then
  echo "========== Stage F0: prepare HF BEDLAM SAM2 patch masks =========="
  OUTPUT_ROOT="${HF_SAM2_MASK_ROOT}" \
  DEVICE="${TRAIN_DEVICE}" \
  MAX_NPZ_FILES="${HF_SAM2_MAX_NPZ_FILES:-0}" \
  MAX_FRAMES="${HF_SAM2_MAX_FRAMES:-0}" \
  MAX_OUTPUT_FRAMES="${HF_SAM2_MAX_OUTPUT_FRAMES:-0}" \
  OVERWRITE="${HF_SAM2_OVERWRITE:-0}" \
  LOG_INTERVAL="${HF_SAM2_LOG_INTERVAL:-100}" \
  bash scripts/preprocess/prepare_hf_bedlam_sam2_patch_masks.sh
fi

if [[ "${RUN_3DPW_PREPROCESS}" == "1" ]]; then
  echo "========== Stage F1: prepare 3DPW SAM2 patch masks =========="
  OUTPUT_ROOT="${THREEDPW_SAM2_MASK_ROOT}" \
  SPLITS="${THREEDPW_SAM2_SPLITS:-train validation test}" \
  DEVICE="${TRAIN_DEVICE}" \
  MAX_FRAMES="${THREEDPW_SAM2_MAX_FRAMES:-}" \
  OVERWRITE="${THREEDPW_SAM2_OVERWRITE:-0}" \
  LOG_INTERVAL="${THREEDPW_SAM2_LOG_INTERVAL:-100}" \
  bash scripts/preprocess/prepare_3dpw_sam2_patch_masks.sh
fi

if [[ "${RUN_HF_TRAIN}" == "1" ]]; then
  echo "========== Stage F2: HF BEDLAM SAM2 mask-intersection pretrain =========="
  PATH_CONFIG="${PATH_CONFIG}" \
  TRAIN_CONFIG="${HF_CONFIG}" \
  OUT_DIR="${HF_OUT_DIR}" \
  BATCH_SIZE="${HF_BATCH_SIZE}" \
  EPOCHS="${HF_EPOCHS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  PREFETCH_FACTOR="${PREFETCH_FACTOR}" \
  PERSISTENT_WORKERS="${PERSISTENT_WORKERS}" \
  RESUME="${INIT_CKPT}" \
  RESET_EPOCH=true \
  RESUME_OPTIMIZER=false \
  SAVE_LATEST=true \
  SAVE_FINAL=true \
  SAVE_TOP_K=3 \
  SAM2_PATCH_MASKS_ROOT="${HF_SAM2_MASK_ROOT}" \
  DEVICE="${TRAIN_DEVICE}" \
  bash scripts/train/train_smpl_base_hf_bedlam_ray_refine.sh
fi

HF_CKPT="${HF_CKPT:-}"
if [[ -z "${HF_CKPT}" ]]; then
  if [[ "${RUN_HF_TRAIN}" == "1" ]]; then
    HF_CKPT="${HF_OUT_DIR}/checkpoint_latest.pt"
  else
    HF_CKPT="${INIT_CKPT}"
  fi
fi
if [[ ! -f "${HF_CKPT}" ]]; then
  if [[ -f "${HF_OUT_DIR}/checkpoint_final.pt" ]]; then
    HF_CKPT="${HF_OUT_DIR}/checkpoint_final.pt"
  else
    echo "[ERROR] Missing HF checkpoint: ${HF_CKPT}" >&2
    exit 1
  fi
fi
echo "HF checkpoint       : ${HF_CKPT}"

if [[ "${RUN_3DPW_FINETUNE}" == "1" ]]; then
  echo "========== Stage G: 3DPW SAM2 mask-intersection long fine-tune =========="
  PATH_CONFIG="${PATH_CONFIG}" \
  TRAIN_CONFIG="${THREEDPW_CONFIG}" \
  OUT_DIR="${THREEDPW_OUT_DIR}" \
  BATCH_SIZE="${THREEDPW_BATCH_SIZE}" \
  EPOCHS="${THREEDPW_EPOCHS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  PREFETCH_FACTOR="${PREFETCH_FACTOR}" \
  PERSISTENT_WORKERS="${PERSISTENT_WORKERS}" \
  RESUME="${HF_CKPT}" \
  RESET_EPOCH=true \
  RESUME_OPTIMIZER=false \
  SAVE_TOP_K=3 \
  CHECKPOINT_MONITOR=loss_total \
  CHECKPOINT_MONITOR_MODE=min \
  SAM2_PATCH_MASKS_ROOT="${THREEDPW_SAM2_MASK_ROOT}" \
  DEVICE="${TRAIN_DEVICE}" \
  bash scripts/train/train_smpl_base_3dpw_ray_refine.sh
fi

THREEDPW_CKPT="${THREEDPW_CKPT:-${THREEDPW_OUT_DIR}/checkpoint_top01.pt}"
if [[ ! -f "${THREEDPW_CKPT}" ]]; then
  if [[ -f "${THREEDPW_OUT_DIR}/checkpoint_latest.pt" ]]; then
    THREEDPW_CKPT="${THREEDPW_OUT_DIR}/checkpoint_latest.pt"
  else
    echo "[ERROR] Missing 3DPW checkpoint: ${THREEDPW_CKPT}" >&2
    exit 1
  fi
fi
echo "3DPW checkpoint     : ${THREEDPW_CKPT}"

if [[ "${RUN_EVAL}" == "1" ]]; then
  echo "========== Stage G-val: 3DPW validation eval =========="
  CHECKPOINT="${THREEDPW_CKPT}" \
  TRAIN_CONFIG="${THREEDPW_CONFIG}" \
  PATH_CONFIG="${PATH_CONFIG}" \
  OUT_DIR="${VAL_OUT_DIR}" \
  SPLIT=validation \
  BATCH_SIZE=1 \
  NUM_WORKERS="${NUM_WORKERS}" \
  SAM2_PATCH_MASKS_ROOT="${THREEDPW_SAM2_MASK_ROOT}" \
  DEVICE="${TRAIN_DEVICE}" \
  bash scripts/eval/evaluate_3dpw_smpl_base_metrics.sh

  echo "========== Stage G-test: 3DPW test eval =========="
  CHECKPOINT="${THREEDPW_CKPT}" \
  TRAIN_CONFIG="${THREEDPW_CONFIG}" \
  PATH_CONFIG="${PATH_CONFIG}" \
  OUT_DIR="${TEST_OUT_DIR}" \
  SPLIT=test \
  BATCH_SIZE=1 \
  NUM_WORKERS="${NUM_WORKERS}" \
  SAM2_PATCH_MASKS_ROOT="${THREEDPW_SAM2_MASK_ROOT}" \
  DEVICE="${TRAIN_DEVICE}" \
  bash scripts/eval/evaluate_3dpw_smpl_base_metrics.sh
fi

echo "========== HF BEDLAM SAM2 -> 3DPW SAM2 pipeline finished =========="
echo "HF checkpoint       : ${HF_CKPT}"
echo "3DPW checkpoint     : ${THREEDPW_CKPT}"
echo "Validation eval     : ${VAL_OUT_DIR}"
echo "Test eval           : ${TEST_OUT_DIR}"
