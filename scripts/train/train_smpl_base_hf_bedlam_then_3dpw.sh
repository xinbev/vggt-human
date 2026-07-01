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
HF_CONFIG="${HF_CONFIG:-configs/train_smpl_base_hf_bedlam_ray_refine.yaml}"
THREEDPW_CONFIG="${THREEDPW_CONFIG:-configs/train_smpl_base_3dpw_ray_refine.yaml}"

HF_OUT_DIR="${HF_OUT_DIR:-outputs/train/stageA_hf_bedlam_smpl_base_ray_refine}"
THREEDPW_OUT_DIR="${THREEDPW_OUT_DIR:-outputs/train/stageB_3dpw_smpl_base_ray_refine_from_hf_bedlam}"
VAL_OUT_DIR="${VAL_OUT_DIR:-outputs/eval/stageC_3dpw_validation_from_hf_bedlam}"
TEST_OUT_DIR="${TEST_OUT_DIR:-outputs/eval/stageD_3dpw_test_from_hf_bedlam}"

HF_BATCH_SIZE="${HF_BATCH_SIZE:-8}"
THREEDPW_BATCH_SIZE="${THREEDPW_BATCH_SIZE:-6}"
HF_EPOCHS="${HF_EPOCHS:-15}"
THREEDPW_EPOCHS="${THREEDPW_EPOCHS:-5}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"

RUN_CHECKS="${RUN_CHECKS:-1}"
RUN_HF_TRAIN="${RUN_HF_TRAIN:-1}"
RUN_3DPW_FINETUNE="${RUN_3DPW_FINETUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

echo "========== SMPL Base Pipeline: HF BEDLAM -> 3DPW =========="
echo "CUDA devices       : ${CUDA_VISIBLE_DEVICES}"
echo "Torch device       : ${TRAIN_DEVICE}"
echo "Path config        : ${PATH_CONFIG}"
echo "HF config          : ${HF_CONFIG}"
echo "3DPW config        : ${THREEDPW_CONFIG}"
echo "HF out             : ${HF_OUT_DIR}"
echo "3DPW out           : ${THREEDPW_OUT_DIR}"
echo "HF epochs/batch    : ${HF_EPOCHS} / ${HF_BATCH_SIZE}"
echo "3DPW epochs/batch  : ${THREEDPW_EPOCHS} / ${THREEDPW_BATCH_SIZE}"
echo "Workers            : ${NUM_WORKERS}"
echo "Prefetch/persistent: ${PREFETCH_FACTOR} / ${PERSISTENT_WORKERS}"

if [[ "${RUN_CHECKS}" == "1" ]]; then
  echo "========== Stage 0: HF BEDLAM data smoke check =========="
  MAX_NPZ_FILES="${CHECK_MAX_NPZ_FILES:-1}" \
  MAX_FRAMES="${CHECK_MAX_FRAMES:-50}" \
  BATCH_SIZE=1 \
  PATH_CONFIG="${PATH_CONFIG}" \
  TRAIN_CONFIG="${HF_CONFIG}" \
  bash scripts/diagnostics/check_hf_bedlam_smpl_base_data.sh
fi

if [[ "${RUN_HF_TRAIN}" == "1" ]]; then
  echo "========== Stage A: HF BEDLAM SMPL base pretrain =========="
  PATH_CONFIG="${PATH_CONFIG}" \
  TRAIN_CONFIG="${HF_CONFIG}" \
  OUT_DIR="${HF_OUT_DIR}" \
  BATCH_SIZE="${HF_BATCH_SIZE}" \
  EPOCHS="${HF_EPOCHS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  PREFETCH_FACTOR="${PREFETCH_FACTOR}" \
  PERSISTENT_WORKERS="${PERSISTENT_WORKERS}" \
  SAVE_LATEST=true \
  SAVE_FINAL=true \
  DEVICE="${TRAIN_DEVICE}" \
  bash scripts/train/train_smpl_base_hf_bedlam_ray_refine.sh
fi

HF_CKPT="${HF_CKPT:-${HF_OUT_DIR}/checkpoint_latest.pt}"
if [[ "${RUN_3DPW_FINETUNE}" == "1" ]]; then
  if [[ ! -f "${HF_CKPT}" ]]; then
    if [[ -f "${HF_OUT_DIR}/checkpoint_final.pt" ]]; then
      HF_CKPT="${HF_OUT_DIR}/checkpoint_final.pt"
    else
      echo "[ERROR] HF BEDLAM checkpoint not found: ${HF_CKPT}" >&2
      echo "        Set HF_CKPT=/path/to/checkpoint.pt or enable RUN_HF_TRAIN=1." >&2
      exit 1
    fi
  fi
  echo "HF checkpoint       : ${HF_CKPT}"
else
  echo "HF checkpoint       : skipped"
fi

if [[ "${RUN_3DPW_FINETUNE}" == "1" ]]; then
  echo "========== Stage B: 3DPW fine-tune from HF BEDLAM =========="
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
  DEVICE="${TRAIN_DEVICE}" \
  bash scripts/train/train_smpl_base_3dpw_ray_refine.sh
fi

THREEDPW_CKPT="${THREEDPW_CKPT:-${THREEDPW_OUT_DIR}/checkpoint_top01.pt}"
if [[ ! -f "${THREEDPW_CKPT}" ]]; then
  if [[ -f "${THREEDPW_OUT_DIR}/checkpoint_latest.pt" ]]; then
    THREEDPW_CKPT="${THREEDPW_OUT_DIR}/checkpoint_latest.pt"
  else
    echo "[ERROR] 3DPW checkpoint not found: ${THREEDPW_CKPT}" >&2
    echo "        Set THREEDPW_CKPT=/path/to/checkpoint.pt or enable RUN_3DPW_FINETUNE=1." >&2
    exit 1
  fi
fi
echo "3DPW checkpoint     : ${THREEDPW_CKPT}"

if [[ "${RUN_EVAL}" == "1" ]]; then
  echo "========== Stage C: 3DPW validation eval =========="
  CHECKPOINT="${THREEDPW_CKPT}" \
  TRAIN_CONFIG="${THREEDPW_CONFIG}" \
  PATH_CONFIG="${PATH_CONFIG}" \
  OUT_DIR="${VAL_OUT_DIR}" \
  SPLIT=validation \
  BATCH_SIZE=1 \
  NUM_WORKERS="${NUM_WORKERS}" \
  DEVICE="${TRAIN_DEVICE}" \
  bash scripts/eval/evaluate_3dpw_smpl_base_metrics.sh

  echo "========== Stage D: 3DPW test eval =========="
  CHECKPOINT="${THREEDPW_CKPT}" \
  TRAIN_CONFIG="${THREEDPW_CONFIG}" \
  PATH_CONFIG="${PATH_CONFIG}" \
  OUT_DIR="${TEST_OUT_DIR}" \
  SPLIT=test \
  BATCH_SIZE=1 \
  NUM_WORKERS="${NUM_WORKERS}" \
  DEVICE="${TRAIN_DEVICE}" \
  bash scripts/eval/evaluate_3dpw_smpl_base_metrics.sh
fi

echo "========== Pipeline finished =========="
echo "HF checkpoint   : ${HF_CKPT:-skipped}"
echo "3DPW checkpoint : ${THREEDPW_CKPT}"
echo "Validation eval : ${VAL_OUT_DIR}"
echo "Test eval       : ${TEST_OUT_DIR}"
