#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_temporal_refiner_emdb_3dpw.yaml}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
THREEDPW_ROOT="${THREEDPW_ROOT:-/home/zhw/xyb_space/3DPW/sequenceFiles/train}"
EMDB_ROOT="${EMDB_ROOT:-/home/zhw/xyb_space/emdb}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/smpl_temporal_refiner_emdb_3dpw_v1}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"
WINDOW_SIZE="${WINDOW_SIZE:-9}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-2e-4}"
SAMPLING_MODE="${SAMPLING_MODE:-balanced_dataset}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-vggt-human}"
WANDB_MODE="${WANDB_MODE:-online}"

[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -d "${THREEDPW_ROOT}" ]] || { echo "[ERROR] Missing 3DPW sequenceFiles/train root: ${THREEDPW_ROOT}" >&2; exit 1; }
[[ -d "${EMDB_ROOT}" ]] || { echo "[ERROR] Missing EMDB root: ${EMDB_ROOT}" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SOURCES_JSON="[{\"name\":\"3dpw\",\"root\":\"${THREEDPW_ROOT}\"},{\"name\":\"emdb\",\"root\":\"${EMDB_ROOT}\"}]"
echo "========== Standalone SMPL temporal-refiner training =========="
echo "3DPW: ${THREEDPW_ROOT}"
echo "EMDB : ${EMDB_ROOT}"
echo "Output: ${OUTPUT_DIR}"
echo "Window/batch: ${WINDOW_SIZE} / ${BATCH_SIZE}"
echo "Sampling: ${SAMPLING_MODE}, samples/epoch=${SAMPLES_PER_EPOCH}"
echo "W&B: enabled=${WANDB_ENABLED}, project=${WANDB_PROJECT}, mode=${WANDB_MODE}"

python scripts/train/train_smpl_temporal_refiner.py \
  --config "${TRAIN_CONFIG}" \
  --path-config "${PATH_CONFIG}" \
  --override "data.sources=${SOURCES_JSON}" \
  --override "experiment.output_dir=${OUTPUT_DIR}" \
  --override "data.window_size=${WINDOW_SIZE}" \
  --override "data.min_valid_frames=${WINDOW_SIZE}" \
  --override "data.num_workers=${NUM_WORKERS}" \
  --override "model.window_size=${WINDOW_SIZE}" \
  --override "optim.batch_size=${BATCH_SIZE}" \
  --override "optim.epochs=${EPOCHS}" \
  --override "optim.lr=${LR}" \
  --override "data.sampling_mode=${SAMPLING_MODE}" \
  --override "data.samples_per_epoch=${SAMPLES_PER_EPOCH}" \
  --override "logging.wandb.enabled=${WANDB_ENABLED}" \
  --override "logging.wandb.project=${WANDB_PROJECT}" \
  --override "logging.wandb.mode=${WANDB_MODE}"

echo "Latest checkpoint: ${OUTPUT_DIR}/checkpoint_latest.pt"
echo "Best checkpoint  : ${OUTPUT_DIR}/checkpoint_best.pt"
