#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/train_smpl_pose_stabilizer_v2_mixture.yaml}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
THREEDPW_ROOT="${THREEDPW_ROOT:-/home/zhw/xyb_space/3DPW/sequenceFiles/train}"
EMDB_ROOT="${EMDB_ROOT:-/home/zhw/xyb_space/emdb}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/smpl_temporal_stabilizer_v2_pose_mixture}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-3e-4}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_MODE="${WANDB_MODE:-online}"

[[ -f "${CONFIG}" ]] || { echo "[ERROR] Missing config: ${CONFIG}" >&2; exit 1; }
[[ -d "${THREEDPW_ROOT}" ]] || { echo "[ERROR] Missing 3DPW root: ${THREEDPW_ROOT}" >&2; exit 1; }
[[ -d "${EMDB_ROOT}" ]] || { echo "[ERROR] Missing EMDB root: ${EMDB_ROOT}" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SOURCES_JSON="[{\"name\":\"3dpw\",\"root\":\"${THREEDPW_ROOT}\"},{\"name\":\"emdb\",\"root\":\"${EMDB_ROOT}\"}]"

echo "========== V2 pose stabilizer mixed training =========="
echo "Output: ${OUTPUT_DIR}; batch=${BATCH_SIZE}; epochs=${EPOCHS}"
echo "Mixture: clean=30%, centre=30%, small=25%, medium=15%"

python scripts/train/train_smpl_pose_stabilizer_v2.py \
  --config "${CONFIG}" \
  --path-config "${PATH_CONFIG}" \
  --override "data.sources=${SOURCES_JSON}" \
  --override "experiment.output_dir=${OUTPUT_DIR}" \
  --override "data.num_workers=${NUM_WORKERS}" \
  --override "optim.batch_size=${BATCH_SIZE}" \
  --override "optim.epochs=${EPOCHS}" \
  --override "optim.lr=${LR}" \
  --override "logging.wandb.enabled=${WANDB_ENABLED}" \
  --override "logging.wandb.mode=${WANDB_MODE}"

echo "Latest: ${OUTPUT_DIR}/checkpoint_latest.pt"
echo "Best  : ${OUTPUT_DIR}/checkpoint_best.pt"
