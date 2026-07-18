#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
DATA_ROOT="${DATA_ROOT:-/home/zhw/xyb_space}"
BEDLAM_ROOT="${BEDLAM_ROOT:-${DATA_ROOT}/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
SPLIT_ROOT="${SPLIT_ROOT:-${REPO_ROOT}/outputs/preprocess/hsi_sequence_split_v2}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_stage2_v4_a1_correction.yaml}"
STAGE1_CKPT="${STAGE1_CKPT:-${REPO_ROOT}/outputs/train/stage1_scale_linear_b20_gpu7/checkpoint_top_train_epoch_0003_loss_total_0.171740.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/hsi_stage2_v4_decoupled/stage2a1_correction}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MAX_HUMANS="${MAX_HUMANS:-20}"
NUM_VIEWS="${NUM_VIEWS:-2}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-5}"
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-0}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-0}"
SUBSET_INDICES_CSV="${SUBSET_INDICES_CSV:-}"
SUBSET_REPEAT="${SUBSET_REPEAT:-1}"
SUBSET_MAX_SAMPLES="${SUBSET_MAX_SAMPLES:-0}"
SUBSET_APPLY_TO_VAL="${SUBSET_APPLY_TO_VAL:-false}"
TRAIN_SEQUENCE_MANIFEST="${TRAIN_SEQUENCE_MANIFEST:-${SPLIT_ROOT}/train_sequences.txt}"
VAL_SEQUENCE_MANIFEST="${VAL_SEQUENCE_MANIFEST:-${SPLIT_ROOT}/val_sequences.txt}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-false}"
RESUME_REQUIRED_PREFIXES="${RESUME_REQUIRED_PREFIXES:-hsi_refinement_head.}"
FROZEN_HASH_PREFIXES="${FROZEN_HASH_PREFIXES:-hsi_refinement_head.}"

cd "${REPO_ROOT}"
[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing V4 config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -f "${STAGE1_CKPT}" ]] || { echo "[ERROR] Missing Stage1 checkpoint: ${STAGE1_CKPT}" >&2; exit 1; }
[[ -f "${TRAIN_SEQUENCE_MANIFEST}" ]] || { echo "[ERROR] Missing train manifest: ${TRAIN_SEQUENCE_MANIFEST}" >&2; exit 1; }
[[ -f "${VAL_SEQUENCE_MANIFEST}" ]] || { echo "[ERROR] Missing val manifest: ${VAL_SEQUENCE_MANIFEST}" >&2; exit 1; }
if [[ -f "${OUTPUT_DIR}/metrics_latest.json" && "${ALLOW_EXISTING_OUTPUT}" != "true" ]]; then
  echo "[ERROR] Refusing to overwrite existing V4 output: ${OUTPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ARGS=(
  --path-config "${PATH_CONFIG}"
  --train-config "${TRAIN_CONFIG}"
  --device cuda
  --override "datasets.bedlam_root=${BEDLAM_ROOT}"
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}"
  --override "experiment.output_dir=${OUTPUT_DIR}"
  --override "checkpoint.resume=${STAGE1_CKPT}"
  --override "checkpoint.resume_required_prefixes=${RESUME_REQUIRED_PREFIXES}"
  --override "checkpoint.frozen_hash_prefixes=${FROZEN_HASH_PREFIXES}"
  --override "data.train_sequence_manifest=${TRAIN_SEQUENCE_MANIFEST}"
  --override "data.val_sequence_manifest=${VAL_SEQUENCE_MANIFEST}"
  --override "data.sequence_length=${NUM_VIEWS}"
  --override "data.max_humans=${MAX_HUMANS}"
  --override "data.num_workers=${NUM_WORKERS}"
  --override "model.num_smpl_queries=${MAX_HUMANS}"
  --override "optim.batch_size=${BATCH_SIZE}"
  --override "optim.epochs=${EPOCHS}"
  --override "optim.lr=${LR}"
  --override "optim.max_steps_per_epoch=${MAX_STEPS_PER_EPOCH}"
  --override "optim.max_val_steps=${MAX_VAL_STEPS}"
  --override "data.subset_indices_csv=${SUBSET_INDICES_CSV}"
  --override "data.subset_repeat=${SUBSET_REPEAT}"
  --override "data.subset_max_samples=${SUBSET_MAX_SAMPLES}"
  --override "data.subset_apply_to_val=${SUBSET_APPLY_TO_VAL}"
)

echo "========== HSI Stage2 V4-A1 correction-only =========="
echo "Output      : ${OUTPUT_DIR}"
echo "Stage1 ckpt : ${STAGE1_CKPT}"
echo "GPU         : ${CUDA_VISIBLE_DEVICES_VALUE}"
echo "Batch/views : ${BATCH_SIZE} / ${NUM_VIEWS}"
echo "Epochs/LR   : ${EPOCHS} / ${LR}"
echo "Max steps   : ${MAX_STEPS_PER_EPOCH}"
python scripts/train/train_smpl.py "${ARGS[@]}"
