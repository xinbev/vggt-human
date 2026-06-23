#!/usr/bin/env bash

set -euo pipefail

# SMPL Translation V2 long-sequence training.
# Trains the new ray-offset-depth geometry seed and track-aware temporal
# translation head while keeping the VGGT/SMPL/HSI baseline weights frozen.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_translation_v2_longseq.yaml}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VGGT_CKPT="${VGGT_CKPT:-/home/zhw/lab_users/xyb/home/projects/vggt-omega/checkpoints/vggt_omega_1b_512.pt}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-/home/zhw/xyb_space/SAT-HMR/weights/smpl_data/}"
INIT_CKPT="${INIT_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_temporal_after_translation_ray_refine/stage2_human_momentum_no_worse/checkpoint_latest.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/smpl_translation_v2_longseq}"

NUM_FRAMES="${NUM_FRAMES:-12}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_HUMANS="${MAX_HUMANS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-6}"
LR="${LR:-0.000003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
SUBSET_CSV="${SUBSET_CSV:-}"
SUBSET_REPEAT="${SUBSET_REPEAT:-1}"
SUBSET_MAX_SAMPLES="${SUBSET_MAX_SAMPLES:-0}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing preprocessed boxes: ${PREPROCESSED_ROOT}" >&2; exit 1; }
[[ -f "${VGGT_CKPT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${VGGT_CKPT}" >&2; exit 1; }
[[ -f "${INIT_CKPT}" ]] || { echo "[ERROR] Missing init checkpoint: ${INIT_CKPT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }
if [[ -n "${SUBSET_CSV}" && ! -f "${SUBSET_CSV}" ]]; then
  echo "[ERROR] Missing subset CSV: ${SUBSET_CSV}" >&2
  exit 1
fi

echo "========== SMPL Translation V2 long-sequence training =========="
echo "Init ckpt   : ${INIT_CKPT}"
echo "Output      : ${OUTPUT_DIR}"
echo "Train config: ${TRAIN_CONFIG}"
echo "Frames      : ${NUM_FRAMES}"
echo "Epochs      : ${EPOCHS}"
echo "LR          : ${LR}"
echo "Subset CSV  : ${SUBSET_CSV:-<none>}"
echo "GPU         : ${CUDA_VISIBLE_DEVICES_VALUE}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/train/train_smpl.py \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
  --override "checkpoints.vggt_baseline=${VGGT_CKPT}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
  --override "experiment.output_dir=${OUTPUT_DIR}" \
  --override "data.sequence_length=${NUM_FRAMES}" \
  --override "data.stride=${FRAME_STRIDE}" \
  --override "data.max_humans=${MAX_HUMANS}" \
  --override "data.num_workers=${NUM_WORKERS}" \
  --override "data.subset_indices_csv=${SUBSET_CSV}" \
  --override "data.subset_repeat=${SUBSET_REPEAT}" \
  --override "data.subset_max_samples=${SUBSET_MAX_SAMPLES}" \
  --override "checkpoint.resume=${INIT_CKPT}" \
  --override "checkpoint.load_vggt_baseline=true" \
  --override "checkpoint.resume_strict=false" \
  --override "checkpoint.resume_optimizer=false" \
  --override "checkpoint.save_final=true" \
  --override "checkpoint.save_epoch_checkpoint=false" \
  --override "checkpoint.save_latest=true" \
  --override "optim.batch_size=${BATCH_SIZE}" \
  --override "optim.epochs=${EPOCHS}" \
  --override "optim.lr=${LR}" \
  --override "optim.weight_decay=${WEIGHT_DECAY}" \
  --override "optim.grad_clip_norm=${GRAD_CLIP_NORM}" \
  --override "optim.log_interval=${LOG_INTERVAL}" \
  --override "optim.save_interval=0" \
  --override "model.num_smpl_queries=${MAX_HUMANS}" \
  --override "model.smpl_translation_output_mode=ray_offset_depth" \
  --override "model.smpl_enable_temporal_translation=true" \
  --override "model.smpl_temporal_translation_use_world=true" \
  --override "model.smpl_enable_translation_refine=false" \
  --override "model.enable_hsi_refine=false" \
  --override "model.enable_depth=false" \
  --override "model.train_smpl_translation_decode_heads=true" \
  --override "model.train_smpl_temporal_translation=true"

echo "========== SMPL Translation V2 training finished =========="
echo "Checkpoint: ${OUTPUT_DIR}/checkpoint_latest.pt"
