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

NUM_FRAMES="${NUM_FRAMES:-16}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_HUMANS="${MAX_HUMANS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-6}"
EPOCHS="${EPOCHS:-8}"
LR="${LR:-0.000005}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
SAVE_FINAL="${SAVE_FINAL:-true}"
SAVE_EPOCH_CHECKPOINT="${SAVE_EPOCH_CHECKPOINT:-false}"
SAVE_LATEST="${SAVE_LATEST:-true}"
RESET_EPOCH="${RESET_EPOCH:-true}"
SUBSET_CSV="${SUBSET_CSV:-}"
SUBSET_REPEAT="${SUBSET_REPEAT:-1}"
SUBSET_MAX_SAMPLES="${SUBSET_MAX_SAMPLES:-0}"
ENABLE_TEMPORAL_TRANSLATION="${ENABLE_TEMPORAL_TRANSLATION:-true}"
TRAIN_TRANSLATION_DECODE_HEADS="${TRAIN_TRANSLATION_DECODE_HEADS:-true}"
TRAIN_TEMPORAL_TRANSLATION="${TRAIN_TEMPORAL_TRANSLATION:-true}"
TEMPORAL_USE_WORLD="${TEMPORAL_USE_WORLD:-true}"
TEMPORAL_MAX_VELOCITY_DELTA_M="${TEMPORAL_MAX_VELOCITY_DELTA_M:-0.35}"
TEMPORAL_GATE_BIAS="${TEMPORAL_GATE_BIAS:-2.5}"
DECODE_MAX_LOG_DEPTH_DELTA="${DECODE_MAX_LOG_DEPTH_DELTA:-1.10}"
DECODE_MAX_RAY_DELTA_M="${DECODE_MAX_RAY_DELTA_M:-1.50}"
DECODE_MAX_TANGENT_OFFSET_M="${DECODE_MAX_TANGENT_OFFSET_M:-1.00}"

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
echo "Temporal    : enable=${ENABLE_TEMPORAL_TRANSLATION} train=${TRAIN_TEMPORAL_TRANSLATION} world=${TEMPORAL_USE_WORLD}"
echo "Checkpoints : final=${SAVE_FINAL} epoch=${SAVE_EPOCH_CHECKPOINT} latest=${SAVE_LATEST} reset_epoch=${RESET_EPOCH}"
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
  --override "checkpoint.reset_epoch=${RESET_EPOCH}" \
  --override "checkpoint.save_final=${SAVE_FINAL}" \
  --override "checkpoint.save_epoch_checkpoint=${SAVE_EPOCH_CHECKPOINT}" \
  --override "checkpoint.save_latest=${SAVE_LATEST}" \
  --override "optim.batch_size=${BATCH_SIZE}" \
  --override "optim.epochs=${EPOCHS}" \
  --override "optim.lr=${LR}" \
  --override "optim.weight_decay=${WEIGHT_DECAY}" \
  --override "optim.grad_clip_norm=${GRAD_CLIP_NORM}" \
  --override "optim.log_interval=${LOG_INTERVAL}" \
  --override "optim.save_interval=0" \
  --override "model.num_smpl_queries=${MAX_HUMANS}" \
  --override "model.smpl_translation_output_mode=ray_offset_depth" \
  --override "model.smpl_translation_decode_max_log_depth_delta=${DECODE_MAX_LOG_DEPTH_DELTA}" \
  --override "model.smpl_translation_decode_max_ray_delta_m=${DECODE_MAX_RAY_DELTA_M}" \
  --override "model.smpl_translation_decode_max_tangent_offset_m=${DECODE_MAX_TANGENT_OFFSET_M}" \
  --override "model.smpl_enable_temporal_translation=${ENABLE_TEMPORAL_TRANSLATION}" \
  --override "model.smpl_temporal_translation_use_world=${TEMPORAL_USE_WORLD}" \
  --override "model.smpl_temporal_translation_max_velocity_delta_m=${TEMPORAL_MAX_VELOCITY_DELTA_M}" \
  --override "model.smpl_temporal_translation_gate_bias=${TEMPORAL_GATE_BIAS}" \
  --override "model.smpl_enable_translation_refine=false" \
  --override "model.enable_hsi_refine=false" \
  --override "model.enable_depth=false" \
  --override "model.train_smpl_translation_decode_heads=${TRAIN_TRANSLATION_DECODE_HEADS}" \
  --override "model.train_smpl_temporal_translation=${TRAIN_TEMPORAL_TRANSLATION}"

echo "========== SMPL Translation V2 training finished =========="
echo "Checkpoint: ${OUTPUT_DIR}/checkpoint_latest.pt"
