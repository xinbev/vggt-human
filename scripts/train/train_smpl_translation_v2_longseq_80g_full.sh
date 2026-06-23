#!/usr/bin/env bash

set -euo pipefail

# One-command 80GB-GPU schedule for SMPL Translation V2.
# Runs: 27f memory smoke test -> geometry seed warmup -> temporal 24f
# training -> 27f polish -> full long-window and bad-frame focused eval.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_translation_v2_longseq.yaml}"
VGGT_CKPT="${VGGT_CKPT:-/home/zhw/lab_users/xyb/home/projects/vggt-omega/checkpoints/vggt_omega_1b_512.pt}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-/home/zhw/xyb_space/SAT-HMR/weights/smpl_data/}"
BASE_INIT_CKPT="${INIT_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_temporal_after_translation_ray_refine/stage2_human_momentum_no_worse/checkpoint_latest.pt}"

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-6}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${REPO_ROOT}/outputs/train/smpl_translation_v2_longseq_80g}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${REPO_ROOT}/outputs/eval/smpl_translation_v2_longseq_80g}"

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
MAX_HUMANS="${MAX_HUMANS:-20}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"

RUN_MEMCHECK="${RUN_MEMCHECK:-true}"
RUN_STAGE_A="${RUN_STAGE_A:-true}"
RUN_STAGE_B="${RUN_STAGE_B:-true}"
RUN_STAGE_C="${RUN_STAGE_C:-true}"
RUN_EVAL="${RUN_EVAL:-true}"

MEMCHECK_NUM_FRAMES="${MEMCHECK_NUM_FRAMES:-27}"
MEMCHECK_EPOCHS="${MEMCHECK_EPOCHS:-1}"
MEMCHECK_LR="${MEMCHECK_LR:-0.000003}"
MEMCHECK_SUBSET_MAX_SAMPLES="${MEMCHECK_SUBSET_MAX_SAMPLES:-16}"

STAGE_A_NUM_FRAMES="${STAGE_A_NUM_FRAMES:-16}"
STAGE_A_EPOCHS="${STAGE_A_EPOCHS:-4}"
STAGE_A_LR="${STAGE_A_LR:-0.000008}"

STAGE_B_NUM_FRAMES="${STAGE_B_NUM_FRAMES:-24}"
STAGE_B_EPOCHS="${STAGE_B_EPOCHS:-6}"
STAGE_B_LR="${STAGE_B_LR:-0.000003}"
STAGE_B_TEMPORAL_MAX_VELOCITY_DELTA_M="${STAGE_B_TEMPORAL_MAX_VELOCITY_DELTA_M:-0.35}"
STAGE_B_TEMPORAL_GATE_BIAS="${STAGE_B_TEMPORAL_GATE_BIAS:-2.5}"

STAGE_C_NUM_FRAMES="${STAGE_C_NUM_FRAMES:-27}"
STAGE_C_EPOCHS="${STAGE_C_EPOCHS:-2}"
STAGE_C_LR="${STAGE_C_LR:-0.0000015}"
STAGE_C_TEMPORAL_MAX_VELOCITY_DELTA_M="${STAGE_C_TEMPORAL_MAX_VELOCITY_DELTA_M:-0.25}"
STAGE_C_TEMPORAL_GATE_BIAS="${STAGE_C_TEMPORAL_GATE_BIAS:-3.0}"

EVAL_NUM_FRAMES="${EVAL_NUM_FRAMES:-${STAGE_C_NUM_FRAMES}}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
EVAL_BAD_MAX_SAMPLES="${EVAL_BAD_MAX_SAMPLES:-0}"

TRAIN_SCRIPT="${REPO_ROOT}/scripts/train/train_smpl_translation_v2_longseq.sh"
EVAL_SCRIPT="${REPO_ROOT}/scripts/eval/eval_smpl_translation_v2_longseq.sh"

STAGE_A_DIR="${BASE_OUTPUT_DIR}/stageA_seed_${STAGE_A_NUM_FRAMES}f"
STAGE_B_DIR="${BASE_OUTPUT_DIR}/stageB_temporal_${STAGE_B_NUM_FRAMES}f"
STAGE_C_DIR="${BASE_OUTPUT_DIR}/stageC_temporal_${STAGE_C_NUM_FRAMES}f_polish"
MEMCHECK_DIR="${BASE_OUTPUT_DIR}/memcheck_${MEMCHECK_NUM_FRAMES}f"

STAGE_B_INIT_CKPT="${STAGE_B_INIT_CKPT:-${STAGE_A_DIR}/checkpoint_latest.pt}"
STAGE_C_INIT_CKPT="${STAGE_C_INIT_CKPT:-${STAGE_B_DIR}/checkpoint_latest.pt}"
FINAL_CKPT="${FINAL_CKPT:-${STAGE_C_DIR}/checkpoint_latest.pt}"

export REPO_ROOT
export BEDLAM_ROOT
export PREPROCESSED_ROOT
export PATH_CONFIG
export TRAIN_CONFIG
export VGGT_CKPT
export SMPL_MODEL_DIR
export CUDA_VISIBLE_DEVICES_VALUE
export PYTORCH_CUDA_ALLOC_CONF

cd "${REPO_ROOT}"

echo "========== SMPL Translation V2 80GB full schedule =========="
echo "GPU        : ${CUDA_VISIBLE_DEVICES_VALUE}"
echo "Base init  : ${BASE_INIT_CKPT}"
echo "Output root: ${BASE_OUTPUT_DIR}"
echo "Eval root  : ${EVAL_OUTPUT_ROOT}"
echo "Batch      : ${BATCH_SIZE}"
echo "Workers    : train=${NUM_WORKERS} eval=${EVAL_NUM_WORKERS}"

if [[ "${RUN_MEMCHECK}" == "true" ]]; then
  echo "========== Stage 0: memory smoke test (${MEMCHECK_NUM_FRAMES}f) =========="
  INIT_CKPT="${BASE_INIT_CKPT}" \
  OUTPUT_DIR="${MEMCHECK_DIR}" \
  NUM_FRAMES="${MEMCHECK_NUM_FRAMES}" \
  FRAME_STRIDE="${FRAME_STRIDE}" \
  MAX_HUMANS="${MAX_HUMANS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  EPOCHS="${MEMCHECK_EPOCHS}" \
  LR="${MEMCHECK_LR}" \
  SUBSET_MAX_SAMPLES="${MEMCHECK_SUBSET_MAX_SAMPLES}" \
  RESET_EPOCH=true \
  SAVE_FINAL=false \
  SAVE_EPOCH_CHECKPOINT=false \
  SAVE_LATEST=true \
  bash "${TRAIN_SCRIPT}"
fi

if [[ "${RUN_STAGE_A}" == "true" ]]; then
  echo "========== Stage A: geometry seed warmup (${STAGE_A_NUM_FRAMES}f) =========="
  INIT_CKPT="${BASE_INIT_CKPT}" \
  OUTPUT_DIR="${STAGE_A_DIR}" \
  NUM_FRAMES="${STAGE_A_NUM_FRAMES}" \
  FRAME_STRIDE="${FRAME_STRIDE}" \
  MAX_HUMANS="${MAX_HUMANS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  EPOCHS="${STAGE_A_EPOCHS}" \
  LR="${STAGE_A_LR}" \
  ENABLE_TEMPORAL_TRANSLATION=false \
  TRAIN_TEMPORAL_TRANSLATION=false \
  TRAIN_TRANSLATION_DECODE_HEADS=true \
  RESET_EPOCH=true \
  SAVE_FINAL=true \
  SAVE_EPOCH_CHECKPOINT=false \
  SAVE_LATEST=true \
  bash "${TRAIN_SCRIPT}"
fi

if [[ "${RUN_STAGE_B}" == "true" ]]; then
  echo "========== Stage B: temporal trajectory training (${STAGE_B_NUM_FRAMES}f) =========="
  INIT_CKPT="${STAGE_B_INIT_CKPT}" \
  OUTPUT_DIR="${STAGE_B_DIR}" \
  NUM_FRAMES="${STAGE_B_NUM_FRAMES}" \
  FRAME_STRIDE="${FRAME_STRIDE}" \
  MAX_HUMANS="${MAX_HUMANS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  EPOCHS="${STAGE_B_EPOCHS}" \
  LR="${STAGE_B_LR}" \
  ENABLE_TEMPORAL_TRANSLATION=true \
  TRAIN_TEMPORAL_TRANSLATION=true \
  TRAIN_TRANSLATION_DECODE_HEADS=true \
  TEMPORAL_MAX_VELOCITY_DELTA_M="${STAGE_B_TEMPORAL_MAX_VELOCITY_DELTA_M}" \
  TEMPORAL_GATE_BIAS="${STAGE_B_TEMPORAL_GATE_BIAS}" \
  RESET_EPOCH=true \
  SAVE_FINAL=true \
  SAVE_EPOCH_CHECKPOINT=false \
  SAVE_LATEST=true \
  bash "${TRAIN_SCRIPT}"
fi

if [[ "${RUN_STAGE_C}" == "true" ]]; then
  echo "========== Stage C: 27f temporal polish (${STAGE_C_NUM_FRAMES}f) =========="
  INIT_CKPT="${STAGE_C_INIT_CKPT}" \
  OUTPUT_DIR="${STAGE_C_DIR}" \
  NUM_FRAMES="${STAGE_C_NUM_FRAMES}" \
  FRAME_STRIDE="${FRAME_STRIDE}" \
  MAX_HUMANS="${MAX_HUMANS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  EPOCHS="${STAGE_C_EPOCHS}" \
  LR="${STAGE_C_LR}" \
  ENABLE_TEMPORAL_TRANSLATION=true \
  TRAIN_TEMPORAL_TRANSLATION=true \
  TRAIN_TRANSLATION_DECODE_HEADS=true \
  TEMPORAL_MAX_VELOCITY_DELTA_M="${STAGE_C_TEMPORAL_MAX_VELOCITY_DELTA_M}" \
  TEMPORAL_GATE_BIAS="${STAGE_C_TEMPORAL_GATE_BIAS}" \
  RESET_EPOCH=true \
  SAVE_FINAL=true \
  SAVE_EPOCH_CHECKPOINT=false \
  SAVE_LATEST=true \
  bash "${TRAIN_SCRIPT}"
fi

if [[ "${RUN_EVAL}" == "true" ]]; then
  echo "========== Final eval (${EVAL_NUM_FRAMES}f) =========="
  SMPL_CKPT="${FINAL_CKPT}" \
  OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}" \
  NUM_FRAMES="${EVAL_NUM_FRAMES}" \
  FRAME_STRIDE="${FRAME_STRIDE}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  NUM_WORKERS="${EVAL_NUM_WORKERS}" \
  MAX_SAMPLES="${EVAL_MAX_SAMPLES}" \
  BAD_MAX_SAMPLES="${EVAL_BAD_MAX_SAMPLES}" \
  bash "${EVAL_SCRIPT}"
fi

echo "========== 80GB schedule finished =========="
echo "Final checkpoint: ${FINAL_CKPT}"
echo "Eval metrics    : ${EVAL_OUTPUT_ROOT}/all_windows_${EVAL_NUM_FRAMES}f/smpl_translation_metrics.json"
echo "Bad-frame CSV   : ${EVAL_OUTPUT_ROOT}/bad_frame_windows_${EVAL_NUM_FRAMES}f/smpl_translation_person_metrics.csv"
