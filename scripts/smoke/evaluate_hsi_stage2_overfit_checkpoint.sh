#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
DATA_ROOT="${DATA_ROOT:-/home/zhw/xyb_space}"
BEDLAM_ROOT="${BEDLAM_ROOT:-${DATA_ROOT}/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
SPLIT_ROOT="${SPLIT_ROOT:-${REPO_ROOT}/outputs/preprocess/hsi_sequence_split_v2}"
OVERFIT_CKPT="${OVERFIT_CKPT:?Set OVERFIT_CKPT to the completed 1000-step checkpoint_top01.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/debug/hsi_curriculum_v2_stage2a_overfit64_final_eval}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"

cd "${REPO_ROOT}"

RUN_STAGES=2A \
DATA_ROOT="${DATA_ROOT}" \
BEDLAM_ROOT="${BEDLAM_ROOT}" \
PREPROCESSED_ROOT="${PREPROCESSED_ROOT}" \
TRAIN_SEQUENCE_MANIFEST="${SPLIT_ROOT}/train_sequences.txt" \
VAL_SEQUENCE_MANIFEST="${SPLIT_ROOT}/train_sequences.txt" \
SUBSET_INDICES_CSV="${SPLIT_ROOT}/overfit64_indices.csv" \
SUBSET_REPEAT=1 \
SUBSET_MAX_SAMPLES=64 \
SUBSET_APPLY_TO_VAL=true \
STAGE1_CKPT="${OVERFIT_CKPT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
NUM_WORKERS="${NUM_WORKERS:-16}" \
NLF_INTERNAL_BATCH_SIZE="${NLF_INTERNAL_BATCH_SIZE:-192}" \
MAX_HUMANS="${MAX_HUMANS:-20}" \
BATCH_SIZE_2A="${BATCH_SIZE_2A:-24}" \
EPOCHS_2A=1 \
LR_2A=0 \
MAX_STEPS_PER_EPOCH=1 \
MAX_VAL_STEPS=3 \
bash scripts/train/train_smpl_hsi_scale_trans_contact_curriculum.sh

OUTPUT_DIR="${OUTPUT_ROOT}/stage2a_gt_transl" \
GATE_STAGE=stage2 \
GATE_MODE=overfit \
bash scripts/smoke/inspect_hsi_curriculum_metrics.sh
