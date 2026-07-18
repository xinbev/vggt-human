#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_stage3_grounding_gt.yaml}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
BOXES_ROOT="${BOXES_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
CONTACT_TEACHER_ROOT="${CONTACT_TEACHER_ROOT:-${REPO_ROOT}/outputs/preprocess/hsi_contact_teachers_v3_strict}"
VAL_SEQUENCE_MANIFEST="${VAL_SEQUENCE_MANIFEST:-${REPO_ROOT}/outputs/preprocess/hsi_sequence_split_v2/val_sequences.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/hsi_stage3_grounding_g0}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
MAX_BATCHES="${MAX_BATCHES:-100}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-8}"

[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${CONTACT_TEACHER_ROOT}" ]] || { echo "[ERROR] Missing contact teachers: ${CONTACT_TEACHER_ROOT}" >&2; exit 1; }
[[ -f "${VAL_SEQUENCE_MANIFEST}" ]] || { echo "[ERROR] Missing val manifest: ${VAL_SEQUENCE_MANIFEST}" >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

echo "========== G0: analytic grounding audit (no training) =========="
echo "GT geometry       : GT depth + GT K + perturbed GT SMPL"
echo "Contact teachers  : ${CONTACT_TEACHER_ROOT}"
echo "Output            : ${OUTPUT_DIR}/g0_metrics.json"
echo "GPU / batches     : ${CUDA_VISIBLE_DEVICES_VALUE} / ${MAX_BATCHES}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/eval/eval_hsi_grounding_g0.py \
  --path-config configs/path.yaml \
  --train-config "${TRAIN_CONFIG}" \
  --device cuda \
  --max-batches "${MAX_BATCHES}" \
  --output "${OUTPUT_DIR}/g0_metrics.json" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${BOXES_ROOT}" \
  --override "data.contact_teacher_root=${CONTACT_TEACHER_ROOT}" \
  --override "data.val_sequence_manifest=${VAL_SEQUENCE_MANIFEST}" \
  --override "data.num_workers=${NUM_WORKERS}" \
  --override "optim.batch_size=${BATCH_SIZE}"

echo "========== G0 passed: gate training is allowed =========="
