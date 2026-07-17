#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
BOXES_ROOT="${BOXES_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
CONTACT_TEACHER_ROOT="${CONTACT_TEACHER_ROOT:-${REPO_ROOT}/outputs/preprocess/hsi_contact_teachers_v3_strict}"
SEQUENCE_MANIFEST="${SEQUENCE_MANIFEST:-${REPO_ROOT}/outputs/preprocess/hsi_sequence_split_v2/val_sequences.txt}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-${REPO_ROOT}/checkpoints/body_models/smpl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vis/hsi_curriculum_v2_data_audit}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
NUM_SAMPLES="${NUM_SAMPLES:-24}"

cd "${REPO_ROOT}"
[[ -f "${SMPL_MODEL_DIR}/smpl/SMPL_NEUTRAL.pkl" ]] || {
  echo "[ERROR] Missing SMPL model: ${SMPL_MODEL_DIR}/smpl/SMPL_NEUTRAL.pkl" >&2
  exit 1
}
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/vis/visualize_hsi_curriculum_data_audit.py \
  --bedlam-root "${BEDLAM_ROOT}" \
  --boxes-root "${BOXES_ROOT}" \
  --contact-teacher-root "${CONTACT_TEACHER_ROOT}" \
  --sequence-manifest "${SEQUENCE_MANIFEST}" \
  --smpl-model-dir "${SMPL_MODEL_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-samples "${NUM_SAMPLES}"
