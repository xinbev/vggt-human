#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
BOXES_ROOT="${BOXES_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/preprocess/hsi_contact_teachers_v2}"
SEQUENCE_MANIFEST="${SEQUENCE_MANIFEST:-}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-${REPO_ROOT}/checkpoints/body_models/smpl}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-7}"
MAX_HUMANS="${MAX_HUMANS:-20}"
OVERWRITE="${OVERWRITE:-false}"

cd "${REPO_ROOT}"
[[ -f "${SMPL_MODEL_DIR}/smpl/SMPL_NEUTRAL.pkl" ]] || {
  echo "[ERROR] Missing SMPL model: ${SMPL_MODEL_DIR}/smpl/SMPL_NEUTRAL.pkl" >&2
  exit 1
}
ARGS=(
  --bedlam-root "${BEDLAM_ROOT}"
  --boxes-root "${BOXES_ROOT}"
  --smpl-model-dir "${SMPL_MODEL_DIR}"
  --output-root "${OUTPUT_ROOT}"
  --max-humans "${MAX_HUMANS}"
)
[[ -n "${SEQUENCE_MANIFEST}" ]] && ARGS+=(--sequence-manifest "${SEQUENCE_MANIFEST}")
[[ "${OVERWRITE}" == "true" ]] && ARGS+=(--overwrite)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/preprocess/prepare_hsi_contact_teachers.py "${ARGS[@]}"
