#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
DATA_ROOT="${DATA_ROOT:-/home/zhw/xyb_space}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_nlf_provider.yaml}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-6}"

SPLIT="${SPLIT:-Training}"
START_INDEX="${START_INDEX:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
IMAGE_PATH="${IMAGE_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/debug/gt_smpl_scale_teacher_data}"

SOURCE="${SOURCE:-vertices}"
VIS_TOL_M="${VIS_TOL_M:-0.20}"
WINDOW="${WINDOW:-3}"
MAX_POINTS_PER_PERSON="${MAX_POINTS_PER_PERSON:-512}"
MIN_POINTS_PER_PERSON="${MIN_POINTS_PER_PERSON:-32}"
MIN_VISIBLE_POINTS="${MIN_VISIBLE_POINTS:-128}"
MAD_MULT="${MAD_MULT:-2.5}"
MAX_Z_M="${MAX_Z_M:-20.0}"
OVERLAY_MAX_POINTS_PER_PERSON="${OVERLAY_MAX_POINTS_PER_PERSON:-260}"
FILTERED_OVERLAY_MAX_POINTS="${FILTERED_OVERLAY_MAX_POINTS:-160}"

export DATA_ROOT

cd "${REPO_ROOT}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM boxes root: ${PREPROCESSED_ROOT}" >&2; exit 1; }

ARGS=(
  --path-config "${PATH_CONFIG}"
  --train-config "${TRAIN_CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --split "${SPLIT}"
  --start-index "${START_INDEX}"
  --num-samples "${NUM_SAMPLES}"
  --source "${SOURCE}"
  --visibility-tolerance-m "${VIS_TOL_M}"
  --window "${WINDOW}"
  --max-points-per-person "${MAX_POINTS_PER_PERSON}"
  --min-points-per-person "${MIN_POINTS_PER_PERSON}"
  --min-visible-points "${MIN_VISIBLE_POINTS}"
  --mad-multiplier "${MAD_MULT}"
  --max-z-m "${MAX_Z_M}"
  --overlay-max-points-per-person "${OVERLAY_MAX_POINTS_PER_PERSON}"
  --filtered-overlay-max-points "${FILTERED_OVERLAY_MAX_POINTS}"
  --override "datasets.bedlam_root=${BEDLAM_ROOT}"
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}"
  --override "data.require_depth=true"
  --override "data.require_boxes=true"
  --override "model.enable_camera=true"
  --override "model.enable_depth=true"
  --override "model.enable_smpl=false"
  --override "model.enable_hsi_refine=false"
)

if [[ -n "${IMAGE_PATH}" ]]; then
  ARGS+=(--image "${IMAGE_PATH}")
fi

echo "========== GT-SMPL scale teacher data check =========="
echo "Repo        : ${REPO_ROOT}"
echo "BEDLAM      : ${BEDLAM_ROOT}"
echo "Boxes       : ${PREPROCESSED_ROOT}"
echo "Image       : ${IMAGE_PATH:-<dataset index ${START_INDEX}>}"
echo "Num samples : ${NUM_SAMPLES}"
echo "Source      : ${SOURCE}"
echo "Vis tol     : ${VIS_TOL_M}"
echo "Window      : ${WINDOW}"
echo "Max z       : ${MAX_Z_M}"
echo "Output      : ${OUTPUT_DIR}"
echo "GPU visible : ${CUDA_VISIBLE_DEVICES_VALUE}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/diagnostics/check_gt_smpl_scale_teacher_data.py "${ARGS[@]}"

echo "========== GT-SMPL scale teacher data check finished =========="
echo "Summary: ${OUTPUT_DIR}/summary.json"
