#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_hsi_refine.yaml}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VGGT_CKPT="${VGGT_CKPT:-/home/zhw/lab_users/xyb/home/projects/vggt-omega/checkpoints/vggt_omega_1b_512.pt}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-/home/zhw/xyb_space/SAT-HMR/weights/smpl_data/}"
SMPL_CKPT="${SMPL_CKPT:-${REPO_ROOT}/outputs/train/smpl_hsi_refine_20q/checkpoint_epoch_0121.pt}"
IMAGE_PATH="${IMAGE_PATH:-${BEDLAM_ROOT}/Training/20221013_3_250_batch01hand_orbit_bigOffice_seq_000000/rgb/seq_000000_0000.png}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/hsi_sequence_person_diagnostics_0121}"

NUM_FRAMES="${NUM_FRAMES:-27}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
START_INDEX="${START_INDEX:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.10}"
SPLIT="${SPLIT:-Training}"
USE_GT_BOX_PRIOR="${USE_GT_BOX_PRIOR:-true}"
MATCH_SOURCE="${MATCH_SOURCE:-base}"
INTRINSICS_SOURCE="${INTRINSICS_SOURCE:-gt}"
DEPTH_MAX_M="${DEPTH_MAX_M:-30.0}"
ROI_EXPAND="${ROI_EXPAND:-0.75}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing preprocessed boxes: ${PREPROCESSED_ROOT}" >&2; exit 1; }
[[ -f "${VGGT_CKPT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${VGGT_CKPT}" >&2; exit 1; }
[[ -f "${SMPL_CKPT}" ]] || { echo "[ERROR] Missing SMPL checkpoint: ${SMPL_CKPT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }
if [[ -n "${IMAGE_PATH}" ]]; then
  [[ -f "${IMAGE_PATH}" ]] || { echo "[ERROR] Missing start image: ${IMAGE_PATH}" >&2; exit 1; }
fi

echo "========== HSI sequence per-person diagnostics =========="
echo "Checkpoint  : ${SMPL_CKPT}"
echo "Train config: ${TRAIN_CONFIG}"
echo "Image       : ${IMAGE_PATH}"
echo "Output      : ${OUTPUT_DIR}"
echo "Frames      : ${NUM_FRAMES}"
echo "Stride      : ${FRAME_STRIDE}"
echo "Samples     : ${MAX_SAMPLES}"
echo "GT prior    : ${USE_GT_BOX_PRIOR}"
echo "Match source: ${MATCH_SOURCE}"
echo "K source    : ${INTRINSICS_SOURCE}"
echo "Depth max   : ${DEPTH_MAX_M}m"

PRIOR_ARGS=()
if [[ "${USE_GT_BOX_PRIOR}" == "true" ]]; then
  PRIOR_ARGS+=(--use-gt-box-prior)
fi

IMAGE_ARGS=()
if [[ -n "${IMAGE_PATH}" ]]; then
  IMAGE_ARGS+=(--image "${IMAGE_PATH}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/eval/evaluate_hsi_sequence_person_diagnostics.py \
  --checkpoint "${SMPL_CKPT}" \
  --baseline-checkpoint "${VGGT_CKPT}" \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --num-frames "${NUM_FRAMES}" \
  --frame-stride "${FRAME_STRIDE}" \
  --max-samples "${MAX_SAMPLES}" \
  --start-index "${START_INDEX}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --conf-threshold "${CONF_THRESHOLD}" \
  --match-source "${MATCH_SOURCE}" \
  --intrinsics-source "${INTRINSICS_SOURCE}" \
  --depth-max-m "${DEPTH_MAX_M}" \
  --roi-expand "${ROI_EXPAND}" \
  "${PRIOR_ARGS[@]}" \
  "${IMAGE_ARGS[@]}" \
  --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}" \
  --override "datasets.bedlam_root=${BEDLAM_ROOT}" \
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}" \
  --override "data.sequence_length=${NUM_FRAMES}" \
  --override "data.stride=${FRAME_STRIDE}" \
  --override "data.val_split=" \
  --override "data.require_boxes=true" \
  --override "data.require_depth=true" \
  --override "model.enable_camera=true" \
  --override "model.enable_depth=true" \
  --override "model.enable_hsi_refine=true"

echo "========== HSI sequence per-person diagnostics finished =========="
echo "Metrics json : ${OUTPUT_DIR}/hsi_sequence_person_diagnostics.json"
echo "Person csv   : ${OUTPUT_DIR}/hsi_sequence_person_summary.csv"
echo "Frame csv    : ${OUTPUT_DIR}/hsi_sequence_frame_person_metrics.csv"
echo "Depth csv    : ${OUTPUT_DIR}/hsi_sequence_frame_depth_metrics.csv"
echo "Temporal csv : ${OUTPUT_DIR}/hsi_sequence_person_temporal_metrics.csv"
