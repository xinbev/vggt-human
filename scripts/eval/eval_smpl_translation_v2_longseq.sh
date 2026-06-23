#!/usr/bin/env bash

set -euo pipefail

# Evaluate SMPL Translation V2 on long windows and, separately, on windows
# containing previously scanned bad frames.

REPO_ROOT="${REPO_ROOT:-/home/zhw/lab_users/xyb/home/projects/vggt-human}"
BEDLAM_ROOT="${BEDLAM_ROOT:-/home/zhw/xyb_space/bedlam/processed_bedlam}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${REPO_ROOT}/outputs/preprocess/bedlam_boxes}"
PATH_CONFIG="${PATH_CONFIG:-${REPO_ROOT}/configs/path.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/configs/train_smpl_translation_v2_longseq.yaml}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VGGT_CKPT="${VGGT_CKPT:-/home/zhw/lab_users/xyb/home/projects/vggt-omega/checkpoints/vggt_omega_1b_512.pt}"
SMPL_MODEL_DIR="${SMPL_MODEL_DIR:-/home/zhw/xyb_space/SAT-HMR/weights/smpl_data/}"
SMPL_CKPT="${SMPL_CKPT:-${REPO_ROOT}/outputs/train/smpl_translation_v2_longseq/checkpoint_latest.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/eval/smpl_translation_v2_longseq}"
BAD_FRAME_CSV="${BAD_FRAME_CSV:-${REPO_ROOT}/outputs/eval/hsi_bad_translation_scan_after_translation_ray_refine/bad_frame_person_rows.csv}"

NUM_FRAMES="${NUM_FRAMES:-27}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-300}"
BAD_MAX_SAMPLES="${BAD_MAX_SAMPLES:-0}"
START_INDEX="${START_INDEX:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SPLIT="${SPLIT:-Training}"
USE_GT_BOX_PRIOR="${USE_GT_BOX_PRIOR:-true}"
TOP_WORST="${TOP_WORST:-50}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

[[ -f "${PATH_CONFIG}" ]] || { echo "[ERROR] Missing path config: ${PATH_CONFIG}" >&2; exit 1; }
[[ -f "${TRAIN_CONFIG}" ]] || { echo "[ERROR] Missing train config: ${TRAIN_CONFIG}" >&2; exit 1; }
[[ -d "${BEDLAM_ROOT}" ]] || { echo "[ERROR] Missing BEDLAM root: ${BEDLAM_ROOT}" >&2; exit 1; }
[[ -d "${PREPROCESSED_ROOT}" ]] || { echo "[ERROR] Missing preprocessed boxes: ${PREPROCESSED_ROOT}" >&2; exit 1; }
[[ -f "${VGGT_CKPT}" ]] || { echo "[ERROR] Missing VGGT checkpoint: ${VGGT_CKPT}" >&2; exit 1; }
[[ -f "${SMPL_CKPT}" ]] || { echo "[ERROR] Missing SMPL checkpoint: ${SMPL_CKPT}" >&2; exit 1; }
[[ -d "${SMPL_MODEL_DIR}" ]] || { echo "[ERROR] Missing SMPL model dir: ${SMPL_MODEL_DIR}" >&2; exit 1; }

PRIOR_ARGS=()
if [[ "${USE_GT_BOX_PRIOR}" == "true" ]]; then
  PRIOR_ARGS+=(--use-gt-box-prior)
fi

COMMON_OVERRIDES=(
  --override "assets.smpl_model_dir=${SMPL_MODEL_DIR}"
  --override "datasets.bedlam_root=${BEDLAM_ROOT}"
  --override "datasets.bedlam_boxes_root=${PREPROCESSED_ROOT}"
  --override "data.sequence_length=${NUM_FRAMES}"
  --override "data.stride=${FRAME_STRIDE}"
  --override "data.require_boxes=true"
  --override "data.require_smpl=true"
  --override "data.require_depth=false"
  --override "model.enable_camera=true"
  --override "model.enable_depth=false"
  --override "model.enable_hsi_refine=false"
  --override "model.smpl_translation_output_mode=ray_offset_depth"
  --override "model.smpl_enable_temporal_translation=true"
  --override "model.smpl_temporal_translation_use_world=true"
  --override "model.smpl_enable_translation_refine=false"
)

echo "========== SMPL Translation V2 long-window eval =========="
echo "Checkpoint : ${SMPL_CKPT}"
echo "Frames     : ${NUM_FRAMES}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Bad CSV    : ${BAD_FRAME_CSV}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/eval/evaluate_smpl_translation_metrics.py \
  --checkpoint "${SMPL_CKPT}" \
  --baseline-checkpoint "${VGGT_CKPT}" \
  --path-config "${PATH_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --output-dir "${OUTPUT_ROOT}/all_windows_${NUM_FRAMES}f" \
  --split "${SPLIT}" \
  --max-samples "${MAX_SAMPLES}" \
  --start-index "${START_INDEX}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --top-worst "${TOP_WORST}" \
  --log-interval "${LOG_INTERVAL}" \
  "${PRIOR_ARGS[@]}" \
  "${COMMON_OVERRIDES[@]}"

if [[ -f "${BAD_FRAME_CSV}" ]]; then
  echo "========== SMPL Translation V2 bad-frame focused eval =========="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python scripts/eval/evaluate_smpl_translation_metrics.py \
    --checkpoint "${SMPL_CKPT}" \
    --baseline-checkpoint "${VGGT_CKPT}" \
    --path-config "${PATH_CONFIG}" \
    --train-config "${TRAIN_CONFIG}" \
    --output-dir "${OUTPUT_ROOT}/bad_frame_windows_${NUM_FRAMES}f" \
    --split "${SPLIT}" \
    --max-samples "${BAD_MAX_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --top-worst "${TOP_WORST}" \
    --log-interval "${LOG_INTERVAL}" \
    --subset-frame-csv "${BAD_FRAME_CSV}" \
    --subset-sequence-column sequence_name \
    --subset-frame-column frame_name \
    "${PRIOR_ARGS[@]}" \
    "${COMMON_OVERRIDES[@]}"
else
  echo "[WARN] Bad-frame CSV not found; skipped focused eval: ${BAD_FRAME_CSV}"
fi

echo "========== SMPL Translation V2 eval finished =========="
echo "All metrics : ${OUTPUT_ROOT}/all_windows_${NUM_FRAMES}f/smpl_translation_metrics.json"
echo "Bad metrics : ${OUTPUT_ROOT}/bad_frame_windows_${NUM_FRAMES}f/smpl_translation_metrics.json"
echo "Bad rows    : ${OUTPUT_ROOT}/bad_frame_windows_${NUM_FRAMES}f/smpl_translation_person_metrics.csv"
